"""Gate L, part 1: the coefficient generator on the bath phase-space grid.

The last test in this file is the one that matters: it ties the numerical
grid operator to the symbolic Gate J object, so the two layers cannot drift.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from mqcle.coefficient_operator import (build_grid, coefficient_rhs,
                                        hermiticity_error, physical_matrices,
                                        populations_from_rho,
                                        spectral_derivative, trace)
from mqcle.models import ConstantCouplingModel, build_model
from mqcle.propagation import rk4_step
from mqcle.types import TraceConvention
from mqcle.validation import build_symbolic_setup, qcle_reference_rhs

HBAR = 1.0
MASS = 2000.0


def _gaussian_field(grid, Rc=-1.0, Pc=12.0, sR=0.7, sP=3.0, dressed=True):
    """A Hermitian analytic coefficient field, well inside the box."""
    RR, PP = grid.mesh()
    g = np.exp(-((RR - Rc) ** 2) / (2 * sR**2) - ((PP - Pc) ** 2) / (2 * sP**2))
    if dressed:
        g = g * (1.0 + 0.1 * RR + 0.05 * PP)
    M = np.array([[1.0 + 0j, 0.3 + 0.2j], [0.3 - 0.2j, 0.5 + 0j]])
    return g[:, :, None, None] * M[None, None, :, :]


# --------------------------------------------------------------------------
# 1. spectral differentiation
# --------------------------------------------------------------------------
def test_spectral_derivative_of_a_gaussian_is_analytic():
    grid = build_grid(-24.0, 24.0, 256, -40.0, 80.0, 256)
    RR, PP = grid.mesh()
    sR, sP, Rc, Pc = 1.3, 6.0, -2.0, 20.0
    f = np.exp(-((RR - Rc) ** 2) / (2 * sR**2) - ((PP - Pc) ** 2) / (2 * sP**2))

    dR_num = spectral_derivative(f.astype(complex), grid.kR, axis=0).real
    dP_num = spectral_derivative(f.astype(complex), grid.kP, axis=1).real
    dR_ana = -(RR - Rc) / sR**2 * f
    dP_ana = -(PP - Pc) / sP**2 * f

    assert np.max(np.abs(dR_num - dR_ana)) < 1e-10
    assert np.max(np.abs(dP_num - dP_ana)) < 1e-10


def test_spectral_derivative_is_exact_on_a_box_harmonic():
    grid = build_grid(0.0, 2.0, 32, 0.0, 3.0, 16)
    RR, _ = grid.mesh()
    f = np.exp(2j * np.pi * 3 * RR / grid.L_R)
    d = spectral_derivative(f, grid.kR, axis=0)
    assert np.max(np.abs(d - (2j * np.pi * 3 / grid.L_R) * f)) < 1e-12


# --------------------------------------------------------------------------
# 2. free electronic rotation
# --------------------------------------------------------------------------
def test_constant_h_gives_constant_populations_and_exact_coherence_rate():
    """grad_R h = 0 and V_b = 0: populations frozen, coherence at (E1-E0)/hbar."""
    model = ConstantCouplingModel(a0=0.02, b0=-0.01, c0=0.005, k=0.0, mass=MASS)
    grid = build_grid(-20.0, 20.0, 96, -30.0, 70.0, 96)
    h0 = model.h(np.zeros((1, 1)))[0]
    evals = np.linalg.eigvalsh(h0)
    omega = (evals[1] - evals[0]) / HBAR

    # start in the eigenbasis so the coherence is a single pure phase
    _, U = np.linalg.eigh(h0)
    rho_ad = np.array([[0.6 + 0j, 0.4 + 0j], [0.4 + 0j, 0.4 + 0j]])
    M = U @ rho_ad @ U.conj().T
    RR, PP = grid.mesh()
    g = np.exp(-((RR + 1.0) ** 2) / (2 * 0.9**2) - ((PP - 15.0) ** 2) / (2 * 4.0**2))
    rho = g[:, :, None, None] * M[None, None, :, :]

    # propagate with RK4; the bath drift is present but cannot mix the states
    from mqcle.propagation import rk4_step
    cache = physical_matrices(model, grid, TraceConvention.TRACELESS)
    dt, n_steps = 0.5, 200
    pops0 = populations_from_rho(rho, grid)
    state = rho.copy()
    for _ in range(n_steps):
        state = rk4_step(state, dt, grid, model, HBAR,
                         TraceConvention.TRACELESS, cache)

    # the eigenbasis populations are the conserved ones; the diabatic ones
    # oscillate, because the initial state carries an eigenbasis coherence
    ad0 = (U.conj().T @ np.einsum("ijab->ab", rho) @ U) * grid.cell_area
    ad = (U.conj().T @ np.einsum("ijab->ab", state) @ U) * grid.cell_area
    assert np.max(np.abs(np.real(np.diag(ad)) - np.real(np.diag(ad0)))) \
        / abs(np.trace(ad0)) < 1e-12
    # total trace is conserved regardless of basis
    assert abs(np.sum(populations_from_rho(state, grid))
               - np.sum(pops0)) / np.sum(pops0) < 1e-12

    # In the eigenbasis, d_t rho_01 = -(i/hbar)(E_0-E_1) rho_01 = +i omega rho_01,
    # so the coherence is the initial one times exp(+i omega t).
    t = dt * n_steps
    predicted = ad0[0, 1] * np.exp(+1j * omega * t)
    assert abs(ad[0, 1] - predicted) / abs(ad0[0, 1]) < 1e-8


# --------------------------------------------------------------------------
# 3. conservation laws of the right-hand side
# --------------------------------------------------------------------------
@pytest.mark.parametrize("convention",
                         [TraceConvention.TRACELESS, TraceConvention.FULL])
def test_rhs_has_zero_phase_space_trace(convention):
    model = build_model({"name": "tully_dual_avoided_crossing"}, mass=MASS)
    grid = build_grid(-25.0, 25.0, 128, -40.0, 80.0, 128)
    rho = _gaussian_field(grid)
    r = coefficient_rhs(rho, grid, model, HBAR, convention)
    scale = abs(trace(rho, grid))
    assert abs(trace(r, grid)) / scale < 1e-12


def test_rhs_preserves_hermiticity():
    model = build_model({"name": "tully_dual_avoided_crossing"}, mass=MASS)
    grid = build_grid(-25.0, 25.0, 128, -40.0, 80.0, 128)
    rho = _gaussian_field(grid)
    assert hermiticity_error(rho) < 1e-15
    r = coefficient_rhs(rho, grid, model, HBAR)
    # d/dt of a Hermitian field must itself be Hermitian
    assert hermiticity_error(r) / np.max(np.abs(r)) < 1e-12


def test_propagation_keeps_hermiticity_and_trace():
    from mqcle.propagation import rk4_step
    model = build_model({"name": "tully_dual_avoided_crossing"}, mass=MASS)
    grid = build_grid(-25.0, 25.0, 128, -40.0, 80.0, 128)
    rho = _gaussian_field(grid)
    cache = physical_matrices(model, grid, TraceConvention.TRACELESS)
    tr0 = trace(rho, grid)
    state = rho.copy()
    for _ in range(50):
        state = rk4_step(state, 0.5, grid, model, HBAR,
                         TraceConvention.TRACELESS, cache)
    assert hermiticity_error(state) / np.max(np.abs(state)) < 1e-12
    assert abs(trace(state, grid) - tr0) / abs(tr0) < 1e-10


def test_operator_is_independent_of_the_trace_convention():
    """The physical h is reassembled from the split, so both conventions agree."""
    model = build_model({"name": "tully_dual_avoided_crossing"}, mass=MASS)
    grid = build_grid(-25.0, 25.0, 96, -40.0, 80.0, 96)
    rho = _gaussian_field(grid)
    a = coefficient_rhs(rho, grid, model, HBAR, TraceConvention.TRACELESS)
    b = coefficient_rhs(rho, grid, model, HBAR, TraceConvention.FULL)
    assert np.max(np.abs(a - b)) < 1e-14


# --------------------------------------------------------------------------
# 4. the tie to Gate J
# --------------------------------------------------------------------------
def test_grid_operator_matches_the_symbolic_qcle_reference():
    r"""Evaluate the *same* symbolic RHS that Gate J uses, at grid points.

    ``qcle_reference_rhs`` is the single definition shared by
    ``verify_qcle_coefficient_correspondence`` (Gate J, symbolic) and this
    test (Gate L, numerical).  If the grid operator and the symbolic operator
    ever disagree, one of the two gates is wrong.
    """
    s = build_symbolic_setup(2, 1, "traceless")
    R0, P0 = s.R[0], s.P[0]
    A, B, C, D, E0 = 0.1, 0.28, 0.015, 0.06, 0.05
    Rc, Pc, sR, sP = -1.0, 12.0, 0.7, 3.0
    g = sp.exp(-(R0 - Rc) ** 2 / (2 * sR**2) - (P0 - Pc) ** 2 / (2 * sP**2))
    dress = 1 + sp.Rational(1, 10) * R0 + sp.Rational(1, 20) * P0
    Mre = [[1.0, 0.3], [0.3, 0.5]]
    Mim = [[0.0, 0.2], [-0.2, 0.0]]

    subs = {
        sp.Function("h00"): sp.Lambda(R0, sp.Integer(0)),
        sp.Function("h11"): sp.Lambda(R0, -A * sp.exp(-B * R0**2) + E0),
        sp.Function("h01"): sp.Lambda(R0, C * sp.exp(-D * R0**2)),
        sp.Function("Vb"): sp.Lambda(R0, sp.Integer(0)),
    }
    for a in range(2):
        for b in range(2):
            subs[sp.Function(f"rho{a}{b}")] = sp.Lambda(
                (R0, P0), (Mre[a][b] + sp.I * Mim[a][b]) * dress * g)

    expr = qcle_reference_rhs(s).subs(subs).doit()
    expr = expr.subs({s.hbar: HBAR, s.mass: MASS})
    f_sym = sp.lambdify((R0, P0), expr, "numpy")

    model = build_model({"name": "tully_dual_avoided_crossing",
                         "A": A, "B": B, "C": C, "D": D, "E0": E0}, mass=MASS)
    grid = build_grid(-24.0, 24.0, 512, -36.0, 60.0, 512)
    rho = _gaussian_field(grid, Rc=Rc, Pc=Pc, sR=sR, sP=sP, dressed=True)
    num = coefficient_rhs(rho, grid, model, HBAR, TraceConvention.TRACELESS)

    # sample where the field is not numerically negligible
    idx_R = [np.argmin(np.abs(grid.R - v)) for v in (-1.6, -1.0, -0.4, 0.2)]
    idx_P = [np.argmin(np.abs(grid.P - v)) for v in (9.0, 12.0, 15.0)]
    worst = 0.0
    for i in idx_R:
        for j in idx_P:
            ref = np.array(f_sym(grid.R[i], grid.P[j]), dtype=complex)
            worst = max(worst, np.max(np.abs(num[i, j] - ref)))
    assert worst < 1e-10, f"grid vs symbolic mismatch {worst:.3e}"


# --------------------------------------------------------------------------
# 5. the Strang integrator: exact invariants, and agreement with RK4
# --------------------------------------------------------------------------
def test_strang_conserves_trace_and_hermiticity_exactly():
    from mqcle.propagation import StrangPropagator, initial_coefficient_field
    model = build_model({"name": "tully_dual_avoided_crossing"}, mass=MASS)
    grid = build_grid(-45.0, 45.0, 256, 18.0, 62.0, 128)
    rho = initial_coefficient_field(grid, -15.0, 40.0, 0.5)
    tr0 = trace(rho, grid)
    n_steps = 200
    prop = StrangPropagator(grid, model, dt=2.0)
    for _ in range(n_steps):
        rho = prop.step(rho)

    # The splitting conserves the trace *identically*: at k_P = 0 the two
    # exponentials are mutual adjoints, and advection leaves the k_R = 0 mode
    # untouched.  What remains is round-off in the FFT round trips, which
    # accumulates as a random walk.  Measured drift over this grid:
    #   50 steps 8.4e-15 | 200 steps 1.5e-14 | 800 steps 3.6e-14
    # i.e. a factor 4.2 for a factor 16 in step count -- sqrt(n), not n.
    # A fixed 1e-14 would therefore be a wrong tolerance, not a strict one.
    tol = 3e-15 * np.sqrt(n_steps)
    assert abs(trace(rho, grid) - tr0) / abs(tr0) < tol
    assert hermiticity_error(rho) / np.max(np.abs(rho)) < tol


def test_strang_is_stable_where_rk4_is_not():
    """dt above the RK4 limit: RK4 must refuse, Strang must stay bounded."""
    from mqcle.propagation import (StrangPropagator, initial_coefficient_field,
                                   propagate_coefficients)
    from mqcle.coefficient_operator import spectral_radius_estimate
    model = build_model({"name": "tully_dual_avoided_crossing"}, mass=MASS)
    grid = build_grid(-45.0, 45.0, 1024, 18.0, 62.0, 256)
    rad = spectral_radius_estimate(grid, model)
    dt = 2.0 * rad["max_dt_rk4"]
    rho = initial_coefficient_field(grid, -15.0, 40.0, 0.25)

    with pytest.raises(ValueError, match="RK4 is unstable"):
        propagate_coefficients(rho, grid, model, dt, 1, integrator="rk4")

    prop = StrangPropagator(grid, model, dt=dt)
    state = rho.copy()
    for _ in range(50):
        state = prop.step(state)
    assert np.max(np.abs(state)) < 10.0 * np.max(np.abs(rho))


def test_fused_strang_advance_matches_repeated_steps_and_preserves_input():
    """Fusing adjacent exact half-flows changes scheduling, not the method."""
    from mqcle.propagation import (StrangPropagator, grid_observables,
                                   initial_coefficient_field)
    model = build_model({"name": "tully_dual_avoided_crossing"}, mass=MASS)
    grid = build_grid(-45.0, 45.0, 128, -30.0, 50.0, 64)
    rho0 = initial_coefficient_field(grid, -15.0, 10.0, 1.0)
    untouched = rho0.copy()
    prop = StrangPropagator(grid, model, dt=0.5)

    repeated = rho0.copy()
    for _ in range(11):
        repeated = prop.step(repeated)
    fused = prop.advance(rho0, 11)

    assert np.array_equal(rho0, untouched)
    scale = np.max(np.abs(repeated))
    assert np.max(np.abs(fused - repeated)) / scale < 2e-13
    a = grid_observables(repeated, grid, model)
    b = grid_observables(fused, grid, model)
    for key in ("populations", "adiabatic_populations", "trace", "mean_R",
                "mean_P", "boundary_mass_R", "boundary_mass_P"):
        assert np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key]))) < 2e-13


def test_optimized_grid_observables_match_direct_definitions():
    """Summing over P before the R-local basis rotation is exactly linear."""
    from mqcle.propagation import grid_observables
    model = build_model({"name": "tully_dual_avoided_crossing"}, mass=MASS)
    grid = build_grid(-20.0, 20.0, 96, -30.0, 50.0, 64)
    rho = _gaussian_field(grid, Rc=-1.0, Pc=10.0, sR=0.8, sP=3.0)
    got = grid_observables(rho, grid, model)

    h = model.h(grid.R[:, None])
    _, U = np.linalg.eigh(h)
    Ud = np.conj(np.swapaxes(U, -1, -2))[:, None]
    direct_ad = np.real(np.einsum("ijaa->a", Ud @ rho @ U[:, None])) \
        * grid.cell_area
    assert np.max(np.abs(got["adiabatic_populations"] - direct_ad)) < 2e-13
    assert abs(got["trace"] - trace(rho, grid)) < 2e-13
    assert np.max(np.abs(got["populations"]
                         - populations_from_rho(rho, grid))) < 2e-13
    assert abs(got["hermiticity"] - hermiticity_error(rho)) < 2e-13


def test_fused_production_schedule_matches_per_step_callback_path():
    """Saved endpoints are unchanged for a non-divisible final block."""
    from mqcle.propagation import (initial_coefficient_field,
                                   propagate_coefficients)
    model = build_model({"name": "tully_dual_avoided_crossing"}, mass=MASS)
    grid = build_grid(-30.0, 30.0, 64, -20.0, 40.0, 32)
    rho0 = initial_coefficient_field(grid, -10.0, 10.0, 1.0)
    fast = propagate_coefficients(rho0, grid, model, 0.5, 23, n_save=5)
    # Supplying a callback deliberately selects the original per-step path.
    slow = propagate_coefficients(
        rho0, grid, model, 0.5, 23, n_save=5,
        progress=lambda _step, _total: None,
    )
    assert fast.times == slow.times == [0.0, 2.5, 5.0, 7.5, 10.0, 11.5]
    for key in fast.records[0]:
        a, b = fast.column(key), slow.column(key)
        assert np.max(np.abs(a - b)) < 2e-13, key


def test_strang_is_second_order_and_converges_to_rk4():
    """Both integrate the same spatial operator, so they share a dt -> 0 limit."""
    from mqcle.propagation import (StrangPropagator, initial_coefficient_field,
                                   grid_observables)
    model = build_model({"name": "tully_dual_avoided_crossing"}, mass=MASS)
    grid = build_grid(-45.0, 45.0, 256, 18.0, 62.0, 128)
    rho0 = initial_coefficient_field(grid, -15.0, 40.0, 0.5)
    T = 200.0

    def run(dt):
        prop = StrangPropagator(grid, model, dt=dt)
        r = rho0.copy()
        for _ in range(int(round(T / dt))):
            r = prop.step(r)
        return r

    ref = run(0.0625)
    e = {dt: np.max(np.abs(run(dt) - ref)) for dt in (1.0, 0.5, 0.25)}
    order = np.log2(e[1.0] / e[0.5])
    order2 = np.log2(e[0.5] / e[0.25])
    assert 1.8 < order < 2.2, f"measured Strang order {order:.3f}"
    assert 1.8 < order2 < 2.2, f"measured Strang order {order2:.3f}"

    # and the two integrators agree at a dt where both are accurate and stable
    cache = physical_matrices(model, grid, TraceConvention.TRACELESS)
    r = rho0.copy()
    for _ in range(int(round(T / 0.25))):
        r = rk4_step(r, 0.25, grid, model, HBAR, TraceConvention.TRACELESS, cache)
    assert np.max(np.abs(r - ref)) / np.max(np.abs(ref)) < 1e-5
