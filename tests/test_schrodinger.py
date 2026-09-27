"""The independent quantum reference, and its agreement with the coefficient
solver in the one regime where the QCLE is exact."""
from __future__ import annotations

import numpy as np
import pytest

from mqcle.coefficient_operator import build_grid
from mqcle.models import ConstantCouplingModel, build_model
from mqcle.propagation import (grid_observables, initial_coefficient_field,
                               rk4_step)
from mqcle.schrodinger import (SplitOperatorPropagator, build_schrodinger_grid,
                               gaussian_wavepacket, wavefunction_observables)
from mqcle.types import TraceConvention

HBAR = 1.0
MASS = 2000.0


# --------------------------------------------------------------------------
# unitarity and analytic limits
# --------------------------------------------------------------------------
def test_norm_is_conserved_to_machine_precision():
    model = build_model({"name": "tully_dual_avoided_crossing"}, mass=MASS)
    grid = build_schrodinger_grid(-40.0, 40.0, 2048)
    psi = gaussian_wavepacket(grid, -15.0, 40.0, 0.25)
    prop = SplitOperatorPropagator(grid, model, dt=1.0)
    n0 = wavefunction_observables(psi, grid, model)["norm"]
    for _ in range(300):
        psi = prop.step(psi)
    n1 = wavefunction_observables(psi, grid, model)["norm"]
    # Every split factor is unitary.  The measured defect is FFT roundoff and
    # accumulates approximately as a random walk, so use the same n-step
    # scaling as the production gate rather than a step-count-independent cap.
    assert abs(n1 - n0) < 1e-13 * np.sqrt(300)


def test_free_particle_matches_the_analytic_spreading_gaussian():
    """V = 0: |psi|^2 stays Gaussian with the textbook width and centre."""
    free = ConstantCouplingModel(a0=0.0, b0=0.0, c0=0.0, k=0.0, mass=MASS)
    grid = build_schrodinger_grid(-60.0, 60.0, 4096)
    R0, P0, s0 = -15.0, 40.0, 0.5
    psi = gaussian_wavepacket(grid, R0, P0, s0)
    dt, n_steps = 2.0, 400
    prop = SplitOperatorPropagator(grid, free, dt=dt)
    for _ in range(n_steps):
        psi = prop.step(psi)
    t = dt * n_steps

    dens = np.abs(psi[:, 0]) ** 2
    nrm = np.sum(dens) * grid.dR
    mean = np.sum(dens * grid.R) * grid.dR / nrm
    var = np.sum(dens * (grid.R - mean) ** 2) * grid.dR / nrm

    sig_t = s0 * np.sqrt(1.0 + (HBAR * t / (2.0 * MASS * s0**2)) ** 2)
    assert abs(mean - (R0 + P0 * t / MASS)) < 1e-9
    assert abs(np.sqrt(var) - sig_t) / sig_t < 1e-10


def test_constant_h_gives_the_exact_two_level_phase():
    model = ConstantCouplingModel(a0=0.02, b0=-0.01, c0=0.005, k=0.0, mass=MASS)
    grid = build_schrodinger_grid(-40.0, 40.0, 1024)
    psi = gaussian_wavepacket(grid, -5.0, 10.0, 1.0)
    h0 = model.h(np.zeros((1, 1)))[0]
    w, U = np.linalg.eigh(h0)
    omega = (w[1] - w[0]) / HBAR

    chi0 = np.einsum("ba,rb->ra", np.conj(U), psi)
    dt, n_steps = 1.0, 250
    prop = SplitOperatorPropagator(grid, model, dt=dt)
    for _ in range(n_steps):
        psi = prop.step(psi)
    chi = np.einsum("ba,rb->ra", np.conj(U), psi)
    t = dt * n_steps

    # eigenbasis populations are frozen
    p0 = np.sum(np.abs(chi0) ** 2, axis=0) * grid.dR
    p1 = np.sum(np.abs(chi) ** 2, axis=0) * grid.dR
    assert np.max(np.abs(p1 - p0)) < 1e-12
    # eigenbasis coherence <chi_0|chi_1> picks up exp(-i omega t)
    c0 = np.sum(np.conj(chi0[:, 0]) * chi0[:, 1]) * grid.dR
    c1 = np.sum(np.conj(chi[:, 0]) * chi[:, 1]) * grid.dR
    # the free-particle kinetic phase is common to both components and cancels
    assert abs(c1 - c0 * np.exp(-1j * omega * t)) / abs(c0) < 1e-9


# --------------------------------------------------------------------------
# the tie between the two solvers
# --------------------------------------------------------------------------
def test_coefficient_solver_equals_schrodinger_when_qcle_is_exact():
    r"""For an ``R``-independent ``h`` and ``V_b=0`` the QCLE is exact.

    The Wigner transform of the free Schroedinger evolution obeys
    :math:`\partial_tW=-(P/M)\partial_RW` exactly, and a constant ``h``
    commutes with itself, so in this regime the coefficient solver and the
    exact quantum reference must agree to solver accuracy -- not merely to
    QCLE accuracy.  This is the one place where a disagreement can only be a
    bug.
    """
    model = ConstantCouplingModel(a0=0.02, b0=-0.01, c0=0.005, k=0.0, mass=MASS)
    R0, P0, sR, init = -5.0, 20.0, 1.0, 0

    sg = build_schrodinger_grid(-40.0, 40.0, 2048)
    psi = gaussian_wavepacket(sg, R0, P0, sR, init_state=init)
    prop = SplitOperatorPropagator(sg, model, dt=0.25)

    cg = build_grid(-40.0, 40.0, 256, P0 - 20.0, P0 + 20.0, 256)
    rho = initial_coefficient_field(cg, R0, P0, sR, init_state=init)
    cache = None

    dt, n_steps = 0.25, 1200
    for _ in range(n_steps):
        psi = prop.step(psi)
        rho = rk4_step(rho, dt, cg, model, HBAR, TraceConvention.TRACELESS, cache)

    q = wavefunction_observables(psi, sg, model)
    c = grid_observables(rho, cg, model)
    assert np.max(np.abs(np.asarray(c["populations"])
                         - np.asarray(q["populations"]))) < 1e-8
    assert abs(c["coherence_re"] - q["coherence_re"]) < 1e-8
    assert abs(c["coherence_im"] + q["coherence_im"]) < 1e-8, \
        "rho_W^{01} = conj(<psi_0|psi_1>): the imaginary parts are opposite"
    assert abs(c["mean_R"] - q["mean_R"]) < 1e-6
