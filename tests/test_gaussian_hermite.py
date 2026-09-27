r"""Mandatory verification gates GH-01 .. GH-15 for the Gaussian--Hermite branch.

Every gate in ``LOW_LEVEL_LLM_MASTER_INSTRUCTIONS.md`` is implemented here.
No production run is authorised until all of them pass.

The gates are ordered from pure algebra to propagation:

============ ==========================================================
GH-01        Gaussian conjugation identity, symbolically
GH-02        analytic ``T_ij`` recurrence vs symbolic differentiation
GH-03        rotation recurrence and antisymmetric cancellation
GH-04        no-raising property for K = 0..4
GH-05        analytic matrix elements vs Gauss--Hermite quadrature
GH-06        physical frame: 11 modes, rank 4, idempotence, round trip
GH-07        infinitesimal physical-frame invariance
GH-08        reduction to the subsystem QCLE right-hand side
GH-09        spot check against the independent full mapping operator
GH-10        propagation agreement with the subsystem-QCLE reference
GH-11        basis covariance: the operator conjugates under U(N), and the
             induced map on coefficients preserves total Hermite order
GH-12        the optimised real kernel is the same operator: one application
             and a whole RK4 trajectory agree with the reference to round-off
GH-13        DB = I exactly, and D L_GH B = L_QCLE on every physical direction
GH-14        no matrix element raises Hermite order or changes parity, and
             W R is antisymmetric
GH-15        the Arnoldi exponential is the production propagator: it matches
             the independent reference and beats RK4 at equal cost
============ ==========================================================
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import sympy as sp

from mqcle.coefficient_operator import (build_grid, coefficient_rhs,
                                        physical_matrices)
from mqcle.constants import CDTYPE, HBAR
from mqcle.hermite.basis import he_multi
from mqcle.hermite.covariance import (hermite_rotation_matrix,
                                      mapping_real_form, rotate_density,
                                      rotation_O2, unitary_U2)
from mqcle.hermite.frame import (hermite_metric, hermite_to_rho,
                                 physical_frame_matrix,
                                 physical_frame_residual, physical_projector,
                                 rho_to_hermite, weighted_pseudoinverse)
from mqcle.hermite.operator import (active_modes, force_matrix,
                                    d_real_hamiltonian_dR, hermite_rhs,
                                    hermite_T_matrices, mode_index_map,
                                    real_hamiltonian, rotation_basis_matrices,
                                    rotation_matrix, symplectic_J)
from mqcle.hermite.kernel import (RealHermiteKernel, generator_is_real,
                                  krylov_arnoldi_real, rk4_real,
                                  state_is_real)
from mqcle.hermite.propagator import (HermiteGenerator, propagate_krylov,
                                      propagate_rk4)
from mqcle.models import TullyDualAvoidedCrossing
from mqcle.propagation import StrangPropagator


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _tully():
    return TullyDualAvoidedCrossing(mass=2000.0)


def _gauss_moment(k: int) -> int:
    r"""Standard-normal moment :math:`\int\varphi(z)z^k\,dz=(k-1)!!` for even k."""
    if k % 2:
        return 0
    m = 1
    for j in range(1, k, 2):
        m *= j
    return m


def _hermite_expand(expr, zsyms, modes):
    r"""Exact Hermite coefficients by orthogonal projection.

    :math:`c_{\bm n}[f]=\frac{1}{\bm n!}\int dz\,\varphi\He_{\bm n}f`,
    evaluated exactly through standard-normal monomial moments rather than by
    peeling, so the result does not depend on any elimination order.
    """
    out = {}
    for n in modes:
        prod_poly = sp.Poly(sp.expand(he_multi(n, zsyms) * expr), *zsyms)
        integral = 0
        for mono, c in zip(prod_poly.monoms(), prod_poly.coeffs()):
            w = 1
            for k in mono:
                w *= _gauss_moment(k)
                if w == 0:
                    break
            if w:
                integral += c * w
        if integral != 0:
            fact = 1
            for k in n:
                fact *= sp.factorial(k)
            out[tuple(n)] = sp.nsimplify(integral / fact)
    return out


def _small_grid(n_R=8, n_P=8):
    return build_grid(-6.0, 6.0, n_R, 20.0, 60.0, n_P)


# ==========================================================================
# GH-01
# ==========================================================================
def test_gh01_gaussian_conjugation_identity():
    r"""phi^{-1} d_i d_j (phi q) - (z_i z_j - delta_ij) q = -T_ij q."""
    d = 4
    z = sp.symbols(f"z0:{d}", real=True)
    phi = sp.exp(-sum(zi ** 2 for zi in z) / 2)
    q = sp.Function("q")(*z)

    for i, j in itertools.product(range(d), repeat=2):
        lhs = sp.expand(sp.simplify(
            (phi ** -1) * sp.diff(phi * q, z[i], z[j])))
        rhs = sp.expand(sp.diff(q, z[i], z[j])
                        - z[i] * sp.diff(q, z[j])
                        - z[j] * sp.diff(q, z[i])
                        + (z[i] * z[j] - (1 if i == j else 0)) * q)
        assert sp.simplify(lhs - rhs) == 0

        # the combination that appears in the generator
        T = (z[i] * sp.diff(q, z[j]) + z[j] * sp.diff(q, z[i])
             - sp.diff(q, z[i], z[j]))
        combo = sp.expand((z[i] * z[j] - (1 if i == j else 0)) * q - lhs)
        assert sp.simplify(combo - T) == 0


# ==========================================================================
# GH-02
# ==========================================================================
def test_gh02_T_recurrence_matches_symbolic_differentiation():
    """Analytic T^{ij} equals direct differentiation, for every (i, j)."""
    nstates, K = 2, 4
    d = 2 * nstates
    modes = active_modes(nstates, K)
    index = mode_index_map(modes)
    T = hermite_T_matrices(modes)
    z = sp.symbols(f"z0:{d}", real=True)

    rng = np.random.default_rng(20260917)
    sources = [modes[k] for k in rng.choice(len(modes), size=8, replace=False)]

    for n in sources:
        he_n = he_multi(n, z)
        for i, j in itertools.product(range(d), repeat=2):
            sym = sp.expand(z[i] * sp.diff(he_n, z[j])
                            + z[j] * sp.diff(he_n, z[i])
                            - sp.diff(he_n, z[i], z[j]))
            expansion = _hermite_expand(sym, z, modes)
            a = index[tuple(n)]
            for m, c in expansion.items():
                assert T[i, j, index[m], a] == pytest.approx(float(c)), \
                    f"T[{i},{j}] mismatch at source {n} target {m}"
            # every nonzero analytic entry must appear in the expansion
            nz = np.nonzero(T[i, j, :, a])[0]
            for b in nz:
                assert tuple(modes[b]) in expansion


# ==========================================================================
# GH-03
# ==========================================================================
def test_gh03_rotation_recurrence_and_antisymmetric_cancellation():
    """The order-lowering remainder of -K_ij z_j d_i cancels on contraction."""
    nstates, K = 2, 4
    d = 2 * nstates
    modes = active_modes(nstates, K)
    index = mode_index_map(modes)
    z = sp.symbols(f"z0:{d}", real=True)

    # integer entries so the antisymmetric cancellation is exact rather than
    # exact-to-round-off: the gate is about algebra, not floating point
    rng = np.random.default_rng(7)
    A = rng.integers(-5, 6, size=(d, d))
    Kmat = (A - A.T).astype(float)
    assert np.array_equal(Kmat, -Kmat.T)
    N_ij = rotation_basis_matrices(modes)
    Rmat = -np.einsum("ij,ijmn->mn", Kmat, N_ij, optimize=True)

    for n in [modes[k] for k in rng.choice(len(modes), size=6, replace=False)]:
        he_n = he_multi(n, z)
        sym = sp.expand(-sum(sp.Integer(int(Kmat[i, j])) * z[j] * sp.diff(he_n, z[i])
                             for i in range(d) for j in range(d)))
        expansion = _hermite_expand(sym, z, modes)
        a = index[tuple(n)]
        for m, c in expansion.items():
            assert Rmat[index[m], a] == pytest.approx(float(c), abs=1e-10)
        # the |n|-2 block must be empty: that is the cancellation
        for m in expansion:
            assert sum(m) == sum(n), \
                f"rotation produced order {sum(m)} from {sum(n)}"


# ==========================================================================
# GH-04
# ==========================================================================
@pytest.mark.parametrize("K", [0, 1, 2, 3, 4])
def test_gh04_no_raising(K):
    """No nonzero target of R or T^{ij} has order above its source."""
    nstates = 2
    modes = active_modes(nstates, K)
    if not modes:
        pytest.skip("empty mode set")
    orders = np.array([sum(n) for n in modes])
    T = hermite_T_matrices(modes)
    N_ij = rotation_basis_matrices(modes)

    for mats in (T, N_ij):
        d = mats.shape[0]
        for i, j in itertools.product(range(d), repeat=2):
            tgt, src = np.nonzero(mats[i, j])
            assert np.all(orders[tgt] <= orders[src]), \
                f"order raised by matrix ({i},{j}) at K={K}"


# ==========================================================================
# GH-05
# ==========================================================================
def test_gh05_analytic_vs_gauss_hermite_quadrature():
    """Selected matrix elements reproduced by independent tensor quadrature."""
    nstates, K = 2, 2
    d = 2 * nstates
    modes = active_modes(nstates, K, parity="even")
    index = mode_index_map(modes)
    T = hermite_T_matrices(modes)

    nq = 6
    t, w = np.polynomial.hermite.hermgauss(nq)
    znodes = np.sqrt(2.0) * t
    wstd = w / np.sqrt(np.pi)

    z = sp.symbols(f"z0:{d}", real=True)
    fact = hermite_metric(modes)

    rng = np.random.default_rng(11)
    pairs = [(int(a), int(b)) for a, b in
             rng.integers(0, len(modes), size=(6, 2))]

    for i, j in [(0, 0), (0, 2), (1, 3), (2, 3)]:
        for a, b in pairs:
            n, m = modes[b], modes[a]
            he_n = he_multi(n, z)
            op = sp.expand(z[i] * sp.diff(he_n, z[j])
                           + z[j] * sp.diff(he_n, z[i])
                           - sp.diff(he_n, z[i], z[j]))
            integrand = sp.lambdify(z, sp.expand(op * he_multi(m, z)), "numpy")

            grids = np.meshgrid(*([znodes] * d), indexing="ij")
            weights = 1.0
            for ax in range(d):
                shape = [1] * d
                shape[ax] = nq
                weights = weights * wstd.reshape(shape)
            quad = float(np.sum(weights * integrand(*grids))) / fact[a]

            assert quad == pytest.approx(float(T[i, j, a, b]), abs=1e-8), \
                f"quadrature disagrees for T[{i},{j}] element ({m}|{n})"


# ==========================================================================
# GH-06
# ==========================================================================
def test_gh06_physical_frame():
    """Eleven even modes, rank 4, idempotent projector, exact round trip."""
    nstates = 2
    modes = active_modes(nstates, 2, parity="even")
    assert len(modes) == 11
    assert len(modes) == 1 + (2 * nstates) * (2 * nstates + 1) // 2

    B = physical_frame_matrix(nstates, modes)
    W = hermite_metric(modes)
    assert B.shape == (11, 4)
    assert np.linalg.matrix_rank(B) == nstates ** 2

    P = physical_projector(B, W)
    assert np.allclose(P @ P, P, atol=1e-12)

    rng = np.random.default_rng(3)
    for _ in range(20):
        a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        rho = 0.5 * (a + a.conj().T)                   # Hermitian
        c = rho_to_hermite(rho, B)
        # c_0 is the trace
        assert c[0] == pytest.approx(np.trace(rho), abs=1e-12)
        back = hermite_to_rho(c, B, W)
        assert np.allclose(back, rho, atol=1e-12)
        # a physical coefficient vector has zero residual
        assert physical_frame_residual(c, B, W) < 1e-12

    # orientation of the imaginary coherence, not only the populations
    rho = np.array([[0.0, 1j], [-1j, 0.0]], dtype=CDTYPE)
    assert np.allclose(hermite_to_rho(rho_to_hermite(rho, B), B, W), rho,
                       atol=1e-12)


# ==========================================================================
# GH-07
# ==========================================================================
def test_gh07_infinitesimal_physical_invariance():
    """L_H maps the physical frame into itself, for random smooth bath data."""
    model = _tully()
    grid = _small_grid()
    gen = HermiteGenerator.from_model(model, grid, K=2, parity="even")
    B = physical_frame_matrix(model.nstates, gen.modes)
    W = hermite_metric(gen.modes)

    rng = np.random.default_rng(5)
    n_R, n_P = grid.shape
    a = (rng.standard_normal((n_R, n_P, 2, 2))
         + 1j * rng.standard_normal((n_R, n_P, 2, 2)))
    rho = 0.5 * (a + np.conj(np.swapaxes(a, -1, -2)))
    C = gen.to_state(rho_to_hermite(rho, B))

    assert physical_frame_residual(gen.from_state(C), B, W) < 1e-12
    dC = gen.rhs(C)
    assert physical_frame_residual(gen.from_state(dC), B, W) < 1e-10


# ==========================================================================
# GH-08
# ==========================================================================
def test_gh08_reduces_to_subsystem_qcle_rhs():
    """Hermite derivative, read out, equals the subsystem-QCLE derivative.

    This is a validation theorem, not the implementation route: the Hermite
    branch never forms ``rho_W`` while propagating.
    """
    model = _tully()
    grid = _small_grid(16, 16)
    gen = HermiteGenerator.from_model(model, grid, K=2, parity="even")
    B = physical_frame_matrix(model.nstates, gen.modes)
    W = hermite_metric(gen.modes)
    Bplus = weighted_pseudoinverse(B, W)

    rng = np.random.default_rng(17)
    n_R, n_P = grid.shape
    # smooth, band-limited physical field so the spectral derivatives agree
    Rg, Pg = np.meshgrid(grid.R, grid.P, indexing="ij")
    env = np.exp(-((Rg / 3.0) ** 2) - ((Pg - 40.0) / 8.0) ** 2)
    a = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2)))
    herm = 0.5 * (a + a.conj().T)
    rho = env[..., None, None] * herm[None, None, :, :]

    C = gen.to_state(rho_to_hermite(rho, B))
    dC = gen.from_state(gen.rhs(C))
    d_rho_hermite = hermite_to_rho(dC, B, W, Bplus=Bplus)

    cache = physical_matrices(model, grid)
    d_rho_direct = coefficient_rhs(rho, grid, model, cache=cache)

    scale = np.max(np.abs(d_rho_direct))
    err = np.max(np.abs(d_rho_hermite - d_rho_direct)) / scale
    assert err < 1e-10, f"relative mismatch {err:.3e}"


# ==========================================================================
# GH-09
# ==========================================================================
def test_gh09_full_mapping_operator_spot_check():
    r"""Reconstruct ``\tilde rho = phi q`` and compare with the full operator.

    The full mapping generator is evaluated independently by
    :mod:`mqcle.full_operator` on the symbolic projected density; the Hermite
    route never touches it.
    """
    from mqcle.fields import SymbolicField, symbols_for
    from mqcle.full_operator import full_generator_on_field

    model = _tully()
    nstates = model.nstates
    modes = active_modes(nstates, 2, parity="even")
    B = physical_frame_matrix(nstates, modes)
    W = hermite_metric(modes)

    r, p, Rs, Ps = symbols_for(nstates, model.nbath)
    R, P = Rs[0], Ps[0]

    # a smooth polynomial physical density, so sympy can differentiate exactly
    rho_sym = [[sp.Rational(1, 2) + R / 10 + P / 100,
                sp.Rational(1, 5) - R / 20 + sp.I * P / 50],
               [sp.Rational(1, 5) - R / 20 - sp.I * P / 50,
                sp.Rational(1, 2) - R / 10 - P / 100]]

    from mqcle.fields import projected_density_expr
    expr = projected_density_expr(rho_sym, nstates, model.nbath, hbar=HBAR)
    field = SymbolicField(expr, nstates, model.nbath)

    rng = np.random.default_rng(23)
    for _ in range(4):
        Rv = float(rng.uniform(-2.0, 2.0))
        Pv = float(rng.uniform(20.0, 30.0))
        xv = rng.standard_normal(2 * nstates) * 0.7

        parts = full_generator_on_field(field, xv, np.array([Rv]),
                                        np.array([Pv]), model, hbar=HBAR)
        drho_m = complex(np.asarray(parts.total).reshape(()))

        # x-space Gaussian: q = rho_m / phi_x
        phi_x = np.exp(-np.sum(xv ** 2) / HBAR) / (np.pi * HBAR) ** nstates
        dq_full = drho_m / phi_x

        # Hermite route at the same point
        rho_num = np.array(
            [[complex(sp.N(rho_sym[a][b].subs({R: Rv, P: Pv})))
              for b in range(nstates)] for a in range(nstates)],
            dtype=CDTYPE)
        # build a one-point grid-free derivative using sympy for d/dR, d/dP
        drho_dR = np.array(
            [[complex(sp.N(sp.diff(rho_sym[a][b], R).subs({R: Rv, P: Pv})))
              for b in range(nstates)] for a in range(nstates)], dtype=CDTYPE)
        drho_dP = np.array(
            [[complex(sp.N(sp.diff(rho_sym[a][b], P).subs({R: Rv, P: Pv})))
              for b in range(nstates)] for a in range(nstates)], dtype=CDTYPE)

        C = rho_to_hermite(rho_num, B)
        CdR = rho_to_hermite(drho_dR, B)
        CdP = rho_to_hermite(drho_dP, B)

        Rcol = np.array([[Rv]])
        H_real = real_hamiltonian(model.h(Rcol))
        dH_real = d_real_hamiltonian_dR(model.dh_dR(Rcol)[:, 0])
        dvb = model.dbath_dR(Rcol)[..., 0]
        T = hermite_T_matrices(modes)
        Rmat = rotation_matrix(H_real, HBAR, modes)[0]
        Gmat = force_matrix(dvb, dH_real, T, modes)[0]

        dC = (-(Pv / model.mass) * CdR + Rmat @ C + Gmat @ CdP)

        # evaluate dq at the mapping point
        zv = np.sqrt(2.0 / HBAR) * xv
        zs = sp.symbols(f"z0:{2 * nstates}", real=True)
        dq_herm = sum(
            complex(dC[a]) * float(sp.N(he_multi(n, zs).subs(
                {zs[k]: zv[k] for k in range(2 * nstates)})))
            for a, n in enumerate(modes))

        assert dq_herm == pytest.approx(dq_full, rel=1e-8, abs=1e-8), \
            f"full-operator mismatch at R={Rv:.3f} P={Pv:.3f}"


# ==========================================================================
# GH-10
# ==========================================================================
def test_gh10_propagation_agrees_with_qcle_reference():
    """Hermite propagation and the Strang reference converge to each other."""
    model = _tully()
    grid = _small_grid(32, 32)
    gen = HermiteGenerator.from_model(model, grid, K=2, parity="even")
    B = physical_frame_matrix(model.nstates, gen.modes)
    W = hermite_metric(gen.modes)
    Bplus = weighted_pseudoinverse(B, W)

    Rg, Pg = np.meshgrid(grid.R, grid.P, indexing="ij")
    env = np.exp(-((Rg + 1.0) / 2.0) ** 2 - ((Pg - 40.0) / 6.0) ** 2)
    rho0 = np.zeros(grid.shape + (2, 2), dtype=CDTYPE)
    rho0[..., 0, 0] = env

    lam = gen.spectral_radius_estimate()
    dt = 0.25 * lam["rk4_dt_limit"]
    n_steps = 40
    T = dt * n_steps

    C0 = gen.to_state(rho_to_hermite(rho0, B))
    _, states = propagate_rk4(gen, C0, dt, n_steps, save_every=n_steps)
    C_end = gen.from_state(states[-1])
    rho_herm = hermite_to_rho(C_end, B, W, Bplus=Bplus)

    sp_prop = StrangPropagator(grid, model, dt)
    rho_ref = sp_prop.advance(rho0.copy(), n_steps)

    scale = np.max(np.abs(rho_ref))
    err = np.max(np.abs(rho_herm - rho_ref)) / scale
    assert err < 5e-4, f"relative field difference {err:.3e} at T={T:.4f}"

    # the residual must stay at round-off: it is never projected away
    assert physical_frame_residual(C_end, B, W) < 1e-9


# ==========================================================================
# GH-11
# ==========================================================================
#: One real orthogonal element and two genuinely complex ones.  A real ``O``
#: keeps a real symmetric ``h`` real symmetric and so never leaves the case
#: Theorem 1 was first stated for; only a nonzero phase sends ``h`` to a
#: complex Hermitian matrix and exercises Corollary 1.  A gate built from real
#: rotations alone would pass on an implementation that discards ``Im h``.
_COV_ROTATIONS = [
    ("O(2), theta=0.30", rotation_O2(0.30)),
    ("U(2), theta=0.60 phi=0.90", unitary_U2(0.60, 0.90)),
    ("U(2), theta=pi/4 phi=pi/3", unitary_U2(np.pi / 4, np.pi / 3)),
]


def _random_hermitian(rng, shape):
    a = (rng.standard_normal(shape + (2, 2))
         + 1j * rng.standard_normal(shape + (2, 2)))
    return a + np.conj(np.swapaxes(a, -1, -2))


@pytest.mark.parametrize("name,U", _COV_ROTATIONS,
                         ids=[n for n, _ in _COV_ROTATIONS])
def test_gh11_basis_covariance_of_the_finite_operator(name, U):
    r"""The finite Hermite operator is covariant under :math:`h\mapsto U^\dagger hU`.

    Four statements, in increasing strength, none of which requires a
    propagation:

    1. :math:`\mathcal V=\mathrm{re}(U^\dagger)` is orthogonal, and the induced
       map :math:`\mathsf D` on coefficients is *block diagonal in total
       Hermite order*.  This is the one that matters: it says an orthogonal
       relabelling cannot move weight between orders, so :math:`\mathcal V_K`
       is basis independent and the truncation is not an artefact of the basis
       it was written in.  The reduced field cannot express this statement at
       all, because it has already summed over the orders.
    2. :math:`\mathsf D` represents the group:
       :math:`\mathsf D(\mathcal V)\mathsf D(\mathcal V^{\mathsf T})=I`.
    3. The physical frame intertwines:
       :math:`B\,\mathrm{vec}(U^\dagger\rho U)=\mathsf D\,B\,\mathrm{vec}(\rho)`
       for every Hermitian :math:`\rho`, so a rotation of the density and a
       rotation of the coefficients are the same operation.
    4. The generator conjugates:
       :math:`\mathsf R(U^\dagger hU)=\mathsf D\mathsf R(h)\mathsf D^{-1}` and
       likewise for :math:`\mathsf G`.  This is covariance of
       Eq. (finite-Hermite) itself, matrix element by matrix element.

    All four are exact identities checked to round-off, not tolerances chosen
    to pass.  The Hamiltonians are random Hermitian matrices rather than the
    benchmark model's: a model whose :math:`\mathbb H` happens to be sparse
    could satisfy a weaker statement by accident.
    """
    rng = np.random.default_rng(20260918)
    modes = active_modes(2, 2, parity="even")
    M = len(modes)
    B = physical_frame_matrix(2, modes)
    T_mats = hermite_T_matrices(modes)

    V = mapping_real_form(U)
    assert np.max(np.abs(V @ V.T - np.eye(4))) < 1e-13, "V is not orthogonal"

    D = hermite_rotation_matrix(modes, V)
    assert D.shape == (M, M)

    # 1. no mixing between total orders
    order = np.array([sum(n) for n in modes])
    off = order[:, None] != order[None, :]
    assert np.max(np.abs(D[off])) < 1e-12, (
        f"{name}: basis change moves weight between Hermite orders, "
        f"max {np.max(np.abs(D[off])):.3e}")

    # 2. group property, checked against an independent construction of the
    #    inverse rather than against a numerical matrix inverse
    D_inv = hermite_rotation_matrix(modes, V.T)
    assert np.max(np.abs(D @ D_inv - np.eye(M))) < 1e-12

    # 3. the physical frame intertwines, on random Hermitian fields
    rho = _random_hermitian(rng, (3, 2))
    lhs = rho_to_hermite(rotate_density(rho, U), B)
    rhs = np.einsum("mn,rpn->rpm", D, rho_to_hermite(rho, B))
    scale = max(np.max(np.abs(lhs)), 1.0)
    assert np.max(np.abs(lhs - rhs)) / scale < 1e-12, (
        f"{name}: B does not intertwine the rotation")

    # 4. the generator conjugates -- the covariance statement itself
    for _ in range(4):
        h = _random_hermitian(rng, (1,))
        dh = _random_hermitian(rng, (1,))
        hU = np.conj(U).T @ h @ U
        dhU = np.conj(U).T @ dh @ U

        R0 = rotation_matrix(real_hamiltonian(h), HBAR, modes)[0]
        RU = rotation_matrix(real_hamiltonian(hU), HBAR, modes)[0]
        G0 = force_matrix(np.ones(1), d_real_hamiltonian_dR(dh),
                          T_mats, modes)[0]
        GU = force_matrix(np.ones(1), d_real_hamiltonian_dR(dhU),
                          T_mats, modes)[0]

        dR = np.max(np.abs(RU - D @ R0 @ D_inv)) / max(np.max(np.abs(R0)), 1.0)
        dG = np.max(np.abs(GU - D @ G0 @ D_inv)) / max(np.max(np.abs(G0)), 1.0)
        assert dR < 1e-12, f"{name}: rotation matrix not covariant, {dR:.3e}"
        assert dG < 1e-12, f"{name}: force matrix not covariant, {dG:.3e}"


def test_gh11_covariance_survives_propagation():
    r"""The same problem in a rotated basis propagates to the same answer.

    Statement 4 above makes this a corollary rather than an independent fact,
    but the corollary is what the manuscript quotes, and it is the form in
    which a fault in the *assembly* -- the wrong :math:`\mathbb H` fed to one
    channel, a stale mode list, a transposed frame -- would show up.

    The packet starts at the crossing.  Away from it :math:`\nabla_Rh` is
    numerically zero, both channels of the coupled term vanish, and the check
    reduces to free advection plus a rotation, which nobody doubts is
    covariant.  ``scripts/07c_basis_covariance.py`` records the same trap for
    the reduced branch.
    """
    model = _tully()
    grid = build_grid(-6.0, 6.0, 32, 20.0, 60.0, 32)
    U = unitary_U2(0.60, 0.90)

    # the coupled channel must actually be active over the propagated region
    dh = model.dh_dR(grid.R[:, None])
    assert np.max(np.abs(dh)) > 1e-3, "no gradient: the check would be vacuous"

    gen = HermiteGenerator.from_model(model, grid, K=2, parity="even")
    genU = HermiteGenerator.from_model(_RotatedTully(model, U), grid,
                                       K=2, parity="even")
    B = physical_frame_matrix(model.nstates, gen.modes)
    W = hermite_metric(gen.modes)
    Bplus = weighted_pseudoinverse(B, W)

    Rg, Pg = np.meshgrid(grid.R, grid.P, indexing="ij")
    env = np.exp(-(Rg / 1.5) ** 2 - ((Pg - 40.0) / 6.0) ** 2)
    rho0 = np.zeros(grid.shape + (2, 2), dtype=CDTYPE)
    rho0[..., 0, 0] = env

    dt = 0.25 * gen.spectral_radius_estimate()["rk4_dt_limit"]
    n = 40

    def run(g, r0):
        C0 = g.to_state(rho_to_hermite(r0, B))
        _, states = propagate_rk4(g, C0, dt, n, save_every=n)
        return hermite_to_rho(g.from_state(states[-1]), B, W, Bplus=Bplus)

    ref = run(gen, rho0)
    got = run(genU, rotate_density(rho0, U))
    scale = np.max(np.abs(ref))
    err = np.max(np.abs(got - rotate_density(ref, U))) / scale
    assert err < 1e-10, f"basis covariance broken after propagation: {err:.3e}"


class _RotatedTully(TullyDualAvoidedCrossing):
    r"""``base`` in a rotated subsystem basis, :math:`h\mapsto U^\dagger hU`.

    Only the subsystem matrix rotates.  The bath potential is a multiple of the
    identity in subsystem space and is therefore untouched -- which is exactly
    why the rotation is a symmetry of the problem and not a different model.
    """

    def __init__(self, base, U):
        self.base, self.U = base, np.asarray(U, dtype=complex)
        self.nstates, self.nbath = base.nstates, base.nbath
        self.mass = base.mass
        self.name = f"{base.name}_rotated"

    def h(self, R):
        return np.conj(self.U).T @ self.base.h(R) @ self.U

    def dh_dR(self, R):
        return np.conj(self.U).T @ self.base.dh_dR(R) @ self.U

    def bath_potential(self, R):
        return self.base.bath_potential(R)

    def dbath_dR(self, R):
        return self.base.dbath_dR(R)


# --------------------------------------------------------------------------
# GH-12  the optimised kernel is the same operator
# --------------------------------------------------------------------------
# mqcle.hermite.kernel evaluates L_H in real arithmetic, with cache-blocked
# spectral derivatives and a dense batched mode contraction.  Its licence to
# do so rests on two facts that are checked here rather than assumed: the
# generator is *exactly* real, and a physical-frame state is *exactly* real.
# Nothing else about the method changes -- same grid, same Nyquist
# convention, same matrices, same RK4 tableau, same step.  The gate is that a
# whole trajectory, not merely one application, agrees with the reference.
def _physical_rho0(grid, R0=-1.0, P0=40.0):
    Rg, Pg = np.meshgrid(grid.R, grid.P, indexing="ij")
    env = np.exp(-((Rg - R0) / 2.0) ** 2 - ((Pg - P0) / 6.0) ** 2)
    rho = np.zeros(grid.shape + (2, 2), dtype=CDTYPE)
    rho[..., 0, 0] = env
    rho[..., 0, 1] = 0.25 * env * (1.0 + 0.5j)
    rho[..., 1, 0] = np.conj(rho[..., 0, 1])
    rho[..., 1, 1] = 0.5 * env
    return rho


def test_gh12_generator_and_physical_state_are_exactly_real():
    """The hypotheses behind the real kernel, as exact statements."""
    model = _tully()
    grid = _small_grid(32, 32)
    gen = HermiteGenerator.from_model(model, grid, K=2, parity="even")

    assert np.max(np.abs(np.imag(gen.Rmat))) == 0.0
    assert np.max(np.abs(np.imag(gen.Gmat))) == 0.0
    assert generator_is_real(gen)
    for _, _, coeff in gen._R_blocks + gen._G_blocks:
        assert np.max(np.abs(np.imag(coeff))) == 0.0

    B = physical_frame_matrix(model.nstates, gen.modes)
    C0 = gen.to_state(rho_to_hermite(_physical_rho0(grid), B))
    # B itself is complex; the *combination* B vec(rho) is real for Hermitian
    # rho, which is what makes the real path legitimate.
    assert np.max(np.abs(np.imag(B))) > 0.0
    assert np.max(np.abs(np.imag(C0))) == 0.0
    assert state_is_real(C0)


def test_gh12_kernel_matches_reference_rhs():
    """One application of L_H, on a generic (not physical) real field."""
    model = _tully()
    grid = _small_grid(48, 32)
    gen = HermiteGenerator.from_model(model, grid, K=2, parity="even")
    rng = np.random.default_rng(12)
    C = rng.standard_normal((len(gen.modes),) + grid.shape)

    with RealHermiteKernel(gen, nthread=2) as kern:
        got = kern.rhs(C)
    ref = gen.rhs(C.astype(CDTYPE))

    scale = np.max(np.abs(ref))
    err = np.max(np.abs(got - np.real(ref))) / scale
    # the reference carries its own O(1e-16) imaginary noise; the kernel must
    # agree with its real part at least that well
    assert err < 1e-13, f"kernel disagrees with the reference rhs: {err:.3e}"
    assert np.max(np.abs(np.imag(ref))) / scale < 1e-13


@pytest.mark.parametrize("nthread", [1, 3])
def test_gh12_kernel_is_thread_count_invariant(nthread):
    """Threading must not change a single bit of the result."""
    model = _tully()
    grid = _small_grid(64, 32)
    gen = HermiteGenerator.from_model(model, grid, K=2, parity="even")
    rng = np.random.default_rng(13)
    C = rng.standard_normal((len(gen.modes),) + grid.shape)
    with RealHermiteKernel(gen, nthread=1) as k1:
        a = k1.rhs(C)
    with RealHermiteKernel(gen, nthread=nthread) as kn:
        b = kn.rhs(C)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("r_slab,p_slab,chunk", [(8, 16, 4), (64, 256, 8)])
def test_gh12_kernel_is_blocking_invariant(r_slab, p_slab, chunk):
    """The cache blocking is a loop transformation, not a numerical choice."""
    model = _tully()
    grid = _small_grid(64, 32)
    gen = HermiteGenerator.from_model(model, grid, K=2, parity="even")
    rng = np.random.default_rng(14)
    C = rng.standard_normal((len(gen.modes),) + grid.shape)
    with RealHermiteKernel(gen, nthread=2, r_slab=1024, p_slab=1024,
                           chunk=1024) as k0:
        a = k0.rhs(C)
    with RealHermiteKernel(gen, nthread=2, r_slab=r_slab, p_slab=p_slab,
                           chunk=chunk) as kb:
        b = kb.rhs(C)
    assert np.array_equal(a, b)


def test_gh12_trajectory_matches_reference_rk4():
    """40 RK4 steps -- 160 applications -- from a physical initial state."""
    model = _tully()
    grid = _small_grid(48, 48)
    gen = HermiteGenerator.from_model(model, grid, K=2, parity="even")
    B = physical_frame_matrix(model.nstates, gen.modes)
    W = hermite_metric(gen.modes)
    Bplus = weighted_pseudoinverse(B, W)

    C0 = gen.to_state(rho_to_hermite(_physical_rho0(grid), B))
    dt = 0.25 * gen.spectral_radius_estimate()["rk4_dt_limit"]
    n_steps = 40

    _, states = propagate_rk4(gen, C0, dt, n_steps, save_every=n_steps)
    C_ref = states[-1]
    with RealHermiteKernel(gen, nthread=2) as kern:
        C_fast = rk4_real(kern, np.ascontiguousarray(np.real(C0)), dt, n_steps)

    scale = np.max(np.abs(np.real(C_ref)))
    err = np.max(np.abs(C_fast - np.real(C_ref))) / scale
    assert err < 1e-12, f"trajectories diverge: {err:.3e}"

    # and the observable the paper actually reports
    rho_ref = hermite_to_rho(gen.from_state(C_ref), B, W, Bplus=Bplus)
    rho_fast = hermite_to_rho(gen.from_state(C_fast.astype(CDTYPE)), B, W,
                              Bplus=Bplus)
    drho = np.max(np.abs(rho_fast - rho_ref)) / np.max(np.abs(rho_ref))
    assert drho < 1e-12, f"terminal rho_W differs by {drho:.3e}"


def test_gh12_kernel_refuses_a_complex_generator():
    """The guard must fire rather than silently discarding an imaginary part."""
    model = _tully()
    grid = _small_grid(16, 16)
    gen = HermiteGenerator.from_model(model, grid, K=2, parity="even")
    gen.Rmat = gen.Rmat + 1e-30j
    assert not generator_is_real(gen)
    with pytest.raises(ValueError, match="not exactly real"):
        RealHermiteKernel(gen, nthread=1)


# --------------------------------------------------------------------------
# GH-13  the readout is an exact left inverse, and the projected generator
#        is the subsystem-QCLE generator on every physical direction
# --------------------------------------------------------------------------
def test_gh13_readout_is_an_exact_left_inverse():
    r"""``D B = I`` as a matrix identity, with ``D = (B^dag W B)^-1 B^dag W``."""
    modes = active_modes(2, 2, parity="even")
    B = physical_frame_matrix(2, modes)
    W = hermite_metric(modes)
    D = weighted_pseudoinverse(B, W)

    # the formula itself, not merely the routine's output
    BhW = B.conj().T * np.asarray(W)[None, :]
    D_explicit = np.linalg.inv(BhW @ B) @ BhW
    assert np.allclose(D, D_explicit, atol=1e-13)

    I4 = np.eye(4, dtype=CDTYPE)
    assert np.max(np.abs(D @ B - I4)) < 1e-13, f"DB - I = {np.max(np.abs(D @ B - I4)):.3e}"

    # and P_phys = B D is the projector, idempotent and W-self-adjoint
    P = B @ D
    assert np.max(np.abs(P @ P - P)) < 1e-13
    Wd = np.diag(np.asarray(W).astype(CDTYPE))
    assert np.max(np.abs(Wd @ P - (Wd @ P).conj().T)) < 1e-12


def test_gh13_projected_generator_equals_qcle_on_every_direction():
    r"""``D L_GH B = L_QCLE`` column by column, not on one random state.

    The four physical directions are exercised separately, so a generator that
    happened to be right on one Hermitian combination and wrong on another
    cannot pass.
    """
    model = _tully()
    grid = _small_grid(16, 16)
    gen = HermiteGenerator.from_model(model, grid, K=2, parity="even")
    B = physical_frame_matrix(model.nstates, gen.modes)
    W = hermite_metric(gen.modes)
    Bplus = weighted_pseudoinverse(B, W)
    cache = physical_matrices(model, grid)

    Rg, Pg = np.meshgrid(grid.R, grid.P, indexing="ij")
    env = np.exp(-((Rg / 3.0) ** 2) - ((Pg - 40.0) / 8.0) ** 2)

    directions = {
        "rho_00": np.array([[1, 0], [0, 0]], dtype=CDTYPE),
        "rho_11": np.array([[0, 0], [0, 1]], dtype=CDTYPE),
        "coh_re": np.array([[0, 1], [1, 0]], dtype=CDTYPE),
        "coh_im": np.array([[0, 1j], [-1j, 0]], dtype=CDTYPE),
    }
    worst = 0.0
    for name, herm in directions.items():
        rho = env[..., None, None] * herm[None, None, :, :]
        C = gen.to_state(rho_to_hermite(rho, B))
        d_rho_hermite = hermite_to_rho(gen.from_state(gen.rhs(C)), B, W,
                                       Bplus=Bplus)
        d_rho_direct = coefficient_rhs(rho, grid, model, cache=cache)
        scale = np.max(np.abs(d_rho_direct))
        err = float(np.max(np.abs(d_rho_hermite - d_rho_direct)) / scale)
        worst = max(worst, err)
        assert err < 1e-10, f"D L_GH B != L_QCLE on {name}: {err:.3e}"
    assert worst > 0.0


# --------------------------------------------------------------------------
# GH-14  closure and parity as structural statements about the matrices
# --------------------------------------------------------------------------
# The propagated statement is measured by scripts/07k_gh_closure_validation.py
# in bases that contain the forbidden sectors.  What is checked here is the
# matrix-level reason: in a basis carrying orders 0..K, no matrix element of
# the generator connects a source of order |n| to a target of order |m| > |n|.
@pytest.mark.parametrize("K", [2, 4, 6])
def test_gh14_no_matrix_element_raises_hermite_order(K):
    model = _tully()
    grid = _small_grid(8, 8)
    gen = HermiteGenerator.from_model(model, grid, K=K, parity="even")
    orders = np.array([sum(n) for n in gen.modes])
    raising = orders[:, None] > orders[None, :]        # target > source
    assert raising.any(), "no raising block exists in this basis; vacuous test"
    for name, mat in (("R", gen.Rmat), ("G", gen.Gmat)):
        blocked = np.abs(mat)[:, raising]
        assert np.max(blocked) == 0.0, (
            f"{name} has a nonzero order-raising element, max "
            f"{np.max(blocked):.3e}")
    # and the lowering that DOES exist must be nonzero, or the test is vacuous
    lowering = orders[:, None] == orders[None, :] - 2
    assert np.max(np.abs(gen.Gmat)[:, lowering]) > 0.0


def test_gh14_no_matrix_element_changes_parity():
    """With both parities present, the generator is block diagonal in parity."""
    model = _tully()
    grid = _small_grid(8, 8)
    gen = HermiteGenerator.from_model(model, grid, K=3, parity=None)
    par = np.array([sum(n) % 2 for n in gen.modes])
    mixing = par[:, None] != par[None, :]
    assert mixing.any()
    for name, mat in (("R", gen.Rmat), ("G", gen.Gmat)):
        assert np.max(np.abs(mat)[:, mixing]) == 0.0, (
            f"{name} mixes parity")


def test_gh14_rotation_is_antisymmetric_in_the_hermite_metric():
    r""":math:`W\mathsf R` is antisymmetric: the rotation is a rotation.

    This is the matrix-level statement behind order preservation and behind
    :math:`\mathrm{tr}\,\mathsf R=0`.  The force matrix is *not* symmetric in
    the same metric, which is why the generator as a whole is non-normal and
    why no Lanczos or Chebyshev shortcut applies to it.
    """
    model = _tully()
    grid = _small_grid(8, 8)
    gen = HermiteGenerator.from_model(model, grid, K=2, parity="even")
    W = np.asarray(hermite_metric(gen.modes))
    sw = np.sqrt(W)
    Rw = (sw[None, :, None] * np.real(gen.Rmat)) / sw[None, None, :]
    asym = np.max(np.abs(Rw + Rw.transpose(0, 2, 1))) / np.max(np.abs(Rw))
    assert asym < 1e-13, f"W-antisymmetry of R broken: {asym:.3e}"
    assert np.max(np.abs(np.trace(gen.Rmat, axis1=1, axis2=2))) < 1e-13


# --------------------------------------------------------------------------
# GH-15  the Arnoldi exponential is the production propagator
# --------------------------------------------------------------------------
def test_gh15_arnoldi_matches_the_reference_exponential():
    """The production route against scipy's independent exponential action."""
    model = _tully()
    grid = _small_grid(32, 32)
    gen = HermiteGenerator.from_model(model, grid, K=2, parity="even")
    B = physical_frame_matrix(model.nstates, gen.modes)
    C0 = gen.to_state(rho_to_hermite(_physical_rho0(grid), B))
    T = 40.0

    ref = propagate_krylov(gen, C0, [T])[0]
    with RealHermiteKernel(gen, nthread=2) as kern:
        got, stats = krylov_arnoldi_real(kern, np.ascontiguousarray(np.real(C0)),
                                         [T], m=20, tol=1e-12)
    err = np.linalg.norm(got - np.real(ref)) / np.linalg.norm(ref)
    assert err < 1e-10, f"Arnoldi vs reference exponential: {err:.3e}"
    assert stats["applications"] > 0


def test_gh15_arnoldi_is_more_accurate_than_rk4_at_equal_cost():
    """The reason the exponential is production and RK4 is the cross-check."""
    model = _tully()
    grid = _small_grid(32, 32)
    gen = HermiteGenerator.from_model(model, grid, K=2, parity="even")
    B = physical_frame_matrix(model.nstates, gen.modes)
    C0r = np.ascontiguousarray(
        np.real(gen.to_state(rho_to_hermite(_physical_rho0(grid), B))))
    T = 40.0
    ref = np.real(propagate_krylov(gen, C0r.astype(CDTYPE), [T])[0])
    nref = np.linalg.norm(ref)

    dt = 0.5 * gen.spectral_radius_estimate()["rk4_dt_limit"]
    n = max(1, int(np.ceil(T / dt)))
    with RealHermiteKernel(gen, nthread=2) as kern:
        C_rk4 = rk4_real(kern, C0r.copy(), T / n, n)
        C_ar, stats = krylov_arnoldi_real(kern, C0r, [T], m=20, tol=1e-10)
    e_rk4 = np.linalg.norm(C_rk4 - ref) / nref
    e_ar = np.linalg.norm(C_ar - ref) / nref
    assert stats["applications"] <= 4 * n, (
        f"Arnoldi used {stats['applications']} applications against RK4's "
        f"{4 * n}")
    assert e_ar < e_rk4, f"Arnoldi {e_ar:.3e} not better than RK4 {e_rk4:.3e}"


def test_gh15_real_kernel_adjoint_identity():
    r""":math:`\langle b,\mathbb L_Ha\rangle=\langle\mathbb L_H^{\mathsf T}b,a\rangle`."""
    model = _tully()
    grid = _small_grid(24, 16)
    gen = HermiteGenerator.from_model(model, grid, K=2, parity="even")
    rng = np.random.default_rng(15)
    a = rng.standard_normal((len(gen.modes),) + grid.shape)
    b = rng.standard_normal(a.shape)
    with RealHermiteKernel(gen, nthread=2) as kern:
        lhs = float(np.sum(b * kern.rhs(a)))
        rhs = float(np.sum(kern.rhs_adjoint(b) * a))
        ref = gen.rhs_adjoint(b.astype(CDTYPE))
        assert np.max(np.abs(kern.rhs_adjoint(b) - np.real(ref))) \
            / np.max(np.abs(np.real(ref))) < 1e-13
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs)) < 1e-12
