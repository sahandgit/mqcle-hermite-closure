r"""The projected-density PBME estimator (PART II / PART XXV).

Four properties are required of it, and each is checked here.

1. It recovers the subsystem density at ``t = 0`` without bias.
2. It agrees with the exact answer for an ``R``-independent problem, where the
   mapping flow is one common linear rotation and the PBME operator is exact.
3. It handles *signed* weights.
4. It never divides the weights by their stochastic sum.

Property 4 is the easiest to get wrong and the hardest to notice: dividing by
:math:`\sum_iw_i` looks like a harmless normalisation, turns the estimator into
a ratio estimator with an :math:`O(1/N_s)` bias, and with signed weights has a
denominator that is itself noisy.  The test that catches it feeds in an
ensemble whose weights deliberately do not sum to one and checks that the
answer scales with them.
"""
from __future__ import annotations

import numpy as np
import pytest

from mqcle.models import ConstantCouplingModel, build_model
from mqcle.observables import rho_W_from_ensemble
from mqcle.projector import wigner_aa_dagger_all
from mqcle.propagation import (block_statistics,
                               propagate_pbme_ensemble,
                               propagate_pbme_projected_ensemble,
                               sample_matched_ensemble,
                               sample_projected_ensemble)
from mqcle.sampling import (SIGN_OPTIMAL_Z, projected_importance_weights,
                            _sample_tilted_action)
from mqcle.types import Ensemble, TraceConvention

HBAR = 1.0
SCHEMES = ["sign_optimal", "wigner_seo"]


def _tully():
    return build_model({"name": "tully_dual_avoided_crossing"}, mass=2000.0)


# --------------------------------------------------------------------------
# 1. unbiased at t = 0
# --------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("init_state", [0, 1])
def test_estimator_recovers_rho_W_at_t0(scheme, init_state):
    rng = np.random.default_rng(11 + init_state)
    ens = sample_projected_ensemble(rng, 400_000, nstates=2,
                                    init_state=init_state, R0=-15.0, P0=40.0,
                                    sigma_R=0.25, hbar=HBAR, scheme=scheme)
    rho = rho_W_from_ensemble(ens.x, ens.weights, HBAR, 2)
    exact = np.zeros((2, 2), dtype=complex)
    exact[init_state, init_state] = 1.0
    # the error must be consistent with the estimator's own standard error,
    # not with an arbitrary tolerance
    c = wigner_aa_dagger_all(ens.x, HBAR, 2)
    per = ens.n_samples * ens.weights[:, None, None] * c
    sem = np.std(np.real(per), axis=0, ddof=1) / np.sqrt(ens.n_samples)
    assert np.max(np.abs(np.real(rho) - exact)) < 6.0 * np.max(sem)
    assert abs(np.trace(rho).real - 1.0) < 6.0 * np.max(sem)


def test_phi_weights_are_the_projector_polynomial():
    """Under the phi proposal the weight is Pi_{lam lam}/N_s -- an identity."""
    rng = np.random.default_rng(3)
    ens = sample_projected_ensemble(rng, 5000, nstates=2, init_state=0,
                                    R0=0.0, P0=0.0, sigma_R=1.0, hbar=HBAR,
                                    scheme="wigner_seo")
    rho0 = np.zeros((2, 2), dtype=complex)
    rho0[0, 0] = 1.0
    u = projected_importance_weights(ens.x, rho0, HBAR, 2)
    assert np.allclose(ens.weights, np.real(u) / ens.n_samples, atol=0, rtol=1e-14)
    closed = 2.0 / HBAR * (ens.x[:, 0] ** 2 + ens.x[:, 2] ** 2 - HBAR / 2.0)
    assert np.allclose(np.real(u), closed, rtol=1e-13, atol=1e-13)


def test_sign_optimal_weights_are_bounded_and_two_valued():
    rng = np.random.default_rng(4)
    ens = sample_projected_ensemble(rng, 20_000, nstates=2, init_state=0,
                                    R0=0.0, P0=0.0, sigma_R=1.0, hbar=HBAR,
                                    scheme="sign_optimal")
    w = ens.weights * ens.n_samples
    assert np.allclose(np.abs(w), SIGN_OPTIMAL_Z, rtol=1e-13)
    assert set(np.unique(np.sign(w))) <= {-1.0, 1.0}
    assert abs(SIGN_OPTIMAL_Z - (4.0 * np.exp(-0.5) - 1.0)) < 1e-15


def test_tilted_action_law_matches_its_closed_form():
    """The proposal's radial law, against the exact CDF."""
    from scipy.integrate import quad
    t = _sample_tilted_action(np.random.default_rng(2), 400_000)
    norm = 2 * np.exp(-0.5) - 0.5
    for tq in (0.2, 0.5, 1.0, 2.0, 4.0):
        exact = quad(lambda u: np.exp(-u) * abs(u - 0.5) / norm, 0, tq)[0]
        assert abs(float(np.mean(t <= tq)) - exact) < 0.005, tq


def test_the_two_proposals_agree_within_their_errors():
    """Different proposals, same estimand: the agreement is a real check."""
    m = _tully()
    out = {}
    for scheme, n in (("sign_optimal", 200_000), ("wigner_seo", 200_000)):
        ens = sample_projected_ensemble(np.random.default_rng(41), n, nstates=2,
                                        init_state=0, R0=-3.0, P0=40.0,
                                        sigma_R=0.25, hbar=HBAR, scheme=scheme)
        h = propagate_pbme_projected_ensemble(ens, m, 0.5, 600, hbar=HBAR,
                                              n_save=600, n_blocks=40)
        out[scheme] = block_statistics(
            np.real(np.diagonal(h.arrays["block_rho"], axis1=-2, axis2=-1)))
    a, b = out["sign_optimal"], out["wigner_seo"]
    gap = np.abs(a["mean"][-1] - b["mean"][-1])
    tol = 3.0 * np.sqrt(a["sem"][-1] ** 2 + b["sem"][-1] ** 2)
    assert np.all(gap < tol), (gap, tol)
    # and the sign-optimal proposal is the one with the smaller error
    assert np.max(a["sem"][-1]) < np.max(b["sem"][-1])


# --------------------------------------------------------------------------
# 2. exact for an R-independent problem
# --------------------------------------------------------------------------
def _expm_2x2(A):
    w, V = np.linalg.eig(A)
    return V @ np.diag(np.exp(w)) @ np.linalg.inv(V)


def test_exact_for_R_independent_hamiltonian():
    r"""With ``h`` constant the excess term vanishes, PBME is the full MQCLE,
    and the mapping flow is one common rotation, so the estimator must
    reproduce the exact two-level solution -- not merely something close."""
    m = ConstantCouplingModel(a0=0.02, b0=-0.01, c0=0.005, mass=2000.0)
    rng = np.random.default_rng(5)
    ens = sample_projected_ensemble(rng, 200_000, nstates=2, init_state=0,
                                    R0=0.0, P0=0.0, sigma_R=1.0, hbar=HBAR)
    hist = propagate_pbme_projected_ensemble(ens, m, 0.5, 400, hbar=HBAR,
                                             n_save=100, n_blocks=20)
    h = m.h(np.zeros((1, 1)))[0]
    rho_exact = []
    for t in hist.times:
        U = _expm_2x2(-1j * h * t / HBAR)
        rho_exact.append(np.real(np.diag(U @ np.diag([1.0, 0.0]) @ U.conj().T)))
    rho_exact = np.array(rho_exact)
    st = block_statistics(np.real(np.diagonal(hist.arrays["block_rho"],
                                              axis1=-2, axis2=-1)))
    err = np.abs(st["mean"] - rho_exact)
    assert np.all(err <= 4.0 * np.maximum(st["sem"], 1e-12)), (err, st["sem"])

    # A sharper statement holds here and is checked to round-off rather than to
    # a sampling tolerance.  With ``h`` constant every trajectory shares the
    # same mapping rotation, so the estimator carries its OWN initial sample
    # exactly: rho_hat(t) = U^dagger rho_hat(0) U.  Any deviation is the
    # propagator's error, not the ensemble's.
    #
    # This is also why the t=0 control variate is NOT exact for this problem:
    # it subtracts rho_hat(0), whereas what the rotation reproduces is
    # U^dagger rho_hat(0) U.  The two differ by the rotated sampling error, and
    # the control variate is a variance reduction, not an identity.
    blocks = hist.arrays["block_rho"]
    rho_hat0 = blocks[0].sum(axis=0)
    for k, t in enumerate(hist.times):
        U = _expm_2x2(-1j * h * t / HBAR)
        assert np.max(np.abs(blocks[k].sum(axis=0)
                             - U @ rho_hat0 @ U.conj().T)) < 1e-11, t


# --------------------------------------------------------------------------
# 3. signed weights
# --------------------------------------------------------------------------
def test_phi_proposal_negative_fraction_is_one_minus_exp_minus_half():
    rng = np.random.default_rng(7)
    ens = sample_projected_ensemble(rng, 200_000, nstates=2, init_state=0,
                                    R0=-15.0, P0=40.0, sigma_R=0.25, hbar=HBAR,
                                    scheme="wigner_seo")
    frac = float(np.mean(ens.weights < 0.0))
    assert abs(frac - (1.0 - np.exp(-0.5))) < 0.01
    assert ens.meta["signed_weight_fraction_negative"] == pytest.approx(frac)
    # a vacuity guard: the assertion above would also pass if the negative
    # weights were tiny, so check that they actually carry weight
    assert np.sum(np.abs(ens.weights[ens.weights < 0])) > 0.1 * np.sum(
        np.abs(ens.weights))


def test_sign_optimal_negative_fraction_is_the_tilted_mass():
    r"""Under q the negative mass is (e^{-1/2}-1/2)/(2e^{-1/2}-1/2)."""
    rng = np.random.default_rng(8)
    ens = sample_projected_ensemble(rng, 200_000, nstates=2, init_state=0,
                                    R0=-15.0, P0=40.0, sigma_R=0.25, hbar=HBAR,
                                    scheme="sign_optimal")
    expect = (np.exp(-0.5) - 0.5) / (2 * np.exp(-0.5) - 0.5)
    assert abs(float(np.mean(ens.weights < 0.0)) - expect) < 0.01


@pytest.mark.parametrize("rho0", [
    np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex),          # real coherence
    np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=complex),       # imaginary
    np.array([[0.7, 0.2 + 0.3j], [0.2 - 0.3j, 0.3]], dtype=complex),
], ids=["real coherence", "imaginary coherence", "general Hermitian"])
def test_weight_is_real_for_every_hermitian_initial_density(rho0):
    r"""The weight of a *physical* initial state is real, and that is a theorem.

    With :math:`\rho_W=A+iB`, :math:`A` symmetric and :math:`B` antisymmetric,
    and :math:`\Pi=\frac2\hbar(u-iL-\frac\hbar2\delta)` with :math:`u`
    symmetric and :math:`L` antisymmetric, the cross terms
    :math:`\sum uB` and :math:`\sum LA` vanish by symmetry and what survives is
    real.  Equivalently: the projected mapping density of a density matrix is a
    real function.  An earlier version of this test asserted the opposite for a
    coherent state and failed, which is how the identity was noticed.
    """
    rng = np.random.default_rng(9)
    x = rng.normal(0.0, np.sqrt(HBAR / 2), size=(4000, 4))
    u = projected_importance_weights(x, rho0, HBAR, 2)
    assert np.max(np.abs(np.imag(u))) == 0.0
    assert np.max(np.abs(np.real(u))) > 1.0        # anti-vacuity


def test_complex_weights_survive_for_a_non_hermitian_input():
    """The complex-safe path exists for the case that is genuinely complex."""
    rng = np.random.default_rng(10)
    x = rng.normal(0.0, np.sqrt(HBAR / 2), size=(2000, 4))
    rho0 = np.array([[0.5, 0.4 + 0.1j], [0.9 - 0.7j, 0.5]], dtype=complex)
    u = projected_importance_weights(x, rho0, HBAR, 2)
    assert np.max(np.abs(np.imag(u))) > 1.0


# --------------------------------------------------------------------------
# 4. no normalisation by the stochastic weight sum
# --------------------------------------------------------------------------
def test_estimator_is_linear_in_the_weights():
    """Scaling every weight by ``a`` must scale the answer by ``a``.

    A ratio estimator ``sum w c / sum w`` is invariant under this scaling, so
    this single assertion rules the whole family out.
    """
    m = _tully()
    rng = np.random.default_rng(13)
    ens = sample_projected_ensemble(rng, 20_000, nstates=2, init_state=0,
                                    R0=-15.0, P0=40.0, sigma_R=0.25, hbar=HBAR)
    scaled = Ensemble(R=ens.R, P=ens.P, x=ens.x, weights=3.0 * ens.weights,
                      meta=dict(ens.meta))
    kw = dict(hbar=HBAR, convention=TraceConvention.TRACELESS, n_save=20)
    a = propagate_pbme_projected_ensemble(ens, m, 0.5, 20, **kw)
    b = propagate_pbme_projected_ensemble(scaled, m, 0.5, 20, **kw)
    pa = np.array([r["populations"] for r in a.records])
    pb = np.array([r["populations"] for r in b.records])
    assert np.allclose(pb, 3.0 * pa, rtol=1e-13, atol=1e-15)
    assert a.meta["weights_renormalised_by_their_sum"] is False


def test_trace_drift_is_roundoff_only():
    r"""``sum_nu c_{nu nu}`` is a function of the mapping radius alone, and the
    frozen-``R`` rotation is orthogonal, so the estimated trace cannot drift.
    This is the check that would catch a stepper that silently rescales ``x``.
    """
    m = _tully()
    rng = np.random.default_rng(17)
    ens = sample_projected_ensemble(rng, 20_000, nstates=2, init_state=0,
                                    R0=-3.0, P0=40.0, sigma_R=0.25, hbar=HBAR)
    hist = propagate_pbme_projected_ensemble(ens, m, 0.5, 400, hbar=HBAR,
                                             n_save=50)
    tr = np.real([r["trace"] for r in hist.records])
    assert np.max(np.abs(tr - tr[0])) < 1e-11
    assert hist.records[-1]["mapping_radius_drift"] < 1e-10


# --------------------------------------------------------------------------
# the two branches must not be interchangeable
# --------------------------------------------------------------------------
def test_branches_reject_each_others_ensembles():
    m = _tully()
    rng = np.random.default_rng(19)
    kw = dict(nstates=2, init_state=0, R0=-15.0, P0=40.0, sigma_R=0.25, hbar=HBAR)
    foc = sample_matched_ensemble(rng, 200, **kw)
    prj = sample_projected_ensemble(rng, 200, **kw)
    with pytest.raises(ValueError, match="importance"):
        propagate_pbme_projected_ensemble(foc, m, 0.5, 2)
    with pytest.raises(ValueError, match="focused"):
        propagate_pbme_ensemble(prj, m, 0.5, 2)


def test_focused_and_projected_disagree_on_R_dependent_dynamics():
    """The two PBME branches are *different* quantities.

    If this ever passed as an equality, separating ``E_discard`` from
    ``E_focused`` in the manuscript would be vacuous, so the difference is
    asserted to be resolvable rather than assumed.
    """
    m = _tully()
    kw = dict(nstates=2, init_state=0, R0=-2.0, P0=40.0, sigma_R=0.25, hbar=HBAR)
    foc = sample_matched_ensemble(np.random.default_rng(23), 100_000, **kw)
    prj = sample_projected_ensemble(np.random.default_rng(29), 200_000, **kw)
    a = propagate_pbme_ensemble(foc, m, 0.5, 600, hbar=HBAR, n_save=600)
    b = propagate_pbme_projected_ensemble(prj, m, 0.5, 600, hbar=HBAR,
                                          n_save=600, n_blocks=40)
    st = block_statistics(np.real(np.diagonal(b.arrays["block_rho"],
                                              axis1=-2, axis2=-1)))
    gap = np.abs(np.asarray(a.records[-1]["populations"]) - st["mean"][-1])
    assert np.max(gap) > 3.0 * np.max(st["sem"][-1])


def test_block_subset_mean_is_the_direct_weighted_average():
    r"""The ``k``-block mean must equal the plain estimator over those samples.

    Each block sum carries the global :math:`1/N_s`, so one block's own
    estimator is ``B * block`` for the *total* block count ``B``, not ``k *
    block`` for the subset size.  Scaling by the subset size makes every subset
    mean and standard error too small by ``B/k``, which shows up as a standard
    error that grows with sample size.  This identity pins it down exactly
    rather than by a trend.
    """
    n, B = 64_000, 32
    ens = sample_projected_ensemble(np.random.default_rng(3), n, nstates=2,
                                    init_state=0, R0=-15.0, P0=40.0,
                                    sigma_R=0.25, hbar=HBAR)
    c = wigner_aa_dagger_all(ens.x, HBAR, 2)
    wc = ens.weights[:, None, None].astype(complex) * c
    blocks = np.swapaxes(wc.reshape(1, B, n // B, 2, 2).sum(axis=2), -1, -2)
    d = np.real(np.diagonal(blocks, axis1=-2, axis2=-1))
    prev = None
    for k in (1, 2, 4, 8, 16, 32):
        st = block_statistics(d, k)
        m = k * (n // B)
        direct = np.real(np.diagonal(
            rho_W_from_ensemble(ens.x[:m], ens.weights[:m] * B / k, HBAR, 2)))
        assert np.allclose(st["mean"][0], direct, rtol=1e-12, atol=1e-14), k
        if prev is not None and k > 2:
            # and the standard error falls, which the buggy version reversed
            assert st["sem"][0, 0] < prev
        prev = st["sem"][0, 0]


def test_block_statistics_scales_as_one_over_sqrt_n():
    rng = np.random.default_rng(31)
    ens = sample_projected_ensemble(rng, 160_000, nstates=2, init_state=0,
                                    R0=-15.0, P0=40.0, sigma_R=0.25, hbar=HBAR)
    hist = propagate_pbme_projected_ensemble(ens, _tully(), 0.5, 20, hbar=HBAR,
                                             n_save=20, n_blocks=32)
    d = np.real(np.diagonal(hist.arrays["block_rho"], axis1=-2, axis2=-1))
    s4 = block_statistics(d, 4)["sem"][-1, 0]
    s32 = block_statistics(d, 32)["sem"][-1, 0]
    # eight times the samples -> sqrt(8) = 2.83 times smaller, within the
    # sampling error of a variance estimated from four blocks
    assert 1.5 < s4 / s32 < 5.5
