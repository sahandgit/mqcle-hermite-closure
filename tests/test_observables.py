"""Module 7: readout of rho_W from grids, quadrature and ensembles."""
from __future__ import annotations

import numpy as np
import pytest

from mqcle.observables import (coherence, gauss_hermite_weights,
                               hermiticity_error, lebesgue_nodes,
                               observables_from_rho_W, phi_expectation_nodes,
                               populations, rho_W_from_focused_ensemble,
                               rho_W_from_quadrature, trace_norm)
from mqcle.projector import projected_density
from mqcle.sampling import sample_mapping_focused

N, HBAR = 2, 1.0
RHO = np.array([[0.6 + 0j, 0.25 - 0.15j], [0.25 + 0.15j, 0.4 + 0j]])


def test_gauss_hermite_weights_integrate_the_gaussian():
    r"""$\int dx\, e^{-a|x|^2} = (\pi/a)^{d/2}$."""
    for a in (1.0, 2.0, 0.5):
        _, w = gauss_hermite_weights(2 * N, a, npts=12)
        assert abs(np.sum(w) - (np.pi / a) ** (N)) < 1e-10


def test_phi_expectation_weights_sum_to_one():
    _, w = phi_expectation_nodes(N, HBAR, npts=12)
    assert abs(np.sum(w) - 1.0) < 1e-12


def test_quadrature_readout_is_exact_for_the_projected_density():
    back = rho_W_from_quadrature(
        lambda xx: projected_density(xx, RHO, HBAR, N), HBAR, N, npts=16)
    assert np.max(np.abs(back - RHO)) < 1e-11


@pytest.mark.parametrize("init", [0, 1])
def test_focused_readout_reproduces_a_pure_state_exactly(init):
    """Focused sampling has the right second moments, and c is quadratic, so
    the plain average is exact for the initial density -- no importance
    weights, and none may be applied."""
    n = 20_000
    x = sample_mapping_focused(np.random.default_rng(51), n, N, init, HBAR)
    rho = rho_W_from_focused_ensemble(x, hbar=HBAR, nstates=N)
    want = np.zeros((N, N), dtype=complex)
    want[init, init] = 1.0
    # The POPULATIONS are exact: the action radii are fixed, so the diagonal
    # of c is the same at every sample point and no averaging is involved.
    assert np.max(np.abs(np.diagonal(rho) - np.diagonal(want))) < 1e-12
    # The COHERENCE involves cross terms r_a r_b whose mean vanishes only in
    # expectation over the uniform angles, so it carries Monte-Carlo error.
    assert np.max(np.abs(rho - want)) < 5.0 / np.sqrt(n)


def test_focused_readout_accepts_explicit_weights():
    x = sample_mapping_focused(np.random.default_rng(52), 4000, N, 0, HBAR)
    w = np.full(x.shape[0], 1.0 / x.shape[0])
    a = rho_W_from_focused_ensemble(x, hbar=HBAR, nstates=N)
    b = rho_W_from_focused_ensemble(x, weights=w, hbar=HBAR, nstates=N)
    assert np.max(np.abs(a - b)) < 1e-14


def test_scalar_observables():
    assert np.allclose(populations(RHO), [0.6, 0.4])
    re, im = coherence(RHO, 0, 1)
    assert abs(re - 0.25) < 1e-15 and abs(im + 0.15) < 1e-15
    assert abs(trace_norm(RHO) - 1.0) < 1e-15
    assert hermiticity_error(RHO) < 1e-15
    bad = RHO.copy()
    bad[0, 1] += 0.1
    assert hermiticity_error(bad) > 0.05


def test_observables_bundle_reports_the_matrix_elements():
    out = observables_from_rho_W(RHO)
    assert abs(out["rho00"] - 0.6) < 1e-15
    assert abs(out["rho11"] - 0.4) < 1e-15
    assert abs(out["re_rho01"] - 0.25) < 1e-15
    assert abs(out["im_rho01"] + 0.15) < 1e-15
    assert abs(out["trace"] - 1.0) < 1e-15
    assert out["hermiticity_error"] < 1e-15
