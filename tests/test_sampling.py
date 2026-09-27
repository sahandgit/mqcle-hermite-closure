"""Module 4: initial-condition sampling, focused and SEO."""
from __future__ import annotations

import numpy as np
import pytest

from mqcle.sampling import (focused_radii, mapping_actions,
                            sample_bath_wigner_gaussian,
                            sample_initial_ensemble, sample_mapping_focused,
                            sample_mapping_wigner_seo)

N, HBAR = 2, 1.0


def test_focused_radii_are_the_semiclassical_actions():
    r"""$r^2 + p^2 = 2\hbar(n_\lambda + 1/2)$: 3 hbar for the occupied state,
    1 hbar for the others.  The radii are their square roots."""
    for init in range(N):
        rad = focused_radii(N, init, HBAR)
        want = np.array([3.0 if i == init else 1.0 for i in range(N)]) * HBAR
        assert np.max(np.abs(rad**2 - want)) < 1e-14


def test_focused_sampling_puts_every_point_on_its_action_circle():
    for init in range(N):
        x = sample_mapping_focused(np.random.default_rng(21), 5000, N, init, HBAR)
        acts = mapping_actions(x, N)
        want = np.array([3.0 if i == init else 1.0 for i in range(N)]) * HBAR
        assert np.max(np.abs(acts - want)) < 1e-12


def test_focused_angles_are_uniform_so_the_cross_moments_vanish():
    x = sample_mapping_focused(np.random.default_rng(22), 200_000, N, 0, HBAR)
    r, p = x[:, :N], x[:, N:]
    n = x.shape[0]
    assert np.max(np.abs(r.T @ p / n)) < 2e-2
    off = r.T @ r / n
    np.fill_diagonal(off, 0.0)
    assert np.max(np.abs(off)) < 2e-2


def test_seo_sampling_reproduces_the_phi_second_moments():
    x = sample_mapping_wigner_seo(np.random.default_rng(23), 400_000, N, HBAR)
    for a in (x[:, :N], x[:, N:]):
        cov = a.T @ a / a.shape[0]
        assert np.max(np.abs(cov - 0.5 * HBAR * np.eye(N))) < 6e-3


def test_bath_sampling_uses_the_minimum_uncertainty_widths():
    R0, P0, sR = -15.0, 40.0, 0.25
    R, P = sample_bath_wigner_gaussian(np.random.default_rng(24), 400_000,
                                       R0, P0, sR, HBAR)
    assert abs(R.mean() - R0) < 5e-3 and abs(R.std() - sR) < 5e-3
    sP = HBAR / (2 * sR)
    assert abs(P.mean() - P0) < 5e-2 and abs(P.std() - sP) / sP < 5e-3


def test_ensemble_is_internally_consistent_and_reproducible():
    kw = dict(nstates=N, init_state=0, R0=-15.0, P0=40.0, sigma_R=0.25,
              scheme="focused", hbar=HBAR)
    a = sample_initial_ensemble(np.random.default_rng(25), 1000, **kw)
    b = sample_initial_ensemble(np.random.default_rng(25), 1000, **kw)
    assert a.n_samples == 1000
    assert np.array_equal(a.x, b.x) and np.array_equal(a.R, b.R)
    assert abs(a.weights.sum() - 1.0) < 1e-12
    assert a.R.shape == (1000, 1) and a.x.shape == (1000, 2 * N)


def test_unknown_scheme_is_rejected():
    with pytest.raises((ValueError, KeyError)):
        sample_initial_ensemble(np.random.default_rng(26), 10, nstates=N,
                                init_state=0, R0=0.0, P0=1.0, sigma_R=1.0,
                                scheme="not_a_scheme")
