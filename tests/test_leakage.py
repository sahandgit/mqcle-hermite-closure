"""Gate M: the leakage functional and its sample-based estimator."""
from __future__ import annotations

import numpy as np
import pytest

from mqcle.coefficient_operator import build_grid, spectral_derivative
from mqcle.leakage import (kde_grad_P_rho_W, kde_rho_W, leakage_from_grid,
                           leakage_coefficient_forms, silverman_bandwidth)
from mqcle.models import ConstantCouplingModel, build_model
from mqcle.propagation import initial_coefficient_field
from mqcle.sampling import sample_initial_ensemble

HBAR, MASS = 1.0, 2000.0


def test_only_degree_four_multi_indices_survive():
    """(I - P_2) of a degree-4 amplitude leaves exactly the degree-4 part."""
    keys, _ = leakage_coefficient_forms(2, 1, "traceless")
    assert len(keys) == 31
    assert {sum(k) for k in keys} == {4}


def test_leakage_vanishes_identically_when_grad_R_h_is_zero():
    """Structural: every coefficient is bilinear in grad_R h and grad_P rho."""
    free = ConstantCouplingModel(a0=0.02, b0=-0.01, c0=0.005, k=0.0, mass=MASS)
    grid = build_grid(-20.0, 20.0, 128, 20.0, 60.0, 64)
    rho = initial_coefficient_field(grid, -1.0, 40.0, 1.0)
    out = leakage_from_grid(rho, grid, free)
    assert out["norm"] == 0.0
    assert out["rho_norm"] > 0.0


def test_leakage_is_nonzero_in_the_coupling_region_and_small_outside():
    model = build_model({"name": "tully_dual_avoided_crossing"}, mass=MASS)
    grid = build_grid(-45.0, 45.0, 256, 20.0, 60.0, 128)
    inside = leakage_from_grid(
        initial_coefficient_field(grid, 0.0, 40.0, 0.5), grid, model)["relative"]
    outside = leakage_from_grid(
        initial_coefficient_field(grid, -30.0, 40.0, 0.5), grid, model)["relative"]
    assert inside > 1e-3
    assert outside < inside / 100.0


def test_leakage_is_independent_of_the_evaluation_grid():
    model = build_model({"name": "tully_dual_avoided_crossing"}, mass=MASS)
    vals = []
    for n_R, n_P in ((256, 128), (512, 256)):
        g = build_grid(-45.0, 45.0, n_R, 20.0, 60.0, n_P)
        vals.append(leakage_from_grid(
            initial_coefficient_field(g, 0.0, 40.0, 0.5), g, model)["relative"])
    assert abs(vals[0] - vals[1]) / vals[1] < 1e-6


# --------------------------------------------------------------------------
# the sample-based estimator
# --------------------------------------------------------------------------
def test_binned_kde_matches_the_direct_kde_above_two_cells():
    """The binned estimator deconvolves the deposition window exactly.

    Measured relative agreement with the direct O(n_grid * n) sum:
      h = 1 cell  1.4e-2 | h = 2 cells 1.2e-3 | h = 3 cells 3.3e-4
    so the two-cell floor used by scripts/08 is where the binning stops
    contributing at the percent level.
    """
    grid = build_grid(-20.0, 20.0, 128, 20.0, 60.0, 64)
    rng = np.random.default_rng(1)
    ens = sample_initial_ensemble(rng, 3000, nstates=2, init_state=0, R0=-1.0,
                                  P0=40.0, sigma_R=1.0, scheme="focused")
    for fn, tol in ((kde_rho_W, 2e-3), (kde_grad_P_rho_W, 3e-3)):
        a, _ = fn(ens.x, ens.R, ens.P, grid, h_R=2 * grid.dR, h_P=2 * grid.dP,
                  method="direct")
        b, _ = fn(ens.x, ens.R, ens.P, grid, h_R=2 * grid.dR, h_P=2 * grid.dP,
                  method="binned")
        assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < tol


def test_kde_density_recovers_the_initial_packet_populations():
    """A sanity floor: the *value* of rho_W is easy to estimate."""
    grid = build_grid(-45.0, 45.0, 512, 20.0, 60.0, 256)
    rng = np.random.default_rng(7)
    ens = sample_initial_ensemble(rng, 50000, nstates=2, init_state=0, R0=-15.0,
                                  P0=40.0, sigma_R=0.5, scheme="focused")
    est, _ = kde_rho_W(ens.x, ens.R, ens.P, grid, h_R=3 * grid.dR,
                       h_P=3 * grid.dP)
    pops = np.real(np.einsum("ijaa->a", est)) * grid.cell_area
    assert abs(pops[0] - 1.0) < 0.02
    assert abs(pops[1]) < 0.02


def test_kde_grad_P_is_much_harder_than_the_density_itself():
    r"""The point of Gate M: value convergence does not give operator
    convergence.  The estimator's relative error on :math:`\nabla_P\rho_W` is
    compared against its relative error on :math:`\rho_W` itself, on the same
    sample, against exact analytic ground truth."""
    grid = build_grid(-45.0, 45.0, 512, 20.0, 60.0, 256)
    R0, P0, sR = -15.0, 40.0, 0.5
    exact_rho = initial_coefficient_field(grid, R0, P0, sR)
    exact_gP = spectral_derivative(exact_rho, grid.kP, axis=1)

    rng = np.random.default_rng(11)
    ens = sample_initial_ensemble(rng, 50000, nstates=2, init_state=0, R0=R0,
                                  P0=P0, sigma_R=sR, scheme="focused")
    kw = dict(h_R=3 * grid.dR, h_P=3 * grid.dP)
    est_rho, _ = kde_rho_W(ens.x, ens.R, ens.P, grid, **kw)
    est_gP, _ = kde_grad_P_rho_W(ens.x, ens.R, ens.P, grid, **kw)

    dens = np.abs(np.trace(exact_rho, axis1=-2, axis2=-1))
    m = dens > 1e-4 * dens.max()

    def rel(e, x):
        return float(np.sqrt(np.mean(np.abs((e - x)[m]) ** 2))
                     / np.sqrt(np.mean(np.abs(x[m]) ** 2)))

    r_val = rel(est_rho[..., 0, 0], exact_rho[..., 0, 0])
    r_op = rel(est_gP[..., 0, 0], exact_gP[..., 0, 0])
    assert r_op > r_val, (f"derivative estimate ({r_op:.4f}) should be worse "
                          f"than the value estimate ({r_val:.4f})")


def test_silverman_bandwidth_scales_as_n_to_the_minus_fifth():
    rng = np.random.default_rng(3)
    hs = [silverman_bandwidth(rng.normal(0.0, 2.0, size=n))
          for n in (1000, 100000)]
    assert 0.9 < (hs[0] / hs[1]) / 100.0 ** 0.2 < 1.1
