"""Convergence checks for the full-quantum Tully-II reference."""
from __future__ import annotations

import numpy as np

from mqcle.models import build_model
from mqcle.schrodinger import (SplitOperatorPropagator,
                               build_schrodinger_grid,
                               gaussian_wavepacket,
                               wavefunction_observables)

MASS = 2000.0


def _final(R_box, n_R, dt, T=400.0):
    model = build_model({"name": "tully_dual_avoided_crossing"}, mass=MASS)
    grid = build_schrodinger_grid(*R_box, n_R)
    psi = gaussian_wavepacket(grid, -15.0, 40.0, 0.25)
    prop = SplitOperatorPropagator(grid, model, dt=dt)
    for _ in range(int(round(T / dt))):
        psi = prop.step(psi)
    return wavefunction_observables(psi, grid, model)


def test_split_operator_is_second_order_in_dt():
    ref = _final((-40.0, 40.0), 2048, 0.125)["populations"]
    coarse = _final((-40.0, 40.0), 2048, 2.0)["populations"]
    half = _final((-40.0, 40.0), 2048, 1.0)["populations"]
    fine = _final((-40.0, 40.0), 2048, 0.5)["populations"]
    order = np.log2(np.max(np.abs(coarse - ref)) /
                    np.max(np.abs(half - ref)))
    assert 1.7 < order < 2.3, f"measured order {order:.3f}"
    assert np.max(np.abs(fine - ref)) < np.max(np.abs(half - ref))


def test_population_is_stable_under_spatial_refinement():
    production = _final((-40.0, 40.0), 2048, 0.5)["populations"]
    refined = _final((-40.0, 40.0), 4096, 0.5)["populations"]
    assert np.max(np.abs(production - refined)) < 1e-7


def test_population_is_independent_of_an_enlarged_box_at_fixed_resolution():
    base = _final((-40.0, 40.0), 2048, 0.5)["populations"]
    enlarged = _final((-60.0, 60.0), 3072, 0.5)["populations"]
    assert np.max(np.abs(base - enlarged)) < 1e-7
