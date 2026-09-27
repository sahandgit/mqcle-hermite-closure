"""Consistency checks for every production momentum regime."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mqcle.benchmark_regimes import MOMENTA, REGIMES, TAGS, regime_for


ROOT = Path(__file__).resolve().parents[1]


def test_all_requested_momenta_are_registered_and_tagged():
    assert MOMENTA == (10, 20, 30, 40, 70, 100)
    assert set(TAGS) == set(MOMENTA)
    assert all(regime_for(float(P)) is REGIMES[P] for P in MOMENTA)
    assert regime_for(10.5) is None


def test_grids_resolve_initial_packet_and_phase_space_shear():
    for regime in REGIMES.values():
        LR = regime.R_box[1] - regime.R_box[0]
        LP = 2.0 * regime.P_half
        dR = LR / regime.n_R
        dP = LP / regime.n_P
        assert dR / regime.sigma_R <= 0.36
        assert dP / regime.shear_P_scale <= 0.30
        sigma_P = regime.P0 / 20.0
        lo, hi = regime.P_box
        assert lo <= regime.P0 - 8.0 * sigma_P
        assert hi >= regime.P0 + 8.0 * sigma_P


def test_times_are_exact_integer_numbers_of_steps_and_quantum_grid_is_resolved():
    for regime in REGIMES.values():
        assert np.isclose(regime.T / regime.dt,
                          round(regime.T / regime.dt), atol=1e-12)
        assert np.isclose(regime.T / regime.se_dt,
                          round(regime.T / regime.se_dt), atol=1e-12)
        LR = regime.R_box[1] - regime.R_box[0]
        points_per_wavelength = (2.0 * np.pi * regime.se_n_R
                                 / (LR * regime.P_box[1]))
        assert points_per_wavelength >= 8.0


def test_low_momentum_boxes_include_the_reflected_branch():
    for P0 in (10, 20):
        lo, _ = REGIMES[P0].P_box
        assert lo <= -P0


def test_new_production_settings_are_pinned():
    expected = {
        10: ((-45.0, 45.0), 4096, (-30.0, 50.0), 2048, 6000.0),
        30: ((-45.0, 45.0), 1024, (8.0, 52.0), 512, 2000.0),
        70: ((-35.0, 35.0), 2048, (38.0, 102.0), 1024, 857.5),
    }
    for P0, (R_box, n_R, P_box, n_P, T) in expected.items():
        r = REGIMES[P0]
        assert r.R_box == R_box
        assert r.n_R == n_R
        assert r.P_box == P_box
        assert r.n_P == n_P
        assert r.T == T
        assert r.dt == 0.5


#: macro suffix of each regime in manuscript/paper_values.tex
TAG = {10: "PTen", 20: "PTwenty", 30: "PThirty", 40: "PForty",
       70: "PSeventy", 100: "POneHundred"}


def test_pipeline_and_results_tables_require_all_six_regimes():
    runner = (ROOT / "run_final_pipeline.ps1").read_text()
    results = (ROOT / "manuscript" / "mqcle_hermite_closure.tex").read_text()
    assert set(TAG) == set(MOMENTA)
    for P0 in MOMENTA:
        assert f"08_leakage_P{P0}.json" in runner
        assert f"10_schrodinger_convergence_P{P0}.json" in runner
        assert f"11_pbme_projected_P{P0}.json" in runner
        assert f"11b_pbme_seed_convergence_P{P0}.json" in runner
        # the results table quotes every regime from the generated macros,
        # in the plain or in the table form of each number
        for stems in (("qPop",), ("ghPop",), ("pbPop", "pbPopTab"),
                      ("eQcle", "eQcleTab"), ("ePbme", "ePbmeTab")):
            assert any(f"\\{stem}{TAG[P0]}" in results for stem in stems), (stems, P0)
    assert "Assert-AggregateInputs" in runner
    assert "Assert-PipelineCodeUnchanged" in runner
    assert "07c_p10_refinement_acceptance.py" in runner
    assert "Assert-P10Accepted" in runner
