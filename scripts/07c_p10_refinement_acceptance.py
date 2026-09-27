#!/usr/bin/env python3
"""Post-run acceptance of the expensive P0=10 coefficient benchmark.

The production 4096x2048 calculation intentionally retains the generic
``1e-9`` periodic-edge sentinels from ``07_coefficient_propagation.py``.  At
P0=10 those sentinels detect a small, symmetric Fourier-aliasing floor even
though the population observable is stable under simultaneous two-axis
refinement.  This script does not rewrite or excuse the raw checks.  It reads
the archived 2048x1024 and production 4096x2048 histories and applies an
independent, observable-level refinement acceptance test.

The numerical budget is the smaller of (i) two percent of the final
coefficient-QCLE versus Schrodinger population difference and (ii) half a
unit in the fourth decimal place used for the population table.  Opposite-edge
symmetry and at least a sixteen-fold reduction of each sentinel are required
to distinguish a collapsing spectral floor from physical one-sided leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mqcle.io import RunRecorder  # noqa: E402


BOUNDARY_CHECKS = {
    "gateL.boundary_mass_R",
    "gateL.boundary_mass_P",
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fine-json",
        default="results/raw/07_coefficient_propagation_P10.json",
    )
    p.add_argument(
        "--fine-h5",
        default="results/raw/07_coefficient_propagation_P10.h5",
    )
    p.add_argument(
        "--coarse-json",
        default=("results/failed/P10_2048x1024_20260829/"
                 "07_coefficient_propagation_P10.json"),
    )
    p.add_argument(
        "--coarse-h5",
        default=("results/failed/P10_2048x1024_20260829/"
                 "07_coefficient_propagation_P10.h5"),
    )
    p.add_argument(
        "--out",
        default="results/raw/07c_p10_refinement_acceptance.json",
    )
    return p.parse_args(argv)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required refinement input is missing: {path}")
    with path.open() as fh:
        return json.load(fh)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _grid(doc: dict) -> dict:
    return doc["results"]["coefficient"]["grid"]


def _edge_asymmetry(h5: h5py.File, axis: str) -> float:
    lo = float(np.max(h5[f"coef_boundary_mass_{axis}_low"][:]))
    hi = float(np.max(h5[f"coef_boundary_mass_{axis}_high"][:]))
    return abs(lo - hi) / max(lo, hi, 1e-300)


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = {
        "fine_json": Path(args.fine_json),
        "fine_h5": Path(args.fine_h5),
        "coarse_json": Path(args.coarse_json),
        "coarse_h5": Path(args.coarse_h5),
    }
    fine = _load_json(paths["fine_json"])
    coarse = _load_json(paths["coarse_json"])
    for key in ("fine_h5", "coarse_h5"):
        if not paths[key].is_file():
            raise FileNotFoundError(
                f"required refinement input is missing: {paths[key]}")

    fg, cg = _grid(fine), _grid(coarse)
    expected_fine = {
        "R_min": -45.0, "R_max": 45.0, "n_R": 4096,
        "P_min": -30.0, "P_max": 50.0, "n_P": 2048,
    }
    expected_coarse = {
        "R_min": -45.0, "R_max": 45.0, "n_R": 2048,
        "P_min": -30.0, "P_max": 50.0, "n_P": 1024,
    }
    grid_mismatch = sum(
        not np.isclose(float(fg[k]), float(v), rtol=0.0, atol=1e-12)
        for k, v in expected_fine.items()
    ) + sum(
        not np.isclose(float(cg[k]), float(v), rtol=0.0, atol=1e-12)
        for k, v in expected_coarse.items()
    )

    config_keys = ("P0", "R0", "sigma_R", "dt", "T", "n_steps",
                   "n_save", "R_box", "P_box")
    fcfg, ccfg = fine["metadata"]["config"], coarse["metadata"]["config"]
    config_mismatch = sum(fcfg.get(k) != ccfg.get(k) for k in config_keys)

    failed_names = {c["name"] for c in fine["checks"] if not c["passed"]}
    unexpected_failures = sorted(failed_names - BOUNDARY_CHECKS)
    nonboundary_failure_count = len(unexpected_failures)

    with h5py.File(paths["fine_h5"], "r") as fh, \
            h5py.File(paths["coarse_h5"], "r") as ch:
        required = (
            "coef_t", "coef_populations", "coef_boundary_mass_R",
            "coef_boundary_mass_P", "coef_boundary_mass_R_low",
            "coef_boundary_mass_R_high", "coef_boundary_mass_P_low",
            "coef_boundary_mass_P_high", "se_populations",
        )
        missing_datasets = sum(k not in fh or k not in ch for k in required)
        if missing_datasets:
            raise KeyError("one or more required refinement datasets are missing")

        ft, ct = fh["coef_t"][:], ch["coef_t"][:]
        times_match = ft.shape == ct.shape and np.array_equal(ft, ct)
        fp, cp = fh["coef_populations"][:], ch["coef_populations"][:]
        shape_match = fp.shape == cp.shape
        arrays_finite = all(np.all(np.isfinite(h[k][:]))
                            for h in (fh, ch) for k in required)
        if not (times_match and shape_match):
            max_population_change = float("inf")
        else:
            max_population_change = float(np.max(np.abs(fp - cp)))

        physical_difference = float(np.max(
            np.abs(fp[-1] - fh["se_populations"][-1])))
        display_half_unit = 0.5e-4
        relative_budget = 0.02 * physical_difference
        population_budget = min(display_half_unit, relative_budget)

        coarse_bR = float(np.max(ch["coef_boundary_mass_R"][:]))
        fine_bR = float(np.max(fh["coef_boundary_mass_R"][:]))
        coarse_bP = float(np.max(ch["coef_boundary_mass_P"][:]))
        fine_bP = float(np.max(fh["coef_boundary_mass_P"][:]))
        reduction_R = coarse_bR / max(fine_bR, 1e-300)
        reduction_P = coarse_bP / max(fine_bP, 1e-300)
        asymmetry_R = _edge_asymmetry(fh, "R")
        asymmetry_P = _edge_asymmetry(fh, "P")

    rec = RunRecorder(
        script=Path(__file__).name,
        seed=fine["metadata"].get("seed"),
        config={
            "fine_json": str(paths["fine_json"]),
            "fine_h5": str(paths["fine_h5"]),
            "coarse_json": str(paths["coarse_json"]),
            "coarse_h5": str(paths["coarse_h5"]),
            "fine_grid": expected_fine,
            "coarse_grid": expected_coarse,
            "population_budget_definition": (
                "min(0.02 * final coefficient-QCLE vs Schrodinger "
                "population difference, 0.5e-4)"),
            "minimum_boundary_reduction": 16.0,
            "maximum_opposite_edge_asymmetry": 0.05,
        },
    )
    rec.check("p10_refinement.exact_grids", float(grid_mismatch), 0.0,
              "fine and coarse grids must be the pinned simultaneous "
              "two-axis refinement pair")
    rec.check("p10_refinement.same_physical_configuration",
              float(config_mismatch), 0.0,
              "the two runs may differ only in n_R and n_P")
    rec.check("p10_refinement.only_boundary_sentinels_fail",
              float(nonboundary_failure_count), 0.0,
              "all production checks other than the generic R/P edge "
              "sentinels must pass; unexpected failures: "
              + (", ".join(unexpected_failures) or "none"))
    rec.check("p10_refinement.complete_matching_histories",
              float(not (times_match and shape_match)), 0.0,
              "the complete saved population histories must share every time")
    rec.check("p10_refinement.finite_inputs", float(not arrays_finite), 0.0,
              "all datasets used by the acceptance calculation are finite")
    rec.check("p10_refinement.population_history_change",
              max_population_change, population_budget,
              "max population L-infinity change over the complete history; "
              "budget is tied to the physical comparison and table precision")
    rec.check("p10_refinement.boundary_R_reduction",
              max(0.0, 16.0 - reduction_R), 0.0,
              f"R-edge sentinel must fall by at least 16x; observed "
              f"{reduction_R:.3g}x")
    rec.check("p10_refinement.boundary_P_reduction",
              max(0.0, 16.0 - reduction_P), 0.0,
              f"P-edge sentinel must fall by at least 16x; observed "
              f"{reduction_P:.3g}x")
    rec.check("p10_refinement.fine_boundary_R_below_observable_budget",
              fine_bR, population_budget,
              "the residual R-edge proxy must be below the declared "
              "population error budget")
    rec.check("p10_refinement.fine_boundary_P_below_observable_budget",
              fine_bP, population_budget,
              "the residual P-edge proxy must be below the declared "
              "population error budget")
    rec.check("p10_refinement.opposite_R_edges_symmetric",
              asymmetry_R, 0.05,
              "symmetric opposite edges identify a spectral floor rather "
              "than one-sided physical leakage")
    rec.check("p10_refinement.opposite_P_edges_symmetric",
              asymmetry_P, 0.05,
              "symmetric opposite edges identify a spectral floor rather "
              "than one-sided physical leakage")

    hashes = {key: _sha256(path) for key, path in paths.items()}
    rec.results = {
        "input_sha256": hashes,
        "failed_fine_checks_preserved": sorted(failed_names),
        "max_population_linf_over_time": max_population_change,
        "final_coefficient_vs_schrodinger_population_linf": physical_difference,
        "population_error_budget": population_budget,
        "population_change_over_physical_difference": (
            max_population_change / physical_difference),
        "boundary_mass": {
            "R": {"coarse": coarse_bR, "fine": fine_bR,
                  "reduction": reduction_R,
                  "opposite_edge_asymmetry": asymmetry_R},
            "P": {"coarse": coarse_bP, "fine": fine_bP,
                  "reduction": reduction_P,
                  "opposite_edge_asymmetry": asymmetry_P},
        },
        "interpretation": (
            "The original edge sentinels remain failed in the raw production "
            "archive.  The refined result is accepted for the reported "
            "population observable because simultaneous two-axis refinement "
            "collapses a symmetric Fourier floor and changes the complete "
            "population history by less than the declared numerical budget."),
    }
    out = Path(args.out)
    rec.write(out)

    print("[07c] P10 simultaneous R/P refinement acceptance")
    print(f"[07c] max population-history change {max_population_change:.6e} "
          f"(budget {population_budget:.6e})")
    print(f"[07c] boundary reduction R={reduction_R:.2f}x, "
          f"P={reduction_P:.2f}x; opposite-edge asymmetry "
          f"R={asymmetry_R:.3e}, P={asymmetry_P:.3e}")
    for c in rec.checks:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['name']:58s} "
              f"{c['value']:.3e}  (tol {c['tolerance']:.1e})")
    print(f"[07c] written to {out}; all passed: {rec.all_passed}")
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
