#!/usr/bin/env python3
r"""Convergence of the **full quantum reference itself** (PART VII).

Why this script exists
----------------------
``tests/test_schrodinger.py`` checks the split-operator implementation against
free-particle spreading, a constant electronic Hamiltonian, unitarity, and the
expected second-order behaviour.  Those are implementation checks.  They do not
establish that the *production* Tully-II reference is converged, and the
manuscript quotes coefficient-versus-quantum differences as small as
:math:`2\times10^{-5}`.  A reference whose own discretisation error is of that
size cannot support the number.

Unitarity is not accuracy: every factor of the split-operator propagator is
unitary, so the norm is conserved to machine precision at *any* time step.  The
observables are what must be refined.

What is run, for each production momentum regime
------------------------------------------------
One production run, four one-axis refinements, and one combined finest run:

======================  ==========================================
``production``          exactly the settings ``scripts/07`` uses
``dt/2``, ``dt/4``      temporal refinement at fixed grid and box
``grid x2``             spatial refinement at fixed box and ``dt``
``box x1.5``            enlarged box at fixed ``dR`` and ``dt``
``finest``              all three at once
======================  ==========================================

The reference uncertainty is the change from the production setting to the
combined finest run, and the one-axis runs say which axis, if any, is
responsible.  The temporal order is fitted from the three time steps.

Usage
-----
    python scripts/10_schrodinger_convergence.py --P0 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqcle.io import RunRecorder, load_config                        # noqa: E402
from mqcle.benchmark_regimes import REGIMES                          # noqa: E402
from mqcle.models import build_model                                 # noqa: E402
from mqcle.schrodinger import (SplitOperatorPropagator,              # noqa: E402
                               build_schrodinger_grid,
                               gaussian_wavepacket)

#: The production reference settings of ``scripts/07``, restated here so the
#: convergence study refines *exactly* what the production runs used.  Checked
#: against the archived run in ``main``, which refuses to proceed on a mismatch.
PRODUCTION = {
    float(P0): dict(R_box=regime.R_box, n_R=regime.se_n_R,
                    dt=regime.se_dt, T=regime.T)
    for P0, regime in REGIMES.items()
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="tully_default.yaml")
    p.add_argument("--P0", type=float, required=True)
    p.add_argument("--R0", type=float, default=None)
    p.add_argument("--sigma-R", type=float, default=None)
    p.add_argument("--n-save", type=int, default=40)
    p.add_argument("--box-factor", type=float, default=1.5)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def _run(model, R_box, n_R, dt, T, R0, P0, sigma_R, init_state, hbar, n_save):
    grid = build_schrodinger_grid(R_box[0], R_box[1], n_R)
    psi = gaussian_wavepacket(grid, R0, P0, sigma_R, init_state,
                              model.nstates, hbar)
    prop = SplitOperatorPropagator(grid, model, dt=dt, hbar=hbar)
    n_steps = int(round(T / dt))
    every = max(1, int(round(n_steps / n_save)))
    t0 = time.perf_counter()
    times, recs, _ = prop.propagate(psi, n_steps, n_save=every)
    return {
        "R_box": list(R_box), "n_R": int(n_R), "dR": float(grid.dR),
        "dt": float(dt), "n_steps": int(n_steps),
        "wall_time_s": round(time.perf_counter() - t0, 3),
        "times": [float(t) for t in times],
        "populations": np.array([r["populations"] for r in recs]),
        "adiabatic_populations": np.array([r["adiabatic_populations"]
                                           for r in recs]),
        "norm_drift": float(abs(recs[-1]["norm"] - 1.0)),
        "boundary_mass": float(max(r["boundary_mass"] for r in recs)),
    }


def _compare(a: dict, b: dict) -> dict:
    """Final and worst-in-time population differences between two runs."""
    ta, tb = np.array(a["times"]), np.array(b["times"])
    t = ta if ta.size <= tb.size else tb
    pa = np.stack([np.interp(t, ta, a["populations"][:, k]) for k in range(2)], 1)
    pb = np.stack([np.interp(t, tb, b["populations"][:, k]) for k in range(2)], 1)
    return {
        "final_population_difference":
            float(np.max(np.abs(a["populations"][-1] - b["populations"][-1]))),
        "max_population_difference_over_time": float(np.max(np.abs(pa - pb))),
        "final_adiabatic_population_difference":
            float(np.max(np.abs(a["adiabatic_populations"][-1]
                                - b["adiabatic_populations"][-1]))),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    ic = cfg["initial_conditions"]
    hbar, mass = float(cfg["hbar"]), float(cfg["mass"])
    model = build_model(cfg["model"], mass=mass)
    P0 = float(args.P0)
    if P0 not in PRODUCTION:
        raise SystemExit(f"no production settings recorded for P0={P0}")
    pr = PRODUCTION[P0]
    R0 = float(args.R0 if args.R0 is not None else ic["R0"])
    sigma_R = float(args.sigma_R if args.sigma_R is not None
                    else hbar / (2.0 * (P0 / 20.0)))
    init_state = int(ic["init_state"])
    T = float(pr["T"])

    archived = (Path("results/raw/13_P20_wide_momentum_box.json")
                if P0 == 20.0
                else Path(f"results/raw/07_coefficient_propagation_P{int(P0)}.json"))
    provenance = "settings not cross-checked (no archived production run)"
    if archived.exists():
        d = json.loads(archived.read_text())["results"].get("schrodinger", {})
        if d:
            same = (d["grid"]["n_R"] == pr["n_R"] and d["dt"] == pr["dt"]
                    and abs(d["grid"]["R_min"] - pr["R_box"][0]) < 1e-9)
            provenance = ("matches the archived production reference" if same
                          else "DIFFERS from the archived production reference")
            if not same:
                raise SystemExit(
                    "[10] refusing to run: the settings in PRODUCTION do not "
                    "match the archived production reference for this regime "
                    f"({d['grid']['n_R']} points, dt={d['dt']}, "
                    f"R_min={d['grid']['R_min']}).  Refining the wrong "
                    "settings would say nothing about the number in the paper.")

    rec = RunRecorder(script=Path(__file__).name, seed=int(cfg["seed"]), config={
        **{k: v for k, v in cfg.items() if not k.startswith("_")},
        "P0": P0, "R0": R0, "sigma_R": sigma_R, "T": T,
        "production": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in pr.items()},
        "production_provenance": provenance,
        "box_factor": float(args.box_factor)})

    L = pr["R_box"][1] - pr["R_box"][0]
    c = 0.5 * (pr["R_box"][0] + pr["R_box"][1])
    Lb = L * float(args.box_factor)
    big_box = (c - 0.5 * Lb, c + 0.5 * Lb)
    big_n = int(round(pr["n_R"] * args.box_factor))       # fixed dR

    plan = {
        "production": dict(R_box=pr["R_box"], n_R=pr["n_R"], dt=pr["dt"]),
        "dt_half":    dict(R_box=pr["R_box"], n_R=pr["n_R"], dt=pr["dt"] / 2),
        "dt_quarter": dict(R_box=pr["R_box"], n_R=pr["n_R"], dt=pr["dt"] / 4),
        "grid_x2":    dict(R_box=pr["R_box"], n_R=2 * pr["n_R"], dt=pr["dt"]),
        "box_enlarged": dict(R_box=big_box, n_R=big_n, dt=pr["dt"]),
        "finest":     dict(R_box=big_box, n_R=2 * big_n, dt=pr["dt"] / 4),
    }
    runs = {}
    for name, kw in plan.items():
        runs[name] = _run(model, kw["R_box"], kw["n_R"], kw["dt"], T, R0, P0,
                          sigma_R, init_state, hbar, args.n_save)
        r = runs[name]
        print(f"[10] {name:13s} n_R={r['n_R']:6d} dR={r['dR']:.5f} "
              f"dt={r['dt']:.5f}  pops={np.round(r['populations'][-1], 9)} "
              f"({r['wall_time_s']:.0f}s)", flush=True)

    p1 = runs["production"]["populations"][-1]
    p2 = runs["dt_half"]["populations"][-1]
    p4 = runs["dt_quarter"]["populations"][-1]
    d12 = float(np.max(np.abs(p1 - p2)))
    d24 = float(np.max(np.abs(p2 - p4)))
    order = float(np.log2(d12 / d24)) if d24 > 0 else float("inf")
    remaining_dt = (d24 / (2.0 ** order - 1.0)
                    if np.isfinite(order) and order > 0 else d24)

    comparisons = {name: _compare(runs[name], runs["production"])
                   for name in runs if name != "production"}
    u_final = comparisons["finest"]["final_population_difference"]
    u_time = comparisons["finest"]["max_population_difference_over_time"]

    results = {
        "P0": P0, "T": T,
        "runs": {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                     for kk, vv in v.items()} for k, v in runs.items()},
        "comparisons_against_production": comparisons,
        "temporal": {"d(dt, dt/2)": d12, "d(dt/2, dt/4)": d24,
                     "fitted_order": order,
                     "richardson_remaining_at_dt_over_4": remaining_dt},
        "reference_uncertainty": {
            "final_populations": u_final,
            "max_over_time": u_time,
            "definition": "|production - finest| where finest is dt/4, twice "
                          "the grid points per unit length, and an enlarged "
                          "box, all at once",
            "converged_value_final_populations":
                runs["finest"]["populations"][-1].tolist(),
            "production_value_final_populations":
                runs["production"]["populations"][-1].tolist()},
    }
    rec.results = results

    rec.check("schrodinger.temporal_order", max(0.0, 1.7 - order), 0.0,
              f"fitted temporal order {order:.2f}; Strang splitting is second "
              f"order and at least 1.7 is required (faster is acceptable)")
    rec.check("schrodinger.norm_drift",
              max(r["norm_drift"] for r in runs.values()), 1e-9,
              "unitarity of every run; a check on the implementation, NOT on "
              "its accuracy")
    rec.check("schrodinger.boundary_mass",
              max(r["boundary_mass"] for r in runs.values()), 1e-8,
              "reference density reaching the periodic boundary")
    print(f"[10] P0={P0}: temporal order {order:.2f}; reference uncertainty "
          f"{u_final:.3e} (final), {u_time:.3e} (worst in time)")

    out = Path(args.out or f"results/raw/10_schrodinger_convergence_P{int(P0)}.json")
    rec.write(out)
    print(f"[10] written to {out};  checks all passed: {rec.all_passed}")
    for cc in rec.checks:
        print(f"  {'PASS' if cc['passed'] else 'FAIL'}  {cc['name']:34s} "
              f"{cc['value']:.3e}  (tol {cc['tolerance']:.1e})")
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
