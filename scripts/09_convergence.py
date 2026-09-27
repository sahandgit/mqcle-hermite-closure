#!/usr/bin/env python3
"""Gate L acceptance -- measured convergence of the coefficient solver.

The directive's rule: *report the measured order, do not assume it*, and if
time-step convergence fails, do not report the run.  This script measures

1. **time step** -- ``dt`` refinement at fixed grid, against a finer-``dt``
   reference.  The Strang splitting is expected to be second order and RK4
   fourth order, but both numbers here are measured, not assumed.
2. **spatial resolution** -- ``n_R`` at fixed ``n_P``, then ``n_P`` at fixed
   ``n_R``, each against its own finest member.
3. **box independence** -- the box is periodic, so the answer must not depend
   on where the (empty) boundary sits.  Boxes are enlarged with ``dR`` and
   ``dP`` held fixed, so only the box changes, not the resolution.

The observable is the final diabatic population vector; the reported error is
its max-norm difference.  Everything is written to
``results/raw/09_convergence.json`` and ``results/tables/*.csv``.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqcle.coefficient_operator import (build_grid,                  # noqa: E402
                                        spectral_radius_estimate)
from mqcle.io import RunRecorder, load_config                        # noqa: E402
from mqcle.models import build_model                                 # noqa: E402
from mqcle.propagation import (grid_observables,                     # noqa: E402
                               initial_coefficient_field,
                               propagate_coefficients)
from mqcle.types import TraceConvention                              # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="tully_default.yaml")
    p.add_argument("--P0", type=float, default=40.0)
    p.add_argument("--R0", type=float, default=-15.0)
    p.add_argument("--T", type=float, default=None, help="default 2 M |R0| / P0")
    p.add_argument("--integrator", choices=("strang", "rk4"), default="strang")
    p.add_argument("--out", default="results/raw/09_convergence.json")
    p.add_argument("--quick", action="store_true",
                   help="shorter T and smaller sweeps, for a smoke test")
    return p.parse_args(argv)


def _run(model, hbar, convention, R0, P0, sigma_R, init_state, T, dt,
         R_half, n_R, P_half, n_P, integrator):
    grid = build_grid(-R_half, R_half, n_R, P0 - P_half, P0 + P_half, n_P)
    rho0 = initial_coefficient_field(grid, R0, P0, sigma_R, init_state,
                                     model.nstates, hbar)
    n_steps = int(round(T / dt))
    t0 = time.perf_counter()
    hist = propagate_coefficients(rho0, grid, model, dt, n_steps, hbar=hbar,
                                  convention=convention, n_save=n_steps,
                                  integrator=integrator)
    wall = time.perf_counter() - t0
    fin = hist.records[-1]
    return {"populations": np.asarray(fin["populations"], dtype=float),
            "adiabatic": np.asarray(fin["adiabatic_populations"], dtype=float),
            "trace": float(np.real(fin["trace"])),
            "trace_drift": float(abs(np.real(fin["trace"])
                                     - np.real(hist.records[0]["trace"]))),
            "hermiticity": float(max(r["hermiticity"] for r in hist.records)),
            "boundary_mass": float(max(r["boundary_mass"] for r in hist.records)),
            "wall_time_s": wall,
            "cfl_number": float(hist.meta["cfl_number"]),
            "n_steps": n_steps}


def _orders(hs, errs):
    """Least-squares slope of log(err) against log(h), plus pairwise orders."""
    hs, errs = np.asarray(hs, float), np.asarray(errs, float)
    ok = errs > 0
    fit = float(np.polyfit(np.log(hs[ok]), np.log(errs[ok]), 1)[0]) if ok.sum() > 1 else float("nan")
    pair = [float(np.log2(errs[i] / errs[i + 1]) / np.log2(hs[i] / hs[i + 1]))
            for i in range(len(hs) - 1) if errs[i] > 0 and errs[i + 1] > 0]
    return fit, pair


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    hbar, mass = float(cfg["hbar"]), float(cfg["mass"])
    convention = TraceConvention(cfg["trace_convention"])
    model = build_model(cfg["model"], mass=mass)
    P0, R0 = args.P0, args.R0
    sigma_R = hbar / (2.0 * (P0 / 20.0))
    init_state = int(cfg["initial_conditions"]["init_state"])
    T = float(args.T if args.T is not None else 2.0 * mass * abs(R0) / P0)
    if args.quick:
        T = min(T, 300.0)

    def run(dt, R_half, n_R, P_half, n_P):
        return _run(model, hbar, convention, R0, P0, sigma_R, init_state, T, dt,
                    R_half, n_R, P_half, n_P, args.integrator)

    rec = RunRecorder(script=Path(__file__).name, seed=int(cfg["seed"]), config={
        "P0": P0, "R0": R0, "sigma_R": sigma_R, "T": T,
        "integrator": args.integrator, "quick": bool(args.quick),
        "trace_convention": convention.value, "mass": mass, "hbar": hbar,
        "observable": "final diabatic population vector, max-norm difference"})
    results: dict = {}

    # -- 1. time step ----------------------------------------------------
    dts = [1.0, 0.5, 0.25] if args.quick else [4.0, 2.0, 1.0, 0.5, 0.25]
    dt_ref = dts[-1] / 2.0
    nR_dt, nP_dt, Rh, Ph = (256, 64, 45.0, 22.0) if args.quick else (512, 128, 45.0, 22.0)
    print(f"[09] time-step study on a {nR_dt}x{nP_dt} grid, T={T}, "
          f"reference dt={dt_ref}")
    ref = run(dt_ref, Rh, nR_dt, Ph, nP_dt)
    rows = []
    for dt in dts:
        r = run(dt, Rh, nR_dt, Ph, nP_dt)
        err = float(np.max(np.abs(r["populations"] - ref["populations"])))
        rows.append({"dt": dt, "error": err, "cfl": r["cfl_number"],
                     "trace_drift": r["trace_drift"],
                     "wall_time_s": r["wall_time_s"], "n_steps": r["n_steps"]})
        print(f"     dt={dt:<7g} err={err:.4e}  cfl={r['cfl_number']:.3g}  "
              f"{r['wall_time_s']:.1f} s")
    fit, pair = _orders([r["dt"] for r in rows], [r["error"] for r in rows])
    expected = 2.0 if args.integrator == "strang" else 4.0
    results["time_step"] = {
        "grid": {"n_R": nR_dt, "n_P": nP_dt, "R_half": Rh, "P_half": Ph},
        "reference_dt": dt_ref, "rows": rows,
        "fitted_order": fit, "pairwise_orders": pair,
        "expected_order": expected,
        "reference_populations": ref["populations"].tolist()}
    print(f"[09] measured time-step order: fit {fit:.3f}, "
          f"pairwise {['%.3f' % o for o in pair]}  (expected {expected})")
    rec.check("gateL.time_step_order", abs(fit - expected), 0.35,
              f"fitted order {fit:.3f} vs expected {expected} for "
              f"integrator='{args.integrator}'")
    rec.check("gateL.time_step_error_decreases",
              0.0 if all(rows[i]["error"] > rows[i + 1]["error"]
                         for i in range(len(rows) - 1)) else 1.0, 0.5,
              "the error must fall monotonically under dt refinement")

    # -- 2. spatial resolution -------------------------------------------
    dt_fix = 0.5
    nRs = [128, 256, 512] if args.quick else [256, 512, 1024, 2048]
    nP_fix = 64 if args.quick else 512
    print(f"[09] n_R study at n_P={nP_fix}, dt={dt_fix}")
    runs = {n: run(dt_fix, Rh, n, Ph, nP_fix) for n in nRs}
    fine = runs[nRs[-1]]
    rows = []
    for n in nRs[:-1]:
        err = float(np.max(np.abs(runs[n]["populations"] - fine["populations"])))
        rows.append({"n_R": n, "dR": 2 * Rh / n, "dR_over_sigma_R": 2 * Rh / n / sigma_R,
                     "error": err, "wall_time_s": runs[n]["wall_time_s"]})
        print(f"     n_R={n:<6d} dR/sigma_R={2*Rh/n/sigma_R:.3f}  err={err:.4e}")
    results["n_R"] = {"n_P": nP_fix, "dt": dt_fix, "reference_n_R": nRs[-1],
                      "rows": rows,
                      "reference_populations": fine["populations"].tolist()}
    rec.check("gateL.n_R_converged", rows[-1]["error"], 1e-6,
              f"|pop(n_R={rows[-1]['n_R']}) - pop(n_R={nRs[-1]})|")

    nPs = [32, 64, 128] if args.quick else [128, 256, 512, 1024]
    nR_fix = 256 if args.quick else 1024
    print(f"[09] n_P study at n_R={nR_fix}, dt={dt_fix}")
    sigma_P = hbar / (2.0 * sigma_R)
    runsP = {n: run(dt_fix, Rh, nR_fix, Ph, n) for n in nPs}
    fineP = runsP[nPs[-1]]
    rows = []
    for n in nPs[:-1]:
        err = float(np.max(np.abs(runsP[n]["populations"] - fineP["populations"])))
        rows.append({"n_P": n, "dP": 2 * Ph / n, "dP_over_sigma_P": 2 * Ph / n / sigma_P,
                     "error": err, "wall_time_s": runsP[n]["wall_time_s"]})
        print(f"     n_P={n:<6d} dP/sigma_P={2*Ph/n/sigma_P:.3f}  err={err:.4e}")
    results["n_P"] = {"n_R": nR_fix, "dt": dt_fix, "reference_n_P": nPs[-1],
                      "rows": rows,
                      "reference_populations": fineP["populations"].tolist()}
    rec.check("gateL.n_P_converged", rows[-1]["error"], 1e-6,
              f"|pop(n_P={rows[-1]['n_P']}) - pop(n_P={nPs[-1]})|")

    # -- 3. box independence ---------------------------------------------
    # dR and dP are held FIXED while the box grows, so only the boundary moves.
    print("[09] box-independence study (dR, dP held fixed)")
    base_Rh, base_nR, base_Ph, base_nP = Rh, (256 if args.quick else 512), Ph, \
        (64 if args.quick else 128)
    dR0, dP0 = 2 * base_Rh / base_nR, 2 * base_Ph / base_nP
    boxes = [(base_Rh, base_Ph), (base_Rh * 1.5, base_Ph * 1.5),
             (base_Rh * 2.0, base_Ph * 2.0)]
    rows, base_pop = [], None
    for Rh_i, Ph_i in boxes:
        n_R_i = int(round(2 * Rh_i / dR0))
        n_P_i = int(round(2 * Ph_i / dP0))
        r = run(dt_fix, Rh_i, n_R_i, Ph_i, n_P_i)
        if base_pop is None:
            base_pop = r["populations"]
        err = float(np.max(np.abs(r["populations"] - base_pop)))
        rows.append({"R_half": Rh_i, "P_half": Ph_i, "n_R": n_R_i, "n_P": n_P_i,
                     "error_vs_smallest_box": err,
                     "boundary_mass": r["boundary_mass"],
                     "wall_time_s": r["wall_time_s"]})
        print(f"     R_half={Rh_i:<6g} P_half={Ph_i:<6g} err={err:.4e}  "
              f"boundary_mass={r['boundary_mass']:.2e}")
    results["box"] = {"dR": dR0, "dP": dP0, "rows": rows}
    rec.check("gateL.box_independence",
              max(r["error_vs_smallest_box"] for r in rows), 1e-8,
              "final populations must not depend on where the empty boundary sits")

    rec.results = results
    out = Path(args.out)
    rec.write(out)

    tables = Path("results/tables")
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "convergence_time_step.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results["time_step"]["rows"][0]))
        w.writeheader()
        w.writerows(results["time_step"]["rows"])
    with open(tables / "convergence_grid.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "n", "spacing", "spacing_over_sigma", "error", "wall_time_s"])
        for r in results["n_R"]["rows"]:
            w.writerow(["n_R", r["n_R"], r["dR"], r["dR_over_sigma_R"],
                        r["error"], r["wall_time_s"]])
        for r in results["n_P"]["rows"]:
            w.writerow(["n_P", r["n_P"], r["dP"], r["dP_over_sigma_P"],
                        r["error"], r["wall_time_s"]])
    with open(tables / "convergence_box.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results["box"]["rows"][0]))
        w.writeheader()
        w.writerows(results["box"]["rows"])

    print(f"[09] checks: {len(rec.checks)}  all passed: {rec.all_passed}")
    for c in rec.checks:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['name']:32s} "
              f"{c['value']:.3e}  (tol {c['tolerance']:.1e})  {c['note']}")
    print(f"[09] written to {out} and results/tables/convergence_*.csv")
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
