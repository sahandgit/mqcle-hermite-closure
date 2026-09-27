#!/usr/bin/env python3
"""Observable grid-independence at one initial momentum.

Motivation.  At ``P0 = 20`` the ``boundary_mass_P`` sentinel does not reach
its 1e-9 tolerance even on the production grid, while the reported observable
appears unchanged.  Rather than loosen the sentinel, this script measures the
thing the sentinel is a proxy for: how much the reported observable actually
moves under grid refinement.  The comparison in the manuscript is meaningful
only if that motion is far below the difference being reported.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqcle.coefficient_operator import build_grid                    # noqa: E402
from mqcle.io import RunRecorder, load_config                        # noqa: E402
from mqcle.models import build_model                                 # noqa: E402
from mqcle.propagation import (grid_observables,                     # noqa: E402
                               initial_coefficient_field,
                               propagate_coefficients)
from mqcle.types import TraceConvention                              # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="tully_default.yaml")
    p.add_argument("--P0", type=float, default=20.0)
    p.add_argument("--R0", type=float, default=-15.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--n-R", type=int, default=512)
    p.add_argument("--n-P", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--R-half", type=float, default=45.0)
    p.add_argument("--P-half", type=float, default=20.0)
    p.add_argument("--reported-error", type=float, default=None,
                   help="the manuscript's quoted error for this P0; the grid "
                        "motion is judged against it")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    hbar, mass = float(cfg["hbar"]), float(cfg["mass"])
    conv = TraceConvention(cfg["trace_convention"])
    model = build_model(cfg["model"], mass=mass)
    P0, R0 = args.P0, args.R0
    sigma_R = hbar / (2.0 * (P0 / 20.0))
    init = int(cfg["initial_conditions"]["init_state"])
    T = 2.0 * mass * abs(R0) / P0
    n_steps = int(round(T / args.dt))

    rec = RunRecorder(script=Path(__file__).name, seed=int(cfg["seed"]), config={
        "P0": P0, "R0": R0, "sigma_R": sigma_R, "T": T, "dt": args.dt,
        "n_R": args.n_R, "n_P_list": args.n_P, "R_half": args.R_half,
        "P_half": args.P_half, "trace_convention": conv.value})

    rows = []
    print(f"[07b] P0={P0}  n_R={args.n_R}  T={T}  dt={args.dt}")
    for n_P in args.n_P:
        g = build_grid(-args.R_half, args.R_half, args.n_R,
                       P0 - args.P_half, P0 + args.P_half, n_P)
        rho = initial_coefficient_field(g, R0, P0, sigma_R, init,
                                        model.nstates, hbar)
        t0 = time.perf_counter()
        h = propagate_coefficients(rho, g, model, args.dt, n_steps, hbar=hbar,
                                   convention=conv, n_save=n_steps)
        fin = h.records[-1]
        rows.append({"n_P": n_P, "dP": 2 * args.P_half / n_P,
                     "populations": np.asarray(fin["populations"]).tolist(),
                     "boundary_mass_P": float(max(r["boundary_mass_P"]
                                                  for r in h.records)),
                     "boundary_mass_R": float(max(r["boundary_mass_R"]
                                                  for r in h.records)),
                     "wall_time_s": time.perf_counter() - t0})
        print(f"      n_P={n_P:<6d} dP={rows[-1]['dP']:.4f}  "
              f"pop={np.round(rows[-1]['populations'], 6)}  "
              f"bmass_P={rows[-1]['boundary_mass_P']:.2e}  "
              f"{rows[-1]['wall_time_s']:.0f} s", flush=True)

    pops = np.array([r["populations"] for r in rows])
    spread = float(np.max(np.abs(pops - pops[-1])))
    results = {"rows": rows, "observable_spread": spread,
               "note": "max |population(n_P) - population(finest)| over the "
                       "refinement; this is the quantity the boundary sentinel "
                       "is a proxy for"}
    if args.reported_error:
        results["reported_error"] = args.reported_error
        results["spread_over_reported_error"] = spread / args.reported_error
        rec.check("gateL.observable_grid_independence",
                  spread / args.reported_error, 0.01,
                  f"grid motion {spread:.3e} against the reported error "
                  f"{args.reported_error:.3e}; the comparison is only "
                  f"meaningful if this ratio is small")
    else:
        rec.check("gateL.observable_grid_independence", spread, 1e-5,
                  f"max population change under n_P refinement")
    rec.results = results
    out = Path(args.out or f"results/raw/07b_grid_independence_P{int(round(P0))}.json")
    rec.write(out)
    print(f"[07b] observable spread {spread:.3e}"
          + (f"  = {spread/args.reported_error:.4f} x the reported error"
             if args.reported_error else ""))
    for c in rec.checks:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['name']}  "
              f"{c['value']:.3e}  (tol {c['tolerance']:.1e})")
    print(f"[07b] written to {out}")
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
