#!/usr/bin/env python3
r"""Gate GH-10 at production scale: Hermite branch against the Strang reference.

Two implementations of one equation, sharing no line of right-hand side:

* the Gaussian--Hermite coefficient system, :math:`M=11` coefficients advanced
  unsplit through Eq. (finite-Hermite) by Runge--Kutta;
* the subsystem-QCLE reference, four real fields advanced by exactly split
  Strang steps.

Neither is a second physical theory.  What their agreement establishes is that
the algebra which produced the finite operator is right and that both
discretisations are converged enough for the difference between them to be
discretisation rather than disagreement.  The signature of that is not a small
number at one resolution -- it is a number that *falls* under refinement, which
is why the study is run on a sequence of grids and why the ratio, not the
value, is the check.

The Hermite branch takes the largest step its spectral radius allows; the
reference takes the production step, which its unconditional stability permits.
The two therefore do not share a time step and must not: forcing a common step
would measure the worse of the two integrators rather than the equation.

Timings are recorded per grid.  They are the source of the manuscript's
statement that this branch is *more* expensive than the one it replaces, which
the paper reports rather than hides.

Writes ``results/raw/07g_gh_crosscheck_P40_convergence.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# The kernel threads over modes and over R chunks; BLAS must not thread inside
# those workers as well.  Environment variables are the portable mechanism and
# only take effect before numpy is imported, so this must stay above the
# imports below.  See mqcle.hermite.kernel for the measurement behind it.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mqcle.coefficient_operator import (build_grid,                  # noqa: E402
                                        populations_from_rho)
from mqcle.constants import CDTYPE, HBAR                             # noqa: E402
from mqcle.hermite.frame import (hermite_metric, hermite_to_rho,     # noqa: E402
                                 physical_frame_matrix,
                                 physical_frame_residual,
                                 rho_to_hermite, weighted_pseudoinverse)
from mqcle.hermite.kernel import (RealHermiteKernel,                 # noqa: E402
                                  generator_is_real, rk4_real,
                                  state_is_real)
from mqcle.hermite.propagator import (HermiteGenerator,              # noqa: E402
                                      propagate_rk4)
from mqcle.models import TullyDualAvoidedCrossing                    # noqa: E402
from mqcle.propagation import StrangPropagator                       # noqa: E402

METHOD = "gaussian_hermite_coefficient"


def _initial(grid, R0, P0, sigma_R):
    sigma_P = HBAR / (2.0 * sigma_R)
    Rg, Pg = np.meshgrid(grid.R, grid.P, indexing="ij")
    w = (np.exp(-((Rg - R0) ** 2) / (2.0 * sigma_R ** 2)
                - ((Pg - P0) ** 2) / (2.0 * sigma_P ** 2)) / (np.pi * HBAR))
    rho0 = np.zeros(grid.shape + (2, 2), dtype=CDTYPE)
    rho0[..., 0, 0] = w
    return rho0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--P0", type=float, default=40.0)
    ap.add_argument("--R0", type=float, default=-15.0)
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--dt-strang", type=float, default=0.5,
                    help="the production step of the reference branch")
    ap.add_argument("--cfl", type=float, default=0.5,
                    help="fraction of the Runge--Kutta stability limit")
    ap.add_argument("--grids", default="256x128,512x256",
                    help="comma-separated n_R x n_P, coarsest first")
    ap.add_argument("--threads", type=int, default=None,
                    help="worker threads for the production kernel timing")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    grids = []
    for spec in args.grids.split(","):
        a, b = spec.lower().split("x")
        grids.append((int(a), int(b)))
    if len(grids) < 2:
        raise SystemExit("[10g] at least two grids are needed: the check is "
                         "that the difference falls under refinement")

    out = Path(args.out or ROOT / "results" / "raw"
               / f"07g_gh_crosscheck_P{int(round(args.P0))}_convergence.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    model = TullyDualAvoidedCrossing(mass=2000.0)
    sigma_R = 10.0 / args.P0
    results = {}

    for (n_R, n_P) in grids:
        grid = build_grid(-45.0, 45.0, n_R,
                          args.P0 - 22.0, args.P0 + 22.0, n_P)
        rho0 = _initial(grid, args.R0, args.P0, sigma_R)

        # ---- Hermite branch ---------------------------------------------
        gen = HermiteGenerator.from_model(model, grid, K=2, parity="even")
        B = physical_frame_matrix(model.nstates, gen.modes)
        W = hermite_metric(gen.modes)
        Bplus = weighted_pseudoinverse(B, W)
        dt_h = args.cfl * gen.spectral_radius_estimate()["rk4_dt_limit"]
        n_h = max(1, int(np.ceil(args.T / dt_h)))
        dt_h = args.T / n_h
        C0 = gen.to_state(rho_to_hermite(rho0, B))
        t0 = time.perf_counter()
        _, states = propagate_rk4(gen, C0, dt_h, n_h, save_every=n_h)
        t_h = time.perf_counter() - t0
        C_end = gen.from_state(states[-1])
        rho_h = hermite_to_rho(C_end, B, W, Bplus=Bplus)
        resid = float(physical_frame_residual(C_end, B, W))

        # ---- the same Hermite trajectory on the production kernel --------
        # Same operator, same dt, same number of steps, same tableau; real
        # arithmetic, cache-blocked transforms and a dense mode contraction.
        # Timed here so that the cost the paper quotes is the cost of the code
        # that produced the paper's numbers, and so that the agreement is
        # measured on a whole trajectory at the same place as the cost.
        t_f = float("nan")
        fast_reldiff = float("nan")
        if generator_is_real(gen) and state_is_real(C0):
            kern = RealHermiteKernel(gen, nthread=args.threads)
            C_f = np.ascontiguousarray(np.real(C0))
            t0 = time.perf_counter()
            C_f = rk4_real(kern, C_f, dt_h, n_h)
            t_f = time.perf_counter() - t0
            kern.close()
            ref_scale = float(np.max(np.abs(np.real(states[-1]))))
            fast_reldiff = float(np.max(np.abs(C_f - np.real(states[-1])))
                                 / ref_scale)
            del kern, C_f

        # ---- Strang reference -------------------------------------------
        n_s = max(1, int(np.ceil(args.T / args.dt_strang)))
        dt_s = args.T / n_s
        prop = StrangPropagator(grid, model, dt_s)
        t0 = time.perf_counter()
        rho_s = prop.advance(rho0.copy(), n_s)
        t_s = time.perf_counter() - t0

        p_h = np.real(populations_from_rho(rho_h, grid))
        p_s = np.real(populations_from_rho(rho_s, grid))
        scale = float(np.max(np.abs(rho_s)))
        row = {
            "pop_hermite": p_h.tolist(), "pop_strang": p_s.tolist(),
            "pop_absdiff": float(np.max(np.abs(p_h - p_s))),
            "field_reldiff": float(np.max(np.abs(rho_h - rho_s)) / scale),
            "residual": resid,
            "t_hermite_s": t_h, "t_strang_s": t_s,
            "t_hermite_fast_s": t_f,
            "fast_kernel_vs_reference_reldiff": fast_reldiff,
            "dt_hermite": dt_h, "n_steps_hermite": n_h,
            "dt_strang": dt_s,
        }
        results[f"{n_R}x{n_P}"] = row
        print(f"[10g] {n_R}x{n_P}  populations {row['pop_absdiff']:.3e}  "
              f"field {row['field_reldiff']:.3e}  residual {resid:.2e}  "
              f"Hermite/Strang wall {t_h / t_s:.1f}x  "
              f"fast kernel {t_h / t_f:.2f}x faster, agrees to "
              f"{fast_reldiff:.2e}", flush=True)
        del gen, C0, states, C_end, rho_h, rho_s, prop

    keys = sorted(results, key=lambda k: int(k.split("x")[0]))
    coarse, fine = results[keys[0]], results[keys[-1]]
    ratio = fine["field_reldiff"] / coarse["field_reldiff"]
    checks = [
        {"name": f"population_agreement_{keys[-1].split('x')[0]}",
         "value": fine["pop_absdiff"], "tolerance": 1e-8},
        # the value that matters is the RATIO: a small difference at one
        # resolution proves nothing, a difference that falls under refinement
        # is two discretisations converging to one another
        {"name": "field_difference_decreases_under_refinement",
         "value": ratio, "tolerance": 1.0},
        {"name": f"physical_frame_residual_{keys[-1].split('x')[0]}",
         "value": fine["residual"], "tolerance": 1e-12},
        # the production kernel is only allowed to be faster, not different
        {"name": "production_kernel_matches_reference",
         "value": fine["fast_kernel_vs_reference_reldiff"],
         "tolerance": 1e-12},
    ]
    for c in checks:
        c["passed"] = bool(c["value"] < c["tolerance"])

    payload = {
        "metadata": {
            "method": METHOD, "status": "completed",
            "purpose": "GH-10 production-scale cross-check vs "
                       "subsystem_qcle_grid_reference",
            "timestamp_utc": datetime.now(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "script": Path(__file__).name,
            "source_modules": ["mqcle.hermite.operator", "mqcle.hermite.frame",
                               "mqcle.hermite.propagator",
                               "mqcle.hermite.kernel",
                               "mqcle.propagation"],
            "command_line": " ".join(sys.argv),
            "P0": args.P0, "R0": args.R0, "T": args.T,
            "model": "tully_dual_avoided_crossing", "mass": model.mass,
            "hbar": HBAR, "sigma_R": sigma_R,
            "grids": [f"{a}x{b}" for a, b in grids],
            "reference_method": "subsystem_qcle_grid_reference "
                                "(StrangPropagator)",
            "platform": platform.platform(),
            "python": sys.version, "numpy": np.__version__,
            "output_files": {"json": str(out)},
        },
        "results": results,
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
    }
    out.write_text(json.dumps(payload, indent=2))
    print("[10g] checks: " + ", ".join(
        f"{c['name']}={c['value']:.3e}{'' if c['passed'] else ' FAIL'}"
        for c in checks))
    print(f"[10g] field difference falls by {1 / ratio:.1f}x from "
          f"{keys[0]} to {keys[-1]}")
    print(f"[10g] wrote {out}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
