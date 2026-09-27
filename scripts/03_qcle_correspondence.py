#!/usr/bin/env python3
r"""Gate J: the closed coefficient equations must be the subsystem-basis QCLE.

Solves

.. math::
    \mathcal A_m\rho_m^{\mathcal P}
      =\phi\sum_{\lambda\lambda'}\Pi_{\lambda\lambda'}C_{\lambda'\lambda}

for the coefficient fields by orthogonal projection onto the physical frame,
then subtracts the subsystem QCLE

.. math::
    \partial_t\rho_W=-\frac{i}{\hbar}[h,\rho_W]-\frac{P}{M}\nabla_R\rho_W
      -\tfrac12\bigl(F\nabla_P\rho_W+\nabla_P\rho_W F\bigr),
    \qquad F=-\nabla_R(V_b\mathbb 1+h),

element by element.  Every residual must be exactly zero.  The reference uses
the physical ``h`` and ``V_b``, so the gate also detects an inconsistent trace
convention.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from mqcle.io import RunRecorder
from mqcle.validation import (build_symbolic_setup,
                              verify_qcle_coefficient_correspondence)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nstates", type=int, default=2)
    ap.add_argument("--nbath", type=int, default=1)
    ap.add_argument("--conventions", nargs="*", default=["traceless", "full"])
    ap.add_argument("--out", default="results/raw/03_qcle_correspondence.json")
    args = ap.parse_args()

    rec = RunRecorder(script=Path(__file__).name, config=vars(args), seed=None)
    cases = []
    for conv in args.conventions:
        t0 = time.perf_counter()
        s = build_symbolic_setup(args.nstates, args.nbath, conv)
        out = verify_qcle_coefficient_correspondence(s)
        out.update({"convention": conv, "nstates": args.nstates,
                    "nbath": args.nbath,
                    "seconds": round(time.perf_counter() - t0, 2)})
        cases.append(out)
        print(f"[03] convention={conv:9s} frame_residual_zero="
              f"{out['frame_residual_is_zero']}  all_residuals_zero="
              f"{out['all_zero']}  ({out['seconds']}s)", flush=True)
        rec.check(f"{conv}.frame_residual_zero",
                  0.0 if out["frame_residual_is_zero"] else 1.0, 0.5)
        rec.check(f"{conv}.qcle_residuals_all_zero",
                  0.0 if out["all_zero"] else 1.0, 0.5)
    rec.results["cases"] = cases
    rec.results["reference_equation"] = (
        "d_t rho_W = -(i/hbar)[h, rho_W] - (P/M) d_R rho_W "
        "- (1/2)(F d_P rho_W + d_P rho_W F),  F = -d_R(V_b I + h)")
    rec.write(args.out)
    print(f"\n[03] checks: {len(rec.checks)}  all passed: {rec.all_passed}")
    print(f"[03] written to {args.out}")
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
