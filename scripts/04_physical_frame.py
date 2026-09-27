#!/usr/bin/env python3
r"""Gate I: physical-frame invariance, which is *not* implied by V_2 invariance.

The physical frame is
:math:`\mathcal F_{\rm phys}=\mathrm{span}\{\phi\Pi_{\lambda\lambda'}\}`, of
complex dimension :math:`N^{2}`.  For :math:`N=2` the even part of :math:`V_2`
alone has dimension 11, so :math:`\mathcal A_mV_2\subseteq V_2` leaves open
whether the dynamics stays in the four-dimensional physical frame.

This script computes, by orthogonal projection in
:math:`\langle f,g\rangle_\varphi`,

.. math::
    \epsilon_{\rm phys}=\bigl\|q-\mathcal P_{\rm phys}q\bigr\|,
    \qquad q=\mathcal A\rho_m^{\mathcal P}/\phi,

for the full generator (expected exactly zero) and for PBME alone (expected
nonzero).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import sympy as sp

from mqcle.io import RunRecorder
from mqcle.validation import (apply_excess, apply_full, apply_pbme,
                              build_symbolic_setup, gaussian_expectation,
                              physical_frame_projection)

OPS = {"full": apply_full, "pbme": apply_pbme, "excess": apply_excess}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nstates", type=int, default=2)
    ap.add_argument("--nbath", type=int, default=1)
    ap.add_argument("--convention", default="traceless")
    ap.add_argument("--operators", nargs="*", default=["full", "pbme"])
    ap.add_argument("--out", default="results/raw/04_physical_frame.json")
    args = ap.parse_args()

    rec = RunRecorder(script=Path(__file__).name, config=vars(args), seed=None)
    s = build_symbolic_setup(args.nstates, args.nbath, args.convention)
    n2 = args.nstates ** 2
    dim_even_V2 = 1 + (2 * args.nstates) * (2 * args.nstates + 1) // 2
    rec.results["dimensions"] = {
        "mapping_phase_space": 2 * args.nstates,
        "physical_frame_complex_dim": n2,
        "even_part_of_V2_dim": dim_even_V2,
        "note": ("The physical frame is a proper subspace of V_2, so V_2 "
                 "invariance does not imply frame invariance; this script "
                 "settles the latter directly."),
    }

    # sanity: projecting the density itself must return rho_W unchanged
    coeff, _, res0 = physical_frame_projection(s, s.rho_m)
    ident = all(sp.simplify(coeff[a][b] - s.rho[b][a]) == 0
                for a in range(args.nstates) for b in range(args.nstates))
    rec.check("frame_projection_is_identity_on_rho_m",
              0.0 if (ident and sp.simplify(res0) == 0) else 1.0, 0.5)

    cases = {}
    for name in args.operators:
        t0 = time.perf_counter()
        out = sp.expand(OPS[name](s, s.rho_m))
        _, _, residual = physical_frame_projection(s, out)
        residual = sp.simplify(residual)
        is_zero = residual == 0
        norm2 = gaussian_expectation(
            sp.expand(residual * sp.conjugate(residual).replace(
                lambda e: isinstance(e, sp.conjugate), lambda e: e.args[0])),
            s.z) if not is_zero else sp.Integer(0)
        cases[name] = {
            "residual_is_zero": bool(is_zero),
            "residual_expression": str(residual)[:4000],
            "squared_norm_expression": str(sp.simplify(norm2))[:4000],
            "seconds": round(time.perf_counter() - t0, 2),
        }
        print(f"[04] {name:7s} epsilon_phys == 0 : {bool(is_zero)}  "
              f"({cases[name]['seconds']}s)", flush=True)
    rec.results["cases"] = cases

    if "full" in cases:
        rec.check("full_generator_frame_residual_zero",
                  0.0 if cases["full"]["residual_is_zero"] else 1.0, 0.5,
                  note="the physical statement, distinct from V_2 invariance")
    if "pbme" in cases:
        rec.check("pbme_frame_residual_nonzero",
                  0.0 if not cases["pbme"]["residual_is_zero"] else 1.0, 0.5)

    rec.write(args.out)
    print(f"\n[04] checks: {len(rec.checks)}  all passed: {rec.all_passed}")
    print(f"[04] written to {args.out}")
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
