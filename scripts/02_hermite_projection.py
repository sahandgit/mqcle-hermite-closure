#!/usr/bin/env python3
r"""Gate H: the **true** Hermite projection, and the ``V_K`` regression audit.

Script 01 reports ordinary monomial degrees.  This script computes the actual
orthogonal quantities in the inner product
:math:`\langle f,g\rangle_\varphi=\int dz\,\varphi f^{*}g` with
:math:`\varphi(z)=(2\pi)^{-N}e^{-z^{\mathsf T}z/2}` and
:math:`z=\sqrt{2/\hbar}\,x`:

* Hermite orders actually present in each generator's output;
* the residual :math:`(I-\mathsf P_2)q`;
* the comparison against :math:`\mathrm{Top}_4[q]`, which an earlier version of
  this pipeline reported under the name of the Hermite defect.  They are
  different functions;
* the ``V_K`` audit: apply each generator to explicit basis densities
  :math:`\phi\,\mathrm{He}_{\mathbf n}f(R,P)` with :math:`|n|=K` and record the
  highest output Hermite order.

    python scripts/02_hermite_projection.py --vk-orders 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import sympy as sp

from mqcle.hermite.basis import he_multi
from mqcle.io import RunRecorder
from mqcle.validation import (apply_excess, apply_full, apply_pbme,
                              build_symbolic_setup, hermite_defect,
                              mapping_hermite_orders,
                              mapping_polynomial_degrees, pbme_hermite_defect,
                              pbme_principal_polynomial_leakage, vk_audit)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nstates", type=int, default=2)
    ap.add_argument("--nbath", type=int, default=1)
    ap.add_argument("--convention", default="traceless")
    ap.add_argument("--vk-orders", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--vk-max-indices", type=int, default=None,
                    help="cap the basis functions per order (None = all)")
    ap.add_argument("--out", default="results/raw/02_hermite_projection.json")
    args = ap.parse_args()

    rec = RunRecorder(script=Path(__file__).name, config=vars(args), seed=None)
    s = build_symbolic_setup(args.nstates, args.nbath, args.convention)
    rec.results["inner_product"] = ("<f,g>_phi = int dz phi(z) conj(f) g, "
                                    "phi = (2 pi)^{-N} exp(-z^T z / 2), "
                                    "z = sqrt(2/hbar) x")

    # -- orders present, per operator ------------------------------------
    orders = {}
    for name, op in (("pbme", apply_pbme), ("excess", apply_excess),
                     ("full", apply_full)):
        t0 = time.perf_counter()
        out = sp.expand(op(s, s.rho_m))
        orders[name] = {
            "polynomial_degrees": mapping_polynomial_degrees(s, out),
            "hermite_orders": mapping_hermite_orders(s, out),
            "hermite_defect_K2_is_zero": bool(
                sp.simplify(hermite_defect(s, out, 2)) == 0),
            "seconds": round(time.perf_counter() - t0, 2),
        }
        print(f"[02] {name:7s} poly {orders[name]['polynomial_degrees']}  "
              f"hermite {orders[name]['hermite_orders']}  "
              f"(I-P2)=0 {orders[name]['hermite_defect_K2_is_zero']}", flush=True)
    rec.results["operator_orders"] = orders
    rec.results["initial_density"] = {
        "polynomial_degrees": mapping_polynomial_degrees(s, s.rho_m),
        "hermite_orders": mapping_hermite_orders(s, s.rho_m)}

    # -- Top_4 versus the true residual ----------------------------------
    print("[02] comparing Top_4 against (I - P_2) ...", flush=True)
    top = pbme_principal_polynomial_leakage(s, above=2, standardised=True)
    her = pbme_hermite_defect(s, max_order=2)
    diff = sp.expand(sp.simplify(top - her))
    diff_degrees = ([] if diff == 0 else
                    sorted({int(sum(m)) for m in sp.Poly(diff, *s.z).monoms()}))
    P = sp.Poly(top, *s.z)
    wick = sum(c * he_multi(m, s.z) for m, c in zip(P.monoms(), P.coeffs()))
    wick_matches = bool(sp.simplify(sp.expand(wick - her)) == 0)

    rec.results["pbme_leakage"] = {
        "top4_equals_hermite_residual": bool(diff == 0),
        "difference_monomial_degrees": diff_degrees,
        "hermite_residual_is_wick_ordered_top4": wick_matches,
        "n_terms_top4": len(sp.Add.make_args(top)),
        "n_terms_hermite_residual": len(sp.Add.make_args(her)),
        "top4_expression": str(top)[:4000],
        "hermite_residual_expression": str(her)[:4000],
        "note": ("Top_4 keeps the degree-4 monomials of the amplitude; "
                 "(I - P_2) keeps the degree-4 Hermite components. They "
                 "differ by degree-0 and degree-2 terms. The corrected "
                 "closed form is the Wick-ordered Top_4: replace each "
                 "monomial z^alpha of degree 4 by He_alpha(z)."),
    }
    rec.check("top4_and_hermite_residual_differ",
              0.0 if diff != 0 else 1.0, 0.5,
              note="records that the two quantities are NOT the same function")
    rec.check("hermite_residual_equals_wick_ordered_top4",
              0.0 if wick_matches else 1.0, 0.5)
    rec.check("full_generator_hermite_defect_K2_zero",
              0.0 if orders["full"]["hermite_defect_K2_is_zero"] else 1.0, 0.5)
    rec.check("pbme_hermite_defect_K2_nonzero",
              0.0 if not orders["pbme"]["hermite_defect_K2_is_zero"] else 1.0, 0.5)

    # -- V_K audit --------------------------------------------------------
    if args.vk_orders:
        print(f"[02] V_K audit for K = {args.vk_orders} ...", flush=True)
        t0 = time.perf_counter()
        audit = vk_audit(s, args.vk_orders, max_indices=args.vk_max_indices)
        rec.results["vk_audit"] = audit
        rec.results["vk_audit_seconds"] = round(time.perf_counter() - t0, 1)
        for K, r in audit.items():
            print(f"     K={K}: checked {r['n_basis_functions_checked']}/"
                  f"{r['n_basis_functions_total']}  full_max={r['full_max']}  "
                  f"pbme_max={r['pbme_max']}  excess_max={r['excess_max']}",
                  flush=True)
            rec.check(f"vk.K{K}.full_within_VK",
                      0.0 if r["full_within_VK"] else 1.0, 0.5)
            if int(K) >= 1:
                rec.check(f"vk.K{K}.pbme_raises_by_two",
                          0.0 if r["pbme_raises_by_two"] else 1.0, 0.5,
                          note="PBME alone leaves V_K")

    rec.write(args.out)
    print(f"\n[02] checks: {len(rec.checks)}  all passed: {rec.all_passed}")
    for c in rec.checks:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['name']}")
    print(f"[02] written to {args.out}")
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
