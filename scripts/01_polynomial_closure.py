#!/usr/bin/env python3
r"""Gate G: ordinary-polynomial closure audit.

Applies the Poisson part, the excess part and the full generator to the exact
projected mapping density

.. math:: \rho_m^{\mathcal P}=\phi(x)\,P_2(x;X),

factors out the Gaussian, and records the set of **ordinary total mapping
polynomial degrees** present after one and two applications.

These are ordinary monomial degrees, not Hermite orders.  Degree <= K implies
membership in V_K, because the Hermite change of basis is triangular, so this
audit is sufficient to establish V_2 invariance -- but it says nothing about
how the amplitude is distributed among Hermite orders 0..K, and its
"degree > 2 part" is Top_4, not the orthogonal residual (I - P_2).  Those
quantities are computed by scripts/02_hermite_projection.py.

This is the decision-tree question of PART XL -- *does the original MMST
representation already give practical Hermite closure?* -- answered by exact
computer algebra with arbitrary symbolic ``h(R)``, ``V_b(R)`` and
``rho_W(R,P)``, so the answer is a proof for each ``(nstates, nbath)`` listed
rather than a numerical spot check.

    python scripts/01_polynomial_closure.py --powers 2 --extra-cases
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import sympy as sp

from mqcle.io import RunRecorder
from mqcle.validation import (apply_excess, apply_full, apply_pbme,
                              build_symbolic_setup, mapping_hermite_orders,
                              mapping_polynomial_degrees)

OPS = {"pbme": apply_pbme, "excess": apply_excess, "full": apply_full}

#: PART XIII asks for L_PB, L' and L_m once each, then L_m^2 and, budget
#: permitting, L_m^3.  Repeated application of the excess term *alone* is not a
#: physically meaningful object, and its expression swell is severe: one
#: application on the degree-4 output ran past ten minutes without finishing.
#: Only these two operators are iterated.
ITERATED = ("pbme", "full")


def degree_sequence(setup, op, powers: int, budget_s: float):
    """Degrees of ``A^k rho_m`` for k = 1..powers, stopping on the time budget."""
    expr, seq = setup.rho_m, []
    for k in range(1, powers + 1):
        t0 = time.perf_counter()
        expr = sp.expand(op(setup, expr))
        degs = mapping_polynomial_degrees(setup, expr)
        dt = time.perf_counter() - t0
        seq.append({"power": k, "degrees": degs, "max_degree": max(degs),
                    "n_ops": int(sp.count_ops(expr)), "seconds": round(dt, 2)})
        print(f"      power {k}: degrees {degs}  ({dt:.1f}s, "
              f"{sp.count_ops(expr)} ops)", flush=True)
        if dt > budget_s:
            seq[-1]["stopped_here"] = "time budget exceeded"
            break
    return seq


def pbme_top_degree_closed_form(setup) -> dict:
    r"""Closed form of the **ordinary degree-4** remainder left by the PBME part.

    This is Top_4, not the orthogonal Hermite residual (I - P_2); see
    scripts/02_hermite_projection.py for the corrected quantity.

    .. math::
        \mathrm{Top}_4\bigl[\mathcal A_{\rm PB}\rho_m^{\mathcal P}/\phi\bigr]
        =\frac{1}{\hbar^{2}}\sum_{\lambda\lambda'\mu\nu}
         u_{\lambda\lambda'}\bigl(u_{\mu\nu}-iL_{\mu\nu}\bigr)
         \nabla_Rh^{\lambda\lambda'}\!\cdot\!\nabla_P\rho_W^{\nu\mu}.
    """
    s = setup
    q = sp.expand(sp.cancel(sp.expand(apply_pbme(s, s.rho_m)) / s.phi))
    xs = list(s.r) + list(s.p)
    poly = sp.Poly(q, *xs)
    leak = sum(c * sp.prod([v**e for v, e in zip(xs, m)])
               for m, c in zip(poly.monoms(), poly.coeffs()) if sum(m) > 2)

    def u(a, b):
        return s.r[a] * s.r[b] + s.p[a] * s.p[b]

    def L(a, b):
        return s.r[a] * s.p[b] - s.r[b] * s.p[a]

    n, nb = s.nstates, s.nbath
    claim = sum(u(a, b) * (u(c, d) - sp.I * L(c, d))
                * sp.diff(s.h_map[a, b], s.R[k]) * sp.diff(s.rho[d][c], s.P[k])
                for a in range(n) for b in range(n)
                for c in range(n) for d in range(n)
                for k in range(nb)) / s.hbar**2
    residual = sp.simplify(sp.expand(leak - claim))
    return {"residual_is_zero": residual == 0, "residual": str(residual)[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--powers", type=int, default=2)
    ap.add_argument("--budget-seconds", type=float, default=60.0)
    ap.add_argument("--extra-cases", action="store_true",
                    help="also run N=3 and two bath modes at power 1")
    ap.add_argument("--only-extra", action="store_true",
                    help="run ONLY the N=3 / two-bath-mode cases, so that the "
                         "primary and secondary sweeps fit separate budgets")
    ap.add_argument("--out", default="results/raw/01_polynomial_closure.json")
    args = ap.parse_args()

    rec = RunRecorder(script=Path(__file__).name,
                      config={"powers": args.powers,
                              "budget_seconds": args.budget_seconds,
                              "extra_cases": args.extra_cases}, seed=None)
    cases = []

    if args.only_extra:
        for nst, nb in ((3, 1), (2, 2)):
            print(f"[01] N={nst} Nb={nb} convention=traceless", flush=True)
            setup = build_symbolic_setup(nst, nb, "traceless")
            cases.append({"nstates": nst, "nbath": nb, "convention": "traceless",
                          "mixed_convention": False,
                          "initial_polynomial_degrees": mapping_polynomial_degrees(setup, setup.rho_m),
                          "operators": {k: degree_sequence(setup, op, 1,
                                                           args.budget_seconds)
                                        for k, op in OPS.items()}})
        rec.results["cases"] = cases
        for case in cases:
            tag = f"N{case['nstates']}_Nb{case['nbath']}_traceless"
            rec.check(f"{tag}.initial_polynomial_degrees_are_0_2",
                      0.0 if case["initial_polynomial_degrees"] == [0, 2] else 1.0, 0.5)
            rec.check(f"{tag}.full_generator_closes",
                      0.0 if case["operators"]["full"][0]["degrees"] == [0, 2] else 1.0,
                      0.5)
            rec.check(f"{tag}.pbme_alone_leaks",
                      0.0 if 4 in case["operators"]["pbme"][0]["degrees"] else 1.0,
                      0.5)
        rec.write(args.out)
        print(f"\n[01] checks: {len(rec.checks)}  all passed: {rec.all_passed}")
        for c in rec.checks:
            print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['name']}")
        print(f"[01] written to {args.out}")
        return 0 if rec.all_passed else 1

    # -- primary case: N = 2, one bath mode, both self-consistent conventions
    for conv in ("traceless", "full"):
        print(f"[01] N=2 Nb=1 convention={conv}", flush=True)
        setup = build_symbolic_setup(2, 1, conv)
        entry = {"nstates": 2, "nbath": 1, "convention": conv,
                 "mixed_convention": False,
                 "initial_polynomial_degrees": mapping_polynomial_degrees(setup, setup.rho_m),
                 "operators": {}}
        for name, op in OPS.items():
            powers = args.powers if name in ITERATED else 1
            print(f"   {name}:", flush=True)
            entry["operators"][name] = degree_sequence(setup, op, powers,
                                                       args.budget_seconds)
        if conv == "traceless":
            entry["pbme_top_degree_closed_form"] = pbme_top_degree_closed_form(setup)
        cases.append(entry)

    # -- the invalid mixed convention, kept as a live counter-example
    print("[01] N=2 Nb=1 MIXED convention (expected to fail closure)", flush=True)
    setup = build_symbolic_setup(2, 1, "full", mixed_convention=True)
    cases.append({"nstates": 2, "nbath": 1, "convention": "full",
                  "mixed_convention": True,
                  "initial_polynomial_degrees": mapping_polynomial_degrees(setup, setup.rho_m),
                  "operators": {"full": degree_sequence(setup, apply_full, 1,
                                                        args.budget_seconds)}})

    if args.extra_cases:
        for nst, nb in ((3, 1), (2, 2)):
            print(f"[01] N={nst} Nb={nb} convention=traceless", flush=True)
            setup = build_symbolic_setup(nst, nb, "traceless")
            cases.append({"nstates": nst, "nbath": nb, "convention": "traceless",
                          "mixed_convention": False,
                          "initial_polynomial_degrees": mapping_polynomial_degrees(setup, setup.rho_m),
                          "operators": {k: degree_sequence(setup, op, 1,
                                                           args.budget_seconds)
                                        for k, op in OPS.items()}})

    rec.results["cases"] = cases

    # -- acceptance checks ------------------------------------------------
    for case in cases:
        tag = (f"N{case['nstates']}_Nb{case['nbath']}_{case['convention']}"
               + ("_mixed" if case["mixed_convention"] else ""))
        rec.check(f"{tag}.initial_polynomial_degrees_are_0_2",
                  0.0 if case["initial_polynomial_degrees"] == [0, 2] else 1.0, 0.5)
        full = case["operators"].get("full")
        if full is None:
            continue
        closes = full[0]["degrees"] == [0, 2]
        expect_close = not case["mixed_convention"]
        rec.check(f"{tag}.full_generator_closure_as_expected",
                  0.0 if closes == expect_close else 1.0, 0.5,
                  note=("full generator must preserve degree <= 2"
                        if expect_close else
                        "mixed convention must break closure"))
    primary = cases[0]
    if "pbme_top_degree_closed_form" in primary:
        rec.check("pbme_top_degree_closed_form_residual_zero",
                  0.0 if primary["pbme_top_degree_closed_form"]["residual_is_zero"] else 1.0,
                  0.5)

    # -- decision-tree answer ---------------------------------------------
    full_seq = primary["operators"]["full"]
    pbme_seq = primary["operators"]["pbme"]
    rec.results["decision_tree"] = {
        "question": "Does original MMST already give practical Hermite closure?",
        "answer": "yes" if all(s["degrees"] == [0, 2] for s in full_seq) else "no",
        "full_generator_max_degrees": [s["max_degree"] for s in full_seq],
        "pbme_only_max_degrees": [s["max_degree"] for s in pbme_seq],
        "consequence": (
            "The full generator preserves ordinary mapping-polynomial degree "
            "<= 2 under "
            "repeated application, hence stays in V_2. Whether it also stays "
            "in the N^2-dimensional physical frame is a separate question, "
            "settled in 04_physical_frame.json. Dropping the excess term "
            "raises the ordinary degree by two per application."),
    }

    rec.write(args.out)
    print(f"\n[01] checks: {len(rec.checks)}  all passed: {rec.all_passed}")
    for c in rec.checks:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['name']}")
    print(f"[01] decision-tree answer: {rec.results['decision_tree']['answer']}")
    print(f"[01] written to {args.out}")
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
