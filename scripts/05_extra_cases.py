#!/usr/bin/env python3
r"""Gate K: the generalisation cases, actually executed and archived.

Runs one full audit for a case other than ``N = 2, N_b = 1``:

* ordinary polynomial degrees for PBME, excess and the full generator;
* true Hermite orders and the ``(I - P_2)`` residual;
* physical-frame residual;
* QCLE coefficient correspondence, when affordable.

Writes ``05_extra_cases_N3.json`` or ``06_extra_cases_Nb2.json`` depending on
the case selected, so that a manuscript claim about ``N = 3`` or two bath
modes cites a completed calculation rather than the existence of a CLI flag.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import sympy as sp

from mqcle.io import RunRecorder
from mqcle.validation import (apply_excess, apply_full, apply_pbme,
                              build_symbolic_setup, hermite_defect,
                              mapping_hermite_orders,
                              mapping_polynomial_degrees,
                              physical_frame_projection,
                              verify_qcle_coefficient_correspondence)

CASES = {"N3": (3, 1, False, "results/raw/05_extra_cases_N3.json"),
         "Nb2": (2, 2, False, "results/raw/06_extra_cases_Nb2.json"),
         "Herm": (2, 1, True, "results/raw/06b_extra_cases_hermitian.json"),
         "HermN3": (3, 1, True, "results/raw/06c_extra_cases_hermitian_N3.json")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES), required=True)
    ap.add_argument("--convention", default="traceless")
    ap.add_argument("--skip-qcle", action="store_true")
    ap.add_argument("--skip-frame", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    nstates, nbath, hermitian, default_out = CASES[args.case]
    out_path = args.out or default_out
    rec = RunRecorder(script=Path(__file__).name,
                      config={**vars(args), "nstates": nstates,
                              "nbath": nbath, "hermitian": hermitian},
                      seed=None)
    s = build_symbolic_setup(nstates, nbath, args.convention,
                             hermitian=hermitian)
    tag = (f"N{nstates}_Nb{nbath}" if not hermitian
           else f"N{nstates}_Nb{nbath}_hermitian")
    print(f"[05] case {args.case}: N={nstates} Nb={nbath} "
          f"hermitian={hermitian} convention={args.convention}", flush=True)

    if hermitian:
        # record that h really is complex Hermitian and not secretly real,
        # so the case cannot silently degenerate to the one already covered.
        #
        # build_symbolic_setup deliberately leaves h00(R) an assumption-free
        # undefined function: declaring it real would change the function
        # object's identity and break every substitution mqcle.leakage keys on
        # (see the NOTE there).  So a question that needs SymPy to KNOW the
        # component functions are real has to say so, here and only here.
        def realise(expr):
            sub = {}
            for e in sp.sympify(expr).atoms(sp.Function):
                nm = getattr(e.func, "__name__", "")
                if nm and nm[0] in "hgV":
                    sub[e] = sp.Symbol(nm, real=True)
            return sp.sympify(expr).subs(sub)

        hR = sp.Matrix(nstates, nstates, lambda i, j: realise(s.h[i, j]))
        hd = sp.simplify(hR - hR.conjugate().T)
        im_nonzero = any(sp.simplify(sp.im(hR[i, j])) != 0
                         for i in range(nstates) for j in range(nstates))
        rec.results["h_is_hermitian"] = bool(hd == sp.zeros(nstates, nstates))
        rec.results["h_has_nonzero_imaginary_part"] = bool(im_nonzero)
        rec.results["bigH_is_symmetric"] = bool(
            sp.simplify(s.bigH - s.bigH.T) == sp.zeros(2 * nstates, 2 * nstates))
        rec.results["H_m_is_real"] = bool(
            sp.simplify(sp.im(realise(sp.expand(s.H_m)))) == 0)
        rec.check(f"{tag}.h_is_hermitian",
                  0.0 if rec.results["h_is_hermitian"] else 1.0, 0.5)
        rec.check(f"{tag}.h_imaginary_part_is_nonzero",
                  0.0 if rec.results["h_has_nonzero_imaginary_part"] else 1.0, 0.5)
        rec.check(f"{tag}.real_form_is_symmetric",
                  0.0 if rec.results["bigH_is_symmetric"] else 1.0, 0.5)
        rec.check(f"{tag}.mapping_hamiltonian_is_real",
                  0.0 if rec.results["H_m_is_real"] else 1.0, 0.5)
        print(f"     h Hermitian {rec.results['h_is_hermitian']}, "
              f"Im h != 0 {rec.results['h_has_nonzero_imaginary_part']}, "
              f"2N form symmetric {rec.results['bigH_is_symmetric']}, "
              f"H_m real {rec.results['H_m_is_real']}", flush=True)

    rec.results["initial_density"] = {
        "polynomial_degrees": mapping_polynomial_degrees(s, s.rho_m),
        "hermite_orders": mapping_hermite_orders(s, s.rho_m)}
    rec.check(f"{tag}.initial_polynomial_degrees_are_0_2",
              0.0 if rec.results["initial_density"]["polynomial_degrees"] == [0, 2]
              else 1.0, 0.5)

    ops = {}
    for name, op in (("pbme", apply_pbme), ("excess", apply_excess),
                     ("full", apply_full)):
        t0 = time.perf_counter()
        out = sp.expand(op(s, s.rho_m))
        entry = {"polynomial_degrees": mapping_polynomial_degrees(s, out),
                 "hermite_orders": mapping_hermite_orders(s, out)}
        entry["hermite_defect_K2_is_zero"] = bool(
            sp.simplify(hermite_defect(s, out, 2)) == 0)
        entry["seconds"] = round(time.perf_counter() - t0, 2)
        ops[name] = entry
        print(f"     {name:7s} poly {entry['polynomial_degrees']}  "
              f"hermite {entry['hermite_orders']}  "
              f"(I-P2)=0 {entry['hermite_defect_K2_is_zero']}  "
              f"({entry['seconds']}s)", flush=True)
    rec.results["operators"] = ops
    rec.check(f"{tag}.full_polynomial_degrees_are_0_2",
              0.0 if ops["full"]["polynomial_degrees"] == [0, 2] else 1.0, 0.5)
    rec.check(f"{tag}.full_hermite_defect_K2_zero",
              0.0 if ops["full"]["hermite_defect_K2_is_zero"] else 1.0, 0.5)
    rec.check(f"{tag}.pbme_leaves_V2",
              0.0 if 4 in ops["pbme"]["polynomial_degrees"] else 1.0, 0.5)

    if not args.skip_frame:
        t0 = time.perf_counter()
        _, _, residual = physical_frame_projection(
            s, sp.expand(apply_full(s, s.rho_m)))
        is_zero = sp.simplify(residual) == 0
        rec.results["physical_frame_residual_is_zero"] = bool(is_zero)
        rec.results["physical_frame_seconds"] = round(time.perf_counter() - t0, 2)
        print(f"     physical-frame residual == 0 : {bool(is_zero)} "
              f"({rec.results['physical_frame_seconds']}s)", flush=True)
        rec.check(f"{tag}.physical_frame_residual_zero",
                  0.0 if is_zero else 1.0, 0.5)

    if not args.skip_qcle:
        t0 = time.perf_counter()
        q = verify_qcle_coefficient_correspondence(s)
        q["seconds"] = round(time.perf_counter() - t0, 2)
        rec.results["qcle_correspondence"] = q
        print(f"     QCLE residuals all zero : {q['all_zero']} "
              f"({q['seconds']}s)", flush=True)
        rec.check(f"{tag}.qcle_residuals_all_zero",
                  0.0 if q["all_zero"] else 1.0, 0.5)

    rec.write(out_path)
    print(f"\n[05] checks: {len(rec.checks)}  all passed: {rec.all_passed}")
    for c in rec.checks:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['name']}")
    print(f"[05] written to {out_path}")
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
