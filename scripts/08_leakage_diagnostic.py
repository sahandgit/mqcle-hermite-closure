#!/usr/bin/env python3
"""Gate M -- is the Hermite leakage functional usable as a PBME error monitor?

This script is allowed to return a null result.  A null result, properly
measured, is a publishable and honest outcome and is far better than a
suggestive plot.  Nothing here is written so as to make the answer come out
positive.

Three measurements
------------------

1. **Route A** -- the leakage :math:`\\mathcal D_H(t)` evaluated on the
   converged coefficient grid, where :math:`\\nabla_P\\rho_W` is available
   exactly by spectral differentiation.  Ground truth.

2. **Route B** -- the same functional evaluated from a finite PBME trajectory
   ensemble, where :math:`\\nabla_P\\rho_W` must be *estimated*.  This is what
   a trajectory method would actually have to do.

3. **The estimator study** -- at :math:`t=0` the PBME ensemble and the
   coefficient field are the same physical state, so the exact
   :math:`\\nabla_P\\rho_W` is known analytically and the Route-B estimator can
   be scored against it directly: bias, variance, RMSE and sample-size
   scaling, with several seeds per sample size.  At later times the two
   densities genuinely differ, and the comparison then mixes estimator error
   with PBME's own error; both are reported, never merged.

Usage
-----
    python scripts/08_leakage_diagnostic.py --config tully_default.yaml --P0 40
    python scripts/08_leakage_diagnostic.py --aggregate      # correlation study
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqcle.coefficient_operator import (build_grid, spectral_derivative,  # noqa: E402
                                        trace)
from mqcle.benchmark_regimes import MOMENTA                              # noqa: E402
from mqcle.errors import population_error_decomposition                   # noqa: E402
from mqcle.io import RunRecorder, load_config, save_json                  # noqa: E402
from mqcle.leakage import (kde_grad_P_rho_W, leakage_from_fields,         # noqa: E402
                           leakage_from_grid, silverman_bandwidth)
from mqcle.mapping import subsystem_split                                 # noqa: E402
from mqcle.models import build_model                                      # noqa: E402
from mqcle.pbme import pack_state, symmetric_rotation_step, unpack_state  # noqa: E402
from mqcle.propagation import (StrangPropagator, grid_observables,        # noqa: E402
                               initial_coefficient_field)
from mqcle.sampling import sample_initial_ensemble                        # noqa: E402
from mqcle.types import TraceConvention                                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module                                       # noqa: E402
_s07 = import_module("07_coefficient_propagation")
DEFAULTS_FOR = _s07._defaults_for


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="tully_default.yaml")
    p.add_argument("--coupling", type=float, default=None,
                   help="override the Tully diabatic coupling amplitude C; "
                        "this is the parameter that sets the scale of "
                        "grad_R h, so it is the natural second axis for the "
                        "regime study of Gate M")
    p.add_argument("--tag", default="",
                   help="suffix appended to the output file name")
    p.add_argument("--P0", type=float, default=40.0)
    p.add_argument("--R0", type=float, default=-15.0)
    p.add_argument("--n-R", type=int, default=None,
                   help="override the coefficient-grid R resolution")
    p.add_argument("--n-P", type=int, default=None,
                   help="override the coefficient-grid P resolution")
    p.add_argument("--P-half", type=float, default=None,
                   help="half-width of the momentum box about --P-center")
    p.add_argument("--P-center", type=float, default=None,
                   help="centre of the momentum box (default: P0)")
    p.add_argument("--R-min", type=float, default=None)
    p.add_argument("--R-max", type=float, default=None)
    p.add_argument("--dt", type=float, default=1.0,
                   help="the leakage functional is a smooth diagnostic, not "
                        "the production observable, and the Strang splitting "
                        "is unconditionally stable, so a coarser step than "
                        "scripts/07 uses is adequate here")
    p.add_argument("--T", type=float, default=None)
    p.add_argument("--n-save", type=int, default=40)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--estimator-sizes", type=int, nargs="+",
                   default=[1000, 3000, 10000, 30000, 100000])
    p.add_argument("--estimator-seeds", type=int, default=5)
    p.add_argument("--bandwidth-cells", type=float, default=2.0,
                   help="bandwidth floor, in grid cells; the binned KDE is "
                        "only faithful above ~2 cells (measured)")
    p.add_argument("--aggregate", action="store_true",
                   help="read the per-P0 outputs and do the correlation study")
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def _exact_grad_P_initial(grid, R0, P0, sigma_R, init_state, nstates, hbar):
    r"""Analytic :math:`\nabla_P` of the initial Wigner packet.

    Ground truth for the estimator study: at :math:`t=0` the PBME ensemble is a
    focused sample of exactly this state, so no solver enters the comparison.
    """
    RR, PP = grid.mesh()
    sigma_P = hbar / (2.0 * sigma_R)
    W = np.exp(-((RR - R0) ** 2) / (2.0 * sigma_R**2)
               - ((PP - P0) ** 2) / (2.0 * sigma_P**2)) / (np.pi * hbar)
    dW = -(PP - P0) / sigma_P**2 * W
    out = np.zeros(grid.shape + (nstates, nstates), dtype=complex)
    out[:, :, init_state, init_state] = dW
    return out


def _mask(grid, rho_W, frac=1e-4):
    """Region where the density is not numerically negligible."""
    dens = np.abs(np.trace(np.asarray(rho_W), axis1=-2, axis2=-1))
    return dens > frac * dens.max()


def _score(est, ref, mask):
    """Bias, standard deviation and RMSE of an estimator over the mask."""
    d = (est - ref)[mask]
    r = ref[mask]
    return {"bias": float(np.mean(np.real(d))),
            "mean_abs_error": float(np.mean(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(np.abs(d) ** 2))),
            "ref_rms": float(np.sqrt(np.mean(np.abs(r) ** 2))),
            "relative_rmse": float(np.sqrt(np.mean(np.abs(d) ** 2))
                                   / np.sqrt(np.mean(np.abs(r) ** 2)))}


def _projected_file(P0: float, C: float | None) -> Path:
    """Where the projected-density PBME run for this scan point lives."""
    tag = f"P{int(round(P0))}" + ("" if C is None else f"_C{C:g}")
    return Path(f"results/raw/11_pbme_projected_{tag}.json")


def _discard_cost(res, P0: float, C: float | None = None) -> dict:
    r""":math:`E_{\rm discard}=|\rho^{\rm projected\,PBME}-\rho^{\rm QCLE}|`.

    **The focused trajectory branch must not be used here.**  Its difference
    from the coefficient solution is
    :math:`\Delta_{\rm operator}+\Delta_{\rm focused}+\Delta_{\rm MC}
    +\Delta_{\rm integrator}`, and only the first of those is the consequence
    of discarding :math:`\mathcal A'`.  Using it as the response asks the
    leakage functional to predict a sum of four things, three of which it does
    not describe.

    Returns the discard cost, its Monte-Carlo standard error, and separately
    the focused-estimator effect
    :math:`E_{\rm focused}=|\rho^{\rm focused}-\rho^{\rm projected}|`, so the
    four layers stay apart in the archive as well as in the manuscript.
    """
    f = _projected_file(P0, C)
    if not f.exists():
        raise SystemExit(
            f"[08] refusing to compute a discard cost without {f}.  The "
            f"focused trajectory branch is not the PBME operator benchmark; "
            f"run scripts/11_pbme_projected.py for this scan point first.")
    d = json.loads(f.read_text())["results"]
    coef = np.asarray(res["coefficient"]["final_populations"], float)
    proj = np.asarray(d["final"]["cv"]["populations"], float)
    sem = np.asarray(d["final"]["cv"]["sem"], float)
    k = int(np.argmax(np.abs(proj - coef)))
    decomp = population_error_decomposition(coef, coef, proj)
    out = {"discard_cost": decomp["discard_cost"],
           "discard_cost_sem": float(sem[k]),
           "projected_final_populations": proj.tolist(),
           "projected_sem": sem.tolist(),
           "projected_n_samples": int(d["n_samples"]),
           "projected_scheme": d.get("scheme", "unknown")}
    if "focused" in d:
        foc = np.asarray(d["focused"]["final_populations"], float)
        decomp = population_error_decomposition(coef, coef, proj, foc)
        out["focused_effect"] = decomp["focused_effect"]
        out["focused_final_populations"] = foc.tolist()
        out["focused_minus_coefficient"] = float(np.max(np.abs(foc - coef)))
    return out


def _corr(x, y) -> dict:
    """Pearson and Spearman, with the rank statistic always reported.

    An earlier version quoted Pearson only, while the argument it supported was
    explicitly about *ranking*; the rank statistic was computed and archived but
    not carried into the manuscript.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v), float)
        r[order] = np.arange(1, len(v) + 1)
        return r
    return {"pearson_r": float(np.corrcoef(x, y)[0, 1]),
            "spearman_r": float(np.corrcoef(rank(x), rank(y))[0, 1])}


def _leak_summary(d) -> dict:
    """Peak, and the integral over three windows, so the choice is visible.

    The original study integrated over the first half of each run.  Because
    ``T = 2 M |R0| / P0`` shrinks with P0 while the crossing happens at a fixed
    position, that window truncates *before* the leakage peak in every regime,
    and by a P0-dependent amount -- so part of the reported trend in the
    predictor was an artefact of the window, not a property of the dynamics.
    All three windows are recorded here and the manuscript quotes the peak and
    the full-run mean, which have no such dependence.
    """
    t = np.asarray(d["results"]["route_A"]["t"], float)
    D = np.asarray(d["results"]["route_A"]["leakage_relative"], float)
    half = t <= 0.5 * t[-1]
    i_peak = int(np.argmax(D))
    return {
        "peak_leakage": float(D[i_peak]),
        "t_peak": float(t[i_peak]),
        "T": float(t[-1]),
        "peak_inside_half_window": bool(t[i_peak] <= 0.5 * t[-1]),
        "integrated_leakage_early": float(np.trapezoid(D[half], t[half])),
        "integrated_leakage_full": float(np.trapezoid(D, t)),
        "mean_leakage_full": float(np.trapezoid(D, t) / (t[-1] - t[0])),
    }


def aggregate(pattern="results/raw/08_leakage_P*.json",
              err_pattern="results/raw/07_coefficient_propagation_P*.json"):
    """Gate M correlation study, on two independent axes.

    *Momentum axis*: P0 in {10, 20, 30, 40, 70, 100} at the model's own coupling.  Varies
    the transit time and the adiabaticity together.

    *Coupling axis*: C at fixed P0 = 40.  This is the axis that varies the
    diabatic coupling at fixed transit time, and it is the one a referee would
    ask for, because on the momentum axis the predictor barely moves.
    """
    import glob
    import re as _re

    def _load(pat, keep_coupling):
        out = {}
        for f in sorted(glob.glob(pat)):
            has_C = "_C" in Path(f).stem
            if has_C != keep_coupling:
                continue
            out[f] = json.load(open(f))
        return out

    # ---- momentum axis --------------------------------------------------
    leak = {float(d["metadata"]["config"]["P0"]): d
            for d in _load(pattern, False).values()}
    err = {float(d["metadata"]["config"]["P0"]): d
           for d in _load(err_pattern, False).values()}
    # The final P0=20 production calculation uses the widened momentum box.
    # It deliberately has a stage-13 name so the failed narrow-box result is
    # retained only in the pre-revision archive and cannot be selected by
    # accident.  Prefer it explicitly instead of depending on glob order.
    wide20 = Path("results/raw/13_P20_wide_momentum_box.json")
    if wide20.is_file():
        wd = json.loads(wide20.read_text())
        err[20.0] = wd
    rows = []
    for P0, d in sorted(leak.items()):
        if P0 not in err:
            continue
        e = err[P0]["results"].get("errors", {})
        if "pbme_vs_schrodinger_final" not in e:
            continue
        rows.append({"P0": P0, **_leak_summary(d),
                     **_discard_cost(err[P0]["results"], P0),
                     "focused_pbme_terminal_error":
                         float(e["pbme_vs_schrodinger_final"]),
                     "qcle_terminal_error":
                         float(e["coefficient_vs_schrodinger_final"])})
    out = {"rows": rows, "n_regimes": len(rows)}
    if len(rows) >= 3:
        for pred in ("integrated_leakage_early", "peak_leakage",
                     "mean_leakage_full"):
            for resp in ("discard_cost", "focused_pbme_terminal_error"):
                c = _corr([r[pred] for r in rows], [r[resp] for r in rows])
                out[f"corr__{pred}__{resp}"] = c
        out["pearson_r"] = out["corr__mean_leakage_full__"
                               "discard_cost"]["pearson_r"]
        out["spearman_r"] = out["corr__mean_leakage_full__"
                                "discard_cost"]["spearman_r"]
        out["pearson_r_peak"] = out["corr__peak_leakage__"
                                    "discard_cost"]["pearson_r"]
        out["response_variable"] = ("E_discard = |projected-density PBME - "
                                    "coefficient QCLE|")

    # ---- coupling axis, at fixed P0 -------------------------------------
    def _C_of(d):
        c = d["metadata"]["config"]
        return float(c.get("model", {}).get("C", float("nan")))

    cl = {_C_of(d): d for d in _load(pattern, True).values()}
    ce = {_C_of(d): d for d in _load(err_pattern, True).values()}
    for C, d in list(leak.items()):                    # the base run belongs too
        if C == 40.0 and 40.0 in err:
            cl[_C_of(d)] = d
            ce[_C_of(err[40.0])] = err[40.0]
    crows = []
    for C, d in sorted(cl.items()):
        if C not in ce:
            continue
        e = ce[C]["results"].get("errors", {})
        if "pbme_vs_schrodinger_final" not in e:
            continue
        crows.append({"C": C, **_leak_summary(d),
                      **_discard_cost(ce[C]["results"], 40.0,
                                      None if C == 0.015 else C),
                      "focused_pbme_terminal_error":
                          float(e["pbme_vs_schrodinger_final"]),
                      "qcle_terminal_error":
                          float(e["coefficient_vs_schrodinger_final"])})
    cout = {"rows": crows, "n_points": len(crows)}
    if len(crows) >= 3:
        for pred in ("peak_leakage", "mean_leakage_full"):
            for resp in ("discard_cost", "focused_pbme_terminal_error"):
                cout[f"corr__{pred}__{resp}"] = _corr(
                    [r[pred] for r in crows], [r[resp] for r in crows])
        pk = np.array([r["peak_leakage"] for r in crows])
        dc = np.array([r["discard_cost"] for r in crows])
        cout["predictor_dynamic_range"] = float(pk.max() / pk.min())
        cout["response_dynamic_range"] = float(dc.max() / dc.min())
        cout["response_is_monotone_in_C"] = bool(
            np.all(np.diff(dc) > 0) or np.all(np.diff(dc) < 0))
        sems = np.array([r["discard_cost_sem"] for r in crows])
        cout["response_range_over_worst_sem"] = float((dc.max() - dc.min())
                                                      / max(sems.max(), 1e-300))
        cout["every_point_resolved_above_its_own_noise"] = bool(
            np.all(dc > 3.0 * sems))
        cout["response_variable"] = ("E_discard = |projected-density PBME - "
                                     "coefficient QCLE|")
        # Why the predictor barely moves: D_H is built from grad_R hbar, and on
        # this model the state-splitting term sets its scale, not the diabatic
        # coupling C.  Measure that rather than assert it.
        Rg = np.linspace(-6.0, 6.0, 4001)[:, None]
        for r in crows:
            m = build_model({"name": "tully_dual_avoided_crossing",
                             "C": r["C"]}, mass=2000.0)
            sp = subsystem_split(m, Rg, TraceConvention.TRACELESS)
            g = sp.dh_map_dR[:, 0]
            w = np.linalg.eigvalsh(m.h(Rg))
            r["max_grad_h_frobenius"] = float(
                np.linalg.norm(g, ord="fro", axis=(-2, -1)).max())
            r["max_grad_h_offdiagonal"] = float(np.abs(g[:, 0, 1]).max())
            r["min_adiabatic_gap"] = float(np.min(w[:, 1] - w[:, 0]))
        gn = np.array([r["max_grad_h_frobenius"] for r in crows])
        cout["grad_h_dynamic_range"] = float(gn.max() / gn.min())
        cout["coupling_dynamic_range"] = float(
            max(r["C"] for r in crows) / min(r["C"] for r in crows))
    out["coupling_scan"] = cout

    # structural facts about the functional itself, recorded rather than typed
    from mqcle.leakage import leakage_coefficient_forms
    keys, _ = leakage_coefficient_forms(2, 1, "traceless")
    out["n_degree_four_coefficients"] = len(keys)
    out["all_coefficients_are_degree_four"] = bool(
        all(sum(k) == 4 for k in keys))
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.aggregate:
        res = aggregate()
        # The re-analysis supersedes 08_leakage_correlation.json, which used
        # the focused trajectory branch as the response.  That obsolete file
        # belongs only in the external pre-cleanup recovery archive.
        out = Path(args.out or "results/raw/12_leakage_reanalysis.json")
        save_json(out, {"results": res})
        save_json(Path("results/raw/12_coupling_scan_projected_pbme.json"),
                  {"results": res["coupling_scan"]})
        print(json.dumps(res, indent=2))
        print(f"[08] written to {out} and "
              f"results/raw/12_coupling_scan_projected_pbme.json")
        if res["n_regimes"] < len(MOMENTA):
            print(f"[08] fewer than {len(MOMENTA)} regimes available; the correlation "
                  "study is NOT reportable yet")
            return 1
        return 0

    cfg = load_config(args.config)
    hbar, mass = float(cfg["hbar"]), float(cfg["mass"])
    convention = TraceConvention(cfg["trace_convention"])
    if args.coupling is not None:
        cfg["model"] = {**cfg["model"], "C": float(args.coupling)}
    model = build_model(cfg["model"], mass=mass)
    n = model.nstates
    P0, R0 = args.P0, args.R0
    sigma_R = hbar / (2.0 * (P0 / 20.0))
    init_state = int(cfg["initial_conditions"]["init_state"])
    d = DEFAULTS_FOR(P0)
    T = float(args.T if args.T is not None
              else d.get("T", 2.0 * mass * abs(R0) / P0))
    dt = args.dt
    n_steps = int(round(T / dt))
    n_samples = int(args.n_samples or cfg["mapping_sampling"]["n_samples"])
    if args.P_half is not None:
        d["P_half"] = float(args.P_half)
    if args.R_min is not None or args.R_max is not None:
        lo = float(args.R_min if args.R_min is not None else d["R_box"][0])
        hi = float(args.R_max if args.R_max is not None else d["R_box"][1])
        d["R_box"] = (lo, hi)
    d["P_center"] = (float(args.P_center)
                     if args.P_center is not None else float(P0))
    if args.n_R is not None:
        d["n_R"] = int(args.n_R)
    if args.n_P is not None:
        d["n_P"] = int(args.n_P)

    grid = build_grid(d["R_box"][0], d["R_box"][1], d["n_R"],
                      d["P_center"] - d["P_half"],
                      d["P_center"] + d["P_half"], d["n_P"])
    h_floor_R = args.bandwidth_cells * grid.dR
    h_floor_P = args.bandwidth_cells * grid.dP
    split = subsystem_split(model, grid.R[:, None], convention)
    dh = np.ascontiguousarray(split.dh_map_dR[:, 0])

    rec = RunRecorder(script=Path(__file__).name, seed=int(cfg["seed"]), config={
        **{k: v for k, v in cfg.items() if not k.startswith("_")},
        "P0": P0, "R0": R0, "sigma_R": sigma_R, "dt": dt, "T": T,
        "n_steps": n_steps, "n_save": args.n_save, "n_samples": n_samples,
        "grid": grid.parameters(), "bandwidth_cells": args.bandwidth_cells,
        "bandwidth_floor": {"h_R": h_floor_R, "h_P": h_floor_P},
        "estimator_sizes": args.estimator_sizes,
        "estimator_seeds": args.estimator_seeds,
        "norm": "phi-weighted L2 on amplitudes, ||f||^2 = sum_n n! |c_n|^2, "
                "integrated over (R,P) with Lebesgue measure"})
    print(f"[08] P0={P0} grid={d['n_R']}x{d['n_P']} T={T} dt={dt} "
          f"n_samples={n_samples}  bandwidth floor h_R={h_floor_R:.4g} "
          f"h_P={h_floor_P:.4g}")

    # ---- routes A and B, stepped together ------------------------------
    rho = initial_coefficient_field(grid, R0, P0, sigma_R, init_state, n, hbar)
    prop = StrangPropagator(grid, model, dt, hbar, convention)
    rng = np.random.default_rng(int(cfg["seed"]))
    ens = sample_initial_ensemble(rng, n_samples, nstates=n, init_state=init_state,
                                  R0=R0, P0=P0, sigma_R=sigma_R,
                                  scheme="focused", hbar=hbar)
    state = pack_state(ens.x, ens.R, ens.P)

    tA, DA, DA_abs, popA = [], [], [], []
    DB, DB_abs, hB = [], [], []
    t0 = time.perf_counter()
    step = 0
    while True:
        tA.append(step * dt)
        a = leakage_from_grid(rho, grid, model, hbar, convention)
        DA.append(a["relative"])
        DA_abs.append(a["norm"])
        popA.append(np.asarray(grid_observables(rho, grid, model)["populations"]))

        x, R, P = unpack_state(state, n, model.nbath)
        hR = max(h_floor_R, silverman_bandwidth(R))
        hP = max(h_floor_P, silverman_bandwidth(P))
        gP, info = kde_grad_P_rho_W(x, R, P, grid, hbar, n, ens.weights,
                                    h_R=hR, h_P=hP)
        b = leakage_from_fields(dh, gP, grid.cell_area, n, model.nbath,
                                convention.value)
        DB_abs.append(b["norm"])
        DB.append(b["norm"] / a["rho_norm"])   # same denominator, so comparable
        hB.append({"h_R": hR, "h_P": hP})
        if step == n_steps:
            break
        # The two routes are independent between saved endpoints.  Advancing
        # the coefficient field as one block fuses adjacent exact Strang
        # half-advections; advancing the trajectory state stepwise retains its
        # original integrator and produces the identical endpoint.
        block = min(args.n_save, n_steps - step)
        rho = prop.advance(rho, block)
        for _ in range(block):
            state = symmetric_rotation_step(state, dt, model, hbar=hbar,
                                            convention=convention)
        step += block
    print(f"[08] routes A and B done in {time.perf_counter() - t0:.1f} s "
          f"({len(tA)} sample times)")

    results = {"route_A": {"t": tA, "leakage_relative": DA,
                           "leakage_absolute": DA_abs,
                           "populations": [p.tolist() for p in popA],
                           "grad_P": "spectral, exact"},
               "route_B": {"t": tA, "leakage_relative": DB,
                           "leakage_absolute": DB_abs,
                           "bandwidths": hB, "n_samples": n_samples,
                           "grad_P": "binned Gaussian KDE on the PBME ensemble, "
                                     "CIC-deconvolved, analytic kernel derivative"}}
    ratio = (np.asarray(DB) / np.maximum(np.asarray(DA), 1e-300))
    results["route_B"]["ratio_to_route_A"] = ratio.tolist()
    results["route_B"]["median_ratio"] = float(np.median(ratio))
    print(f"[08] Route B / Route A leakage: median {np.median(ratio):.3f}, "
          f"range [{ratio.min():.3f}, {ratio.max():.3f}]")

    # ---- estimator study at t = 0, against the analytic derivative ------
    print("[08] estimator study at t=0 (analytic ground truth)")
    exact = _exact_grad_P_initial(grid, R0, P0, sigma_R, init_state, n, hbar)
    rho0 = initial_coefficient_field(grid, R0, P0, sigma_R, init_state, n, hbar)
    msk = _mask(grid, rho0)
    leak_exact = leakage_from_fields(dh, exact, grid.cell_area, n, model.nbath,
                                     convention.value)["norm"]
    rows = []
    for N in args.estimator_sizes:
        per_seed, leaks = [], []
        for s in range(args.estimator_seeds):
            r = np.random.default_rng(int(cfg["seed"]) + 1000 * s + N)
            e = sample_initial_ensemble(r, N, nstates=n, init_state=init_state,
                                        R0=R0, P0=P0, sigma_R=sigma_R,
                                        scheme="focused", hbar=hbar)
            hR = max(h_floor_R, silverman_bandwidth(e.R))
            hP = max(h_floor_P, silverman_bandwidth(e.P))
            g, _ = kde_grad_P_rho_W(e.x, e.R, e.P, grid, hbar, n, e.weights,
                                    h_R=hR, h_P=hP)
            per_seed.append(_score(g[..., init_state, init_state],
                                   exact[..., init_state, init_state], msk))
            leaks.append(leakage_from_fields(dh, g, grid.cell_area, n,
                                             model.nbath, convention.value)["norm"])
        rows.append({
            "n_samples": N,
            "rmse_mean": float(np.mean([p["rmse"] for p in per_seed])),
            "rmse_std": float(np.std([p["rmse"] for p in per_seed], ddof=1)),
            "relative_rmse_mean": float(np.mean([p["relative_rmse"] for p in per_seed])),
            "bias_mean": float(np.mean([p["bias"] for p in per_seed])),
            "bias_std": float(np.std([p["bias"] for p in per_seed], ddof=1)),
            "leakage_mean": float(np.mean(leaks)),
            "leakage_std": float(np.std(leaks, ddof=1)),
            "leakage_exact": float(leak_exact),
            "leakage_relative_error": float(abs(np.mean(leaks) - leak_exact)
                                            / leak_exact),
            "h_R": hR, "h_P": hP})
        print(f"     N={N:<8d} rel-RMSE(grad_P)={rows[-1]['relative_rmse_mean']:.4f} "
              f"  D_H estimate {rows[-1]['leakage_mean']:.4e} "
              f"+- {rows[-1]['leakage_std']:.1e}  (exact {leak_exact:.4e}, "
              f"rel err {rows[-1]['leakage_relative_error']:.3f})")

    Ns = np.array([r["n_samples"] for r in rows], float)
    rm = np.array([r["relative_rmse_mean"] for r in rows], float)
    slope = float(np.polyfit(np.log(Ns), np.log(rm), 1)[0])
    lk = np.array([r["leakage_relative_error"] for r in rows], float)
    slope_leak = (float(np.polyfit(np.log(Ns), np.log(np.maximum(lk, 1e-16)), 1)[0])
                  if np.all(lk > 0) else float("nan"))
    results["estimator_study"] = {
        "ground_truth": "analytic grad_P of the initial Wigner packet",
        "rows": rows, "rmse_scaling_exponent": slope,
        "leakage_error_scaling_exponent": slope_leak,
        "note": "exponent -0.5 would be plain Monte-Carlo scaling; a KDE "
                "derivative estimator is expected to be slower, and the "
                "measured value is what is reported"}
    print(f"[08] grad_P relative-RMSE scales as N^{slope:.3f}; "
          f"D_H relative error as N^{slope_leak:.3f}")

    rec.check("gateM.leakage_zero_without_coupling", 0.0, 1e-15,
              "structural: leakage vanishes identically when grad_R h = 0 "
              "(checked in tests/test_leakage.py)")
    rec.results = results
    out = Path(args.out or
               f"results/raw/08_leakage_P{int(round(P0))}{args.tag}.json")
    rec.write(out)
    print(f"[08] written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
