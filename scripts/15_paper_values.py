#!/usr/bin/env python3
r"""Every number quoted in the main text, from the archived raw results.

Writes
  manuscript/paper_values.tex        LaTeX macros, one per quoted number
  results/tables/paper_values.json   the same numbers with their provenance

Roles of the archived calculations (never interchanged)
  07g_gh_coefficient_propagation_P*   Gaussian--Hermite (GH) propagation of the
                                      exact mapping QCLE.  The QCLE result.
  10_schrodinger_convergence_P*       full quantum reference.  The value used
                                      is the most refined run ('finest': dt/4,
                                      twice the grid density, enlarged box).
  11_pbme_projected_P*                PBME through the unbiased projected-
                                      density estimator (control variate),
                                      with its standard error over seeds.
  10b_qcle_grid_crosscheck_P*         subsystem QCLE, four real fields, RK4.
                                      Equivalence check only.
  07_coefficient_propagation_P* and   subsystem QCLE, Strang splitting at
  13_P20_wide_momentum_box            dt = 0.5.  Equivalence check only.
  08_leakage_P*, 12_leakage_*         quartic defect rate evaluated on the
                                      Strang subsystem-QCLE solution.

Time series are compared only at instants saved by both archives; nothing is
interpolated.  A difference whose two calculations were not run on the same
bath grid is flagged as such and never quoted as an equivalence measure.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT_TEX = ROOT / "manuscript" / "paper_values.tex"
OUT_JSON = ROOT / "results" / "tables" / "paper_values.json"

MOMENTA = (10, 20, 30, 40, 70, 100)
TAG = {10: "PTen", 20: "PTwenty", 30: "PThirty", 40: "PForty",
       70: "PSeventy", 100: "POneHundred"}

MACROS: dict[str, str] = {}
LEDGER: dict[str, dict] = {}


def load(name: str) -> dict:
    path = RAW / name
    if not path.is_file():
        raise SystemExit(f"[15] required archive missing: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def put(name: str, tex: str, value, source: str, note: str = "") -> None:
    if name in MACROS:
        raise SystemExit(f"[15] macro defined twice: {name}")
    if any(bad in tex.lower() for bad in ("nan", "inf")):
        raise SystemExit(f"[15] non-finite value for {name}: {tex}")
    if not name.isalpha():
        raise SystemExit(f"[15] illegal macro name: {name}")
    MACROS[name] = tex
    LEDGER[name] = {"tex": tex, "value": value, "source": source, "note": note}


def sci(x: float, digits: int = 2) -> str:
    """x in LaTeX, `digits` significant figures, scientific below 1e-3."""
    x = float(x)
    if x == 0.0:
        return "0"
    if not math.isfinite(x):
        raise SystemExit(f"[15] non-finite value {x}")
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    m = float(f"{m:.{digits - 1}f}")
    if abs(m) >= 10.0:                      # rounding carried a digit
        m /= 10.0
        e += 1
    if -3 <= e <= 2:
        dec = max(digits - 1 - e, 0)
        return f"{x:.{dec}f}"
    return f"{m:.{digits - 1}f}\\times10^{{{e}}}"


def sci_up(x: float, digits: int = 2) -> str:
    """An upper bound: x rounded UP to `digits` significant figures.

    Used wherever the text says "at most", "below", "to within" or "conserved
    to", so that rounding can never turn a true bound into a false one."""
    x = float(x)
    if x <= 0.0 or not math.isfinite(x):
        raise SystemExit(f"[15] cannot bound {x}")
    e = int(math.floor(math.log10(x)))
    scale = 10 ** (digits - 1)
    m = math.ceil(x / 10 ** e * scale * (1 - 1e-12)) / scale
    if m >= 10.0:
        m /= 10.0
        e += 1
    y = m * 10 ** e
    if y < x * (1 - 1e-9):
        raise SystemExit(f"[15] upward rounding failed for {x}")
    if -3 <= e <= 2:
        return f"{y:.{max(digits - 1 - e, 0)}f}"
    return f"{m:.{digits - 1}f}\\times10^{{{e}}}"

def sci_down(x: float, digits: int = 2) -> str:
    """A lower bound: x rounded DOWN to `digits` significant figures, in the
    notation of :func:`sci_up` (x > 0).  Used for the lower end of a range, so
    that "between a and b" always contains the values it summarizes."""
    from decimal import Decimal, ROUND_FLOOR
    x = float(x)
    if x <= 0.0 or not math.isfinite(x):
        raise SystemExit(f"[15] cannot bound {x} from below")
    d = Decimal(repr(x))
    e = d.adjusted()
    y = d.quantize(Decimal(1).scaleb(e - digits + 1), rounding=ROUND_FLOOR)
    if float(y) > x:
        raise SystemExit(f"[15] downward rounding failed for {x}")
    if -3 <= e <= 2:
        return format(y, f".{max(digits - 1 - e, 0)}f")
    return format(y.scaleb(-e), f".{digits - 1}f") + f"\\times10^{{{e}}}"


def dec_bound(x: float, dec: int, up: bool) -> str:
    """x to `dec` decimals, rounded outward -- up for the upper end of a
    range, down for the lower end -- whatever the sign of x."""
    from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
    y = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-dec),
                                         rounding=ROUND_CEILING if up else ROUND_FLOOR)
    if (up and float(y) < float(x)) or (not up and float(y) > float(x)):
        raise SystemExit(f"[15] outward rounding failed for {x}")
    return format(y, f".{dec}f")


def two_sig_floor(x: float) -> str:
    """A lower bound: x rounded DOWN to two significant figures ("at least")."""
    x = float(x)
    e = int(math.floor(math.log10(x)))
    q = 10.0 ** (e - 1)
    v = math.floor(x / q * (1 + 1e-12)) * q
    return f"{v:.{max(1 - e, 0)}f}"


def two_sig(x: float) -> str:
    """A ratio to two significant figures, as an ordinary number."""
    x = float(x)
    e = int(math.floor(math.log10(abs(x))))
    v = round(x, 1 - e)
    return f"{v:.{max(1 - e, 0)}f}"


def fixed(x: float, dec: int) -> str:
    return f"{float(x):.{dec}f}"


def sci_always(x: float, digits: int = 2) -> str:
    """x as m x 10^e with `digits` significant figures, whatever its size, so
    that a table column reads in one notation."""
    x = float(x)
    if x == 0.0 or not math.isfinite(x):
        raise SystemExit(f"[15] cannot format {x}")
    e = int(math.floor(math.log10(abs(x))))
    m = round(x / 10 ** e, digits - 1)
    if abs(m) >= 10.0:
        m /= 10.0
        e += 1
    return f"{m:.{digits - 1}f}\\times10^{{{e}}}"


def sci_sem(x: float, sem: float, quantum: float = 1e-4) -> str:
    """x with its standard error as m(s) x 10^e, the last digit of m at the
    absolute precision `quantum` and s in units of that digit."""
    n = int(round(float(x) / quantum))
    if n == 0:
        raise SystemExit(f"[15] {x} vanishes at the precision {quantum}")
    xq = n * quantum
    e = int(math.floor(math.log10(abs(xq)) + 1e-12))
    dec = e - int(round(math.log10(quantum)))
    if dec < 0:
        raise SystemExit(f"[15] {x} exceeds the table format")
    return f"{xq / 10 ** e:.{dec}f}({int(round(sem / quantum))})\\times10^{{{e}}}"


def signed_pad(tex: str, value: float) -> str:
    """Pad a non-negative entry by the width of a minus sign, so that a
    left-aligned column of mixed signs aligns on the decimal point."""
    return tex if value < 0 else f"\\phantom{{-}}{tex}"


def common_idx(tA, tB, rtol=1e-9):
    """Indices of the instants saved by both A and B (no interpolation)."""
    tA = np.asarray(tA, float); tB = np.asarray(tB, float)
    iA, iB = [], []
    for i, t in enumerate(tA):
        j = int(np.argmin(np.abs(tB - t)))
        if abs(tB[j] - t) <= rtol * max(1.0, abs(t)):
            iA.append(i); iB.append(j)
    return np.asarray(iA, int), np.asarray(iB, int)


def common(tA, yA, tB, yB, rtol=1e-9):
    """Values of A and B at the instants saved by both (no interpolation)."""
    iA, iB = common_idx(tA, tB, rtol)
    return (np.asarray(yA, float)[iA], np.asarray(yB, float)[iB],
            np.asarray(tA, float)[iA])


def h5(name: str) -> dict:
    import h5py
    with h5py.File(RAW / name, "r") as f:
        return {k: f[k][()] for k in f if isinstance(f[k], h5py.Dataset)
                and f[k].ndim <= 2 and f[k].size < 5_000_000}


def supplementary(rows: dict) -> None:
    """Numbers of the supplementary material, and the numerical-uncertainty
    statements of the main text, from the same archives."""
    # ---------------------------------------- numerical uncertainty of the QCLE
    acc = load("07c_p10_refinement_acceptance.json")
    if not acc["all_passed"]:
        raise SystemExit("[15] P10 refinement acceptance did not pass")
    cfg = acc["metadata"]["config"]["fine_grid"]
    g10 = load("07g_gh_coefficient_propagation_P10.json")["metadata"]["grid"]
    if (cfg["n_R"], cfg["n_P"]) != (g10["n_R"], g10["n_P"]) or \
       [cfg["R_min"], cfg["R_max"]] != list(g10["R_box"]) or [cfg["P_min"], cfg["P_max"]] != list(g10["P_box"]):
        raise SystemExit("[15] P10 refinement study is not on the grid of the GH run")
    ar = acc["results"]
    src_acc = "07c_p10_refinement_acceptance.json"
    for P0 in MOMENTA:
        r = rows[P0]
        if P0 == 10:
            r["u_num"] = ar["max_population_linf_over_time"]
        else:
            if r["grid_spatial"] is None or r["grid_dthalf"] is None or not r["grid_same"]:
                raise SystemExit(f"[15] P{P0}: refinement data missing")
            r["u_num"] = max(r["e_grid"], r["grid_spatial"], r["grid_dthalf"])
    hi = [P for P in MOMENTA if P >= 20]
    put("numUncHigh", sci_up(max(rows[P]["u_num"] for P in hi)), max(rows[P]["u_num"] for P in hi),
        "07g_*; 10b_*", "max over P0>=20 of max(|GH-RK4|, grid halving, step halving) at T")
    put("numUncPTen", sci_up(rows[10]["u_num"]), rows[10]["u_num"], src_acc,
        "Strang 2048x1024 vs 4096x2048, max over time")
    u_all = max(rows[P]["u_num"] for P in MOMENTA)
    put("numUncAll", sci_up(u_all), u_all, "07g_*; 10b_*; " + src_acc, "max over all regimes (upper bound)")
    # the steps of the four-field solver, in two groups
    dts = sorted({rows[P]["grid_dt"] for P in MOMENTA})
    if len(dts) != 2:
        raise SystemExit("[15] the four-field solver uses other than two steps")
    small = [P for P in MOMENTA if rows[P]["grid_dt"] == dts[0]]
    put("gridDtSmall", f"{dts[0]:g}", dts[0], "10b_*")
    put("gridDtLarge", f"{dts[1]:g}", dts[1], "10b_*")
    put("gridDtSmallList", (", ".join(str(P) for P in small[:-1]) + f"\\text{{ and }}{small[-1]}"
                            if len(small) > 1 else str(small[0])),
        small, "10b_*", "regimes run at the smaller step")
    ratio_num = min(rows[P]["e_qcle"] / rows[P]["u_num"] for P in MOMENTA)
    ratio_q = min(rows[P]["e_qcle"] / rows[P]["q_unc"] for P in MOMENTA)
    put("qcleOverNumMin", two_sig_floor(ratio_num), ratio_num, "derived", "min over regimes of E_QCLE / u_num")
    r10 = rows[10]["e_qcle"] / rows[10]["u_num"]
    put("qcleOverNumPTen", two_sig_floor(r10), r10, "derived", "E_QCLE / u_num at P0 = 10")
    put("qcleOverQMin", two_sig_floor(ratio_q), ratio_q, "derived", "min over regimes of E_QCLE / quantum uncertainty")
    put("pTenBoundary", sci_up(max(ar["boundary_mass"]["R"]["fine"], ar["boundary_mass"]["P"]["fine"])),
        max(ar["boundary_mass"]["R"]["fine"], ar["boundary_mass"]["P"]["fine"]), src_acc,
        "fraction of the trace within one cell of the box edge, 4096x2048")
    put("pTenBoundaryCoarse", sci(max(ar["boundary_mass"]["R"]["coarse"], ar["boundary_mass"]["P"]["coarse"])),
        max(ar["boundary_mass"]["R"]["coarse"], ar["boundary_mass"]["P"]["coarse"]), src_acc)
    put("gridSpatialMax", sci_up(max(rows[P]["grid_spatial"] for P in hi)),
        max(rows[P]["grid_spatial"] for P in hi), "10b_*", "grid halved in each direction, at T, P0>=20")
    put("gridDtHalfMax", sci_up(max(rows[P]["grid_dthalf"] for P in hi)),
        max(rows[P]["grid_dthalf"] for P in hi), "10b_*")
    put("gridBoundaryMax", sci_up(max(rows[P]["grid_bmass"] for P in hi)),
        max(rows[P]["grid_bmass"] for P in hi), "10b_*", "P0>=20, production grids")
    put("gridCoarsePTen", sci(load("10b_qcle_grid_crosscheck_P10.json")["results"]
                              ["independent_spatial_refinement"]["final_population_linf"]),
        None, "10b_qcle_grid_crosscheck_P10.json", "1024x512 vs 2048x1024 at T")
    # at P0 = 10 the Strang/RK4 comparison was made on the coarser grid of the
    # four-field run; the text says the difference does not change with the
    # Strang step, which must hold to 1 %
    s10 = rows[10]
    if abs(s10["strang_linf"] - s10["strang_linf_half"]) > 0.01 * s10["strang_linf"]:
        raise SystemExit("[15] P10: Strang/RK4 difference changes with the Strang step")
    for P0 in MOMENTA:
        put(f"strangLinf{TAG[P0]}", sci(rows[P0]["strang_linf"], 2), rows[P0]["strang_linf"], "10b_*",
            "max_t |Strang(0.5) - RK4| on the grid of the four-field run")
        put(f"strangLinfHalf{TAG[P0]}", sci(rows[P0]["strang_linf_half"], 2),
            rows[P0]["strang_linf_half"], "10b_*", "max_t |Strang(0.25) - RK4| on the grid of the four-field run")
        if P0 >= 20:
            put(f"numUnc{TAG[P0]}", sci(rows[P0]["u_num"], 1), rows[P0]["u_num"], "derived")
            put(f"gridSpatial{TAG[P0]}", sci(rows[P0]["grid_spatial"], 1), rows[P0]["grid_spatial"], "10b_*")

    # ------------------------------------------------- Arnoldi subspace sizes
    mh = {rows[P]["krylov_m"] for P in MOMENTA if P >= 20}
    if len(mh) != 1:
        raise SystemExit("[15] Arnoldi subspace differs between the regimes P0 >= 20")
    put("ghMHigh", str(mh.pop()), None, "07g_*", "P0 >= 20")
    put("ghMLow", str(rows[10]["krylov_m"]), rows[10]["krylov_m"], "07g_*P10")

    # ---------------------------------- the solution the defect rate is evaluated on
    dts = {P: load(f"08_leakage_P{P}.json")["metadata"]["config"]["dt"] for P in MOMENTA}
    others = sorted({v for P, v in dts.items() if P != 70})
    if len(others) != 1:
        raise SystemExit("[15] defect-rate steps differ between regimes other than P0 = 70")
    est = [(dts[P] / rows[P]["strang_dt"]) ** 2 * rows[P]["e_strang_max"] for P in MOMENTA]
    put("dhDt", f"{others[0]:g}", others[0], "08_leakage_*", "Strang step of the defect-rate runs")
    put("dhDtPSeventy", f"{dts[70]:g}", dts[70], "08_leakage_P70.json")
    put("dhSplitErr", sci_up(max(est), 1), max(est), "derived",
        "second-order scaling of max_t |GH - Strang(0.5)| to the step of the defect-rate run")

    # ------------------------------------------------- error ratio in time
    # |PBME - Q| against |GH - Q| at the instants saved by all three, where the
    # PBME deviation exceeds twice its standard error
    worst = math.inf
    for P0 in MOMENTA:
        tg, yg = rows[P0]["gh_series"]; tq, yq = rows[P0]["q_series"]
        tp, yp, sp_ = rows[P0]["pb_series"]
        a, b, t1 = common(tg, yg, tq, yq)             # a = QCLE, b = quantum
        c1, pv, _ = common(t1, np.abs(a - b), tp, yp)  # |QCLE - Q|, PBME
        qv, sv, _ = common(t1, b, tp, sp_)             # quantum, PBME standard error
        ep = np.abs(pv - qv)
        m = (ep > 2.0 * sv) & (c1 > 0)
        if not m.any():
            raise SystemExit(f"[15] P{P0}: no resolved common instant")
        rows[P0]["ratio_t_min"] = float(np.min(ep[m] / c1[m]))
        rows[P0]["ratio_t_n"] = int(m.sum())
        worst = min(worst, rows[P0]["ratio_t_min"])
    put("ratioTimeMin", two_sig_floor(worst), worst, "07g_*; 10_*; 11_*",
        "min over regimes and resolved common instants of |PBME-Q| / |GH-Q|")

    # --------------------------------------------------- quantum reference
    lo = [P for P in MOMENTA if rows[P]["q_nr_prod"] == rows[10]["q_nr_prod"]]
    hi_q = [P for P in MOMENTA if P not in lo]
    if lo != [10, 20, 30, 40] or len({rows[P]["q_nr_prod"] for P in hi_q}) != 1:
        raise SystemExit("[15] quantum production grids do not follow the grouping quoted in the text")
    put("qNRProdLow", str(rows[10]["q_nr_prod"]), rows[10]["q_nr_prod"], "10_*", "P0 <= 40")
    put("qNRProdHigh", str(rows[100]["q_nr_prod"]), rows[100]["q_nr_prod"], "10_*", "P0 >= 70")
    if len({rows[P]["q_nr_ref"] for P in lo}) != 1 or len({rows[P]["q_nr_ref"] for P in hi_q}) != 1:
        raise SystemExit("[15] quantum reference grids do not follow the production grouping")
    put("qNRRefLow", str(rows[10]["q_nr_ref"]), rows[10]["q_nr_ref"], "10_*", "reference run, P0 <= 40")
    put("qNRRefHigh", str(rows[100]["q_nr_ref"]), rows[100]["q_nr_ref"], "10_*", "reference run, P0 >= 70")
    if len({rows[P]["q_dt_ref"] for P in MOMENTA}) != 1:
        raise SystemExit("[15] quantum reference time steps differ")
    put("qDtRef", f"{rows[40]['q_dt_ref']:g}", rows[40]["q_dt_ref"], "10_*", "reference run")
    if len({rows[P]["q_dt_prod"] for P in MOMENTA}) != 1:
        raise SystemExit("[15] quantum production time steps differ")
    put("qDtProd", f"{rows[40]['q_dt_prod']:g}", rows[40]["q_dt_prod"], "10_*")
    oq = [rows[P]["q_order"] for P in MOMENTA]
    put("qOrderMin", dec_bound(min(oq), 2, False), min(oq), "10_*"); put("qOrderMax", dec_bound(max(oq), 2, True), max(oq), "10_*")
    put("qNormDrift", sci_up(max(rows[P]["q_norm_drift"] for P in MOMENTA), 1),
        max(rows[P]["q_norm_drift"] for P in MOMENTA), "10_*")

    # ------------------------------------------------------------ PBME
    nb = sorted({rows[P]["pb_blocks"] for P in MOMENTA})
    bs = {rows[P]["pb_block_size"] for P in MOMENTA}
    if len(bs) != 1 or rows[40]["pb_blocks"] != nb[-1]:
        raise SystemExit("[15] PBME block structure differs between regimes")
    put("pbNBlocks", str(nb[-1]), nb[-1], "11_*"); put("pbNBlocksMin", str(nb[0]), nb[0], "11_*")
    put("pbBlockSize", f"{bs.pop():,}".replace(",", "\\,"), rows[40]["pb_block_size"], "11_*")
    put("pbNSeedsMin", str(min(rows[P]["pb_seeds"] for P in MOMENTA)),
        min(rows[P]["pb_seeds"] for P in MOMENTA), "11_*")
    sr = [rows[P]["pb_sem_seed"] / rows[P]["pb_sem"] for P in MOMENTA]
    put("semRatioMin", dec_bound(min(sr), 2, False), min(sr), "11_*", "SEM over seed groups / SEM over blocks")
    put("semRatioMax", dec_bound(max(sr), 2, True), max(sr), "11_*")
    sl = [rows[P]["pb_slope"] for P in MOMENTA]
    put("pbSlopeMin", dec_bound(min(sl), 2, False), min(sl), "11_*", "log-log slope of the block SEM, >= 4 blocks")
    put("pbSlopeMax", dec_bound(max(sl), 2, True), max(sl), "11_*")
    put("pbNegFrac", f"{max(rows[P]['pb_negfrac'] for P in MOMENTA):.3f}",
        max(rows[P]["pb_negfrac"] for P in MOMENTA), "11_*", "fraction of samples with negative weight")
    tsc = load("11c_pbme_timestep_convergence.json")
    if not tsc["all_passed"]:
        raise SystemExit("[15] PBME time-step study did not pass")
    reg = tsc["results"]["regimes"]["40.0"]
    d1 = reg["pairs"]["0.5->0.25"]["final_population_difference"][1]
    put("pbDtChange", sci(d1, 2), d1, "11c_pbme_timestep_convergence.json", "dt 0.5 -> 0.25, rho22(T)")
    put("pbDtOrder", f"{reg['observed_order']:.2f}", reg["observed_order"], "11c_pbme_timestep_convergence.json")
    put("pbDtSamples", f"{tsc['results']['n_samples']:,}".replace(",", "\\,"),
        tsc["results"]["n_samples"], "11c_pbme_timestep_convergence.json")
    put("pbDtOverSem", sci(d1 / rows[40]["pb_sem"], 1), d1 / rows[40]["pb_sem"], "derived")

    # --------------------------------------------- kernel gradient estimator
    ex_, rel_lo, rel_hi, rel_first = [], [], [], []
    for P0 in MOMENTA:
        es = load(f"08_leakage_P{P0}.json")["results"]["estimator_study"]
        ex_.append(es["rmse_scaling_exponent"])
        rel_first.append(es["rows"][0]["relative_rmse_mean"])
        rel_hi.append(es["rows"][-1]["relative_rmse_mean"])
        n_here = [rw["n_samples"] for rw in es["rows"]]
        if P0 != MOMENTA[0] and n_here != ns:
            raise SystemExit("[15] estimator study sizes differ between regimes")
        ns = n_here
    put("kdeExpMin", dec_bound(min(ex_), 2, False), min(ex_), "08_leakage_*"); put("kdeExpMax", dec_bound(max(ex_), 2, True), max(ex_), "08_leakage_*")
    put("kdeRelFirstMin", dec_bound(min(rel_first), 2, False), min(rel_first), "08_leakage_*")
    put("kdeRelFirstMax", dec_bound(max(rel_first), 2, True), max(rel_first), "08_leakage_*")
    put("kdeRelLastMin", dec_bound(min(rel_hi), 2, False), min(rel_hi), "08_leakage_*")
    put("kdeRelLastMax", dec_bound(max(rel_hi), 2, True), max(rel_hi), "08_leakage_*")
    for nm, v in (("kdeNMin", ns[0]), ("kdeNMax", ns[-1])):
        e = round(math.log10(v))
        if 10 ** e != v:
            raise SystemExit("[15] estimator sizes are not powers of ten")
        put(nm, f"10^{{{e}}}", v, "08_leakage_*")
    put("kdeSeeds", str(load("08_leakage_P40.json")["metadata"]["config"]["estimator_seeds"]), None, "08_leakage_P40.json")

    # ----------------------------------------- integrator check of the GH code
    g9 = load("09g_gh_convergence_P40.json")
    if not g9["all_passed"]:
        raise SystemExit("[15] 09g did not pass")
    ts = g9["results"]["time_step"]
    put("ghRkOrder", f"{ts['fitted_order_state']:.2f}", ts["fitted_order_state"], "09g_gh_convergence_P40.json")
    put("ghRkErr", sci(ts["rows"][0]["error_state"]), ts["rows"][0]["error_state"], "09g_gh_convergence_P40.json",
        "relative state difference RK4 - exponential at the smallest step")
    put("ghRkDt", f"{ts['rows'][0]['dt']:.3f}", ts["rows"][0]["dt"], "09g_gh_convergence_P40.json")
    put("ghRkT", f"{g9['metadata']['T']:g}", g9["metadata"]["T"], "09g_gh_convergence_P40.json")
    rg = g9["metadata"]["reference_grid"]
    put("ghRkGrid", f"{rg[0]}\\times{rg[1]}", rg, "09g_gh_convergence_P40.json")

    # ------------------------------------------------ basis covariance (S9)
    cg = load("07cg_gh_basis_covariance.json")
    if not cg["all_passed"]:
        raise SystemExit("[15] covariance check did not pass")
    w, md = cg["results"]["worst"], cg["metadata"]
    put("covNRot", str(len(md["rotations"])), len(md["rotations"]), "07cg")
    put("covNComplex", str(md["n_complex_rotations"]), md["n_complex_rotations"], "07cg")
    put("covNRand", str(md["n_random_hamiltonians"]), md["n_random_hamiltonians"], "07cg")
    put("covOrderMix", sci_up(w["order_mixing"], 1), w["order_mixing"], "07cg", "upper bound")
    put("covGroup", sci_up(w["group_property"], 1), w["group_property"], "07cg", "upper bound")
    put("covFrame", sci_up(w["frame_intertwining"], 1), w["frame_intertwining"], "07cg", "upper bound")
    put("covOperator", sci_up(w["operator_conjugation"], 1), w["operator_conjugation"], "07cg", "upper bound")
    put("covProp", sci_up(w["propagated"], 1), w["propagated"], "07cg", "upper bound")
    put("covSep", f"{w['least_separation']:.2f}", w["least_separation"], "07cg")
    put("covSteps", str(md["time"]["n_steps"]), md["time"]["n_steps"], "07cg")
    put("covT", f"{md['time']['T']:.1f}", md["time"]["T"], "07cg")
    put("covGrid", f"{md['grid']['n_R']}\\times{md['grid']['n_P']}", None, "07cg")

    # ------------------------------------------------- symbolic archives (S5)
    p3 = load("01c_powers3.json")
    dt_ = p3["results"]["decision_tree"]
    if dt_["full_generator_max_degrees"] != [2, 2, 2] or dt_["pbme_only_max_degrees"] != [4, 6, 8]:
        raise SystemExit("[15] repeated-application degrees differ from the text")
    put("symPowers", "3", 3, "01c_powers3.json")
    vk = {**load("02_hermite_projection.json")["results"]["vk_audit"],
          **load("02c_vk_exhaustive.json")["results"]["vk_audit"]}
    if not all(v["full_within_VK"] and v["pbme_raises_by_two"] for v in vk.values()):
        raise SystemExit("[15] V_K audit failed")
    put("symVkMax", str(max(int(k) for k in vk)), max(int(k) for k in vk), "02_*, 02c_*")
    put("symVkCount", str(sum(v["n_basis_functions_checked"] for v in vk.values())),
        sum(v["n_basis_functions_checked"] for v in vk.values()), "02_*, 02c_*")
    lk = load("02_hermite_projection.json")["results"]["pbme_leakage"]
    if not lk["hermite_residual_is_wick_ordered_top4"] or lk["top4_equals_hermite_residual"]:
        raise SystemExit("[15] Wick-ordering statement not supported")
    put("symTermsTop", str(lk["n_terms_top4"]), lk["n_terms_top4"], "02_hermite_projection.json")
    put("symTermsHerm", str(lk["n_terms_hermite_residual"]), lk["n_terms_hermite_residual"], "02_hermite_projection.json")
    for name in ("03_qcle_correspondence", "04_physical_frame", "05_extra_cases_N3", "06_extra_cases_Nb2",
                 "06b_extra_cases_hermitian", "06c_extra_cases_hermitian_N3"):
        if not load(name + ".json")["all_passed"]:
            raise SystemExit(f"[15] {name} did not pass")
    s17 = load("17_generator_structure.json")
    if not s17["all_passed"]:
        raise SystemExit("[15] 17_generator_structure did not pass")
    c2, c3 = s17["results"]["C2_identities"], s17["results"]["C3_coefficient_operator"][0]
    c4 = s17["results"]["C4_pbme_deviation"]
    put("symIdCases", str(c2["n_cases"]), c2["n_cases"], "17_generator_structure.json")
    put("symIdOrder", str(c2["order_max"]), c2["order_max"], "17_generator_structure.json")
    put("symOpBasis", str(c3["n_basis_functions"]), c3["n_basis_functions"], "17_generator_structure.json")
    put("symOpOrder", str(c3["order_max"]), c3["order_max"], "17_generator_structure.json")
    n4 = {row["n_order4_functions_occupied"] for row in c4}
    if n4 != {int(MACROS["nQuarticModes"])}:
        raise SystemExit("[15] order-4 count differs between 12_ and 17_")
    put("symOrderFourTotal", str(c4[0]["n_order4_functions"]), c4[0]["n_order4_functions"], "17_generator_structure.json")


#: regimes compared in phase space (scripts/18, 19); at P0 = 10 the terminal
#: fields carry structure finer than the stored grid
PS_MOMENTA = (20, 30, 70, 100)
PS_COMP = {"A": "One", "C": "Two", "bR": "Re", "bI": "Im"}


def phase_space(rows: dict) -> None:
    """Terminal phase-space densities: quantum, QCLE and projected PBME."""
    sys.path.insert(0, str(ROOT / "src"))
    from mqcle.io import load_config
    cfg = load_config("tully_default.yaml")
    M, E0 = float(cfg["mass"]), float(cfg["model"]["E0"])
    neg_q, outside, floors, tails, solver, repro, trace = [], [], [], [], [], [], []
    traj_out, pb_out, l2_over_floor = [], [], []
    strang_peak, strang_steps, grid_same_measures = [], set(), []
    for P0 in PS_MOMENTA:
        tg = TAG[P0]
        a18 = load(f"18_pbme_phase_space_P{P0}.json")
        a19 = load(f"19_phase_space_comparison_P{P0}.json")
        if not (a18["all_passed"] and a19["all_passed"]):
            raise SystemExit(f"[15] P{P0}: phase-space archives did not pass their checks")
        src = f"19_phase_space_comparison_P{P0}.json"
        r18, r19 = a18["results"], a19["results"]
        # the PBME field is the ensemble of scripts/11, trajectory for trajectory
        if r18["n_samples"] != rows[P0]["pb_n"]:
            raise SystemExit(f"[15] P{P0}: the PBME field does not use the whole scripts/11 ensemble")
        shift = (P0 - math.sqrt(P0 ** 2 - 2.0 * M * E0)) / (P0 / 20.0)
        rows[P0]["ps_shift"] = shift
        put(f"psShift{tg}", two_sig(shift), shift, "config",
            "asymptotic momentum difference of the two surfaces over sigma_P")
        fr, cr = r19["full_resolution"], r19["cells_comparison"]
        for k, nm in PS_COMP.items():
            c = cr[k]
            rows[P0][f"ps_l2_qcle_{k}"] = c["rel_l2_qcle_quantum"]
            rows[P0][f"ps_l2_pbme_{k}"] = c["rel_l2_pbme_quantum"]
            put(f"psLtwoQcle{nm}{tg}", fixed(c["rel_l2_qcle_quantum"], 2), c["rel_l2_qcle_quantum"], src,
                "relative L2 distance, QCLE vs quantum, on the PBME cells")
            put(f"psLtwoPbme{nm}{tg}", fixed(c["rel_l2_pbme_quantum"], 2), c["rel_l2_pbme_quantum"], src,
                "relative L2 distance, PBME vs quantum, on the PBME cells")
            put(f"psLtwoPbmeQcle{nm}{tg}", fixed(c["rel_l2_pbme_qcle"], 2), c["rel_l2_pbme_qcle"], src,
                "relative L2 distance, PBME vs QCLE, on the PBME cells")
            put(f"psLtwoFull{nm}{tg}", fixed(fr[k]["rel_l2_qcle_quantum"], 2),
                fr[k]["rel_l2_qcle_quantum"], src, "relative L2 distance, QCLE vs quantum, full grid")
            floors.append(c["pbme_noise_floor_vs_quantum"])
            l2_over_floor.append(c["rel_l2_pbme_quantum"] / c["pbme_noise_floor_vs_quantum"])
        for k, nm in (("A", "One"), ("C", "Two")):
            f, c = fr[k], cr[k]
            neg_q.append(f["negative_volume_quantum"])
            put(f"psNegQ{nm}{tg}", sci(f["negative_volume_quantum"], 2), f["negative_volume_quantum"], src,
                "negative volume of the quantum diagonal density")
            put(f"psNegQcle{nm}{tg}", sci(f["negative_volume_qcle"], 2), f["negative_volume_qcle"], src,
                "negative volume of the QCLE diagonal density")
            put(f"psMomQcle{nm}{tg}", sci(c["momentum_l1_qcle"], 2), c["momentum_l1_qcle"], src,
                "L1 distance of the momentum distribution from the quantum one, / population, cells")
            put(f"psMomPbme{nm}{tg}", sci(c["momentum_l1_pbme"], 2), c["momentum_l1_pbme"], src,
                "same for the projected PBME")
            put(f"psMomFloor{nm}{tg}", sci_up(c["momentum_l1_pbme_noise_floor"], 1),
                c["momentum_l1_pbme_noise_floor"], src, "expected L1 of the PBME noise alone (upper bound)")
            rows[P0][f"ps_mom_qcle_{k}"] = c["momentum_l1_qcle"]
            rows[P0][f"ps_mom_pbme_{k}"] = c["momentum_l1_pbme"]
            rows[P0][f"ps_mom_floor_{k}"] = c["momentum_l1_pbme_noise_floor"]
            rows[P0][f"ps_neg_qcle_{k}"] = f["negative_volume_qcle"]
            rows[P0][f"ps_neg_q_{k}"] = f["negative_volume_quantum"]
        # widths and centres of the momentum distributions, on the cells;
        # moments are quoted only where no distribution has a sizeable
        # negative part, which would make them meaningless
        positive = {}
        for k, nm in (("A", "One"), ("C", "Two")):
            c = cr[k]
            positive[k] = max(c[f"momentum_negative_mass_{m}"]
                              for m in ("quantum", "qcle", "pbme")) < 0.05
            if not positive[k]:
                continue
            wr = c["momentum_std_pbme"] / c["momentum_std_quantum"]
            wq = c["momentum_std_qcle"] / c["momentum_std_quantum"]
            rows[P0][f"ps_width_pbme_{k}"] = wr
            rows[P0][f"ps_width_qcle_{k}"] = wq
            put(f"psWidthPbme{nm}{tg}", fixed(wr, 2), wr, src,
                "width of the PBME momentum distribution / exact width")
            put(f"psWidthQcle{nm}{tg}", fixed(wq, 2), wq, src,
                "width of the QCLE momentum distribution / exact width")
        if positive["A"] and positive["C"]:
            split = {m: cr["A"][f"momentum_mean_{m}"] - cr["C"][f"momentum_mean_{m}"]
                     for m in ("quantum", "qcle", "pbme")}
            split_sem = math.hypot(cr["A"]["momentum_mean_pbme_sem"],
                                   cr["C"]["momentum_mean_pbme_sem"])
            for m, nm in (("quantum", "Q"), ("qcle", "Qcle"), ("pbme", "Pbme")):
                rows[P0][f"ps_split_{m}"] = split[m]
                put(f"psSplit{nm}{tg}", fixed(split[m], 2), split[m], src,
                    "mean momentum of state 1 minus that of state 2")
            put(f"psSplitPbmeSem{tg}", sci_up(split_sem, 1), split_sem, src, "jackknife over blocks")
        outside.append(r19["window"]["fraction_abs_density_outside_cells"])
        # the independent solutions of the subsystem QCLE, whole densities
        sv = r19["solvers"]
        gpk = max(v["max_over_peak"] for v in sv["grid"].values())
        gl2 = max(v["rel_l2"] for v in sv["grid"].values())
        rows[P0]["ps_grid_peak"] = gpk
        put(f"psGridPeak{tg}", sci_up(gpk, 2), gpk, src,
            "largest |four-field - coefficient| over the peak density, any component")
        put(f"psGridLtwo{tg}", sci_up(gl2, 2), gl2, src,
            "largest relative L2 distance, four-field vs coefficient, any component")
        spk = max(v["max_over_peak"] for v in sv["strang"].values())
        sl2 = max(v["rel_l2"] for v in sv["strang"].values())
        put(f"psStrangPeak{tg}", sci_up(spk, 2), spk, src,
            "largest |Strang - coefficient| over the peak density, any component")
        put(f"psStrangLtwo{tg}", sci_up(sl2, 2), sl2, src,
            "largest relative L2 distance, Strang vs coefficient, any component")
        strang_peak.append(spk)
        strang_steps.add(float(sv["strang_step"]))
        # the four-field difference against the Runge--Kutta error of free flight
        rk = sv["grid_rk4_free_flight"]["components"]
        cpk = max(v["corrected_max_over_peak"] for v in rk.values())
        rows[P0]["ps_grid_corr"] = cpk
        rows[P0]["ps_rk_resid"] = max(v["residual_rel_l2"] for v in rk.values())
        rows[P0]["ps_rk_pred"] = max(v["predicted_max_over_peak"] for v in rk.values())
        put(f"psGridCorr{tg}", sci_up(cpk, 2), cpk, src,
            "largest |four-field - coefficient - predicted Runge-Kutta error of "
            "free flight| over the peak density, any component")
        for k in ("A", "C"):
            grid_same_measures += [
                abs(cr[k]["momentum_l1_grid"] - cr[k]["momentum_l1_qcle"]),
                abs(fr[k]["negative_volume_grid"] - fr[k]["negative_volume_qcle"])]
        for k in PS_COMP:
            tails.append(max(fr[k]["spectral_tail_qcle"], fr[k]["spectral_tail_quantum"]))
            solver.append(fr[k]["rel_l2_strang_qcle"])
        repro.append(r18["reproduction_max_abs_difference"])
        chk = {c["name"]: c["value"] for c in a19["checks"]}
        trace.append(chk["phase.quantum_trace"])
        traj_out.append(r18["fraction_trajectories_outside_window"])
        pb_out.append(max(r19["window"]["pbme_outside_abs_over_population"]))
        cells = r18["cells"]
        put(f"psCellR{tg}", f"{cells['dR']:.3g}", cells["dR"], f"18_pbme_phase_space_P{P0}.json")
        put(f"psCellP{tg}", f"{cells['dP']:.3g}", cells["dP"], f"18_pbme_phase_space_P{P0}.json")
        rows[P0]["ps_min_qcle_C"] = fr["C"]["momentum_min_qcle"]
        rows[P0]["ps_peak_q_C"] = fr["C"]["momentum_peak_quantum"]
    put("psNegQMax", sci_up(max(neg_q), 1), max(neg_q), "19_*", "largest negative volume of an exact diagonal density")
    put("psTailMax", sci_up(max(tails), 1), max(tails), "19_*",
        "largest root spectral power beyond 3/4 of Nyquist, quantum and QCLE fields")
    put("psSolverMax", sci_up(max(solver), 1), max(solver), "19_*",
        "largest relative L2 distance between the Strang and coefficient fields")
    put("psReproMax", sci_up(max(repro), 1), max(repro), "18_*",
        "largest difference of a re-propagated chunk total from scripts/11")
    put("psTraceErr", sci_up(max(trace), 1), max(trace), "19_*", "|trace - 1| of the quantum field")
    if len(strang_steps) != 1:
        raise SystemExit("[15] the Strang fields of scripts/10b differ in step between regimes")
    put("psStrangStep", f"{strang_steps.pop():g}", None, "10b_*", "step of the Strang fields")
    put("psStrangPeakMax", sci_up(max(strang_peak), 1), max(strang_peak), "19_*",
        "largest |Strang - coefficient| over the peak density, all regimes")
    low = [rows[P]["ps_grid_peak"] for P in PS_MOMENTA if P <= 30]
    put("psGridPeakLowMax", sci_up(max(low), 1), max(low), "19_*",
        "largest |four-field - coefficient| over the peak density at P0 = 20, 30")
    put("psGridSameMeasures", sci_up(max(grid_same_measures), 1), max(grid_same_measures), "19_*",
        "largest difference of N_a or D_a between the four-field and coefficient solutions")
    # Where the Runge--Kutta error of free flight is the larger part of the
    # four-field difference (P0 >= 30; at P0 = 20 it is below the difference,
    # which the text states with its own numbers).
    rk_regimes = [P for P in PS_MOMENTA if P >= 30]
    put("psRkList", ", ".join(str(P) for P in rk_regimes[:-1]) + r"\text{ and }"
        + str(rk_regimes[-1]), None, "19_*", "regimes of the Runge-Kutta comparison")
    resid = max(rows[P]["ps_rk_resid"] for P in rk_regimes)
    put("psRkResidMax", sci_up(resid, 2), resid, "19_*",
        "largest relative L2 residual of the four-field - coefficient difference "
        "after subtracting the predicted Runge-Kutta error of free flight, "
        "P0 >= 30, any component")
    after = max(rows[P]["ps_grid_corr"] for P in rk_regimes)
    put("psRkAfterMax", sci_up(after, 1), after, "19_*",
        "largest remaining |four-field - coefficient| over the peak density after "
        "that subtraction, P0 >= 30")
    put("psRkPredPTwenty", sci(rows[20]["ps_rk_pred"], 2), rows[20]["ps_rk_pred"],
        "19_phase_space_comparison_P20.json",
        "largest predicted Runge-Kutta error of free flight over the peak density, P0 = 20")
    if not rows[20]["ps_rk_pred"] < rows[20]["ps_grid_peak"]:
        raise SystemExit("[15] the text states that at P0 = 20 the predicted "
                         "Runge-Kutta error is below the measured difference")
    put("psTrajOutsideMax", sci_up(max(traj_out), 1), max(traj_out), "18_*",
        "largest fraction of PBME trajectories outside the cells")
    put("psOutsidePbmeMax", sci_up(max(pb_out), 1), max(pb_out), "19_*",
        "largest change of a PBME D_a the trajectories outside the cells can cause")
    # why P0 = 10 is not compared: its terminal QCLE field is not resolved
    import h5py
    with h5py.File(RAW / "07g_gh_coefficient_propagation_P10.h5", "r") as g10:
        t10 = []
        for k in ("A", "C"):
            a = g10[f"gh_final_{k}"][()]
            F = np.abs(np.fft.fft2(a)) ** 2
            kR = np.abs(np.fft.fftfreq(a.shape[0])) * 2.0
            kP = np.abs(np.fft.fftfreq(a.shape[1])) * 2.0
            m = (kR[:, None] > 0.75) | (kP[None, :] > 0.75)
            t10.append(float(np.sqrt(F[m].sum() / F.sum())))
    put("psTailPTen", two_sig_floor(min(t10)), min(t10), "07g_gh_coefficient_propagation_P10.h5",
        "root spectral power beyond 3/4 of Nyquist of the terminal QCLE diagonal densities at P0=10 (lower bound)")
    put("psOutsideMax", sci_up(max(outside), 1), max(outside), "19_*",
        "largest fraction of |rho| of a grid field outside the PBME cells")
    put("psFloorMax", sci_up(max(floors), 2), max(floors), "19_*",
        "largest statistical floor of a PBME cell distance, relative to the quantum norm")
    put("psLtwoOverFloorMin", f"{math.floor(min(l2_over_floor)):d}", min(l2_over_floor), "19_*",
        "smallest ratio of a PBME cell L2 distance to its statistical floor (rounded down)")
    # regimes in which the cell-averaged QCLE density is further from the exact
    # one than the PBME density, in the relative L2 distance
    def _andlist(Ps):
        Ps = [str(P) for P in Ps]
        # for use inside math mode, where a bare "and" would be set in italics
        return Ps[0] if len(Ps) == 1 else ", ".join(Ps[:-1]) + "\\text{ and }" + Ps[-1]
    worse2 = [P for P in PS_MOMENTA if rows[P]["ps_l2_qcle_C"] > rows[P]["ps_l2_pbme_C"]]
    worse12 = [P for P in PS_MOMENTA
               if rows[P]["ps_l2_qcle_bR"] > rows[P]["ps_l2_pbme_bR"]
               and rows[P]["ps_l2_qcle_bI"] > rows[P]["ps_l2_pbme_bI"]]
    worse1 = [P for P in PS_MOMENTA if rows[P]["ps_l2_qcle_A"] > rows[P]["ps_l2_pbme_A"]]
    if not worse2 or not worse12 or worse1:
        raise SystemExit("[15] the L2 ordering quoted in the text no longer holds")
    put("psLtwoQcleWorseTwoList", _andlist(worse2), worse2, "19_*",
        "regimes where the QCLE rho22 is further from the exact one than the PBME's (L2, cells)")
    put("psLtwoQcleWorseCohList", _andlist(worse12), worse12, "19_*",
        "regimes where the QCLE coherence is further from the exact one than the PBME's (L2, cells)")
    n1 = [rows[P]["ps_neg_qcle_A"] for P in PS_MOMENTA]
    put("psNegQcleOneMin", sci_down(min(n1), 2), min(n1), "19_*", "smallest QCLE negative volume of state 1")
    put("psNegQcleOneMax", sci_up(max(n1), 2), max(n1), "19_*", "largest QCLE negative volume of state 1")
    hi_ps = [P for P in PS_MOMENTA if rows[P]["ps_shift"] < 0.5]
    dh = max(rows[P][f"ps_mom_qcle_{k}"] for P in hi_ps for k in ("A", "C"))
    put("psMomQcleHighMax", sci_up(dh, 2), dh, "19_*",
        "largest QCLE D_a where Delta P < sigma_P / 2 (upper bound)")
    put("psShiftHighList", _andlist(hi_ps), hi_ps, "config", "regimes with Delta P < sigma_P / 2")
    d1 = [rows[P]["ps_mom_qcle_A"] for P in PS_MOMENTA]
    put("psMomQcleOneMax", sci_up(max(d1), 2), max(d1), "19_*", "largest QCLE D_1 (upper bound)")
    ratios = [rows[P][f"ps_mom_pbme_{k}"] / rows[P][f"ps_mom_qcle_{k}"]
              for P in PS_MOMENTA for k in ("A", "C")]
    put("psMomRatioMin", two_sig_floor(min(ratios)), min(ratios), "19_*",
        "smallest D_PBME / D_QCLE over regimes and states (lower bound)")
    put("psMomRatioMax", sci_up(max(ratios), 2), max(ratios), "19_*",
        "largest D_PBME / D_QCLE over regimes and states (upper bound)")
    mf = max(rows[P][f"ps_mom_floor_{k}"] for P in PS_MOMENTA for k in ("A", "C"))
    put("psMomFloorMax", sci_up(mf, 2), mf, "19_*",
        "largest statistical floor of a PBME momentum-distribution distance")
    of = min(rows[P][f"ps_mom_pbme_{k}"] / rows[P][f"ps_mom_floor_{k}"]
             for P in PS_MOMENTA for k in ("A", "C"))
    put("psMomOverFloorMin", f"{math.floor(of):d}", of, "19_*",
        "smallest ratio of a PBME D_a to its statistical floor (rounded down)")
    r20 = rows[20]
    put("psMomMinQcleTwoPTwenty", fixed(r20["ps_min_qcle_C"], 3), r20["ps_min_qcle_C"],
        "19_phase_space_comparison_P20.json", "minimum of the QCLE momentum distribution of state 2")
    put("psMomPeakQTwoPTwenty", fixed(r20["ps_peak_q_C"], 3), r20["ps_peak_q_C"],
        "19_phase_space_comparison_P20.json", "peak of the exact momentum distribution of state 2")


def main() -> int:
    rows = {}
    for P0 in MOMENTA:
        tg = TAG[P0]
        r: dict = {}
        # ---------------------------------------------------------- GH
        g = load(f"07g_gh_coefficient_propagation_P{P0}.json")
        md = g["metadata"]
        if md.get("method") != "gaussian_hermite_coefficient" or not g["all_passed"]:
            raise SystemExit(f"[15] P{P0}: GH archive is not a passed GH run")
        hist = g["results"]["history"]
        t_gh = np.array([h["t"] for h in hist])
        p_gh = np.array([h["populations"] for h in hist])
        a_gh = np.array([h["adiabatic_populations"] for h in hist])
        fin = g["results"]["final"]
        grid = md["grid"]
        r["gh_grid"] = (grid["n_R"], grid["n_P"])
        r["T"] = md["time"]["T"]
        r["gh_pop2"] = fin["populations"][1]
        r["gh_adpop2"] = fin["adiabatic_populations"][1]
        r["frame"] = max(h["physical_frame_residual"] for h in hist)
        r["trace"] = max(abs(h["trace"] - 1.0) for h in hist)
        r["parity"] = max(h["parity_leakage"] for h in hist)
        r["herm"] = max(h["hermiticity_error"] for h in hist)
        r["gh_wall_h"] = md["wall_time_s"] / 3600.0
        ks = md["time"]["krylov_stats"]
        r["krylov_m"] = md["time"]["krylov_m"]
        r["krylov_tol"] = md["time"]["krylov_tol"]
        r["substeps"] = ks["substeps"]
        r["applications"] = ks["applications"]
        r["n_modes"] = md["hermite"]["n_modes"]
        src_gh = f"07g_gh_coefficient_propagation_P{P0}.json"
        put(f"TT{tg}", f"{r['T']:g}", r["T"], src_gh, "final time")
        put(f"sigR{tg}", f"{10.0 / P0:.3g}", 10.0 / P0, "config: sigma_R = 10/P0")
        put(f"ghGrid{tg}", f"{grid['n_R']}\\times{grid['n_P']}", r["gh_grid"], src_gh)
        put(f"ghRbox{tg}", f"[{grid['R_box'][0]:g},{grid['R_box'][1]:g}]", grid["R_box"], src_gh)
        put(f"ghPbox{tg}", f"[{grid['P_box'][0]:g},{grid['P_box'][1]:g}]", grid["P_box"], src_gh)
        put(f"ghHours{tg}", f"{r['gh_wall_h']:.2g}" if r["gh_wall_h"] < 10 else f"{r['gh_wall_h']:.0f}",
            r["gh_wall_h"], src_gh, "hours")
        put(f"ghM{tg}", str(r["krylov_m"]), r["krylov_m"], src_gh)
        put(f"ghSubsteps{tg}", str(r["substeps"]), r["substeps"], src_gh)
        put(f"ghFrame{tg}", sci(r["frame"]), r["frame"], src_gh, "max_t W-norm frame residual")
        put(f"ghTrace{tg}", sci(r["trace"]), r["trace"], src_gh, "max_t |Tr rho_W - 1|")
        put(f"ghPop{tg}", fixed(r["gh_pop2"], 6), r["gh_pop2"], src_gh, "rho_W^22(T)")
        if r["parity"] != 0.0 or r["herm"] != 0.0:
            raise SystemExit(f"[15] P{P0}: parity/Hermiticity not exactly zero")

        # ---------------------------------------------------- quantum
        q = load(f"10_schrodinger_convergence_P{P0}.json")["results"]
        fq = q["runs"]["finest"]
        t_q = np.array(fq["times"]); p_q = np.array(fq["populations"])
        a_q = np.array(fq["adiabatic_populations"])
        q_pop2 = q["reference_uncertainty"]["converged_value_final_populations"][1]
        if abs(q_pop2 - p_q[-1, 1]) > 1e-14:
            raise SystemExit(f"[15] P{P0}: converged value is not the finest run")
        tmp = q["temporal"]
        q_unc = max(tmp["richardson_remaining_at_dt_over_4"],
                    q["comparisons_against_production"]["grid_x2"]["final_population_difference"],
                    q["comparisons_against_production"]["box_enlarged"]["final_population_difference"])
        r["q_pop2"] = q_pop2; r["q_unc"] = q_unc
        r["q_adpop2"] = a_q[-1, 1]
        src_q = f"10_schrodinger_convergence_P{P0}.json"
        put(f"qPop{tg}", fixed(q_pop2, 6), q_pop2, src_q, "finest run, rho^22(T)")
        put(f"qUnc{tg}", sci(q_unc, 1), q_unc, src_q,
            "max(Richardson remainder at dt/4, grid x2 and box differences)")
        put(f"qNR{tg}", str(fq["n_R"]), fq["n_R"], src_q)
        r["q_nr_ref"] = fq["n_R"]; r["q_dt_ref"] = fq["dt"]
        put(f"qDt{tg}", f"{fq['dt']:g}", fq["dt"], src_q)
        qp = q["runs"]["production"]
        r["q_nr_prod"] = qp["n_R"]; r["q_dt_prod"] = qp["dt"]
        r["q_order"] = tmp["fitted_order"]
        r["q_norm_drift"] = max(v["norm_drift"] for v in q["runs"].values())
        put(f"qNRProd{tg}", str(qp["n_R"]), qp["n_R"], src_q, "production run")

        # E_QCLE = |GH - quantum|
        a, b, tc = common(t_gh, p_gh[:, 1], t_q, p_q[:, 1])
        if len(tc) < 11:
            raise SystemExit(f"[15] P{P0}: GH and quantum share only {len(tc)} instants")
        r["e_qcle"] = abs(r["gh_pop2"] - q_pop2)
        r["e_qcle_max"] = float(np.max(np.abs(a - b)))
        if r["e_qcle"] < 5.0 * q_unc:
            raise SystemExit(f"[15] P{P0}: E_QCLE not resolved above the reference uncertainty")
        put(f"eQcle{tg}", sci(r["e_qcle"]), r["e_qcle"], f"{src_gh}; {src_q}", "|GH - quantum| at T")
        put(f"eQcleMax{tg}", sci(r["e_qcle_max"]), r["e_qcle_max"], f"{src_gh}; {src_q}",
            f"max over {len(tc)} common instants")

        # ------------------------------------------------ projected PBME
        pj = load(f"11_pbme_projected_P{P0}.json")["results"]
        if pj["branch"] != "pbme_projected" or pj["weights_renormalised_by_their_sum"]:
            raise SystemExit(f"[15] P{P0}: not the unbiased projected estimator")
        pb_pop2 = pj["final"]["cv"]["populations"][1]
        pb_sem = pj["final"]["cv"]["sem"][1]
        r["pb_pop2"] = pb_pop2; r["pb_sem"] = pb_sem
        r["pb_adpop2"] = np.asarray(pj["adiabatic"]["cv"]["mean"])[-1, 1]
        r["pb_adsem"] = np.asarray(pj["adiabatic"]["cv"]["sem"])[-1, 1]
        r["foc_pop2"] = pj["focused"]["final_populations"][1]
        r["pb_n"] = pj["n_samples"]; r["pb_seeds"] = pj["n_seeds"]
        r["pb_blocks"] = pj["n_blocks"]; r["pb_block_size"] = pj["block_size"]
        r["pb_sem_seed"] = pj["across_seed"]["sem_cv"][1]
        r["pb_negfrac"] = pj["diagnostics"]["fraction_negative_weight"]["max"]
        conv = [c for c in pj["sample_size_convergence"]
                if c["estimator"] == "cv" and c["n_blocks"] >= 4]
        nn = np.array([c["n_samples"] for c in conv], float)
        se = np.array([c["sem"][1] for c in conv], float)
        r["pb_slope"] = float(np.polyfit(np.log(nn), np.log(se), 1)[0])
        r["pb_series"] = (np.array(pj["times"]), np.array(pj["diabatic"]["cv"]["mean"])[:, 1],
                          np.array(pj["diabatic"]["cv"]["sem"])[:, 1])
        r["gh_series"] = (t_gh, p_gh[:, 1])
        r["q_series"] = (t_q, p_q[:, 1])
        r["foc_n"] = pj["focused"]["n_samples"]
        src_pb = f"11_pbme_projected_P{P0}.json"
        put(f"pbPop{tg}", fixed(pb_pop2, 4), pb_pop2, src_pb, "control-variate estimate")
        put(f"pbSem{tg}", fixed(pb_sem, 4), pb_sem, src_pb, "one standard error over blocks")
        put(f"pbSemDigits{tg}", str(int(round(pb_sem * 1e4))), pb_sem, src_pb,
            "standard error in units of the last quoted digit (1e-4)")
        put(f"focPop{tg}", fixed(r["foc_pop2"], 4), r["foc_pop2"], src_pb, "focused PBME")
        r["e_pbme"] = abs(pb_pop2 - r["gh_pop2"])
        put(f"ePbme{tg}", sci(r["e_pbme"]), r["e_pbme"], f"{src_pb}; {src_gh}", "|proj PBME - GH| at T")
        put(f"ePbmeSem{tg}", sci(pb_sem, 1), pb_sem, src_pb)
        r["ratio"] = r["e_pbme"] / r["e_qcle"]
        put(f"ratio{tg}", two_sig(r["ratio"]), r["ratio"], "derived", "E_PBME / E_QCLE")
        # the forms quoted in the results table: one notation per column
        put(f"pbPopTab{tg}", signed_pad(f"{fixed(pb_pop2, 4)}({int(round(pb_sem * 1e4))})", pb_pop2),
            pb_pop2, src_pb, "table form: value(standard error in units of 1e-4)")
        put(f"focPopTab{tg}", signed_pad(fixed(r["foc_pop2"], 4), r["foc_pop2"]),
            r["foc_pop2"], src_pb, "table form")
        put(f"eQcleTab{tg}", sci_always(r["e_qcle"], 2), r["e_qcle"], f"{src_gh}; {src_q}",
            "table form of |GH - quantum| at T")
        put(f"ePbmeTab{tg}", sci_sem(r["e_pbme"], pb_sem), r["e_pbme"], f"{src_pb}; {src_gh}",
            "table form of |proj PBME - GH| at T with one standard error")

        # ------------------------------------------------ grid QCLE (10b)
        x = load(f"10b_qcle_grid_crosscheck_P{P0}.json")["results"]
        xp = x["runs"]["production"]
        xh = h5(f"10b_qcle_grid_crosscheck_P{P0}.h5")
        same = (xp["grid"]["n_R"], xp["grid"]["n_P"]) == r["gh_grid"] and \
               abs(xp["T"] - r["T"]) < 1e-9
        r["grid_same"] = same
        r["grid_pop2"] = xp["final_populations"][1]
        r["grid_dt"] = xp["dt"]
        r["grid_wall_h"] = xp["wall_time_s"] / 3600.0
        a, b, tc = common(t_gh, p_gh[:, 1], xh["grid_t"], xh["grid_populations"][:, 1])
        r["e_grid"] = abs(r["gh_pop2"] - r["grid_pop2"])
        r["e_grid_max"] = float(np.max(np.abs(a - b)))
        r["e_grid_n"] = len(tc)
        src_x = f"10b_qcle_grid_crosscheck_P{P0}.json"
        dth = x["runs"].get("dt_half")
        r["grid_dthalf"] = (abs(dth["final_populations"][1] - r["grid_pop2"]) if dth else None)
        sref = x.get("independent_spatial_refinement") or {}
        r["grid_spatial"] = sref.get("final_population_linf")
        r["grid_bmass"] = max(xp["max_boundary_mass_R"], xp["max_boundary_mass_P"])
        ct = x["coefficient_temporal_error"]
        r["strang_linf"] = ct["cross_code_linf_production_dt"]
        r["strang_linf_half"] = ct["cross_code_linf_refined_dt"]
        r["strang_resolvable"] = bool(ct["cross_code_order_is_resolvable"])
        put(f"gridGrid{tg}", f"{xp['grid']['n_R']}\\times{xp['grid']['n_P']}",
            (xp["grid"]["n_R"], xp["grid"]["n_P"]), src_x)
        put(f"gridDt{tg}", f"{xp['dt']:g}", xp["dt"], src_x)
        put(f"eGrid{tg}", sci(r["e_grid"]), r["e_grid"], f"{src_gh}; {src_x}",
            "|GH - grid QCLE| at T" + ("" if same else " (DIFFERENT bath grids)"))
        put(f"eGridMax{tg}", sci(r["e_grid_max"]), r["e_grid_max"], f"{src_gh}; {src_x}",
            f"max over {len(tc)} common instants")
        if dth:
            put(f"gridDtHalf{tg}", sci(r["grid_dthalf"], 1), r["grid_dthalf"], src_x,
                "|pop2(dt/2) - pop2(dt)| of the grid QCLE")
        if same:
            put(f"ghOverGrid{tg}", f"{r['gh_wall_h'] / r['grid_wall_h']:.1f}",
                r["gh_wall_h"] / r["grid_wall_h"], f"{src_gh}; {src_x}", "wall-time ratio, same machine and grid")

        # The four-field solver was run at P0 = 10 only on a grid coarser by
        # two in each direction.  Its difference from the coefficient
        # propagation compares solutions on different grids: it is reported
        # (Table SIII) but not used as a comparison of the two solvers.
        r["grid_common"] = same
        r["e_grid_c"] = r["e_grid"] if same else None
        r["e_grid_c_max"] = r["e_grid_max"] if same else None

        # ------------------------------------------------ Strang (07_/13_)
        sname = "13_P20_wide_momentum_box" if P0 == 20 else f"07_coefficient_propagation_P{P0}"
        s = load(sname + ".json")
        sc = s["results"]["coefficient"]
        if sc.get("integrator") != "strang":
            raise SystemExit(f"[15] P{P0}: {sname} is not the Strang run")
        sh = h5(sname + ".h5")
        s_same = (sc["grid"]["n_R"], sc["grid"]["n_P"]) == r["gh_grid"]
        if not s_same:
            raise SystemExit(f"[15] P{P0}: Strang and GH grids differ")
        r["strang_pop2"] = sc["final_populations"][1]
        r["strang_cfl"] = sc["cfl_number"]
        r["strang_dt"] = s["metadata"]["config"]["dt"]
        a, b, tc = common(t_gh, p_gh[:, 1], sh["coef_t"], sh["coef_populations"][:, 1])
        r["e_strang"] = abs(r["gh_pop2"] - r["strang_pop2"])
        r["e_strang_max"] = float(np.max(np.abs(a - b)))
        src_s = sname + ".json"
        put(f"eStrang{tg}", sci(r["e_strang"]), r["e_strang"], f"{src_gh}; {src_s}", "|GH - Strang| at T")
        put(f"eStrangMax{tg}", sci(r["e_strang_max"]), r["e_strang_max"], f"{src_gh}; {src_s}",
            f"max over {len(tc)} common instants")
        put(f"strangCfl{tg}", f"{r['strang_cfl']:.1f}", r["strang_cfl"], src_s)
        put(f"strangDt{tg}", f"{r['strang_dt']:g}", r["strang_dt"], src_s)

        # ------------------------------------------------ electronic coherence
        # rho^12(t) = int dR dP rho_W^12 from every calculation, compared with
        # the quantum run archived with the Strang solution (saved every
        # 10 a.u.) at the instants saved by both.  The wave-function code
        # stores <psi_1|psi_2> = conj(rho^12): its imaginary part changes sign.
        t_qd = np.asarray(sh["se_t"], float)
        c_q = sh["se_coherence_re"] - 1j * sh["se_coherence_im"]
        if abs(t_qd[-1] - r["T"]) > 1e-9 or abs(c_q[0]) > 1e-12:
            raise SystemExit(f"[15] P{P0}: quantum coherence series does not span [0, T] from zero")
        sols = {"coefficient propagation": (t_gh, np.array(
                    [h["coherence_re"] + 1j * h["coherence_im"] for h in hist])),
                "four-field RK4": (xh["grid_t"], xh["grid_coherence_re"] + 1j * xh["grid_coherence_im"]),
                "Strang": (sh["coef_t"], sh["coef_coherence_re"] + 1j * sh["coef_coherence_im"])}
        dq = {}
        for name_, (tt, cc) in sols.items():
            iA, iB = common_idx(tt, t_qd)
            if len(iA) < 2:
                raise SystemExit(f"[15] P{P0}: {name_} shares no instants with the quantum run")
            dq[name_] = (float(np.max(np.abs(np.asarray(cc)[iA] - c_q[iB]))), len(iA))
        r["coh_qcle"] = max(v[0] for v in dq.values())
        # coefficient propagation against the four-field solution, same grid
        iA, iB = common_idx(t_gh, xh["grid_t"])
        c_rk = xh["grid_coherence_re"] + 1j * xh["grid_coherence_im"]
        r["coh_gh_rk4"] = float(np.max(np.abs(sols["coefficient propagation"][1][iA] - c_rk[iB])))
        t_pb = np.asarray(pj["times"], float)
        c_pb = np.asarray(pj["coherence"]["re"]["mean"]) + 1j * np.asarray(pj["coherence"]["im"]["mean"])
        s_pb = np.hypot(pj["coherence"]["re"]["sem"], pj["coherence"]["im"]["sem"])
        iA, iB = common_idx(t_pb, t_qd)
        d_pb = np.abs(c_pb[iA] - c_q[iB])
        k_ = int(np.argmax(d_pb))
        r["coh_pbme"] = float(d_pb[k_])
        r["coh_pbme_over_sem"] = float(d_pb[k_] / s_pb[iA][k_])
        # while the quantum coherence is still an order of magnitude below the
        # standard error of the PBME estimate, that estimate must scatter
        # about zero at the level of its standard error: the root mean square
        # of |deviation| / standard error is 1 for pure noise
        pre = np.abs(c_q[iB]) < 0.1 * s_pb[iA]
        if pre.sum() < 5:
            raise SystemExit(f"[15] P{P0}: fewer than 5 instants before the coherence forms")
        rms_pre = float(np.sqrt(np.mean((d_pb[pre] / s_pb[iA][pre]) ** 2)))
        if not 0.5 <= rms_pre <= 1.5:
            raise SystemExit(f"[15] P{P0}: PBME coherence before the coherence forms is not "
                             f"at the level of its standard error (rms ratio {rms_pre:.2f})")
        r["coh_rms_pre"] = (rms_pre, int(pre.sum()))
        r["coh_sem_pre"] = (float(np.min(s_pb[iA][pre])), float(np.max(s_pb[iA][pre])))
        fz = load(f"20_focused_pbme_observables_P{P0}.json")
        if not fz["all_passed"] or fz["metadata"]["config"]["source"] != src_pb or \
           fz["results"]["n_samples"] != pj["focused"]["n_samples"]:
            raise SystemExit(f"[15] P{P0}: focused PBME observables do not belong to {src_pb}")
        fr = fz["results"]
        r["foc_repro"] = next(float(c["value"]) for c in fz["checks"]
                              if c["name"] == "focused.reproduces_archived_populations")
        c_fo = np.asarray(fr["coherence_re"]) + 1j * np.asarray(fr["coherence_im"])
        iA, iB = common_idx(fr["times"], t_qd)
        r["coh_foc"] = float(np.max(np.abs(c_fo[iA] - c_q[iB])))
        r["coh_peak"] = float(np.max(np.abs(c_q)))
        r["coh_T"] = float(np.abs(c_q[-1]))
        src_c = f"{src_gh}; 10b_*; {src_s}; {src_pb}; 20_focused_pbme_observables_P{P0}.json"
        put(f"cohQcle{tg}", sci(r["coh_qcle"]), r["coh_qcle"], src_c,
            "max over the three QCLE solvers and their instants shared with the quantum run of "
            "|rho12 - rho12_quantum|; " + "; ".join(f"{k}: {v[0]:.3e} ({v[1]} instants)"
                                                    for k, v in dq.items()))
        put(f"cohPbme{tg}", sci(r["coh_pbme"]), r["coh_pbme"], src_c,
            f"max |rho12_PBME - rho12_quantum|, {r['coh_pbme_over_sem']:.1f} standard errors")
        put(f"cohFoc{tg}", sci(r["coh_foc"]), r["coh_foc"], src_c, "max |rho12_focused - rho12_quantum|")
        put(f"cohT{tg}", sci(r["coh_T"]), r["coh_T"], sname + ".h5", "|rho12_quantum(T)|")
        rows[P0] = r

    # ------------------------------------------------------------ ranges
    def rng(key, Ps=MOMENTA):
        v = [rows[P][key] for P in Ps]
        return min(v), max(v)

    lo, hi = rng("frame")
    put("ghFrameMax", sci_up(hi), hi, "07g_*", "max over regimes and time (upper bound)")
    lo, hi = rng("trace")
    put("ghTraceMax", sci_up(hi), hi, "07g_*", "upper bound")
    same_P = [P for P in MOMENTA if rows[P]["grid_same"]]
    diff_P = [P for P in MOMENTA if not rows[P]["grid_same"]]
    common_P = [P for P in MOMENTA if rows[P]["grid_common"]]
    lo, hi = rng("e_grid_c_max", common_P)
    put("eGridMaxAll", sci_up(hi), hi, "07g_*; 10b_*",
        "max over the regimes compared on a common grid and their common instants (upper bound)")
    put("gridSameList", ", ".join(str(P) for P in same_P), same_P, "10b_*")
    put("gridDiffList", ", ".join(str(P) for P in diff_P) or "none", diff_P, "10b_*")
    lo, hi = rng("e_strang")
    put("eStrangMin", sci_down(lo), lo, "07g_*; 07_*"); put("eStrangMax", sci_up(hi), hi, "07g_*; 07_*")
    lo, hi = rng("e_qcle")
    put("eQcleMin", sci_down(lo), lo, "derived"); put("eQcleMaxAll", sci_up(hi), hi, "derived")
    lo, hi = rng("e_pbme")
    put("ePbmeMin", sci_down(lo), lo, "derived"); put("ePbmeMaxAll", sci_up(hi), hi, "derived")
    lo, hi = rng("ratio")
    put("ratioMin", two_sig_floor(lo), lo, "derived", "lower bound"); put("ratioMax", sci_up(hi, 2), hi, "derived", "upper bound")
    lo, hi = rng("e_grid_c", common_P)
    put("eGridFinMax", sci_up(hi), hi, "07g_*; 10b_*",
        "max over the regimes compared on a common grid, at T (upper bound)")
    # electronic coherence rho^12(t), all regimes
    lo, hi = rng("coh_peak")
    put("cohPeakMin", dec_bound(lo, 2, False), lo, "07_*; 13_P20_*", "min over regimes of max_t |rho12_quantum|")
    put("cohPeakMax", dec_bound(hi, 2, True), hi, "07_*; 13_P20_*", "max over regimes of max_t |rho12_quantum|")
    cr = [rows[P]["coh_pbme"] / rows[P]["coh_qcle"] for P in MOMENTA]
    if min(cr) <= 1.0:
        raise SystemExit("[15] the PBME coherence is not further from the quantum one in every regime")
    put("cohRatioMin", two_sig_floor(min(cr)), min(cr), "derived",
        "min over regimes of max_t|rho12_PBME - Q| / max_t|rho12_QCLE - Q| (lower bound)")
    put("cohRatioMax", sci_up(max(cr), 2), max(cr), "derived", "upper bound")
    lo, hi = rng("coh_pbme_over_sem")
    put("cohPbmeOverSemMin", two_sig_floor(lo), lo, "11_*",
        "min over regimes of the largest PBME coherence deviation in units of its standard error")
    lo, hi = rng("coh_qcle")
    put("cohQcleMaxAll", sci_up(hi), hi, "derived", "max over regimes (upper bound)")
    lo, hi = rng("coh_gh_rk4", same_P)
    put("cohGridMax", sci_up(hi), hi, "07g_*; 10b_*",
        "max over the same-grid regimes and their common instants of |rho12_GH - rho12_RK4|")
    lo, hi = rng("foc_repro")
    put("focReproMax", sci_up(hi, 1), hi, "20_*",
        "focused ensemble propagated again: largest change of its diabatic populations")
    lo, hi = rng("coh_foc")
    put("cohFocMin", sci_down(lo), lo, "20_*; 07_*; 13_P20_*", "min over regimes of max_t |focused - Q|")
    put("cohFocMax", sci_up(hi), hi, "20_*; 07_*; 13_P20_*", "max over regimes of max_t |focused - Q|")
    if min(rows[P]["coh_foc"] / rows[P]["coh_qcle"] for P in MOMENTA) <= 1.0:
        raise SystemExit("[15] the focused PBME coherence is not further from the quantum one in every regime")
    lo = min(rows[P]["coh_sem_pre"][0] for P in MOMENTA)
    hi = max(rows[P]["coh_sem_pre"][1] for P in MOMENTA)
    put("cohSemMin", sci_down(lo), lo, "11_*",
        "PBME coherence standard error before the coherence forms; rms |dev|/sem there: "
        + ", ".join(f"P{P} {rows[P]['coh_rms_pre'][0]:.2f} ({rows[P]['coh_rms_pre'][1]})"
                    for P in MOMENTA))
    put("cohSemMax", sci_up(hi), hi, "11_*", "PBME coherence standard error before the coherence forms")
    wr = [rows[P]["gh_wall_h"] / rows[P]["grid_wall_h"] for P in same_P]
    put("ghOverGridMin", dec_bound(min(wr), 1, False), min(wr), "07g_*; 10b_*")
    put("ghOverGridMax", dec_bound(max(wr), 1, True), max(wr), "07g_*; 10b_*")
    ratios = []
    for P in same_P:
        ct = load(f"10b_qcle_grid_crosscheck_P{P}.json")["results"]["coefficient_temporal_error"]
        if not ct.get("cross_code_order_is_resolvable"):
            raise SystemExit(f"[15] P{P}: Strang halving study not resolvable")
        ratios.append(ct["cross_code_linf_production_dt"] / ct["cross_code_linf_refined_dt"])
    put("strangHalveMin", dec_bound(min(ratios), 2, False), min(ratios), "10b_*: Strang re-run at dt/2")
    put("strangHalveMax", dec_bound(max(ratios), 2, True), max(ratios), "10b_*: Strang re-run at dt/2")
    lo, hi = rng("q_unc")
    put("qUncMax", sci_up(hi), hi, "10_*", "upper bound")
    put("pbNSamples", f"{rows[40]['pb_n']:,}".replace(",", "\\,"), rows[40]["pb_n"], "11_*")
    put("pbNSeeds", str(rows[40]["pb_seeds"]), rows[40]["pb_seeds"], "11_*")
    ns = sorted({rows[P]["pb_n"] for P in MOMENTA})
    put("pbNSamplesMin", f"{ns[0]:,}".replace(",", "\\,"), ns[0], "11_*")
    put("focNSamples", f"{rows[40]['foc_n']:,}".replace(",", "\\,"), rows[40]["foc_n"], "11_*")
    put("nModesGH", str(rows[40]["n_modes"]), rows[40]["n_modes"], "07g_*")
    put("krylovTol", sci(rows[40]["krylov_tol"], 1), rows[40]["krylov_tol"], "07g_*")
    # the Strang splitting error at P0 = 100 against the QCLE error there
    put("strangOverQclePOneHundred", f"{rows[100]['e_strang'] / rows[100]['e_qcle']:.1f}",
        rows[100]["e_strang"] / rows[100]["e_qcle"], "derived")

    # focused against projected-density PBME, and against the quantum result
    fd = [abs(rows[P]["foc_pop2"] - rows[P]["pb_pop2"]) for P in MOMENTA]
    closer = sum(abs(rows[P]["foc_pop2"] - rows[P]["q_pop2"]) < abs(rows[P]["pb_pop2"] - rows[P]["q_pop2"])
                 for P in MOMENTA)
    put("focDiffMin", dec_bound(min(fd), 4, False), min(fd), "11_*", "min |focused - projected| in rho22(T)")
    put("focDiffMax", dec_bound(max(fd), 4, True), max(fd), "11_*", "max |focused - projected| in rho22(T)")
    put("focCloserCount", ("none", "one", "two", "three", "four", "five", "six")[closer], closer,
        "11_*; 10_*", "regimes where focused is closer to quantum, as a word for the running text")
    if min(fd) < 3 * max(rows[P]["pb_sem"] for P in MOMENTA):
        raise SystemExit("[15] focused and projected estimates are not resolved from each other")

    # negative PBME population at P0 = 10, diabatic and adiabatic
    r10 = rows[10]
    put("pbPopNegSigmaPTen", f"{abs(r10['pb_pop2']) / r10['pb_sem']:.0f}",
        abs(r10["pb_pop2"]) / r10["pb_sem"], "11_pbme_projected_P10.json")
    put("pbAdPopPTen", fixed(r10["pb_adpop2"], 4), r10["pb_adpop2"], "11_pbme_projected_P10.json")
    put("pbAdSemPTen", fixed(r10["pb_adsem"], 4), r10["pb_adsem"], "11_pbme_projected_P10.json")
    put("ghAdPopPTen", fixed(r10["gh_adpop2"], 4), r10["gh_adpop2"], "07g_*P10.json")
    put("qAdPopPTen", fixed(r10["q_adpop2"], 4), r10["q_adpop2"], "10_*P10.json")

    # ------------------------------------------------ closure validation
    k = load("07k_gh_closure_validation.json")
    kr = k["results"]
    if not k["all_passed"]:
        raise SystemExit("[15] closure validation did not pass")
    put("clModes", str(kr["closure"]["basis"]["n_modes"]), kr["closure"]["basis"]["n_modes"], "07k")
    put("clModesFour", str(kr["closure"]["basis"]["n_modes_order4"]), kr["closure"]["basis"]["n_modes_order4"], "07k")
    put("clEFourMax", sci(kr["closure"]["E4_max"]), kr["closure"]["E4_max"], "07k")
    put("clCtlEFour", f"{kr['closure_control']['E4_initial']:.2f}", kr["closure_control"]["E4_initial"], "07k")
    put("clCtlETwo", f"{kr['closure_control']['E2_final']:.2f}", kr["closure_control"]["E2_final"], "07k")
    put("clCtlT", f"{kr['closure_control']['T']:g}", kr["closure_control"]["T"], "07k")
    put("parModes", str(kr["parity"]["basis"]["n_modes"]), kr["parity"]["basis"]["n_modes"], "07k")
    put("parModesOdd", str(kr["parity"]["basis"]["n_modes_odd"]), kr["parity"]["basis"]["n_modes_odd"], "07k")
    put("parEOddMax", sci(kr["parity"]["E_odd_max"]), kr["parity"]["E_odd_max"], "07k")
    put("parCtlEOdd", f"{kr['parity_control']['E_odd_initial']:.2f}", kr["parity_control"]["E_odd_initial"], "07k")
    put("parCtlEOddFinal", f"{kr['parity_control']['E_odd_final']:.2f}", kr["parity_control"]["E_odd_final"], "07k")
    put("clFrame", sci_up(kr["physical_frame"]["max_residual_W_metric"]), kr["physical_frame"]["max_residual_W_metric"], "07k", "upper bound")
    kg = k["metadata"]["grid"]
    put("clGrid", f"{kg['n_R']}\\times{kg['n_P']}", (kg["n_R"], kg["n_P"]), "07k")
    put("clT", f"{k['metadata']['T']:g}", k["metadata"]["T"], "07k")
    put("clPZero", f"{k['metadata']['P0']:g}", k["metadata"]["P0"], "07k")

    # ------------------------------------------------ quartic defect rate
    lk = load("12_leakage_reanalysis.json")["results"]
    peak = {int(rw["P0"]): rw["peak_leakage"] for rw in lk["rows"]}
    mean = {int(rw["P0"]): rw["mean_leakage_full"] for rw in lk["rows"]}
    for P0 in MOMENTA:
        put(f"dhPeak{TAG[P0]}", sci(peak[P0]), peak[P0], "12_leakage_reanalysis.json")
        put(f"dhMean{TAG[P0]}", sci(mean[P0]), mean[P0], "12_leakage_reanalysis.json")
    # the range of the peaks for P0 >= 20, rounded outward
    pk_hi = [peak[P] for P in MOMENTA if P >= 20]
    put("dhPeakRangeLo", sci_down(min(pk_hi), 2), min(pk_hi), "12_leakage_reanalysis.json",
        "smallest peak defect rate for P0 >= 20 (lower end of the range)")
    put("dhPeakRangeHi", sci_up(max(pk_hi), 2), max(pk_hi), "12_leakage_reanalysis.json",
        "largest peak defect rate for P0 >= 20 (upper end of the range)")

    def corr(xv, yv):
        xv = np.asarray(xv, float); yv = np.asarray(yv, float)
        pr = float(np.corrcoef(xv, yv)[0, 1])
        rk = lambda v: np.argsort(np.argsort(v)).astype(float)
        sr = float(np.corrcoef(rk(xv), rk(yv))[0, 1])
        return pr, sr

    pr, sr = corr([peak[P] for P in MOMENTA], [rows[P]["e_pbme"] for P in MOMENTA])
    put("corrMomPearson", f"{pr:.2f}", pr, "derived: peak D_H vs E_PBME(GH)")
    put("corrMomSpearman", f"{sr:.2f}", sr, "derived")
    cs = lk["coupling_scan"]
    Cs = [rw["C"] for rw in cs["rows"]]
    cpk = [rw["peak_leakage"] for rw in cs["rows"]]
    cdc = [rw["discard_cost"] for rw in cs["rows"]]
    csem = [rw["discard_cost_sem"] for rw in cs["rows"]]
    pr, sr = corr(cpk, cdc)
    put("corrScanPearson", f"{pr:.2f}", pr, "derived: coupling scan, peak D_H vs E_discard(Strang)")
    put("corrScanSpearman", f"{sr:.2f}", sr, "derived")
    put("scanCList", ", ".join(f"{c:g}" for c in Cs), Cs, "12_leakage_reanalysis.json")
    put("scanPeakRange", f"{max(cpk) / min(cpk):.2f}", max(cpk) / min(cpk), "derived")
    resolved = [d / s for d, s in zip(cdc, csem)]
    put("scanRespMax", sci(max(cdc)), max(cdc), "12_leakage_reanalysis.json")
    put("scanRespMaxC", f"{Cs[int(np.argmax(cdc))]:g}", Cs[int(np.argmax(cdc))], "derived")
    put("scanResolved", str(sum(v >= 3.0 for v in resolved)), sum(v >= 3.0 for v in resolved), "derived: >= 3 SEM")
    put("scanPoints", str(len(Cs)), len(Cs), "derived")
    unres = [(c, d + 2 * s) for c, d, s in zip(Cs, cdc, csem) if d < 3 * s]
    if len(unres) == 1:
        put("scanBoundC", f"{unres[0][0]:g}", unres[0][0], "derived")
        put("scanBound", sci_up(unres[0][1]), unres[0][1], "derived: response + 2 SEM (upper bound)")
    mom = [P for P in MOMENTA if P >= 20]
    pk = [peak[P] for P in mom]; ep = [rows[P]["e_pbme"] for P in mom]
    put("dhRangeMom", f"{max(pk) / min(pk):.2f}", max(pk) / min(pk), "derived, P0 >= 20")
    put("ePbmeRangeMom", f"{max(ep) / min(ep):.0f}", max(ep) / min(ep), "derived, P0 >= 20")
    put("scanGradSpan", f"{cs['grad_h_dynamic_range']:.2f}", cs["grad_h_dynamic_range"], "12_leakage_reanalysis.json")
    put("nQuarticModes", str(lk["n_degree_four_coefficients"]), lk["n_degree_four_coefficients"], "12_leakage_reanalysis.json")
    put("strangScanErr", sci(rows[40]["e_strang"]), rows[40]["e_strang"], "Strang error at P0=40, context for the scan")

    supplementary(rows)
    phase_space(rows)

    # --------------------------------------------------------------- write
    lines = ["% Generated by scripts/15_paper_values.py -- do not edit by hand.",
             "% Every macro is one number of the main text; its provenance is",
             "% results/tables/paper_values.json under the same name."]
    for name in sorted(MACROS):
        lines.append(f"\\newcommand{{\\{name}}}{{{MACROS[name]}}}")
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"rows": {str(k): {kk: (vv if not isinstance(vv, np.generic) else vv.item())
                                                        for kk, vv in v.items() if not kk.endswith("_series")}
                                             for k, v in rows.items()},
                                    "macros": LEDGER}, indent=1, default=float), encoding="utf-8")
    print(f"[15] wrote {len(MACROS)} macros to {OUT_TEX.relative_to(ROOT)}")
    for P0 in MOMENTA:
        r = rows[P0]
        print(f"  P0={P0:>3}  Q={r['q_pop2']:.6f}  GH={r['gh_pop2']:.6f}  PBME={r['pb_pop2']:.4f}+-{r['pb_sem']:.4f}"
              f"  E_QCLE={r['e_qcle']:.2e}  E_PBME={r['e_pbme']:.2e}  ratio={r['ratio']:.0f}"
              f"  |GH-grid|={r['e_grid']:.1e}{'' if r['grid_same'] else '(diff grid)'}  |GH-Strang|={r['e_strang']:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
