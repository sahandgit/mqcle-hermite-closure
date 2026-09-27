#!/usr/bin/env python3
r"""Main-text figures, and the supplementary figures that show QCLE physics.

Every series is read from results/raw; derived numbers are read from
results/tables/paper_values.json, which scripts/15_paper_values.py writes, so
a figure and the sentence quoting it cannot disagree.

Roles (see scripts/15_paper_values.py): the QCLE curve is the Gaussian--Hermite
(GH) propagation; the full quantum curve is the converged split-operator
reference; PBME is the unbiased projected-density estimator.  The grid-QCLE and
Strang solutions of the subsystem QCLE appear only as equivalence checks.

Differences between two time series are formed only at instants that both
archives saved.  Nothing is interpolated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
FIG = ROOT / "results" / "figures"
LEDGER = ROOT / "results" / "tables" / "paper_values.json"
sys.path.insert(0, str(ROOT / "src"))

MOMENTA = (10, 20, 30, 40, 70, 100)
ONE, TWO = 3.37, 6.9                      # AIP column widths (inches)

# One colour per method in every figure.  The five method hues pass the
# all-pairs colour-vision and normal-vision separation checks of the data-viz
# palette validator on a white surface; the RK4 green is below 3:1 contrast and
# is always drawn with a square marker.  The quantum reference is black.
C_Q, C_GH, C_PB, C_FOC = "#000000", "#2a78d6", "#D55E00", "#882255"
C_GRID, C_STR, C_UNC = "#1baf7a", "#4a3aa7", "#999999"
REG = {P: plt.cm.viridis(x) for P, x in zip(MOMENTA, (0.0, 0.2, 0.4, 0.58, 0.74, 0.88))}
MARK = {10: "o", 20: "s", 30: "^", 40: "D", 70: "v", 100: "P"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "legend.handlelength": 1.8,
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
    "mathtext.fontset": "stix", "font.family": "STIXGeneral",
    "axes.formatter.use_mathtext": True,
})


# ------------------------------------------------------------------ helpers
def jload(name: str) -> dict:
    p = RAW / name
    if not p.is_file():
        raise SystemExit(f"[16] required archive missing: {name}")
    return json.loads(p.read_text(encoding="utf-8"))


def h5load(name: str, keys) -> dict:
    import h5py
    with h5py.File(RAW / name, "r") as f:
        return {k: np.asarray(f[k][()]) for k in keys}


def _common_idx(tA, tB, rtol=1e-9):
    """Indices of the instants saved by both series (no interpolation)."""
    tA = np.asarray(tA, float); tB = np.asarray(tB, float)
    iA, iB = [], []
    for i, t in enumerate(tA):
        j = int(np.argmin(np.abs(tB - t)))
        if abs(tB[j] - t) <= rtol * max(1.0, abs(t)):
            iA.append(i); iB.append(j)
    return np.asarray(iA, int), np.asarray(iB, int)


def common(tA, yA, tB, yB, rtol=1e-9):
    tA = np.asarray(tA, float); tB = np.asarray(tB, float)
    iA, iB = [], []
    for i, t in enumerate(tA):
        j = int(np.argmin(np.abs(tB - t)))
        if abs(tB[j] - t) <= rtol * max(1.0, abs(t)):
            iA.append(i); iB.append(j)
    return tA[iA], np.asarray(yA, float)[iA], np.asarray(yB, float)[iB]


def save(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=220)
    plt.close(fig)
    print(f"[16] wrote results/figures/{stem}.pdf")


def panel_label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, fontweight="bold")


def ledger() -> dict:
    if not LEDGER.is_file():
        raise SystemExit("[16] run scripts/15_paper_values.py first")
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def gh_history(P0):
    g = jload(f"07g_gh_coefficient_propagation_P{P0}.json")
    hist = g["results"]["history"]
    out = {k: np.array([h[k] for h in hist]) for k in
           ("t", "populations", "adiabatic_populations", "coherence_re",
            "coherence_im", "physical_frame_residual", "trace")}
    out["T"] = g["metadata"]["time"]["T"]
    return out


def quantum_finest(P0):
    q = jload(f"10_schrodinger_convergence_P{P0}.json")["results"]["runs"]["finest"]
    return {"t": np.array(q["times"]), "pop": np.array(q["populations"]),
            "ad": np.array(q["adiabatic_populations"])}


def quantum_dense(P0):
    name = ("13_P20_wide_momentum_box.h5" if P0 == 20
            else f"07_coefficient_propagation_P{P0}.h5")
    return h5load(name, ["se_t", "se_populations", "se_adiabatic_populations",
                         "se_coherence_re", "se_coherence_im"])


def strang_series(P0):
    """Strang solution of the subsystem QCLE at the production step."""
    name = ("13_P20_wide_momentum_box.h5" if P0 == 20
            else f"07_coefficient_propagation_P{P0}.h5")
    return h5load(name, ["coef_t", "coef_populations", "coef_adiabatic_populations",
                         "coef_coherence_re", "coef_coherence_im"])


def rk4_series(P0):
    """Four-field Runge--Kutta solution of the subsystem QCLE."""
    return h5load(f"10b_qcle_grid_crosscheck_P{P0}.h5",
                  ["grid_t", "grid_populations", "grid_adiabatic_populations",
                   "grid_coherence_re", "grid_coherence_im"])


def focused_series(P0):
    """Focused PBME: the ensemble of scripts/11, read out in full by scripts/20."""
    d = jload(f"20_focused_pbme_observables_P{P0}.json")
    if not d.get("all_passed"):
        raise SystemExit(f"[16] 20_focused_pbme_observables_P{P0}.json did not pass")
    r = d["results"]
    return {"t": np.array(r["times"]), "pop": np.array(r["populations"]),
            "ad": np.array(r["adiabatic_populations"]),
            "cre": np.array(r["coherence_re"]), "cim": np.array(r["coherence_im"])}


def pbme(P0):
    r = jload(f"11_pbme_projected_P{P0}.json")["results"]
    f = r["focused"]
    return {"t": np.array(r["times"]),
            "pop": np.array(r["diabatic"]["cv"]["mean"]),
            "sem": np.array(r["diabatic"]["cv"]["sem"]),
            "ad": np.array(r["adiabatic"]["cv"]["mean"]),
            "adsem": np.array(r["adiabatic"]["cv"]["sem"]),
            "cre": np.array(r["coherence"]["re"]["mean"]),
            "cim": np.array(r["coherence"]["im"]["mean"]),
            "cresem": np.array(r["coherence"]["re"]["sem"]),
            "cimsem": np.array(r["coherence"]["im"]["sem"]),
            "ft": np.array(f["times"]), "fpop": np.array(f["populations"])}


# ============================================================ Fig. 1 (b,c)
def fig1_selection_rules():
    """Mode-coupling pattern of the force channel, exact and truncated.

    Built from the production operator module in the K = 4 even basis for
    N = 2.  The truncated (Poisson bracket) channel is assembled independently
    from the multiplication operator z_i z_j - delta_ij on the Hermite basis,
    and the script asserts that it differs from the exact channel by the pure
    raising ladder He_n -> He_{n+e_i+e_j}, which is the statement of Sec. II.
    """
    from mqcle.hermite.operator import (active_modes, hermite_T_matrices,
                                        mode_index_map, rotation_basis_matrices)
    modes = active_modes(2, 4, "even")
    idx = mode_index_map(modes)
    M, d = len(modes), 4
    T = hermite_T_matrices(modes)
    Nij = rotation_basis_matrices(modes)
    # z_i z_j - delta_ij on He_n, from z_a He_n = He_{n+e_a} + n_a He_{n-e_a}
    Z = np.zeros((d, d, M, M)); U = np.zeros((d, d, M, M))
    for a, n in enumerate(modes):
        for i in range(d):
            for j in range(d):
                terms = {}
                def add(m, c):
                    if min(m) >= 0:
                        terms[m] = terms.get(m, 0.0) + c
                # z_j He_n
                first = {}
                m = list(n); m[j] += 1; first[tuple(m)] = 1.0
                if n[j] > 0:
                    m = list(n); m[j] -= 1; first[tuple(m)] = first.get(tuple(m), 0.0) + n[j]
                for mm, c in first.items():
                    m1 = list(mm); m1[i] += 1; add(tuple(m1), c)
                    if mm[i] > 0:
                        m2 = list(mm); m2[i] -= 1; add(tuple(m2), c * mm[i])
                if i == j:
                    add(tuple(n), -1.0)
                for m, c in terms.items():
                    b = idx.get(m)
                    if b is not None and c != 0.0:
                        Z[i, j, b, a] += c
                m = list(n); m[i] += 1; m[j] += 1
                b = idx.get(tuple(m))
                if b is not None:
                    U[i, j, b, a] = 1.0
    diff = Z - T - U
    if np.max(np.abs(diff)) != 0.0:
        raise SystemExit("[16] (z_i z_j - delta_ij) - T^{ij} is not the pure raising ladder")
    print("[16] checked: (z_i z_j - delta_ij) = T^{ij} + raise-by-two, exactly, in the K=4 basis")
    # real symmetric h with every element R dependent: dH = diag(dh, dh)
    pairs = [(i, j) for i in range(d) for j in range(d) if (i < 2) == (j < 2)]
    Kpairs = [(i, j) for i in range(d) for j in range(d) if (i < 2) != (j < 2)]
    exact = np.zeros((M, M)); trunc = np.zeros((M, M)); rot = np.zeros((M, M))
    for i, j in pairs:
        exact += np.abs(T[i, j]); trunc += np.abs(Z[i, j])
    for i, j in Kpairs:
        rot += np.abs(Nij[i, j])
    order = np.array([sum(m) for m in modes])
    edges = [0] + [int(np.searchsorted(order, k)) for k in (2, 4)] + [M]

    fig, axes = plt.subplots(1, 2, figsize=(2.95, 2.0), constrained_layout=True)
    for ax, mat, title in ((axes[0], exact + rot, r"full, $i\mathcal{L}_m$"),
                           (axes[1], trunc + rot, r"Poisson bracket, $i\mathcal{L}_m^{\rm PB}$")):
        img = np.full((M, M, 3), 1.0)
        for b in range(M):
            for a in range(M):
                if mat[b, a] > 0:
                    dk = order[b] - order[a]
                    img[b, a] = matplotlib.colors.to_rgb(
                        C_PB if dk > 0 else (C_GH if dk == 0 else C_GRID))
        ax.imshow(img, interpolation="nearest", origin="upper")
        for e in edges[1:-1]:
            ax.axhline(e - 0.5, color="0.35", lw=0.5)
            ax.axvline(e - 0.5, color="0.35", lw=0.5)
        cen = [(edges[k] + edges[k + 1] - 1) / 2 for k in range(3)]
        ax.set_xticks(cen, ["0", "2", "4"]); ax.set_yticks(cen, ["0", "2", "4"])
        ax.tick_params(length=0, top=False, right=False)
        ax.set_xlabel("source order", fontsize=7)
        ax.set_title(title, fontsize=7)
    axes[0].set_ylabel("target order", fontsize=7)
    fig.text(0.0, 1.0, "(b)", fontsize=8, fontweight="bold", ha="left", va="top")
    handles = [Line2D([], [], ls="none", marker="s", color=c, markersize=4, label=l)
               for c, l in ((C_GH, "order kept"), (C_GRID, "lowered by two"),
                            (C_PB, "raised by two"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12),
               handletextpad=0.2, columnspacing=0.7, fontsize=6.5)
    save(fig, "paper_fig1_selection")


# ================================================================== Fig. 2
def fig2_numerics():
    """(a) physical-frame residual of the GH propagation; (b) error hierarchy."""
    L = ledger()["rows"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(TWO, 2.35), constrained_layout=True,
                               gridspec_kw={"width_ratios": [1.0, 1.1]})
    tol = None
    for P0 in MOMENTA:
        g = gh_history(P0)
        r = np.maximum(g["physical_frame_residual"], 1e-17)
        a.semilogy(g["t"] / g["T"], r, color=REG[P0], marker=MARK[P0], markersize=2.6,
                   markevery=5, lw=1.0, label=f"$P_0={P0}$")
        tol = jload(f"07g_gh_coefficient_propagation_P{P0}.json")["metadata"]["time"]["krylov_tol"]
    a.axhline(tol, color=C_UNC, lw=0.7, ls="--")
    a.text(0.02, tol * 1.5, "requested error per substep", color="0.4", fontsize=6)
    a.set_xlabel("$t/T$"); a.set_ylabel(r"$\epsilon_{\rm phys}(t)$")
    a.set_ylim(1e-17, 1e-8); a.set_xlim(0, 1)
    a.legend(ncol=3, loc="lower right", handlelength=1.4, columnspacing=0.8)
    panel_label(a, "(a)")

    P = np.array(MOMENTA, float)
    def col(key):
        return np.array([L[str(p)][key] for p in MOMENTA], float)
    b.errorbar(P, col("e_pbme"), yerr=2 * col("pb_sem"), color=C_PB, marker="s",
               ms=4, mfc="white", mew=1.0, lw=1.0, capsize=2, elinewidth=0.8,
               label=r"$E_{\rm PBME}$: PBME vs QCLE")
    b.plot(P, col("e_qcle"), color=C_GH, marker="o", ms=4, mfc="white", mew=1.0,
           lw=1.0, label=r"$E_{\rm QCLE}$: QCLE vs quantum")
    b.plot(P, col("e_strang"), color=C_STR, marker="^", ms=4, mfc="white", mew=1.0,
           lw=0.9, ls="--", label=r"vs Strang, $\Delta t=0.5$")
    # four-field RK4 only where it ran on the grid of the coefficient
    # propagation; at P0 = 10 it ran on a grid coarser by two, and a
    # difference between solutions on different grids is not a comparison
    # of the solvers (it is listed in Table SIII)
    common_ = np.array([bool(L[str(p)].get("grid_common")) for p in MOMENTA], bool)
    e_c = np.array([L[str(p)]["e_grid_c"] if L[str(p)].get("grid_common") else np.nan
                    for p in MOMENTA], float)
    b.plot(P[common_], e_c[common_], color=C_GRID, marker="D", ms=3.6,
           mfc="white", mew=1.0, lw=0.9, ls="--", label="vs four-field RK4")
    b.plot(P, col("q_unc"), color=C_UNC, marker="x", ms=3.6, lw=0.8, ls=":",
           label="quantum reference: uncertainty")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xticks(P, [str(int(p)) for p in P]); b.minorticks_off()
    b.set_xlabel(r"$P_0$ (a.u.)"); b.set_ylabel(r"$|\Delta\rho^{22}(T)|$")
    b.set_ylim(1e-12, 1.0); b.set_xlim(8.5, 118)
    hs, ls_ = b.get_legend_handles_labels()
    order = sorted(range(len(ls_)), key=lambda i: (not ls_[i].startswith("$E_{\\rm PBME}"),
                                                   not ls_[i].startswith("$E_{\\rm QCLE}"), i))
    b.legend([hs[i] for i in order], [ls_[i] for i in order], loc="upper left",
             bbox_to_anchor=(1.01, 1.0), fontsize=7, handlelength=2.0, borderaxespad=0.0)
    panel_label(b, "(b)")
    save(fig, "paper_fig2_numerics")


# ================================================================== Fig. 3
PANEL = "abcdef"


def _sparse(t, T, fracs):
    """Indices of the saved instants nearest to the fractions of T given, so
    that the three QCLE solutions carry markers at interleaved times."""
    t = np.asarray(t, float) / T
    return np.array(sorted({int(np.argmin(np.abs(t - f))) for f in fracs}))


def _method_handles(focused_label=True):
    h = [Line2D([], [], color=C_Q, lw=1.1, label="full quantum"),
         Line2D([], [], ls="none", marker="o", ms=4.4, mfc="white", mec=C_GH, mew=1.1,
                label="QCLE, coefficient propagation"),
         Line2D([], [], ls="none", marker="s", ms=4.0, mfc="white", mec=C_GRID, mew=1.1,
                label="QCLE, four-field RK4"),
         Line2D([], [], ls="none", marker="^", ms=4.6, mfc="white", mec=C_STR, mew=1.1,
                label="QCLE, Strang"),
         Line2D([], [], color=C_PB, lw=2.4, label=r"PBME, projected density ($\pm2\sigma$)")]
    if focused_label:
        h.append(Line2D([], [], color=C_FOC, lw=1.5, ls="-.", label="PBME, focused"))
    return h


def _draw(ax, T, curves):
    """curves: list of (kind, t, y[, sem]) with kind in quantum, gh, rk4,
    strang, pbme, focused; markers of the QCLE solutions are interleaved."""
    for kind, t, y, *rest in curves:
        t = np.asarray(t, float); y = np.asarray(y, float)
        # the reference is drawn last among the lines, thin, so that the
        # broader PBME line shows around it where the two agree and beside it
        # where they do not; the markers of the QCLE solutions sit on top
        if kind == "quantum":
            ax.plot(t / T, y, color=C_Q, lw=1.1, zorder=6)
        elif kind == "pbme":
            sem = rest[0]
            ax.fill_between(t / T, y - 2 * sem, y + 2 * sem, color=C_PB, alpha=0.25,
                            lw=0, zorder=1)
            ax.plot(t / T, y, color=C_PB, lw=2.4, zorder=3)
        elif kind == "focused":
            ax.plot(t / T, y, color=C_FOC, lw=1.5, ls="-.", zorder=4)
        else:
            col, mk, ms, off = {"gh": (C_GH, "o", 4.4, 0.0),
                                "rk4": (C_GRID, "s", 4.0, 1.0 / 3.0),
                                "strang": (C_STR, "^", 4.6, 2.0 / 3.0)}[kind]
            i = _sparse(t, T, [(k + off) / 10.0 for k in range(11 if off == 0 else 10)])
            ax.plot(t[i] / T, y[i], ls="none", marker=mk, ms=ms, mfc="white", mec=col,
                    mew=1.1, zorder=7)


#: deviation rows: line style of each calculation
DEV_STYLE = {"rk4": dict(color=C_GRID, ls="--", lw=1.4, zorder=4),
             "strang": dict(color=C_STR, ls=":", lw=1.7, zorder=4),
             "pbme": dict(color=C_PB, ls="-", lw=2.0, zorder=3),
             "focused": dict(color=C_FOC, ls="-.", lw=1.5, zorder=3)}


def _check_trace(P0, arrays):
    for name, arr in arrays.items():
        drift = float(np.max(np.abs(np.asarray(arr).sum(axis=1) - 1.0)))
        if drift > 1e-6:
            raise SystemExit(f"[16] P{P0} {name}: trace drifts by {drift:.1e}; "
                             "rho^11 = 1 - rho^22 would not hold")


def fig3_populations():
    """Populations of both diabatic and both adiabatic states in six regimes,
    from every calculation the paper names, and the deviation of rho^22 from
    the quantum result.  Every calculation conserves the trace (checked), so
    rho^11 = 1 - rho^22 and the deviation of rho^11 equals that of rho^22."""
    fig = plt.figure(figsize=(TWO, 7.9))
    outer = fig.add_gridspec(2, 1, hspace=0.15, left=0.085, right=0.995, top=0.975,
                             bottom=0.082)
    first = {}
    for blk in range(2):
        gs = outer[blk].subgridspec(3, 3, height_ratios=[1.3, 1.3, 1.15],
                                    hspace=0.07, wspace=0.07)
        for c in range(3):
            k = 3 * blk + c
            P0 = MOMENTA[k]
            g = gh_history(P0); qf = quantum_finest(P0); qd = quantum_dense(P0)
            pb = pbme(P0); fo = focused_series(P0); st = strang_series(P0); rk = rk4_series(P0)
            T = float(g["T"])
            _check_trace(P0, {"QCLE": g["populations"], "quantum": qd["se_populations"],
                              "PBME": pb["pop"], "focused PBME": fo["pop"],
                              "Strang": st["coef_populations"], "RK4": rk["grid_populations"],
                              "QCLE adiabatic": g["adiabatic_populations"],
                              "PBME adiabatic": pb["ad"], "focused adiabatic": fo["ad"]})
            axd = fig.add_subplot(gs[0, c], sharey=first.get("d"))
            axa = fig.add_subplot(gs[1, c], sharex=axd, sharey=first.get("a"))
            axe = fig.add_subplot(gs[2, c], sharex=axd, sharey=first.get("e"))
            first.setdefault("d", axd); first.setdefault("a", axa); first.setdefault("e", axe)
            for ax, kq, kg, kp, ks, kf, kst, kr in (
                    (axd, "se_populations", "populations", "pop", "sem", "pop",
                     "coef_populations", "grid_populations"),
                    (axa, "se_adiabatic_populations", "adiabatic_populations", "ad", "adsem",
                     "ad", "coef_adiabatic_populations", "grid_adiabatic_populations")):
                for s_ in (0, 1):
                    _draw(ax, T, [("quantum", qd["se_t"], qd[kq][:, s_]),
                                  ("pbme", pb["t"], pb[kp][:, s_], pb[ks][:, s_]),
                                  ("focused", fo["t"], fo[kf][:, s_]),
                                  ("gh", g["t"], g[kg][:, s_]),
                                  ("rk4", rk["grid_t"], rk[kr][:, s_]),
                                  ("strang", st["coef_t"], st[kst][:, s_])])
                ax.axhline(0, color="0.6", lw=0.4, ls=":")
                ax.set_xlim(0, 1)
                ax.tick_params(labelbottom=False)
            # deviation of rho^22 from the quantum result, at instants saved by
            # both calculations: the coefficient propagation against the
            # converged reference (its instants), the others against the
            # production run (every 10 a.u.)
            tt, a1, b1 = common(g["t"], g["populations"][:, 1], qf["t"], qf["pop"][:, 1])
            axe.semilogy(tt / T, np.maximum(np.abs(a1 - b1), 1e-12), color=C_GH, lw=1.1,
                         marker="o", ms=3.0, mfc="white", mew=0.9, zorder=5)
            for kind, t_, y_ in (
                    ("rk4", rk["grid_t"], rk["grid_populations"][:, 1]),
                    ("strang", st["coef_t"], st["coef_populations"][:, 1]),
                    ("pbme", pb["t"], pb["pop"][:, 1]),
                    ("focused", fo["t"], fo["pop"][:, 1])):
                tt, a2, b2 = common(t_, y_, qd["se_t"], qd["se_populations"][:, 1])
                axe.semilogy(tt / T, np.maximum(np.abs(a2 - b2), 1e-12), **DEV_STYLE[kind])
            axe.set_ylim(1e-7, 0.5)
            axe.set_xticks([0, 0.5, 1], ["0" if c == 0 else "", "0.5", "1"])
            if blk == 1:
                axe.set_xlabel("$t/T$", labelpad=1)
            if c > 0:
                for ax in (axd, axa, axe):
                    ax.tick_params(labelleft=False)
            else:
                axd.set_ylabel("diabatic")
                axa.set_ylabel("adiabatic")
                axe.set_ylabel(r"$|\Delta\rho^{22}|$")
            axd.text(0.02, 1.03, f"({PANEL[k]}) $P_0={P0}$", transform=axd.transAxes,
                     fontsize=8, ha="left", va="bottom")
            if k == 0:
                for ax, top, bottom in ((axd, r"$\rho^{11}$", r"$\rho^{22}$"),
                                        (axa, "lower", "upper")):
                    ax.text(0.04, 0.85, top, transform=ax.get_yaxis_transform(),
                            fontsize=7.5, ha="left", va="top")
                    ax.text(0.04, 0.05, bottom, transform=ax.get_yaxis_transform(),
                            fontsize=7.5, ha="left", va="bottom")
    first["d"].set_ylim(-0.14, 1.1); first["a"].set_ylim(-0.14, 1.1)
    first["e"].set_yticks([1e-6, 1e-4, 1e-2])
    fig.legend(handles=_method_handles(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.54, 0.0), columnspacing=1.6, handlelength=2.4)
    save(fig, "paper_fig3_populations")


def fig_coherence():
    """Real and imaginary parts of the integrated electronic coherence
    rho^12(t) = int dR dP rho_W^12 in six regimes, from every calculation, and
    |rho^12 - rho^12_quantum| at the instants saved by both calculations."""
    fig = plt.figure(figsize=(TWO, 7.9))
    outer = fig.add_gridspec(2, 1, hspace=0.15, left=0.085, right=0.995, top=0.975,
                             bottom=0.082)
    first_e = None
    for blk in range(2):
        gs = outer[blk].subgridspec(3, 3, height_ratios=[1.3, 1.3, 1.15],
                                    hspace=0.07, wspace=0.30)
        for c in range(3):
            k = 3 * blk + c
            P0 = MOMENTA[k]
            g = gh_history(P0); qd = quantum_dense(P0); pb = pbme(P0)
            fo = focused_series(P0); st = strang_series(P0); rk = rk4_series(P0)
            T = float(g["T"])
            # the wave-function code accumulates <psi_1|psi_2> = conj(rho^12)
            q_im = -qd["se_coherence_im"]
            axr = fig.add_subplot(gs[0, c])
            axi = fig.add_subplot(gs[1, c], sharex=axr, sharey=axr)
            axe = fig.add_subplot(gs[2, c], sharex=axr, sharey=first_e)
            first_e = first_e or axe
            for ax, kq, kg, kp, ks, kf, kst, kr in (
                    (axr, qd["se_coherence_re"], "coherence_re", "cre", "cresem", "cre",
                     "coef_coherence_re", "grid_coherence_re"),
                    (axi, q_im, "coherence_im", "cim", "cimsem", "cim",
                     "coef_coherence_im", "grid_coherence_im")):
                _draw(ax, T, [("quantum", qd["se_t"], kq),
                              ("pbme", pb["t"], pb[kp], pb[ks]),
                              ("focused", fo["t"], fo[kf]),
                              ("gh", g["t"], g[kg]),
                              ("rk4", rk["grid_t"], rk[kr]),
                              ("strang", st["coef_t"], st[kst])])
                ax.axhline(0, color="0.6", lw=0.4, ls=":")
                ax.set_xlim(0, 1)
                ax.tick_params(labelbottom=False)
            lim = 1.12 * max(np.max(np.abs(qd["se_coherence_re"])), np.max(np.abs(q_im)),
                             np.max(np.abs(pb["cre"])), np.max(np.abs(pb["cim"])),
                             np.max(np.abs(fo["cre"])), np.max(np.abs(fo["cim"])))
            axr.set_ylim(-lim, lim)
            # |rho^12 - rho^12_quantum| against the quantum run saved every
            # 10 a.u.; the coefficient propagation shares only some of its
            # instants (at P0 = 70 only t = 0 and T), so it is drawn as markers
            cq = qd["se_coherence_re"] + 1j * q_im
            iA, iB = _common_idx(g["t"], qd["se_t"])
            dg = np.abs(g["coherence_re"][iA] + 1j * g["coherence_im"][iA] - cq[iB])
            axe.semilogy(g["t"][iA] / T, np.maximum(dg, 1e-12), ls="none", marker="o",
                         ms=3.0, mfc="white", mec=C_GH, mew=0.9, zorder=5)
            for kind, t_, c_ in (
                    ("rk4", rk["grid_t"], rk["grid_coherence_re"] + 1j * rk["grid_coherence_im"]),
                    ("strang", st["coef_t"], st["coef_coherence_re"] + 1j * st["coef_coherence_im"]),
                    ("pbme", pb["t"], pb["cre"] + 1j * pb["cim"]),
                    ("focused", fo["t"], fo["cre"] + 1j * fo["cim"])):
                iA, iB = _common_idx(t_, qd["se_t"])
                axe.semilogy(np.asarray(t_)[iA] / T,
                             np.maximum(np.abs(c_[iA] - cq[iB]), 1e-12), **DEV_STYLE[kind])
            axe.set_ylim(1e-7, 0.5)
            axe.set_xticks([0, 0.5, 1], ["0" if c == 0 else "", "0.5", "1"])
            if blk == 1:
                axe.set_xlabel("$t/T$", labelpad=1)
            if c == 0:
                axr.set_ylabel(r"${\rm Re}\,\rho^{12}$")
                axi.set_ylabel(r"${\rm Im}\,\rho^{12}$")
                axe.set_ylabel(r"$|\Delta\rho^{12}|$")
            else:
                axe.tick_params(labelleft=False)
            axr.text(0.02, 1.03, f"({PANEL[k]}) $P_0={P0}$", transform=axr.transAxes,
                     fontsize=8, ha="left", va="bottom")
    first_e.set_yticks([1e-6, 1e-4, 1e-2])
    fig.legend(handles=_method_handles(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.54, 0.0), columnspacing=1.6, handlelength=2.4)
    save(fig, "paper_fig_coherence")


# ================================================================== Fig. 4
def fig4_defect():
    L = ledger()["rows"]
    lk = jload("12_leakage_reanalysis.json")["results"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(TWO, 2.1), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    for P0 in MOMENTA:
        d = jload(f"08_leakage_P{P0}.json")["results"]["route_A"]
        t = np.array(d["t"]); D = np.array(d["leakage_relative"])
        T = jload(f"07g_gh_coefficient_propagation_P{P0}.json")["metadata"]["time"]["T"]
        a.semilogy(t / T, D, color=REG[P0], lw=1.0, marker=MARK[P0], ms=2.4, markevery=4,
                   label=f"$P_0={P0}$")
    a.set_xlabel("$t/T$"); a.set_ylabel(r"$\mathcal{D}_4(t)$  (a.u.)")
    a.set_xlim(0, 1); a.set_ylim(1e-8, 0.3)
    a.legend(ncol=2, loc="lower right", handlelength=1.4, columnspacing=0.8, fontsize=6.5)
    panel_label(a, "(a)")

    rows = {int(rw["P0"]): rw for rw in lk["rows"]}
    P = np.array(MOMENTA, float)
    pk = np.array([rows[p]["peak_leakage"] for p in MOMENTA])
    e = np.array([L[str(p)]["e_pbme"] for p in MOMENTA]); s = np.array([L[str(p)]["pb_sem"] for p in MOMENTA])
    i40 = MOMENTA.index(40)
    b.plot(P, pk / pk[i40], color=C_GH, marker="o", ms=3.6, mfc="white", mew=1.0, lw=1.0,
           label=r"peak $\mathcal{D}_4$")
    b.errorbar(P, e / e[i40], yerr=2 * s / e[i40], color=C_PB, marker="s", ms=3.6, mfc="white",
               mew=1.0, lw=1.0, capsize=1.6, elinewidth=0.7, label=r"$E_{\rm PBME}$")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xticks(P, [str(int(p)) for p in P]); b.minorticks_off()
    b.set_xlabel(r"$P_0$ (a.u.),  $C=0.015$"); b.set_ylabel(r"relative to $P_0=40$, $C=0.015$")
    b.set_ylim(1e-4, 30)
    b.legend(loc="lower left")
    panel_label(b, "(b)")

    cs = lk["coupling_scan"]["rows"]
    C = np.array([rw["C"] for rw in cs]); ck = np.array([rw["peak_leakage"] for rw in cs])
    ce = np.array([rw["discard_cost"] for rw in cs]); csem = np.array([rw["discard_cost_sem"] for rw in cs])
    j = int(np.argmin(np.abs(C - 0.015)))
    c.plot(C, ck / ck[j], color=C_GH, marker="o", ms=3.6, mfc="white", mew=1.0, lw=1.0)
    ok = ce > 3 * csem
    c.errorbar(C[ok], ce[ok] / ce[j], yerr=2 * csem[ok] / ce[j], color=C_PB, marker="s", ms=3.6,
               mfc="white", mew=1.0, lw=1.0, capsize=1.6, elinewidth=0.7)
    for x, y, sg in zip(C[~ok], ce[~ok], csem[~ok]):      # unresolved: 2-sigma upper bound
        ub = (y + 2 * sg) / ce[j]
        c.errorbar([x], [ub], yerr=[[0.6 * ub], [0.0]], color=C_PB, marker="v", ms=3.6,
                   mfc="white", mew=1.0, ls="none", capsize=0, elinewidth=0.7)
    c.plot(C[ok], ce[ok] / ce[j], color=C_PB, lw=1.0)
    c.set_xscale("log"); c.set_yscale("log")
    c.set_xticks(C, [f"{x:g}" for x in C]); c.minorticks_off()
    c.tick_params(axis="x", labelsize=5.8)
    c.set_xlabel(r"$C$ (a.u.),  $P_0=40$"); c.set_ylim(1e-4, 30)
    panel_label(c, "(c)")
    save(fig, "paper_fig4_defect")


# ================================================================ SM figures
def figS_closure():
    k = jload("07k_gh_closure_validation.json")["results"]
    hc = k["closure"]["history"]; hp = k["parity"]["history"]
    t = np.array([h["t"] for h in hc])
    # unweighted content sum_{|n|=k} int dR dP |c_n|^2, the measure in which the
    # control runs are archived and quoted
    E = {o: np.array([h["E_raw"][o] for h in hc]) for o in ("0", "2", "4")}
    tp = np.array([h["t"] for h in hp])
    Eo = np.array([h["E_raw"]["1"] + h["E_raw"]["3"] for h in hp])
    if np.any(E["4"] != 0.0) or np.any(Eo != 0.0):
        raise SystemExit("[16] forbidden sectors are not exactly empty")
    fig, b = plt.subplots(figsize=(ONE, 2.1), constrained_layout=True)
    b.plot(t, E["0"], color=C_GRID, lw=1.1, label="order 0")
    b.plot(t, E["2"], color=C_GH, lw=1.1, label="order 2")
    b.plot(t, E["4"], color=C_PB, lw=1.6, label="order 4 (identically 0)")
    b.plot(tp, Eo, color=C_FOC, lw=0.9, ls="--", label="odd orders (identically 0)")
    b.set_xlabel("$t$ (a.u.)"); b.set_ylabel(r"$\sum_{|\mathbf{n}|=k}\|c_{\mathbf{n}}\|^2$")
    b.set_xlim(t[0], t[-1]); b.set_ylim(-0.02, 1.05 * max(E["0"].max(), E["2"].max()))
    b.legend(loc="center right")
    save(fig, "paper_figS_closure")


def figS_convergence():
    """Convergence of the reference calculations and of the two estimators."""
    L = ledger()["rows"]
    hi = [P for P in MOMENTA if P >= 20]
    fig, axs = plt.subplots(2, 2, figsize=(TWO, 4.4), constrained_layout=True)
    (a, b), (c, d) = axs
    # (a) Strang splitting against the four-field Runge--Kutta solution
    x = np.array(hi, float)
    s1 = np.array([L[str(P)]["strang_linf"] for P in hi])
    s2 = np.array([L[str(P)]["strang_linf_half"] for P in hi])
    a.loglog(x, s1, color=C_STR, marker="s", mfc="white", label=r"Strang, $\Delta t=0.5$")
    a.loglog(x, s2, color=C_STR, marker="s", label=r"Strang, $\Delta t=0.25$")
    q = s1 / s2
    a.text(0.97, 0.05, f"ratio {q.min():.2f}\u2013{q.max():.2f}", transform=a.transAxes,
           ha="right", va="bottom", fontsize=7, color="0.3")
    a.set_xticks(x, [str(int(v)) for v in x]); a.minorticks_off()
    a.set_ylabel(r"$\max_t|\rho^{22}_{\rm Strang}-\rho^{22}_{\rm RK4}|$")
    a.set_xlabel(r"$P_0$ (a.u.)")
    a.legend(loc="upper left")
    # (b) refinement of the grid solver, of the coefficient propagation and of the quantum reference
    xa = np.array(MOMENTA, float)
    b.loglog(x, [L[str(P)]["e_grid"] for P in hi], color=C_GRID, marker="o", mfc="white",
             label="coefficients vs RK4")
    b.loglog(x, [L[str(P)]["grid_spatial"] for P in hi], color=C_GRID, marker="^", lw=0.8,
             label="RK4, grid halved")
    b.loglog(x, [L[str(P)]["grid_dthalf"] for P in hi], color=C_GRID, marker="v", ls=":",
             label="RK4, step halved")
    b.loglog([10], [L["10"]["u_num"]], ls="none", marker="X", color=C_GH, ms=5,
             label=r"grid doubled ($P_0=10$)")
    b.loglog(xa, [L[str(P)]["q_unc"] for P in MOMENTA], color=C_Q, marker="D", ms=2.6, lw=0.8,
             label="quantum reference")
    b.set_xticks(xa, [str(int(v)) for v in xa]); b.minorticks_off()
    b.set_xlabel(r"$P_0$ (a.u.)"); b.set_ylabel(r"change in $\rho^{22}(T)$")
    b.set_ylim(1e-17, 1e-1)
    b.legend(loc="upper right", fontsize=6, ncol=1)
    # (c) statistical error of the projected-density estimator
    for P0 in MOMENTA:
        conv = [q for q in jload(f"11_pbme_projected_P{P0}.json")["results"]["sample_size_convergence"]
                if q["estimator"] == "cv" and q["n_blocks"] >= 4]
        n = np.array([q["n_samples"] for q in conv], float); e = np.array([q["sem"][1] for q in conv])
        c.loglog(n, e, color=REG[P0], marker=MARK[P0], ms=2.8, lw=0.9, label=f"$P_0={P0}$")
    nn = np.array([2e4, 1.2e6]); c.loglog(nn, 1.2e-2 * (nn / 2e4) ** -0.5, color="0.5", ls="--", lw=0.8)
    c.text(1.5e5, 5.5e-3, r"$N^{-1/2}$", fontsize=7, color="0.4")
    c.set_xlabel("trajectories $N$"); c.set_ylabel(r"standard error of $\rho^{22}(T)$")
    c.legend(loc="lower left", fontsize=6, ncol=2)
    # (d) kernel estimate of the momentum gradient of the initial Wigner function
    for P0 in MOMENTA:
        es = jload(f"08_leakage_P{P0}.json")["results"]["estimator_study"]["rows"]
        n = np.array([q["n_samples"] for q in es], float)
        e = np.array([q["relative_rmse_mean"] for q in es])
        d.loglog(n, e, color=REG[P0], marker=MARK[P0], ms=2.8, lw=0.9, label=f"$P_0={P0}$")
    nn = np.array([1e3, 1e5]); d.loglog(nn, 0.5 * (nn / 1e3) ** -0.25, color="0.5", ls="--", lw=0.8)
    d.loglog(nn, 0.5 * (nn / 1e3) ** -0.5, color="0.5", ls=":", lw=0.8)
    d.text(1.25e5, 0.5 * 100 ** -0.25, r"$N^{-1/4}$", fontsize=7, color="0.4", va="center")
    d.text(1.25e5, 0.05, r"$N^{-1/2}$", fontsize=7, color="0.4", va="center")
    d.set_xlabel("trajectories $N$"); d.set_ylabel(r"relative error of $\partial_P\rho_W$")
    d.set_ylim(0.04, 0.8); d.set_xlim(7e2, 3.2e5)
    for ax, lab in zip((a, b, c, d), "abcd"):
        panel_label(ax, f"({lab})")
    save(fig, "paper_figS_convergence")


# ================================================================ phase space
#: regimes compared in phase space; at P0 = 10 the terminal fields carry
#: structure finer than the stored grid and are not compared
PS_REGIMES = (20, 30, 70, 100)
PS_ROWS = (("quantum", "full quantum"), ("qcle", "QCLE\n(coefficients)"),
           ("grid", "QCLE\n(four-field RK4)"), ("pbme", "PBME"))
PS_LABEL = {"A": r"\rho_W^{11}", "C": r"\rho_W^{22}",
            "bR": r"{\rm Re}\,\rho_W^{12}", "bI": r"{\rm Im}\,\rho_W^{12}"}


def phase(P0) -> dict:
    import h5py
    name = f"19_phase_space_comparison_P{P0}.h5"
    if not (RAW / name).is_file():
        raise SystemExit(f"[16] required archive missing: {name}")
    with h5py.File(RAW / name, "r") as f:
        return {k: np.asarray(f[k][()]) for k in f}


def _map_extent(d, comps, frac=0.03, pad=0.05):
    """Box around the region where any of the three densities of any of
    `comps` exceeds `frac` of its own peak."""
    R, P = d["map_R"], d["map_P"]
    Rc = 0.5 * (d["R_edges"][1:] + d["R_edges"][:-1])
    Pc = 0.5 * (d["P_edges"][1:] + d["P_edges"][:-1])
    r_in, p_in = [], []
    for k in comps:
        for m in ("quantum", "qcle"):
            a = np.abs(d[f"map_{m}_{k}"])
            mask = a > frac * a.max()
            r_in += [R[mask.any(axis=1)].min(), R[mask.any(axis=1)].max()]
            p_in += [P[mask.any(axis=0)].min(), P[mask.any(axis=0)].max()]
        a = np.abs(d[f"cells_pbme_{k}"])
        mask = a > frac * a.max()
        r_in += [Rc[mask.any(axis=1)].min(), Rc[mask.any(axis=1)].max()]
        p_in += [Pc[mask.any(axis=0)].min(), Pc[mask.any(axis=0)].max()]
    r0, r1, p0, p1 = min(r_in), max(r_in), min(p_in), max(p_in)
    dr, dp = (r1 - r0) * pad, (p1 - p0) * pad
    return r0 - dr, r1 + dr, p0 - dp, p1 + dp


def _asymptotic_momenta(P0):
    """Momentum of the packet centre on state 1 and, asymptotically, on state 2."""
    from mqcle.io import load_config
    cfg = load_config("tully_default.yaml")
    M, E0 = float(cfg["mass"]), float(cfg["model"]["E0"])
    p2 = np.sqrt(P0 ** 2 - 2.0 * M * E0) if P0 ** 2 > 2.0 * M * E0 else np.nan
    return float(P0), float(p2)


def _phase_block(fig, gs_block, P0, comps, with_momentum, letters, first_block):
    """One regime: a colour bar and three rows of maps per component, and
    optionally the momentum distributions below."""
    d = phase(P0)
    ext = _map_extent(d, comps)
    if with_momentum:
        nmap = len(PS_ROWS)
        inner = gs_block.subgridspec(2, 1, height_ratios=[1.1 * nmap, 1.0], hspace=0.14)
        gmaps = inner[0].subgridspec(nmap + 1, len(comps),
                                     height_ratios=[0.06] + [1.0] * nmap,
                                     hspace=0.07, wspace=0.07)
        gmom = inner[1].subgridspec(1, len(comps), wspace=0.48)
    else:
        nmap = len(PS_ROWS)
        gmaps = gs_block.subgridspec(nmap + 1, len(comps),
                                     height_ratios=[0.06] + [1.0] * nmap,
                                     hspace=0.07, wspace=0.07)
    for cj, k in enumerate(comps):
        vmax = float(max(np.abs(d[f"map_quantum_{k}"]).max(),
                         np.abs(d[f"map_qcle_{k}"]).max()))
        letter = next(letters)
        for ri, (m, lab) in enumerate(PS_ROWS):
            ax = fig.add_subplot(gmaps[ri + 1, cj])
            if m == "pbme":
                im = ax.pcolormesh(d["R_edges"], d["P_edges"], d[f"cells_pbme_{k}"].T,
                                   cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                                   shading="flat", rasterized=True)
            else:
                im = ax.pcolormesh(d["map_R"], d["map_P"], d[f"map_{m}_{k}"].T,
                                   cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                                   shading="nearest", rasterized=True)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ax.tick_params(labelsize=6, length=2)
            if ri < len(PS_ROWS) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"$R$ (a.u.)", labelpad=1)
            if cj == 0:
                ax.set_ylabel(r"$P$ (a.u.)", labelpad=1)
                if first_block:
                    ax.text(-0.50, 0.5, lab, transform=ax.transAxes, rotation=90,
                            ha="center", va="center", fontsize=7.2, linespacing=1.35)
            else:
                ax.set_yticklabels([])
            if ri == 0:
                cax = fig.add_subplot(gmaps[0, cj])
                cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                cb.set_ticks([-vmax, 0.0, vmax])
                cb.ax.set_xticklabels([f"{-vmax:.2g}", "0", f"{vmax:.2g}"], fontsize=5.5)
                cb.ax.xaxis.set_ticks_position("top")
                cb.ax.tick_params(length=1.5, pad=1)
                labs = cb.ax.get_xticklabels()
                labs[0].set_ha("left"); labs[-1].set_ha("right")
                cb.outline.set_linewidth(0.4)
                cax.set_title(f"({letter}) $P_0={P0}$, ${PS_LABEL[k]}$", fontsize=7.5, pad=8)
        if with_momentum:
            ax = fig.add_subplot(gmom[0, cj])
            Pn = d["P_nodes"]
            ax.axhline(0.0, color="0.8", lw=0.5)
            p1, p2 = _asymptotic_momenta(P0)
            ax.axvline(p1 if k == "A" else p2, color="0.55", lw=0.6, ls=":")
            ax.plot(Pn, d[f"momentum_quantum_{k}"], color=C_Q, lw=1.0, label="full quantum")
            ax.plot(Pn, d[f"momentum_qcle_{k}"], color=C_GH, lw=1.0, ls="--",
                    label="QCLE, coefficients")
            ax.plot(Pn, d[f"momentum_grid_{k}"], color=C_GRID, lw=1.2, ls=(0, (1, 1.6)),
                    label="QCLE, grid")
            Pc = 0.5 * (d["P_edges"][1:] + d["P_edges"][:-1])
            ax.errorbar(Pc, d[f"momentum_pbme_{k}"],
                        yerr=2.0 * d[f"momentum_pbme_sem_{k}"], color=C_PB,
                        marker="o", ms=1.8, lw=0.0, elinewidth=0.6, capsize=0,
                        label=r"PBME ($\pm2$ s.e.)")
            ax.set_xlim(ext[2], ext[3])
            ax.tick_params(labelsize=6, length=2)
            ax.set_xlabel(r"$P$ (a.u.)", labelpad=1)
            ax.set_ylabel(rf"$n_{{{'1' if k == 'A' else '2'}}}(P)$", labelpad=1)
            ax.text(0.04, 0.96, f"({next(letters)})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.5, fontweight="bold")


def _phase_figure(regimes, comps, with_momentum, stem, height):
    fig = plt.figure(figsize=(TWO, height))
    outer = fig.add_gridspec(1, len(regimes), left=0.085, right=0.995, top=0.95,
                             bottom=0.085 if with_momentum else 0.055, wspace=0.17)
    # columns are lettered first, then the momentum panels
    n_cols = len(regimes) * len(comps)
    col_letters = list("abcdefghijklmnop"[:n_cols])
    mom_letters = list("abcdefghijklmnop"[n_cols:2 * n_cols])
    for bi, P0 in enumerate(regimes):
        seq = col_letters[bi * len(comps):(bi + 1) * len(comps)]
        if with_momentum:
            seq = [seq[0]] + [mom_letters[bi * len(comps)]]
            seq = [col_letters[bi * len(comps) + c] if j % 2 == 0 else
                   mom_letters[bi * len(comps) + c]
                   for c in range(len(comps)) for j in range(2)]
        _phase_block(fig, outer[0, bi], P0, comps, with_momentum, iter(seq), bi == 0)
    if with_momentum:
        # one legend for the momentum panels, below them, clear of the data
        fig.legend(handles=[
            Line2D([], [], color=C_Q, lw=1.0, label="full quantum"),
            Line2D([], [], color=C_GH, lw=1.0, ls="--", label="QCLE, coefficient propagation"),
            Line2D([], [], color=C_GRID, lw=1.2, ls=(0, (1, 1.6)), label="QCLE, four-field RK4"),
            Line2D([], [], color=C_PB, ls="none", marker="o", ms=3.0,
                   label=r"PBME, projected density ($\pm2$ s.e.)")],
            loc="lower center", ncol=4, bbox_to_anchor=(0.54, 0.0), fontsize=7,
            columnspacing=1.6, handlelength=2.4)
    save(fig, stem)


def fig_phasespace():
    """Terminal phase-space densities at the two lowest compared momenta."""
    _phase_figure((20, 30), ("A", "C"), True, "paper_fig_phasespace", 7.9)


def figS_phasespace_coherence():
    _phase_figure((20, 30), ("bR", "bI"), False, "paper_figS_phasespace_coherence", 6.6)


def figS_phasespace_high():
    _phase_figure((70, 100), ("A", "C"), True, "paper_figS_phasespace_high", 7.9)


def main(argv) -> int:
    todo = {"fig1": fig1_selection_rules, "fig2": fig2_numerics,
            "fig3": fig3_populations, "fig4": fig4_defect,
            "figS_closure": figS_closure,
            "fig_coherence": fig_coherence, "figS_convergence": figS_convergence,
            "fig_phasespace": fig_phasespace,
            "figS_phasespace_coherence": figS_phasespace_coherence,
            "figS_phasespace_high": figS_phasespace_high}
    names = argv[1:] or list(todo)
    failed = []
    for n in names:
        try:
            todo[n]()
        except Exception as exc:             # noqa: BLE001
            failed.append(n)
            print(f"[16] {n} FAILED: {type(exc).__name__}: {exc}")
    if failed:
        print(f"[16] {len(failed)} figure(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
