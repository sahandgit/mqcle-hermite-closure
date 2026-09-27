#!/usr/bin/env python3
r"""Terminal phase-space densities of the three descriptions, compared.

What is compared
----------------
At ``t = T`` these fields are available for a regime:

* ``quantum``   the partial Wigner transform of the converged wavefunction,
                on the coefficient nodes (``scripts/10b``);
* ``QCLE``      the density read out of the coefficient propagation
                (``scripts/07g``), on the same nodes;
* ``grid``      the four-field Runge--Kutta solution of the subsystem QCLE
                (``scripts/10b``), whose solver keeps cell-centred nodes; it
                is moved onto the coefficient nodes by exact Fourier
                translation.  The Strang solution archived by ``scripts/10b``
                is compared as well;
* ``PBME``      the projected-density PBME estimator resolved by phase-space
                cell (``scripts/18``), with per-block estimates;
* ``focused``   the focused-sampling PBME estimator on the same cells
                (``scripts/10b``), for reference.

The grid fields are compared at full resolution and all four on the PBME
cells.  A grid field is reduced onto a cell by its exact average over the
cell: the fields are resolved Fourier series (their spectral tails are
archived), so the average of the interpolant over ``[x_k, x_k + b h)`` is the
multiplier ``exp(i k b h/2) sin(k b h/2)/(k b h/2)`` applied in Fourier space.
The PBME cells have their edges on coefficient nodes, so the reduction and the
histogram cover exactly the same set of points.

Measures
--------
* relative L2 distance of each component from the quantum field;
* negative volume  ``N_a = int max(-rho_W^{aa}, 0) dR dP``;
* momentum and position distributions of each diabatic state,
  ``n_a(P) = int dR rho_W^{aa}`` and ``x_a(R) = int dP rho_W^{aa}``, and their
  L1 distance from the quantum ones in units of the quantum population;
* for the PBME, the statistical floor of each distance, from the blocks.

Usage
-----
    python scripts/19_phase_space_comparison.py --P0 30
"""
from __future__ import annotations

import argparse
import json
import sys
from math import erf, sqrt
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqcle.io import RunRecorder, save_arrays                         # noqa: E402
from mqcle.models import build_model                                  # noqa: E402

COMPONENTS = ("A", "C", "bR", "bI")
DIAGONAL = ("A", "C")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--P0", type=float, required=True)
    p.add_argument("--out", default=None, help="output stem (no extension)")
    p.add_argument("--pbme-field", default=None,
                   help="stem of the scripts/18 archive (default results/raw/"
                        "18_pbme_phase_space_P<P0>)")
    p.add_argument("--map-stride", type=int, default=None,
                   help="node stride of the stored full-resolution maps "
                        "(default: at most 600 nodes along each axis)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# exact cell averages of a resolved Fourier series
# --------------------------------------------------------------------------
def _box_multiplier(n: int, h: float, width: float) -> np.ndarray:
    """Fourier multiplier of the average over ``[x, x + width)``."""
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=h)
    u = 0.5 * k * width
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(u == 0.0, 1.0, np.sin(u) / np.where(u == 0.0, 1.0, u))
    return np.exp(1j * u) * s


def cell_averages(f: np.ndarray, dR: float, dP: float, i0: int, j0: int,
                  n_cR: int, n_cP: int, b: int) -> np.ndarray:
    """Averages of the periodic interpolant of ``f`` over the cells
    ``[R_{i0+kb}, R_{i0+(k+1)b}) x [P_{j0+lb}, P_{j0+(l+1)b})``."""
    nR, nP = f.shape
    F = np.fft.fft2(f)
    F *= _box_multiplier(nR, dR, b * dR)[:, None]
    F *= _box_multiplier(nP, dP, b * dP)[None, :]
    g = np.real(np.fft.ifft2(F))
    return g[i0:i0 + n_cR * b:b, j0:j0 + n_cP * b:b][:n_cR, :n_cP].copy()


def spectral_tail(f: np.ndarray, frac: float = 0.75) -> float:
    """Root fraction of the spectral power beyond ``frac`` of the Nyquist
    wavenumber along either axis."""
    F = np.abs(np.fft.fft2(f)) ** 2
    kR = np.abs(np.fft.fftfreq(f.shape[0])) * 2.0
    kP = np.abs(np.fft.fftfreq(f.shape[1])) * 2.0
    m = (kR[:, None] > frac) | (kP[None, :] > frac)
    tot = F.sum()
    return float(np.sqrt(F[m].sum() / tot)) if tot > 0 else 0.0


def rel_l2(a: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - ref) ** 2) / np.sum(ref ** 2)))


def negative_volume(f: np.ndarray, area: float) -> float:
    return float(np.clip(-f, 0.0, None).sum() * area)


def _gaussian_cell_test(dR: float, dP: float, nR: int, nP: int, b: int) -> float:
    """Maximum error of :func:`cell_averages` on a resolved Gaussian whose cell
    averages are known in closed form."""
    R = np.arange(nR) * dR
    P = np.arange(nP) * dP
    Rc, Pc = 0.5 * nR * dR + 0.3 * dR, 0.5 * nP * dP - 0.2 * dP
    sR, sP = 12.0 * dR, 9.0 * dP
    f = np.exp(-0.5 * ((R[:, None] - Rc) / sR) ** 2
               - 0.5 * ((P[None, :] - Pc) / sP) ** 2)
    i0 = (nR // 2 - 8 * b) // b * b
    j0 = (nP // 2 - 8 * b) // b * b
    ncR, ncP = 16, 16
    got = cell_averages(f, dR, dP, i0, j0, ncR, ncP, b)

    def avg1(a, w, c, s):
        return (sqrt(np.pi / 2.0) * s / w
                * (erf((a + w - c) / (sqrt(2.0) * s)) - erf((a - c) / (sqrt(2.0) * s))))
    exR = np.array([avg1(R[i0 + k * b], b * dR, Rc, sR) for k in range(ncR)])
    exP = np.array([avg1(P[j0 + k * b], b * dP, Pc, sP) for k in range(ncP)])
    return float(np.max(np.abs(got - exR[:, None] * exP[None, :])))


def momentum_moments(n: np.ndarray, P: np.ndarray) -> tuple[float, float]:
    """Mean and root second central moment of a (signed) momentum
    distribution, normalized by its own integral."""
    m0 = float(n.sum())
    m1 = float((n * P).sum() / m0)
    var = float((n * (P - m1) ** 2).sum() / m0)
    return m1, float(np.sqrt(var)) if var > 0 else float("nan")


def main(argv=None) -> int:
    args = parse_args(argv)
    P0i = int(round(args.P0))
    raw = Path("results/raw")
    f07 = raw / f"07g_gh_coefficient_propagation_P{P0i}.h5"
    f10 = raw / f"10b_qcle_grid_crosscheck_P{P0i}.h5"
    j10 = raw / f"10b_qcle_grid_crosscheck_P{P0i}.json"
    s18 = Path(args.pbme_field or raw / f"18_pbme_phase_space_P{P0i}")
    f18, j18 = s18.with_suffix(".h5"), s18.with_suffix(".json")
    j07 = raw / f"07g_gh_coefficient_propagation_P{P0i}.json"
    rec = RunRecorder(script=Path(__file__).name, config={
        "P0": args.P0, "inputs": [str(p) for p in (f07, f10, j10, f18, j18, j07)]})

    # ---- fields ------------------------------------------------------------
    with h5py.File(f07, "r") as g:
        R = g["R"][()]
        P = g["P"][()]
        qcle = {k: g[f"gh_final_{k}"][()] for k in COMPONENTS}
    with h5py.File(f10, "r") as x:
        quantum = {k: x[f"quantum_final_{k}"][()] for k in COMPONENTS}
        strang = {k: x[f"coef_final_{k}"][()] for k in COMPONENTS}
        grid4 = {k: x[f"grid_final_{k}"][()] for k in COMPONENTS}
        focused = {k: x[f"pbme_final_{k}"][()] for k in COMPONENTS}
        t_grid_end = float(x["grid_t"][-1])
    with h5py.File(f18, "r") as y:
        pb_blocks = y["block_fields"][()]                 # (B, 4, ncR, ncP)
    meta10 = json.loads(j10.read_text())["results"]
    meta18 = json.loads(j18.read_text())
    cells = meta18["results"]["cells"]
    if not meta18.get("all_passed", False):
        raise SystemExit(f"{j18} did not pass its own checks")

    grid = meta10["runs"]["production"]["grid"]
    nR, nP = int(grid["n_R"]), int(grid["n_P"])
    if qcle["A"].shape != (nR, nP) or quantum["A"].shape != (nR, nP):
        raise SystemExit("the QCLE and quantum fields are not on one grid")
    Rg = float(grid["R_min"]) + (float(grid["R_max"]) - float(grid["R_min"])) * np.arange(nR) / nR
    Pg = float(grid["P_min"]) + (float(grid["P_max"]) - float(grid["P_min"])) * np.arange(nP) / nP
    rec.check("phase.node_alignment",
              float(max(np.max(np.abs(Rg - R)), np.max(np.abs(Pg - P)))), 1e-12,
              "nodes of the coefficient propagation against the grid on which "
              "the quantum field was evaluated")
    dR, dP = float(R[1] - R[0]), float(P[1] - P[0])
    area = dR * dP

    # ---- the four-field grid solution of the subsystem QCLE -----------------
    # Its solver keeps cell-centred nodes, R_min + (i + 1/2) dR; the offsets
    # recorded by scripts/10b are removed by exact Fourier translation, as
    # there, so that the fields can be differenced node by node.
    tfc = meta10["terminal_field_comparison"]
    offR, offP = float(tfc["node_offset_R"]), float(tfc["node_offset_P"])
    rec.check("phase.grid_node_offset_is_half_cell",
              max(abs(offR - 0.5 * dR), abs(offP - 0.5 * dP)), 1e-12,
              "node offset of the four-field solver recorded by scripts/10b "
              "against half a cell")
    if grid4["A"].shape != (nR, nP):
        raise SystemExit("the four-field solution is not on the grid of the "
                         "coefficient propagation")
    kRg = 2.0 * np.pi * np.fft.fftfreq(nR, d=dR)
    kPg = 2.0 * np.pi * np.fft.fftfreq(nP, d=dP)
    shift = np.exp(-1j * kRg[:, None] * offR) * np.exp(-1j * kPg[None, :] * offP)
    grid4 = {k: np.real(np.fft.ifft2(np.fft.fft2(v) * shift)) for k, v in grid4.items()}
    peak = max(float(np.max(np.abs(qcle[k]))) for k in COMPONENTS)
    solvers = {"grid_step": float(meta10["runs"]["production"]["dt"]),
               "strang_step": float(tfc["coefficient_step"]),
               "density_peak": peak,
               "note": "max_over_peak is max |X - coefficient propagation| over "
                       "the largest |value| of any component of the latter"}
    for name, f in (("grid", grid4), ("strang", strang)):
        solvers[name] = {k: {"rel_l2": rel_l2(f[k], qcle[k]),
                             "max_over_peak": float(np.max(np.abs(f[k] - qcle[k])) / peak)}
                         for k in COMPONENTS}
    rec.check("phase.grid_solver_agrees",
              max(v["max_over_peak"] for v in solvers["grid"].values()), 1e-2,
              "four-field grid solution against the coefficient propagation, "
              "largest pointwise difference over the peak of the density")

    # ---- the time-integration error of the four-field solver ----------------
    # Away from the crossing the forces and the coupling vanish and the
    # four-field equations reduce to free flight: at fixed P, each Fourier
    # component in R of a diagonal density advances over a step dt by exp(z),
    # z = -i k P dt / M, and that of rho^12 = bR + i bI by exp(z + i g dt/hbar),
    # with g = V22 - V11 the asymptotic gap of the model.  Classical
    # Runge--Kutta replaces exp(z) by its Taylor polynomial of degree four,
    # G(z), and is exact at z = 0.  Over the N steps of the run a four-field
    # density therefore differs from the exact one by the field whose Fourier
    # components are [G(z)^N exp(-N z) - 1] times those of the exact density.
    # Evaluated on the coefficient-propagation density this predicts the
    # difference between the two solutions; it neglects the forces and the
    # coupling near the crossing, and how much of the measured difference it
    # accounts for is recorded, not gated.
    cfg10 = json.loads(j10.read_text())["metadata"]["config"]
    mass, hbar = float(cfg10["mass"]), float(cfg10["hbar"])
    model = build_model(cfg10["model"], mass=mass)
    edges = model.h(np.array([[float(grid["R_min"])], [float(grid["R_max"])]]))
    gaps = np.real(edges[:, 1, 1] - edges[:, 0, 0])
    rec.check("phase.rk4_asymptotic_gap",
              float(max(abs(gaps[1] - gaps[0]),
                        np.max(np.abs(np.real(edges[:, 0, 1]))))), 1e-12,
              "the diabatic gap is the same at both edges of the box and the "
              "coupling vanishes there, so free flight is the asymptotic "
              "motion")
    gap = float(gaps[1])
    dt4 = float(meta10["runs"]["production"]["dt"])
    n4 = int(round(t_grid_end / dt4))
    rec.check("phase.rk4_whole_steps", abs(n4 * dt4 - t_grid_end), 1e-9,
              "final time of the four-field run against a whole number of "
              "its steps")

    def _rk4_defect(z: np.ndarray) -> np.ndarray:
        g = 1.0 + z * (1.0 + z * (0.5 + z * (1.0 / 6.0 + z / 24.0)))
        return (g * np.exp(-z)) ** n4 - 1.0

    z_d = -1j * kRg[:, None] * P[None, :] * (dt4 / mass)
    fac = _rk4_defect(z_d)
    predicted = {k: np.real(np.fft.ifft(fac * np.fft.fft(qcle[k], axis=0), axis=0))
                 for k in DIAGONAL}
    fac = _rk4_defect(z_d + 1j * gap * dt4 / hbar)
    pz = np.fft.ifft(fac * np.fft.fft(qcle["bR"] + 1j * qcle["bI"], axis=0), axis=0)
    predicted["bR"], predicted["bI"] = np.real(pz), np.imag(pz)
    del fac, pz, z_d
    rk4 = {}
    for k in COMPONENTS:
        d = grid4[k] - qcle[k]
        rk4[k] = {
            "measured_max_over_peak": float(np.max(np.abs(d)) / peak),
            "predicted_max_over_peak": float(np.max(np.abs(predicted[k])) / peak),
            "residual_rel_l2": float(np.linalg.norm(d - predicted[k])
                                     / np.linalg.norm(d)),
            "corrected_max_over_peak": float(np.max(np.abs(d - predicted[k])) / peak),
        }
    solvers["grid_rk4_free_flight"] = {
        "n_steps": n4, "step": dt4, "mass": mass, "hbar": hbar, "gap": gap,
        "components": rk4,
        "note": "predicted = F^-1{[G(z)^N e^{-Nz} - 1] F(coefficient density)} "
                "along R, z = -i k P dt/M (+ i g dt/hbar for rho^12), "
                "G the degree-4 Taylor polynomial; residual_rel_l2 = "
                "||measured - predicted|| / ||measured||; corrected = "
                "max |measured - predicted| over the density peak"}
    del predicted

    b = int(cells["coarsen_block"])
    i0, j0 = (int(v) for v in cells["index_origin"])
    ncR, ncP = int(cells["n_R"]), int(cells["n_P"])
    carea = cells["dR"] * cells["dP"]

    rec.check("phase.cell_average_exactness",
              _gaussian_cell_test(dR, dP, nR, nP, b), 1e-12,
              "Fourier cell averages of a resolved Gaussian against erf")

    # ---- consistency with the archived populations -----------------------
    fin07 = json.loads(j07.read_text())["results"]["final"]
    pops_ts = np.asarray(fin07["populations"], float)
    coh_ts = np.array([float(fin07["coherence_re"]), float(fin07["coherence_im"])])
    pop_q = np.array([quantum[k].sum() * area for k in DIAGONAL])
    pop_c = np.array([qcle[k].sum() * area for k in DIAGONAL])
    coh_c = np.array([qcle["bR"].sum() * area, qcle["bI"].sum() * area])
    rec.check("phase.qcle_field_population",
              float(max(np.max(np.abs(pop_c - pops_ts)),
                        np.max(np.abs(coh_c - coh_ts)))), 1e-9,
              "integrals of the QCLE field against the archived final "
              "populations and coherence of the coefficient propagation")
    rec.check("phase.quantum_trace",
              float(abs(pop_q.sum() - 1.0)), 1e-8,
              "integral of the quantum Wigner field")

    res: dict = {"P0": args.P0, "solvers": solvers,
                 "grid": {"n_R": nR, "n_P": nP, "dR": dR, "dP": dP},
                 "cells": cells, "populations": {"quantum": pop_q.tolist(),
                                                 "qcle": pop_c.tolist()}}

    # ---- full resolution: QCLE against quantum ---------------------------
    full = {}
    for k in COMPONENTS:
        full[k] = {
            "rel_l2_qcle_quantum": rel_l2(qcle[k], quantum[k]),
            "rel_l2_strang_qcle": rel_l2(strang[k], qcle[k]),
            "spectral_tail_qcle": spectral_tail(qcle[k]),
            "spectral_tail_quantum": spectral_tail(quantum[k]),
            "peak_quantum": float(np.max(np.abs(quantum[k]))),
            "peak_qcle": float(np.max(np.abs(qcle[k]))),
        }
    for a, k in enumerate(DIAGONAL):
        nq, nc = quantum[k].sum(axis=0) * dR, qcle[k].sum(axis=0) * dR
        xq, xc = quantum[k].sum(axis=1) * dP, qcle[k].sum(axis=1) * dP
        ng = grid4[k].sum(axis=0) * dR
        full[k].update({
            "negative_volume_grid": negative_volume(grid4[k], area),
            "momentum_l1_grid": float(np.abs(ng - nq).sum() * dP / pop_q[a]),
            "negative_volume_quantum": negative_volume(quantum[k], area),
            "negative_volume_qcle": negative_volume(qcle[k], area),
            "momentum_l1_qcle": float(np.abs(nc - nq).sum() * dP / pop_q[a]),
            "position_l1_qcle": float(np.abs(xc - xq).sum() * dR / pop_q[a]),
            "momentum_min_quantum": float(nq.min()),
            "momentum_min_qcle": float(nc.min()),
            "momentum_peak_quantum": float(nq.max()),
            "momentum_peak_qcle": float(nc.max()),
            "momentum_negative_mass_qcle": float(np.clip(-nc, 0, None).sum() * dP),
            "momentum_mean_quantum": float((P * nq).sum() / nq.sum()),
            "momentum_mean_qcle": float((P * nc).sum() / nc.sum()),
        })
    res["full_resolution"] = full
    rec.check("phase.qcle_fields_resolved",
              max(v["spectral_tail_qcle"] for v in full.values()), 1e-4,
              "root spectral power of the QCLE fields beyond 3/4 of Nyquist")
    rec.check("phase.quantum_fields_resolved",
              max(v["spectral_tail_quantum"] for v in full.values()), 1e-4,
              "root spectral power of the quantum fields beyond 3/4 of Nyquist")
    rec.check("phase.qcle_solvers_agree",
              max(v["rel_l2_strang_qcle"] for v in full.values()), 1e-4,
              "coefficient propagation against the Strang solution of the "
              "subsystem QCLE at the step of scripts/10b, whole fields")

    # ---- on the PBME cells ------------------------------------------------
    qc = {k: cell_averages(quantum[k], dR, dP, i0, j0, ncR, ncP, b) for k in COMPONENTS}
    cc = {k: cell_averages(qcle[k], dR, dP, i0, j0, ncR, ncP, b) for k in COMPONENTS}
    cg = {k: cell_averages(grid4[k], dR, dP, i0, j0, ncR, ncP, b) for k in COMPONENTS}
    nB = pb_blocks.shape[0]
    pm = pb_blocks.mean(axis=0)
    ps = pb_blocks.std(axis=0, ddof=1) / np.sqrt(nB)
    pb = {k: pm[i] for i, k in enumerate(COMPONENTS)}
    pb_sem = {k: ps[i] for i, k in enumerate(COMPONENTS)}
    fo = {k: np.asarray(focused[k]) for k in COMPONENTS}
    focused_note = "on the PBME cells"
    if fo["A"].shape != (ncR, ncP):
        # the focused field of scripts/10b lives on a sub-window of the same
        # cells; it is exactly zero outside it when no focused trajectory left
        # that window
        pf10 = meta10["other_method_fields"]["pbme_field"]
        oi = int(pf10["index_origin"][0]) - i0
        oj = int(pf10["index_origin"][1]) - j0
        if (int(pf10["coarsen_block"]) != b or float(pf10["fraction_outside_window"]) != 0.0
                or oi % b or oj % b or oi < 0 or oj < 0
                or oi // b + fo["A"].shape[0] > ncR or oj // b + fo["A"].shape[1] > ncP):
            raise SystemExit("the focused field cannot be placed on the PBME cells")
        oi //= b
        oj //= b
        for k in COMPONENTS:
            z = np.zeros((ncR, ncP))
            z[oi:oi + fo[k].shape[0], oj:oj + fo[k].shape[1]] = fo[k]
            fo[k] = z
        focused_note = ("zero-padded from the scripts/10b window, which held every "
                        "focused trajectory")
    # the cells must hold the density, and the reduction must preserve it
    inwin = np.zeros((nR, nP), dtype=bool)
    inwin[i0:i0 + ncR * b, j0:j0 + ncP * b] = True
    outside = max(float(np.abs(f[k][~inwin]).sum() / np.abs(f[k]).sum())
                  for f in (quantum, qcle) for k in COMPONENTS)
    res_window = {"fraction_abs_density_outside_cells": outside}
    rec.check("phase.cells_hold_density", outside, 1e-3,
              "fraction of |rho| of the quantum and QCLE fields outside the cells")
    # PBME weight that fell outside the cells, as recorded by scripts/18
    r18 = meta18["results"]
    out_abs = r18.get("outside_abs_integrals")
    out_signed = r18.get("outside_signed_integrals")
    if out_abs is None:
        # an archive written before these sums were recorded is usable only
        # when no trajectory left the cells, where they vanish identically
        if float(r18["fraction_trajectories_outside_window"]) != 0.0:
            raise SystemExit(f"{j18} predates the record of the weight outside the cells")
        out_abs = out_signed = [0.0] * len(COMPONENTS)
    res_window["pbme_outside_abs_integrals"] = out_abs
    res_window["pbme_outside_signed_integrals"] = out_signed
    out_rel = [float(out_abs[a]) / float(pop_q[a]) for a in range(len(DIAGONAL))]
    res_window["pbme_outside_abs_over_population"] = out_rel
    rec.check("phase.pbme_cells_hold_trajectories", max(out_rel), 1e-2,
              "sum of |w c| over the PBME trajectories outside the cells, in "
              "units of the exact population of the state: the most by which "
              "they can change D_a")
    rec.check("phase.cell_reduction_integral",
              float(max(abs(cc[k].sum() * carea - qcle[k].sum() * area)
                        for k in COMPONENTS)), 1e-3,
              "cell averages times cell area against the integral over the "
              "whole grid; they differ by the density outside the cells")

    cellres = {}
    for k in COMPONENTS:
        nq = float(np.sqrt(np.sum(qc[k] ** 2)))
        nc_ = float(np.sqrt(np.sum(cc[k] ** 2)))
        floor = float(np.sqrt(np.sum(pb_sem[k] ** 2)))
        cellres[k] = {
            "rel_l2_grid_quantum": rel_l2(cg[k], qc[k]),
            "rel_l2_qcle_quantum": rel_l2(cc[k], qc[k]),
            "rel_l2_pbme_quantum": rel_l2(pb[k], qc[k]),
            "rel_l2_pbme_qcle": float(np.sqrt(np.sum((pb[k] - cc[k]) ** 2)) / nc_),
            "rel_l2_focused_quantum": rel_l2(fo[k], qc[k]),
            "pbme_noise_floor_vs_quantum": floor / nq,
            "pbme_noise_floor_vs_qcle": floor / nc_,
        }
    for a, k in enumerate(DIAGONAL):
        mq, mc = qc[k].sum(axis=0) * cells["dR"], cc[k].sum(axis=0) * cells["dR"]
        mp = pb[k].sum(axis=0) * cells["dR"]
        mps = pb_blocks[:, a].sum(axis=1) * cells["dR"]          # (B, ncP)
        mp_sem = mps.std(axis=0, ddof=1) / np.sqrt(nB)
        mf = fo[k].sum(axis=0) * cells["dR"]
        l1 = lambda m: float(np.abs(m - mq).sum() * cells["dP"] / pop_q[a])
        Pc = np.asarray(P[j0] + cells["dP"] * (np.arange(ncP) + 0.5))
        mq_m, mq_s = momentum_moments(mq, Pc)
        mc_m, mc_s = momentum_moments(mc, Pc)
        mp_m, mp_s = momentum_moments(mp, Pc)
        # jackknife over the blocks for the PBME moments
        tot = mps.sum(axis=0)
        jk = np.array([momentum_moments((tot - mps[i]) / (nB - 1), Pc) for i in range(nB)])
        jk_sem = np.sqrt((nB - 1) / nB * ((jk - jk.mean(axis=0)) ** 2).sum(axis=0))
        neg = lambda m: float(np.clip(-m, 0.0, None).sum() * cells["dP"] / pop_q[a])
        cellres[k].update({
            "momentum_negative_mass_quantum": neg(mq),
            "momentum_negative_mass_qcle": neg(mc),
            "momentum_negative_mass_pbme": neg(mp),
            "momentum_negative_mass_pbme_noise_floor":
                float(mp_sem.sum() * cells["dP"] / np.sqrt(2.0 * np.pi) / pop_q[a]),
            "momentum_min_qcle": float(mc.min()), "momentum_argmin_qcle": float(Pc[np.argmin(mc)]),
            "momentum_min_pbme": float(mp.min()), "momentum_argmin_pbme": float(Pc[np.argmin(mp)]),
            "momentum_peak_quantum_cells": float(mq.max()),
            "momentum_argmax_quantum": float(Pc[np.argmax(mq)]),
            "momentum_mean_quantum": mq_m, "momentum_std_quantum": mq_s,
            "momentum_mean_qcle": mc_m, "momentum_std_qcle": mc_s,
            "momentum_mean_pbme": mp_m, "momentum_std_pbme": mp_s,
            "momentum_mean_pbme_sem": float(jk_sem[0]),
            "momentum_std_pbme_sem": float(jk_sem[1]),
            "momentum_l1_qcle": l1(mc),
            "momentum_l1_grid": l1(cg[k].sum(axis=0) * cells["dR"]),
            "momentum_l1_pbme": l1(mp),
            "momentum_l1_focused": l1(mf),
            "momentum_l1_pbme_noise_floor":
                float(np.sqrt(2.0 / np.pi) * mp_sem.sum() * cells["dP"] / pop_q[a]),
            "negative_volume_quantum": negative_volume(qc[k], carea),
            "negative_volume_qcle": negative_volume(cc[k], carea),
            "negative_volume_pbme": negative_volume(pb[k], carea),
            "negative_volume_focused": negative_volume(fo[k], carea),
            "pbme_population_in_cells": float(pb[k].sum() * carea),
        })
    res["cells_comparison"] = cellres
    res["window"] = res_window
    res["focused_field"] = focused_note

    # ---- small arrays for the figures -----------------------------------
    stride = args.map_stride or max(1, int(np.ceil(max(ncR * b, ncP * b) / 600)))
    sl = (slice(i0, i0 + ncR * b, stride), slice(j0, j0 + ncP * b, stride))
    arrays = {"map_R": R[sl[0]], "map_P": P[sl[1]],
              "R_edges": np.asarray(R[i0] + cells["dR"] * np.arange(ncR + 1)),
              "P_edges": np.asarray(P[j0] + cells["dP"] * np.arange(ncP + 1)),
              "P_nodes": P, "R_nodes": R}
    for k in COMPONENTS:
        arrays[f"map_quantum_{k}"] = quantum[k][sl].astype(np.float32)
        arrays[f"map_qcle_{k}"] = qcle[k][sl].astype(np.float32)
        arrays[f"map_grid_{k}"] = grid4[k][sl].astype(np.float32)
        arrays[f"cells_grid_{k}"] = cg[k]
        arrays[f"cells_quantum_{k}"] = qc[k]
        arrays[f"cells_qcle_{k}"] = cc[k]
        arrays[f"cells_pbme_{k}"] = pb[k]
        arrays[f"cells_pbme_sem_{k}"] = pb_sem[k]
        arrays[f"cells_focused_{k}"] = fo[k]
    for a, k in enumerate(DIAGONAL):
        arrays[f"momentum_quantum_{k}"] = quantum[k].sum(axis=0) * dR
        arrays[f"momentum_qcle_{k}"] = qcle[k].sum(axis=0) * dR
        arrays[f"momentum_grid_{k}"] = grid4[k].sum(axis=0) * dR
        arrays[f"position_quantum_{k}"] = quantum[k].sum(axis=1) * dP
        arrays[f"position_qcle_{k}"] = qcle[k].sum(axis=1) * dP
        mps = pb_blocks[:, a].sum(axis=1) * cells["dR"]
        arrays[f"momentum_pbme_{k}"] = mps.mean(axis=0)
        arrays[f"momentum_pbme_sem_{k}"] = mps.std(axis=0, ddof=1) / np.sqrt(nB)
        xps = pb_blocks[:, a].sum(axis=2) * cells["dP"]
        arrays[f"position_pbme_{k}"] = xps.mean(axis=0)
        arrays[f"position_pbme_sem_{k}"] = xps.std(axis=0, ddof=1) / np.sqrt(nB)
    res["map_stride"] = stride
    res["n_pbme_blocks"] = nB
    rec.results = res

    stem = Path(args.out or raw / f"19_phase_space_comparison_P{P0i}")
    save_arrays(stem.with_suffix(".h5"), arrays, attrs=rec.payload()["metadata"])
    rec.write(stem.with_suffix(".json"))
    print(f"[19] P0={args.P0}: written {stem}.h5/.json; checks all passed: "
          f"{rec.all_passed}")
    for chk in rec.checks:
        print(f"  {'PASS' if chk['passed'] else 'FAIL'}  {chk['name']:36s} "
              f"{chk['value']:.3e}  (tol {chk['tolerance']:.1e})")
    for k in COMPONENTS:
        print(f"  {k:2s} solvers vs coefficients: grid relL2 {solvers['grid'][k]['rel_l2']:.2e} "
              f"max/peak {solvers['grid'][k]['max_over_peak']:.2e} | Strang relL2 "
              f"{solvers['strang'][k]['rel_l2']:.2e} max/peak {solvers['strang'][k]['max_over_peak']:.2e}")
    for k in COMPONENTS:
        f_, c_ = full[k], cellres[k]
        print(f"  {k:2s} full: QCLE-Q {f_['rel_l2_qcle_quantum']:.3f} | cells: "
              f"QCLE-Q {c_['rel_l2_qcle_quantum']:.3f}  PBME-Q "
              f"{c_['rel_l2_pbme_quantum']:.3f} (floor "
              f"{c_['pbme_noise_floor_vs_quantum']:.3f})  PBME-QCLE "
              f"{c_['rel_l2_pbme_qcle']:.3f}  focused-Q {c_['rel_l2_focused_quantum']:.3f}")
    for k in DIAGONAL:
        f_, c_ = full[k], cellres[k]
        print(f"  {k} negative volume Q {f_['negative_volume_quantum']:.2e} "
              f"QCLE {f_['negative_volume_qcle']:.2e}; momentum L1 QCLE "
              f"{f_['momentum_l1_qcle']:.3f} | cells PBME "
              f"{c_['momentum_l1_pbme']:.3f} (floor "
              f"{c_['momentum_l1_pbme_noise_floor']:.3f}) focused "
              f"{c_['momentum_l1_focused']:.3f}")
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
