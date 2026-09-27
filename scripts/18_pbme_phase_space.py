#!/usr/bin/env python3
r"""The projected-density PBME resolved in phase space at t = T.

Why this script exists
----------------------
``scripts/11`` reports the PBME diabatic populations as integrals of the
unbiased projected-density estimator,

.. math::
    \rho_W^{\nu'\nu}(R,P,T)=\sum_i w_i\,c_{\nu\nu'}(x_i(T))\,
    \delta(R-R_i(T))\,\delta(P-P_i(T)),

while the grid calculations (``scripts/07g``, ``scripts/10b``) store the whole
terminal fields.  To compare the descriptions on the phase-space density and
not only on its integrals, this script propagates the first chunks of the
``scripts/11`` ensemble again -- the same seeds, chunk size, importance
sampler and integrator, hence the same trajectories -- and resolves the sum by
phase-space cell instead of integrating it.

The cells are the ones on which ``scripts/10b`` binned the focused-PBME field:
blocks of ``coarsen_block`` x ``coarsen_block`` coefficient cells, with edges
on coefficient nodes, so that every grid field can be reduced onto exactly the
same cells.

Statistics
----------
Every chunk is split into equal contiguous blocks and the binned partial sums
of each block are archived as per-block estimates of the whole field.  The
blocks are independent and identically distributed, so the per-cell standard
error, and the noise floor of any distance built from the field, are computed
afterwards without propagating anything twice.

Checks
------
* reproduction: the total of each chunk, over all trajectories, equals the
  seed-replicate population archived by ``scripts/11`` for the same seed;
* the binned field integrates to the in-window total;
* the mapping radius is conserved by the integrator;
* the weight outside the window is negligible.

Usage
-----
    python scripts/18_pbme_phase_space.py --P0 30
    python scripts/18_pbme_phase_space.py --P0 70 --n-chunks 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqcle.benchmark_regimes import regime_for                       # noqa: E402
from mqcle.io import RunRecorder, load_config, save_arrays           # noqa: E402
from mqcle.models import build_model                                 # noqa: E402
from mqcle.pbme import (pack_state, symmetric_rotation_step,         # noqa: E402
                        unpack_state)
from mqcle.projector import wigner_aa_dagger_all                     # noqa: E402
from mqcle.propagation import sample_projected_ensemble              # noqa: E402
from mqcle.types import TraceConvention                              # noqa: E402

COMPONENTS = ("A", "C", "bR", "bI")   # rho^{11}, rho^{22}, Re rho^{12}, Im rho^{12}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--P0", type=float, required=True)
    p.add_argument("--config", default="tully_default.yaml")
    p.add_argument("--n-chunks", type=int, default=None,
                   help="number of scripts/11 chunks to reproduce (default: all)")
    p.add_argument("--blocks-per-chunk", type=int, default=20)
    p.add_argument("--pbme-archive", default=None)
    p.add_argument("--window-archive", default=None)
    p.add_argument("--extend-to", type=float, nargs=4, default=None,
                   metavar=("R_LO", "R_HI", "P_LO", "P_HI"),
                   help="enlarge the scripts/10b window by whole cells until it "
                        "covers this box (clipped to the grid)")
    p.add_argument("--out", default=None, help="output stem (no extension)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    P0i = int(round(args.P0))
    raw = Path("results/raw")
    pb_path = Path(args.pbme_archive or raw / f"11_pbme_projected_P{P0i}.json")
    win_path = Path(args.window_archive
                    or raw / f"10b_qcle_grid_crosscheck_P{P0i}.json")
    pb11 = json.loads(pb_path.read_text())
    win = json.loads(win_path.read_text())

    # ---- the scripts/11 run being reproduced -----------------------------
    c11 = pb11["metadata"]["config"]
    cfg = load_config(args.config)
    hbar, mass = float(cfg["hbar"]), float(cfg["mass"])
    convention = TraceConvention(cfg["trace_convention"])
    model_cfg = dict(cfg["model"])
    if dict(c11["model"]) != model_cfg:
        raise SystemExit(f"model of {pb_path} differs from {args.config}")
    model = build_model(model_cfg, mass=mass)
    ic = cfg["initial_conditions"]
    init_state = int(ic["init_state"])
    P0 = float(c11["P0"])
    R0 = float(c11["R0"])
    sigma_R = float(c11["sigma_R"])
    T = float(c11["T"])
    dt = float(c11["dt"])
    n_steps = int(c11["n_steps"])
    registered = regime_for(P0)
    if registered is not None and abs(registered.T - T) > 1e-12:
        raise SystemExit(f"T={T} of {pb_path} is not the registered {registered.T}")
    if abs(n_steps * dt - T) > 1e-9:
        raise SystemExit("T is not an integer number of steps")
    chunk = int(c11["chunk"])
    n_chunks_11 = int(c11["n_chunks"])
    n_seeds = int(c11["n_seeds"])
    scheme = str(c11["scheme"])
    base_seed = int(c11["seed"])
    chunks_per_seed = n_chunks_11 // n_seeds
    n_run = int(args.n_chunks or n_chunks_11)
    if not 1 <= n_run <= n_chunks_11:
        raise SystemExit(f"--n-chunks must lie in [1, {n_chunks_11}]")
    bpc = int(args.blocks_per_chunk)
    if chunk % bpc:
        raise SystemExit("--blocks-per-chunk must divide the chunk size")
    block_size = chunk // bpc

    # ---- the cells: the focused-field window of scripts/10b ---------------
    pf = win["results"]["other_method_fields"]["pbme_field"]
    g = win["results"]["runs"]["production"]["grid"]
    n_R, n_P = int(g["n_R"]), int(g["n_P"])
    dR = (float(g["R_max"]) - float(g["R_min"])) / n_R
    dP = (float(g["P_max"]) - float(g["P_min"])) / n_P
    i0, j0 = (int(v) for v in pf["index_origin"])
    i1, j1 = (int(v) for v in pf["index_end"])
    blk = int(pf["coarsen_block"])
    nRc, nPc = (i1 - i0) // blk, (j1 - j0) // blk
    if nRc * blk != i1 - i0 or nPc * blk != j1 - j0:
        raise SystemExit("the archived window is not a whole number of blocks")
    window_source = "scripts/10b focused-PBME window"
    if args.extend_to is not None:
        # grow by whole cells, so the scripts/10b cells remain a subset
        R_min, P_min = float(g["R_min"]), float(g["P_min"])
        rlo, rhi, plo, phi = args.extend_to
        while i0 - blk >= 0 and R_min + i0 * dR > rlo:
            i0 -= blk
        while i1 + blk <= n_R and R_min + i1 * dR < rhi:
            i1 += blk
        while j0 - blk >= 0 and P_min + j0 * dP > plo:
            j0 -= blk
        while j1 + blk <= n_P and P_min + j1 * dP < phi:
            j1 += blk
        nRc, nPc = (i1 - i0) // blk, (j1 - j0) // blk
        window_source += f", enlarged to cover R in [{rlo:g}, {rhi:g}], P in [{plo:g}, {phi:g}]"
    R_lo = float(g["R_min"]) + i0 * dR
    P_lo = float(g["P_min"]) + j0 * dP
    dRc, dPc = blk * dR, blk * dP
    cell = dRc * dPc

    rec = RunRecorder(script=Path(__file__).name, seed=base_seed, config={
        **{k: v for k, v in cfg.items() if not k.startswith("_")},
        "P0": P0, "R0": R0, "sigma_R": sigma_R, "T": T, "dt": dt,
        "n_steps": n_steps, "scheme": scheme, "chunk": chunk,
        "n_chunks_reproduced": n_run, "n_chunks_of_source": n_chunks_11,
        "n_seeds_of_source": n_seeds, "blocks_per_chunk": bpc,
        "pbme_archive": str(pb_path), "window_archive": str(win_path)})
    print(f"[18] P0={P0} T={T} dt={dt} n_steps={n_steps}; reproducing "
          f"{n_run}/{n_chunks_11} chunks of {chunk:,} ({scheme}); cells "
          f"{nRc}x{nPc} of {dRc:.4g} x {dPc:.4g}", flush=True)

    reps = {int(r["seed_group"]): r for r in pb11["results"]["seed_replicates"]}
    n_blocks = n_run * bpc
    fields = np.zeros((n_blocks, len(COMPONENTS), nRc, nPc))
    totals = np.zeros((n_blocks, 2, 2), dtype=np.complex128)
    repro, seeds, radius_drift, frac_w_out, frac_n_out = [], [], [], [], []
    direct_inside = np.zeros(len(COMPONENTS))
    outside_signed = np.zeros(len(COMPONENTS))
    outside_abs = np.zeros(len(COMPONENTS))
    t_start = time.perf_counter()
    for ci in range(n_run):
        seed_group = ci // chunks_per_seed
        seed = base_seed + 1000 * seed_group + ci          # exactly as scripts/11
        rng = np.random.default_rng(seed)
        ens = sample_projected_ensemble(rng, chunk, nstates=model.nstates,
                                        init_state=init_state, R0=R0, P0=P0,
                                        sigma_R=sigma_R, hbar=hbar, seed=seed,
                                        scheme=scheme)
        state = pack_state(ens.x, ens.R, ens.P)
        rad0 = np.sum(np.asarray(ens.x, float) ** 2, axis=-1)
        for _ in range(n_steps):
            state = symmetric_rotation_step(state, dt, model, hbar=hbar,
                                            convention=convention)
        x, R, P = unpack_state(state, model.nstates, model.nbath)
        R, P = np.ravel(R), np.ravel(P)
        radius_drift.append(float(np.max(np.abs(np.sum(x ** 2, axis=-1) - rad0))))
        w = np.asarray(ens.weights, dtype=np.complex128)
        c = wigner_aa_dagger_all(x, hbar, model.nstates)          # (m, N, N)
        wc = w[:, None, None] * c
        # rho_W^{nu' nu} = sum_i w_i c_{nu nu'}(x_i): the transpose of the
        # c-contraction, as in scripts/11 and scripts/10b
        vals = (np.real(wc[:, 0, 0]), np.real(wc[:, 1, 1]),
                np.real(wc[:, 1, 0]), np.imag(wc[:, 1, 0]))
        tot = wc.sum(axis=0).T
        if chunks_per_seed == 1 and seed_group in reps:
            ref = np.asarray(reps[seed_group]["final_populations"], float)
            repro.append(float(np.max(np.abs(np.real(np.diag(tot)) - ref))))
        seeds.append(seed)
        iR = np.floor((R - R_lo) / dRc).astype(np.int64)
        iP = np.floor((P - P_lo) / dPc).astype(np.int64)
        inside = (iR >= 0) & (iR < nRc) & (iP >= 0) & (iP < nPc)
        frac_n_out.append(float(1.0 - inside.mean()))
        frac_w_out.append(float(np.sum(np.abs(w[~inside])) / np.sum(np.abs(w))))
        flat = iR * nPc + iP
        # the in-window sum taken directly from the trajectories, to check
        # the binning (edges, cell order and area) independently
        direct_inside += np.array([v[inside].sum() for v in vals]) / n_run
        outside_signed += np.array([v[~inside].sum() for v in vals]) / n_run
        outside_abs += np.array([np.abs(v[~inside]).sum() for v in vals]) / n_run
        for b in range(bpc):
            sl = slice(b * block_size, (b + 1) * block_size)
            ins = inside[sl]
            fb = flat[sl][ins]
            kb = ci * bpc + b
            for k, v in enumerate(vals):
                fields[kb, k] = (np.bincount(fb, weights=v[sl][ins],
                                             minlength=nRc * nPc)
                                 .reshape(nRc, nPc) / cell) * bpc
            totals[kb] = wc[sl].sum(axis=0).T * bpc
        el = time.perf_counter() - t_start
        print(f"[18]   chunk {ci + 1}/{n_run} (seed {seed}) populations "
              f"{np.real(np.diag(tot))}  {el:.0f}s elapsed, "
              f"~{el / (ci + 1) * (n_run - ci - 1):.0f}s left", flush=True)

    mean = fields.mean(axis=0)
    sem = fields.std(axis=0, ddof=1) / np.sqrt(n_blocks)
    tot_mean = totals.mean(axis=0)
    binned = np.array([mean[k].sum() * cell for k in range(len(COMPONENTS))])
    # the in-window total of the same estimator, from the per-block fields
    in_window = np.real(np.array([tot_mean[0, 0], tot_mean[1, 1],
                                  np.real(tot_mean[0, 1]), np.imag(tot_mean[0, 1])]))

    results = {
        "estimator": "projected-density PBME, plain (no control variate)",
        "components": list(COMPONENTS),
        "component_meaning": ["rho_W^{11}", "rho_W^{22}", "Re rho_W^{12}",
                              "Im rho_W^{12}"],
        "n_samples": n_run * chunk, "n_blocks": n_blocks,
        "block_size": block_size, "seeds": seeds,
        "cells": {"n_R": nRc, "n_P": nPc, "dR": dRc, "dP": dPc,
                  "R_lo": R_lo, "P_lo": P_lo, "coarsen_block": blk,
                  "index_origin": [i0, j0], "index_end": [i1, j1],
                  "grid": {"R_min": float(g["R_min"]), "R_max": float(g["R_max"]),
                           "n_R": n_R, "P_min": float(g["P_min"]),
                           "P_max": float(g["P_max"]), "n_P": n_P},
                  "edges": "cell k spans [R_lo + k dR, R_lo + (k+1) dR), "
                           "likewise in P; R_lo and P_lo are coefficient nodes",
                  "source": window_source},
        "total_populations": np.real(np.diag(tot_mean)).tolist(),
        "total_coherence": [float(np.real(tot_mean[0, 1])),
                            float(np.imag(tot_mean[0, 1]))],
        "binned_integrals": binned.tolist(),
        "fraction_trajectories_outside_window": max(frac_n_out),
        "outside_signed_integrals": outside_signed.tolist(),
        "outside_abs_integrals": outside_abs.tolist(),
        "fraction_abs_weight_outside_window": max(frac_w_out),
        "reproduction_max_abs_difference": (max(repro) if repro else None),
        "mean_trajectories_per_cell": float(n_run * chunk * (1.0 - max(frac_n_out))
                                            / (nRc * nPc)),
        "wall_time_s": time.perf_counter() - t_start,
    }
    rec.results = results
    if repro:
        rec.check("pbme_field.reproduces_scripts11_seed_replicates", max(repro),
                  1e-10, "chunk totals against the archived seed-replicate "
                         "populations of the same seeds")
    else:
        rec.check("pbme_field.reproduces_scripts11_seed_replicates",
                  float("nan"), 0.0, "no seed replicate matches one chunk")
    # the binned field must integrate to what the trajectories inside carry
    rec.check("pbme_field.binned_integral_consistency",
              float(np.max(np.abs(binned - direct_inside))), 1e-12,
              "integral of the pooled binned field against the direct sum of "
              "w c over the trajectories inside the cells")
    rec.check("pbme_field.mapping_radius_drift", max(radius_drift), 1e-9,
              "max |x(T)|^2 - |x(0)|^2 over the ensemble")
    rec.check("pbme_field.weight_outside_window",
              max(frac_w_out), 1e-3,
              "fraction of total |weight| outside the cells")
    results["binned_minus_total"] = (binned - in_window).tolist()

    stem = Path(args.out or raw / f"18_pbme_phase_space_P{P0i}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    save_arrays(stem.with_suffix(".h5"), {
        "field_mean": mean, "field_sem": sem, "block_fields": fields,
        "block_totals_re": np.real(totals), "block_totals_im": np.imag(totals),
        "R_edges": R_lo + dRc * np.arange(nRc + 1),
        "P_edges": P_lo + dPc * np.arange(nPc + 1)},
        attrs=rec.payload()["metadata"])
    rec.write(stem.with_suffix(".json"))
    print(f"[18] total populations {results['total_populations']}, binned "
          f"{binned[:2]}, outside |w| {max(frac_w_out):.2e}; written to "
          f"{stem}.h5/.json; checks all passed: {rec.all_passed}", flush=True)
    for chk in rec.checks:
        print(f"  {'PASS' if chk['passed'] else 'FAIL'}  {chk['name']:48s} "
              f"{chk['value']:.3e}  (tol {chk['tolerance']:.1e})")
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
