#!/usr/bin/env python3
r"""The PBME **operator** benchmark: unbiased projected-density estimator.

Why this script exists
----------------------
The practical focused trajectory PBME of ``scripts/07`` differs from the
coefficient QCLE solution by four things at once,

.. math::
    \Delta_{\rm current}=\Delta_{\rm operator}+\Delta_{\rm focused}
                        +\Delta_{\rm MC}+\Delta_{\rm integrator},

so it cannot be called *the cost of discarding* :math:`\mathcal A'` without
qualification.  This script computes the first term on its own.  Mapping
variables are drawn from an importance density and carry the signed weights of
the projected density; the PBME flow is Liouvillian, so

.. math:: \rho_W^{\nu'\nu}(t)=\sum_iw_i\,c_{\nu\nu'}(x_i(t))

is unbiased for the PBME density observable.  The remaining two numerical terms
are then *measured*: this script quantifies :math:`\Delta_{\rm MC}` and
``scripts/11c`` quantifies :math:`\Delta_{\rm integrator}`.

Statistics
----------
The ensemble is propagated in chunks so that memory stays bounded, and each
chunk records per-block partial sums.  Because the blocks are independent and
identically distributed, the sample-size study, the seed replicates, the
standard errors and the confidence intervals are all computed from those
partial sums afterwards -- nothing is propagated twice.

Two estimators are reported for every observable:

``plain``
    :math:`\hat\rho(t)`, the direct weighted average.

``cv``
    :math:`\hat\rho(t)-\hat\rho(0)+\rho_W(0)`, which subtracts the *same
    samples'* error at :math:`t=0`, where the answer is known exactly.  This is
    a control variate, not a normalisation: it is unbiased for every
    :math:`N_s` because :math:`\mathbb E[\hat\rho(0)]=\rho_W(0)` exactly.  It
    helps in proportion to how correlated :math:`\hat\rho(t)` and
    :math:`\hat\rho(0)` are, which is a property of the dynamics and is
    therefore reported rather than assumed.

Weights are never divided by their stochastic sum.

Usage
-----
    python scripts/11_pbme_projected.py --P0 40 --n-samples 1200000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqcle.io import RunRecorder, load_config                        # noqa: E402
from mqcle.benchmark_regimes import regime_for                       # noqa: E402
from mqcle.models import build_model                                 # noqa: E402
from mqcle.propagation import (block_statistics,                     # noqa: E402
                               propagate_pbme_ensemble,
                               propagate_pbme_projected_ensemble,
                               sample_matched_ensemble,
                               sample_projected_ensemble)
from mqcle.types import TraceConvention                              # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="tully_default.yaml")
    p.add_argument("--P0", type=float, default=40.0)
    p.add_argument("--R0", type=float, default=None)
    p.add_argument("--sigma-R", type=float, default=None)
    p.add_argument("--coupling", type=float, default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--T", type=float, default=None)
    p.add_argument("--n-save", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=1_200_000)
    p.add_argument("--chunk", type=int, default=200_000)
    p.add_argument("--n-seeds", type=int, default=6)
    p.add_argument("--blocks-per-chunk", type=int, default=40)
    p.add_argument("--focused-samples", type=int, default=None)
    p.add_argument("--skip-focused", action="store_true")
    p.add_argument("--scheme", default="sign_optimal",
                   choices=("sign_optimal", "wigner_seo"),
                   help="importance proposal.  'sign_optimal' draws from "
                        "phi|Pi|/Z so the weight is bounded (about eleven "
                        "times less variance); 'wigner_seo' draws from phi "
                        "itself.  Both are unbiased; running both is the "
                        "cross-check.")
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def _stats(blocks: np.ndarray, n_blocks: int | None = None) -> dict:
    st = block_statistics(blocks, n_blocks)
    return {k: (np.real(v).tolist() if isinstance(v, np.ndarray) else v)
            for k, v in st.items()}


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    ic = cfg["initial_conditions"]
    hbar, mass = float(cfg["hbar"]), float(cfg["mass"])
    convention = TraceConvention(cfg["trace_convention"])
    if args.coupling is not None:
        cfg["model"] = {**cfg["model"], "C": float(args.coupling)}
    model = build_model(cfg["model"], mass=mass)

    P0 = float(args.P0)
    R0 = float(args.R0 if args.R0 is not None else ic["R0"])
    sigma_R = float(args.sigma_R if args.sigma_R is not None
                    else hbar / (2.0 * (P0 / 20.0)))
    init_state = int(ic["init_state"])
    registered = regime_for(P0)
    T = float(args.T if args.T is not None
              else (registered.T if registered is not None
                    else 2.0 * mass * abs(R0) / P0))
    dt = float(args.dt)
    n_steps = int(round(T / dt))

    chunk = min(int(args.chunk), int(args.n_samples))
    if args.n_samples % chunk:
        raise SystemExit("--chunk must divide --n-samples")
    n_chunks = args.n_samples // chunk
    if n_chunks % args.n_seeds:
        raise SystemExit(f"--n-seeds ({args.n_seeds}) must divide the chunk "
                         f"count ({n_chunks})")
    chunks_per_seed = n_chunks // args.n_seeds
    base_seed = int(cfg["seed"])

    rec = RunRecorder(script=Path(__file__).name, seed=base_seed, config={
        **{k: v for k, v in cfg.items() if not k.startswith("_")},
        "P0": P0, "R0": R0, "sigma_R": sigma_R, "T": T, "dt": dt,
        "n_steps": n_steps, "n_save": args.n_save,
        "n_samples": int(args.n_samples), "chunk": chunk,
        "n_chunks": n_chunks, "n_seeds": int(args.n_seeds),
        "blocks_per_chunk": int(args.blocks_per_chunk),
        "scheme": args.scheme})
    print(f"[11] P0={P0} T={T} dt={dt} n_steps={n_steps}  "
          f"N_s={args.n_samples:,} in {n_chunks} chunks of {chunk:,} "
          f"({args.n_seeds} seed groups, scheme={args.scheme})", flush=True)

    rho0 = np.zeros((model.nstates, model.nstates), dtype=np.complex128)
    rho0[init_state, init_state] = 1.0

    all_rho, all_ad, seed_of_block = [], [], []
    diag = {"trace_t0": [], "trace_drift": [], "energy_drift": [],
            "radius_drift": [], "fraction_negative_weight": [],
            "per_sample_sd_rho00_t0": []}
    times = None
    t_start = time.perf_counter()
    for ci in range(n_chunks):
        seed_group = ci // chunks_per_seed
        seed = base_seed + 1000 * seed_group + ci
        rng = np.random.default_rng(seed)
        ens = sample_projected_ensemble(rng, chunk, nstates=model.nstates,
                                        init_state=init_state, R0=R0, P0=P0,
                                        sigma_R=sigma_R, hbar=hbar, seed=seed,
                                        scheme=args.scheme)
        h = propagate_pbme_projected_ensemble(
            ens, model, dt, n_steps, hbar=hbar, convention=convention,
            n_save=args.n_save, n_blocks=int(args.blocks_per_chunk))
        times = np.asarray(h.times)
        all_rho.append(h.arrays["block_rho"])
        all_ad.append(h.arrays["block_adiabatic"])
        seed_of_block += [seed_group] * int(args.blocks_per_chunk)
        tr = np.real([r["trace"] for r in h.records])
        diag["trace_t0"].append(float(tr[0]))
        diag["trace_drift"].append(float(np.max(np.abs(tr - tr[0]))))
        diag["energy_drift"].append(float(h.records[-1]["energy_drift"]))
        diag["radius_drift"].append(float(h.records[-1]["mapping_radius_drift"]))
        diag["fraction_negative_weight"].append(
            float(ens.meta["signed_weight_fraction_negative"]))
        # the per-sample standard deviation sets the sample budget, so it is
        # archived rather than inferred back out of the standard error
        b0 = np.real(np.diagonal(h.arrays["block_rho"][0], axis1=-2, axis2=-1))
        nb = b0.shape[0]
        diag["per_sample_sd_rho00_t0"].append(
            float(np.std(nb * b0[:, 0], ddof=1) * np.sqrt(chunk / nb)))
        el = time.perf_counter() - t_start
        print(f"[11]   chunk {ci + 1}/{n_chunks} (seed group {seed_group}) "
              f"{el:.0f}s elapsed, ~{el / (ci + 1) * (n_chunks - ci - 1):.0f}s left",
              flush=True)

    # every chunk's blocks carry the same 1/chunk factor, so dividing by the
    # chunk count makes the sum over all blocks the pooled 1/N_s estimator
    B = np.concatenate(all_rho, axis=1) / n_chunks
    A = np.concatenate(all_ad, axis=1) / n_chunks
    seed_of_block = np.asarray(seed_of_block)
    Btot = B.shape[1]
    block_size = args.n_samples // Btot

    cvB = B - B[0][None] + rho0[None, None] / Btot
    cvA = A - A[0][None] + np.real(np.diagonal(rho0))[None, None] / Btot

    results = {
        "branch": "pbme_projected",
        "scheme": args.scheme,
        "sampling": ("phi|Pi|/Z importance sampling of the projected density"
                     if args.scheme == "sign_optimal"
                     else "phi importance sampling of the projected density"),
        "readout": "rho_W_from_ensemble",
        "weights_renormalised_by_their_sum": False,
        "times": times.tolist(),
        "n_samples": int(args.n_samples), "n_blocks": int(Btot),
        "block_size": int(block_size), "n_seeds": int(args.n_seeds),
        "diagnostics": {k: {"min": float(np.min(v)), "max": float(np.max(v)),
                            "per_chunk": v} for k, v in diag.items()},
        "diabatic": {"plain": _stats(np.real(np.diagonal(B, axis1=-2, axis2=-1))),
                     "cv": _stats(np.real(np.diagonal(cvB, axis1=-2, axis2=-1)))},
        "adiabatic": {"plain": _stats(A), "cv": _stats(cvA)},
        "coherence": {"re": _stats(np.real(B[..., 0, 1])),
                      "im": _stats(np.imag(B[..., 0, 1]))},
    }

    # -- sample-size convergence, from nested block subsets ---------------
    conv = []
    # Include the reviewer's requested 5k, 10k, 20k, 50k and 100k points
    # whenever the block size permits them, plus powers of two and production.
    requested = (5_000, 10_000, 20_000, 50_000, 100_000)
    ks = {1, Btot}
    k = 1
    while k <= Btot:
        ks.add(k)
        k *= 2
    ks.update(n // block_size for n in requested
              if n % block_size == 0 and 1 <= n // block_size <= Btot)
    for k in sorted(ks):
        for est, blocks in (("plain", B), ("cv", cvB)):
            st = block_statistics(np.real(np.diagonal(blocks, axis1=-2, axis2=-1)), k)
            conv.append({"estimator": est, "n_samples": int(k * block_size),
                         "n_blocks": int(k),
                         "final_populations": np.real(st["mean"][-1]).tolist(),
                         "sem": np.real(st["sem"][-1]).tolist()})
    results["sample_size_convergence"] = conv

    # -- independent seed replicates --------------------------------------
    reps = []
    for g in range(int(args.n_seeds)):
        sel = seed_of_block == g
        nb = int(sel.sum())
        st = block_statistics(
            np.real(np.diagonal(B[:, sel] * Btot / nb, axis1=-2, axis2=-1)))
        stc = block_statistics(
            np.real(np.diagonal(cvB[:, sel] * Btot / nb, axis1=-2, axis2=-1)))
        reps.append({"seed_group": g, "n_samples": int(nb * block_size),
                     "final_populations": np.real(st["mean"][-1]).tolist(),
                     "sem": np.real(st["sem"][-1]).tolist(),
                     "final_populations_cv": np.real(stc["mean"][-1]).tolist(),
                     "sem_cv": np.real(stc["sem"][-1]).tolist()})
    results["seed_replicates"] = reps
    across = np.array([r["final_populations"] for r in reps])
    across_cv = np.array([r["final_populations_cv"] for r in reps])
    results["across_seed"] = {
        "mean": across.mean(0).tolist(),
        "sd": across.std(0, ddof=1).tolist(),
        "sem": (across.std(0, ddof=1) / np.sqrt(len(reps))).tolist(),
        "mean_cv": across_cv.mean(0).tolist(),
        "sd_cv": across_cv.std(0, ddof=1).tolist(),
        "sem_cv": (across_cv.std(0, ddof=1) / np.sqrt(len(reps))).tolist()}

    # -- the practical focused branch, for the E_focused layer -------------
    if not args.skip_focused:
        nf = int(args.focused_samples or cfg["mapping_sampling"]["n_samples"])
        rngf = np.random.default_rng(base_seed)
        ensf = sample_matched_ensemble(rngf, nf, nstates=model.nstates,
                                       init_state=init_state, R0=R0, P0=P0,
                                       sigma_R=sigma_R, hbar=hbar)
        t0 = time.perf_counter()
        hf = propagate_pbme_ensemble(ensf, model, dt, n_steps, hbar=hbar,
                                     convention=convention, n_save=args.n_save)
        results["focused"] = {
            "branch": "pbme_focused", "n_samples": nf,
            "wall_time_s": time.perf_counter() - t0,
            "times": list(hf.times),
            "populations": [np.asarray(r["populations"]).tolist()
                            for r in hf.records],
            "final_populations": np.asarray(hf.records[-1]["populations"]).tolist()}
        print(f"[11] focused branch ({nf:,} trajectories) final pops "
              f"{results['focused']['final_populations']}", flush=True)

    # -- checks -----------------------------------------------------------
    st = block_statistics(np.real(np.diagonal(B, axis1=-2, axis2=-1)))
    stc = block_statistics(np.real(np.diagonal(cvB, axis1=-2, axis2=-1)))
    fin, sem = np.real(st["mean"][-1]), np.real(st["sem"][-1])
    finc, semc = np.real(stc["mean"][-1]), np.real(stc["sem"][-1])
    results["final"] = {
        "plain": {"populations": fin.tolist(), "sem": sem.tolist(),
                  "ci95": [[float(m - 1.96 * s), float(m + 1.96 * s)]
                           for m, s in zip(fin, sem)]},
        "cv": {"populations": finc.tolist(), "sem": semc.tolist(),
               "ci95": [[float(m - 1.96 * s), float(m + 1.96 * s)]
                        for m, s in zip(finc, semc)]},
        "variance_reduction_factor": [float(a / b) if b else float("inf")
                                      for a, b in zip(sem, semc)]}
    rec.check("pbme.trace_drift", float(np.max(diag["trace_drift"])), 1e-10,
              "sum_nu c_nu nu depends on the mapping radius alone, which the "
              "frozen-R rotation preserves exactly, so this is round-off")
    rec.check("pbme.radius_drift", float(np.max(diag["radius_drift"])), 1e-9,
              "max |x(t)|^2 - |x(0)|^2 over the ensemble")
    sem0 = float(np.max(np.real(st["sem"][0])))
    rec.check("pbme.estimator_unbiased_at_t0",
              float(np.max(np.abs(np.real(np.diagonal(B[0].sum(0)))
                                  - np.real(np.diagonal(rho0))))),
              max(6.0 * sem0, 1e-15),
              "|rho_hat(0) - rho_W(0)| against six of its own standard errors")

    print("[11] final diabatic populations")
    print(f"       plain {fin}  +/- {1.96 * sem} (95%)")
    print(f"       cv    {finc}  +/- {1.96 * semc} (95%)")
    print(f"       variance reduction {results['final']['variance_reduction_factor']}")

    rec.results = results
    tag = f"P{int(round(P0))}{args.tag}"
    out = Path(args.out or f"results/raw/11_pbme_projected_{tag}.json")
    rec.write(out)
    print(f"[11] written to {out};  checks all passed: {rec.all_passed}")
    for c in rec.checks:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['name']:34s} "
              f"{c['value']:.3e}  (tol {c['tolerance']:.1e})")
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
