#!/usr/bin/env python3
r"""Time-step convergence of the projected-density PBME branch (PART IV).

The coefficient solver has time-step, grid and box convergence evidence.  The
trajectory branch had none, so its difference from the coefficient solution
could not be attributed to the operator truncation rather than to the
integrator.  This script supplies it.

Design
------
The three time steps are run **on the same initial ensemble**: same seed, same
:math:`N_s`, same sampled points.  The difference between two of them is then a
*paired* quantity whose Monte-Carlo error largely cancels, which is what makes
the required comparison -- integration error below sampling uncertainty --
meaningful rather than a comparison of two noisy numbers.  Both are reported:
the paired difference with its own standard error across blocks, and the
standard error of the single-run estimate for scale.

Measured, for each pair of time steps: the change in the final diabatic
populations; the :math:`L^2` difference of the whole population time series;
the energy drift and mapping-radius drift of each run; and reversibility.

The mapping radius is preserved exactly by the frozen-:math:`R` rotation, so a
nonzero radius drift is round-off; the energy is *not* a conserved quantity of
the split step and its drift is a genuine integrator diagnostic.

Usage
-----
    python scripts/11c_pbme_timestep_convergence.py --P0 40 20 100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqcle.io import RunRecorder, load_config                        # noqa: E402
from mqcle.models import build_model                                 # noqa: E402
from mqcle.pbme import pack_state, symmetric_rotation_step           # noqa: E402
from mqcle.propagation import (block_statistics,                     # noqa: E402
                               propagate_pbme_projected_ensemble,
                               sample_projected_ensemble)
from mqcle.types import TraceConvention                              # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="tully_default.yaml")
    p.add_argument("--P0", type=float, nargs="+", default=[40.0])
    p.add_argument("--R0", type=float, default=None)
    p.add_argument("--dt", type=float, nargs="+", default=[0.5, 0.25, 0.125])
    p.add_argument("--n-samples", type=int, default=50_000)
    p.add_argument("--n-blocks", type=int, default=25)
    p.add_argument("--n-save", type=int, default=10)
    p.add_argument("--reversibility-steps", type=int, default=400)
    p.add_argument("--out", default="results/raw/11c_pbme_timestep_convergence.json")
    return p.parse_args(argv)


def _reversibility(ens, model, dt, n_steps, hbar, convention) -> float:
    """Forward then backward: the scheme is palindromic, so this is round-off."""
    s0 = pack_state(ens.x[:2000], ens.R[:2000], ens.P[:2000])
    s = s0.copy()
    for _ in range(n_steps):
        s = symmetric_rotation_step(s, dt, model, hbar=hbar, convention=convention)
    for _ in range(n_steps):
        s = symmetric_rotation_step(s, -dt, model, hbar=hbar, convention=convention)
    scale = float(np.max(np.abs(s0)))
    return float(np.max(np.abs(s - s0)) / max(scale, 1e-300))


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    ic = cfg["initial_conditions"]
    hbar, mass = float(cfg["hbar"]), float(cfg["mass"])
    convention = TraceConvention(cfg["trace_convention"])
    model = build_model(cfg["model"], mass=mass)
    init_state = int(ic["init_state"])
    base_seed = int(cfg["seed"])

    rec = RunRecorder(script=Path(__file__).name, seed=base_seed, config={
        **{k: v for k, v in cfg.items() if not k.startswith("_")},
        "P0": list(args.P0), "dt": list(args.dt),
        "n_samples": int(args.n_samples), "n_blocks": int(args.n_blocks),
        "matched_initial_samples": True})

    out: dict = {"regimes": {}, "dt": list(args.dt),
                 "n_samples": int(args.n_samples),
                 "design": "identical initial ensemble at every time step, so "
                           "the pairwise differences are correlated and their "
                           "own standard error is what the integration error "
                           "must be compared against"}
    worst_ratio = 0.0
    for P0 in args.P0:
        R0 = float(args.R0 if args.R0 is not None else ic["R0"])
        sigma_R = hbar / (2.0 * (P0 / 20.0))
        T = 2.0 * mass * abs(R0) / P0
        rng = np.random.default_rng(base_seed + int(P0))
        ens = sample_projected_ensemble(rng, int(args.n_samples),
                                        nstates=model.nstates,
                                        init_state=init_state, R0=R0, P0=P0,
                                        sigma_R=sigma_R, hbar=hbar,
                                        seed=base_seed + int(P0))
        runs = {}
        # one physical save interval shared by every time step, so that the
        # pairwise differences below compare the same instants.  Getting this
        # wrong does not produce a small error: it silently differences t=40
        # against t=20 and reports the dynamics as an integrator defect.
        save_interval = float(args.n_save) * float(max(args.dt))
        for dt in args.dt:
            n_steps = int(round(T / dt))
            n_save = max(1, int(round(save_interval / dt)))
            t0 = time.perf_counter()
            h = propagate_pbme_projected_ensemble(
                ens, model, dt, n_steps, hbar=hbar, convention=convention,
                n_save=n_save, n_blocks=int(args.n_blocks))
            runs[dt] = {
                "times": list(h.times), "n_steps": n_steps,
                "blocks": h.arrays["block_rho"],
                "energy_drift": float(h.records[-1]["energy_drift"]),
                "mapping_radius_drift":
                    float(h.records[-1]["mapping_radius_drift"]),
                "reversibility": _reversibility(ens, model, dt,
                                                int(args.reversibility_steps),
                                                hbar, convention),
                "wall_time_s": round(time.perf_counter() - t0, 1)}
            st = block_statistics(np.real(np.diagonal(runs[dt]["blocks"],
                                                      axis1=-2, axis2=-1)))
            runs[dt]["final_populations"] = np.real(st["mean"][-1]).tolist()
            runs[dt]["sem"] = np.real(st["sem"][-1]).tolist()
            print(f"[11c] P0={P0} dt={dt}: final {runs[dt]['final_populations']} "
                  f"+/- {runs[dt]['sem']}  E drift {runs[dt]['energy_drift']:.2e}  "
                  f"rev {runs[dt]['reversibility']:.2e}  "
                  f"({runs[dt]['wall_time_s']:.0f}s)", flush=True)

        pairs = {}
        for a, b in zip(args.dt[:-1], args.dt[1:]):
            ba = np.real(np.diagonal(runs[a]["blocks"], axis1=-2, axis2=-1))
            bb = np.real(np.diagonal(runs[b]["blocks"], axis1=-2, axis2=-1))
            ta, tb = np.array(runs[a]["times"]), np.array(runs[b]["times"])
            m = min(ba.shape[0], bb.shape[0])
            if not np.allclose(ta[:m], tb[:m], atol=1e-9):
                raise SystemExit(
                    f"[11c] save times do not line up for dt={a} and dt={b}: "
                    f"{ta[:4]} vs {tb[:4]}.  The paired difference would be "
                    f"comparing different instants.")
            d = ba[:m] - bb[:m]                       # paired, per block
            std = block_statistics(d)
            dfin = np.abs(np.real(std["mean"][-1]))
            dsem = np.real(std["sem"][-1])
            l2 = float(np.sqrt(np.trapezoid(
                np.sum(np.real(std["mean"]) ** 2, axis=1), ta[:m])
                / max(ta[m - 1] - ta[0], 1e-300)))
            ratio = float(np.max(dfin)
                          / max(float(np.max(np.real(runs[b]["sem"]))), 1e-300))
            worst_ratio = max(worst_ratio, ratio)
            pairs[f"{a}->{b}"] = {
                "final_population_difference": dfin.tolist(),
                "sem_of_the_difference": dsem.tolist(),
                "time_series_L2": l2,
                "difference_over_single_run_sem": ratio,
                "difference_resolved_above_its_own_noise":
                    bool(np.max(dfin) > 2.0 * np.max(dsem))}
            print(f"[11c] P0={P0} {a}->{b}: |dpop| {dfin} "
                  f"(its own SEM {dsem}), L2 {l2:.3e}, "
                  f"{ratio:.3f} of the single-run SEM", flush=True)
        keys = list(pairs)
        order = None
        if len(keys) == 2:
            d1 = max(pairs[keys[0]]["final_population_difference"])
            d2 = max(pairs[keys[1]]["final_population_difference"])
            if d2 > 0 and pairs[keys[1]]["difference_resolved_above_its_own_noise"]:
                order = float(np.log2(d1 / d2))
        out["regimes"][str(P0)] = {
            "P0": P0, "T": T, "sigma_R": sigma_R,
            "runs": {str(k): {kk: vv for kk, vv in v.items() if kk != "blocks"}
                     for k, v in runs.items()},
            "pairs": pairs, "observed_order": order}

    rec.results = out
    rec.check("pbme.integration_error_below_sampling_error", worst_ratio, 1.0,
              "largest |population change between successive time steps| "
              "divided by the Monte-Carlo standard error of the finer run; "
              "below one means the time step is not what limits the branch")
    rec.check("pbme.reversibility",
              max(r["reversibility"] for reg in out["regimes"].values()
                  for r in reg["runs"].values()), 1e-10,
              "forward then backward; the step is palindromic so this is "
              "round-off")
    rec.check("pbme.mapping_radius_drift",
              max(r["mapping_radius_drift"] for reg in out["regimes"].values()
                  for r in reg["runs"].values()), 1e-8,
              "the frozen-R rotation is orthogonal")
    p = Path(args.out)
    rec.write(p)
    print(f"[11c] written to {p};  checks all passed: {rec.all_passed}")
    for c in rec.checks:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['name']:46s} "
              f"{c['value']:.3e}  (tol {c['tolerance']:.1e})")
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
