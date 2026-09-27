#!/usr/bin/env python3
r"""Focused PBME: adiabatic populations and coherence of the archived ensemble.

``scripts/11`` propagates the focused-sampling PBME ensemble of each regime
but archives only its diabatic populations.  The comparison figures of the
paper show every calculation in every panel, so this script propagates the
same ensemble again -- the same generator seed, sampler, step and saving
interval as the focused branch of ``scripts/11`` -- and keeps the adiabatic
populations and the electronic coherence that the propagator reads out at
every saved instant.

The re-propagation must reproduce the archived diabatic populations; that is
checked, so the new observables belong to the ensemble whose populations the
paper already reports.

Usage
-----
    python scripts/20_focused_pbme_observables.py --P0 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mqcle.io import RunRecorder                                     # noqa: E402
from mqcle.models import build_model                                 # noqa: E402
from mqcle.propagation import (propagate_pbme_ensemble,              # noqa: E402
                               sample_matched_ensemble)
from mqcle.types import TraceConvention                              # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--P0", type=float, required=True)
    p.add_argument("--source", default=None,
                   help="scripts/11 archive (default results/raw/"
                        "11_pbme_projected_P<P0>.json)")
    p.add_argument("--out", default=None,
                   help="output JSON (default results/raw/"
                        "20_focused_pbme_observables_P<P0>.json)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    P0i = int(round(args.P0))
    raw = ROOT / "results" / "raw"
    src = Path(args.source or raw / f"11_pbme_projected_P{P0i}.json")
    out = Path(args.out or raw / f"20_focused_pbme_observables_P{P0i}.json")
    doc = json.loads(src.read_text(encoding="utf-8"))
    cfg = doc["metadata"]["config"]
    arch = doc["results"]["focused"]

    hbar, mass = float(cfg["hbar"]), float(cfg["mass"])
    model = build_model(cfg["model"], mass=mass)
    convention = TraceConvention(cfg["trace_convention"])
    P0, R0, sigma_R = float(cfg["P0"]), float(cfg["R0"]), float(cfg["sigma_R"])
    dt, n_steps, n_save = float(cfg["dt"]), int(cfg["n_steps"]), int(cfg["n_save"])
    init_state = int(cfg["initial_conditions"]["init_state"])
    seed = int(cfg["seed"])
    nf = int(arch["n_samples"])

    rec = RunRecorder(script=Path(__file__).name, seed=seed, config={
        "source": src.name, "P0": P0, "R0": R0, "sigma_R": sigma_R, "dt": dt,
        "n_steps": n_steps, "n_save": n_save, "n_samples": nf,
        "trace_convention": convention.value, "model": cfg["model"],
        "mass": mass, "hbar": hbar})

    # the focused branch of scripts/11, call for call
    rngf = np.random.default_rng(seed)
    ensf = sample_matched_ensemble(rngf, nf, nstates=model.nstates,
                                   init_state=init_state, R0=R0, P0=P0,
                                   sigma_R=sigma_R, hbar=hbar)
    t0 = time.perf_counter()
    hf = propagate_pbme_ensemble(ensf, model, dt, n_steps, hbar=hbar,
                                 convention=convention, n_save=n_save)
    wall = time.perf_counter() - t0

    times = np.asarray(hf.times, float)
    pops = np.array([np.asarray(r["populations"], float) for r in hf.records])
    ad = np.array([np.asarray(r["adiabatic_populations"], float) for r in hf.records])
    cre = np.array([float(r["coherence_re"]) for r in hf.records])
    cim = np.array([float(r["coherence_im"]) for r in hf.records])
    trace = np.array([float(np.real(r["trace"])) for r in hf.records])

    t_arch = np.asarray(arch["times"], float)
    p_arch = np.asarray(arch["populations"], float)
    same_times = times.shape == t_arch.shape and bool(np.all(times == t_arch))
    rec.check("focused.same_saved_times", 0.0 if same_times else 1.0, 0.0,
              "saved instants identical to the archived focused branch")
    rec.check("focused.reproduces_archived_populations",
              float(np.max(np.abs(pops - p_arch))) if same_times else float("inf"),
              1e-12, "diabatic populations against scripts/11, every instant")
    rec.check("focused.trace_conserved", float(np.max(np.abs(trace - 1.0))), 1e-10,
              "trace of the read-out density matrix")
    rec.check("focused.adiabatic_trace", float(np.max(np.abs(ad.sum(axis=1) - 1.0))),
              1e-10, "adiabatic populations sum to one")

    rec.results = {
        "branch": "pbme_focused", "n_samples": nf, "wall_time_s": wall,
        "times": times.tolist(), "populations": pops.tolist(),
        "adiabatic_populations": ad.tolist(),
        "coherence_re": cre.tolist(), "coherence_im": cim.tolist(),
        "final": {"populations": pops[-1].tolist(),
                  "adiabatic_populations": ad[-1].tolist(),
                  "coherence": [float(cre[-1]), float(cim[-1])]}}
    out.parent.mkdir(parents=True, exist_ok=True)
    rec.write(out)
    print(f"[20] P0={P0:g}: {nf:,} focused trajectories, {wall:.0f} s; wrote "
          f"{out.name}; checks all passed: {rec.all_passed}", flush=True)
    for chk in rec.checks:
        print(f"  {'PASS' if chk['passed'] else 'FAIL'}  {chk['name']:<44s} "
              f"{chk['value']:.3e}  (tol {chk['tolerance']:.1e})", flush=True)
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
