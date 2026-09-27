#!/usr/bin/env python3
"""Extract and validate the projected-PBME sample/seed convergence record.

Stage 11 stores the independent block sums, sample-size sequence, seed-group
replicates, and confidence intervals in its production JSON.  This lightweight
stage writes the reviewer-requested ``11b`` artifact without rerunning any
trajectory and fails if the archived uncertainty is not internally consistent.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqcle.io import save_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--P0", type=int, required=True)
    p.add_argument("--source", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def build_summary(doc: dict, source_name: str) -> dict:
    r = doc["results"]
    cv = [row for row in r["sample_size_convergence"]
          if row["estimator"] == "cv"]
    reps = r["seed_replicates"]
    checks = []

    def check(name: str, passed: bool, evidence: str) -> None:
        checks.append({"name": name, "passed": bool(passed),
                       "evidence": evidence})

    check("pbme_statistics.at_least_five_seeds", r["n_seeds"] >= 5,
          f"n_seeds={r['n_seeds']}")
    sample_sizes = sorted({int(row["n_samples"]) for row in cv})
    check("pbme_statistics.sample_size_range",
          bool(sample_sizes) and sample_sizes[0] <= 5_000
          and sample_sizes[-1] >= 100_000,
          f"sample sizes={sample_sizes}")

    usable = [row for row in cv if row["n_blocks"] >= 2]
    sem_scale_ok = False
    slope = float("nan")
    if len(usable) >= 3:
        ns = np.asarray([row["n_samples"] for row in usable], float)
        sem = np.asarray([max(row["sem"]) for row in usable], float)
        positive = np.isfinite(sem) & (sem > 0)
        if positive.sum() >= 3:
            slope = float(np.polyfit(np.log(ns[positive]),
                                     np.log(sem[positive]), 1)[0])
            sem_scale_ok = slope < 0.0
    check("pbme_statistics.sem_decreases_with_sample_size", sem_scale_ok,
          f"log-log slope={slope:.3f}")

    across = np.asarray([row["final_populations_cv"] for row in reps], float)
    expected_sem = across.std(axis=0, ddof=1) / math.sqrt(len(reps))
    archived_sem = np.asarray(r["across_seed"]["sem_cv"], float)
    seed_sem_ok = (len(reps) == r["n_seeds"]
                   and np.allclose(expected_sem, archived_sem,
                                   rtol=1e-12, atol=1e-15))
    check("pbme_statistics.sem_uses_seed_variance", seed_sem_ok,
          f"recomputed={expected_sem.tolist()}, archived={archived_sem.tolist()}")

    final = r["final"]["cv"]
    mean = np.asarray(final["populations"], float)
    sem = np.asarray(final["sem"], float)
    ci = np.asarray(final["ci95"], float)
    expected_ci = np.column_stack((mean - 1.96 * sem, mean + 1.96 * sem))
    ci_ok = np.allclose(ci, expected_ci, rtol=1e-12, atol=1e-15)
    check("pbme_statistics.ci95_uses_recorded_sem", ci_ok,
          "ci95 = mean +/- 1.96*SEM")
    check("pbme_statistics.weights_not_self_normalised",
          r.get("weights_renormalised_by_their_sum") is False,
          "weights_renormalised_by_their_sum=false")

    return {
        "metadata": {**doc["metadata"],
                     "script": Path(__file__).name,
                     "source_file": source_name},
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
        "results": {
            "P0": r.get("P0", doc["metadata"]["config"].get("P0")),
            "source_file": source_name,
            "n_samples": r["n_samples"],
            "n_seeds": r["n_seeds"],
            "sample_size_convergence": r["sample_size_convergence"],
            "seed_replicates": reps,
            "across_seed": r["across_seed"],
            "final": r["final"],
            "sem_loglog_slope": slope,
        },
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    source = Path(args.source) if args.source else (
        RAW / f"11_pbme_projected_P{args.P0}.json")
    if not source.is_file():
        raise SystemExit(f"[11b] missing source: {source}")
    doc = json.loads(source.read_text())
    summary = build_summary(doc, source.name)
    out = Path(args.out) if args.out else (
        RAW / f"11b_pbme_seed_convergence_P{args.P0}.json")
    save_json(out, summary)
    for c in summary["checks"]:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['name']}: "
              f"{c['evidence']}")
    print(f"[11b] written to {out}; all passed: {summary['all_passed']}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
