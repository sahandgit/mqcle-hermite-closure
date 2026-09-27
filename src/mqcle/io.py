"""Config loading, run metadata and raw-data persistence (PART XXXVI).

Every script writes a metadata block recording timestamp, config, seed,
interpreter and package versions, dtype, hardware, command line, wall time and
git commit when available.  No scientific parameter may live in a plotting
script; all of them come from ``configs/``.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

__all__ = ["load_config", "RunRecorder", "save_json", "save_arrays", "REPO_ROOT"]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def load_config(name_or_path: str | Path, config_dir: Path | None = None
                ) -> dict[str, Any]:
    """Load a YAML config, resolving a single-level ``inherit:`` chain."""
    config_dir = Path(config_dir or REPO_ROOT / "configs")
    path = Path(name_or_path)
    if not path.is_file():
        # A path such as ``configs/tully_default.yaml`` is repo-relative and
        # must resolve from any working directory, not only from the root.
        # Without this, running a script from ``scripts/`` sends the fallback
        # below looking for ``configs/configs/tully_default.yaml``, whose error
        # message points at the symptom rather than at the cause.
        rooted = REPO_ROOT / path
        path = rooted if rooted.is_file() else config_dir / str(name_or_path)
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"config not found: {name_or_path!r} is not a file relative to the "
            f"working directory {Path.cwd()}, nor {REPO_ROOT / name_or_path}, "
            f"nor {config_dir / str(name_or_path)}")
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    parent_name = cfg.pop("inherit", None)
    if parent_name:
        parent = load_config(parent_name, config_dir)
        merged = dict(parent)
        for key, val in cfg.items():
            if isinstance(val, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **val}
            else:
                merged[key] = val
        cfg = merged
    cfg["_config_path"] = str(path)
    return cfg


def _versions() -> dict[str, str]:
    import scipy
    import sympy
    mods = {"numpy": np.__version__, "scipy": scipy.__version__,
            "sympy": sympy.__version__}
    for name in ("matplotlib", "pandas", "h5py", "yaml", "torch"):
        try:
            mods[name] = __import__(name).__version__
        except Exception:
            mods[name] = "not installed"
    return mods


@dataclass
class RunRecorder:
    """Accumulates results and metadata for one script run."""

    script: str
    config: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    results: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter)

    def check(self, name: str, value: float, tol: float, note: str = "") -> bool:
        """Record a pass/fail numerical check and return whether it passed."""
        ok = bool(np.isfinite(value) and abs(value) <= tol)
        self.checks.append({"name": name, "value": float(value),
                            "tolerance": float(tol), "passed": ok, "note": note})
        return ok

    @property
    def all_passed(self) -> bool:
        return all(c["passed"] for c in self.checks)

    def metadata(self) -> dict[str, Any]:
        return {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script": self.script,
            "config": self.config,
            "seed": self.seed,
            "python": sys.version,
            "packages": _versions(),
            "dtype": "float64",
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "command_line": " ".join(sys.argv),
            "wall_time_s": round(time.perf_counter() - self._t0, 6),
            "git_commit": _git_commit(),
        }

    def payload(self) -> dict[str, Any]:
        return {"metadata": self.metadata(), "checks": self.checks,
                "all_passed": self.all_passed, "results": self.results}

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, self.payload())
        return path


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, complex):
        return {"re": obj.real, "im": obj.imag}
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_jsonify(payload), fh, indent=2, sort_keys=False)
    return path


def save_arrays(path: str | Path, arrays: dict[str, np.ndarray],
                attrs: dict[str, Any] | None = None) -> Path:
    """Persist arrays to HDF5 with a JSON-encoded metadata attribute."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fh:
        for key, arr in arrays.items():
            fh.create_dataset(key, data=np.asarray(arr))
        fh.attrs["metadata_json"] = json.dumps(_jsonify(attrs or {}))
    return path
