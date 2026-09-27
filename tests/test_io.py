"""Module 8: configuration loading and run provenance."""
from __future__ import annotations

import json

import numpy as np
import pytest

from mqcle.io import REPO_ROOT, RunRecorder, load_config, save_arrays, save_json


def test_config_inheritance_merges_nested_blocks():
    cfg = load_config("tully_default.yaml")
    assert cfg["units"] == "atomic"                 # from base.yaml
    assert cfg["model"]["name"] == "tully_dual_avoided_crossing"
    assert cfg["initial_conditions"]["R0"] == -15.0
    assert "inherit" not in cfg
    assert cfg["_config_path"].endswith("tully_default.yaml")


def test_config_matches_the_archived_effective_parameters():
    """The configs were reconstructed from the archive; this asserts that the
    reconstruction still agrees with what the archived run actually used."""
    p = REPO_ROOT / "results" / "raw" / "00_validate_physics.json"
    if not p.is_file():
        pytest.skip("archive not present")
    archived = json.loads(p.read_text())["metadata"]["config"]
    cfg = load_config("tully_default.yaml")
    for key in ("hbar", "mass", "nstates", "nbath", "trace_convention", "seed"):
        assert cfg[key] == archived[key], key
    for block in ("model", "initial_conditions", "mapping_sampling", "propagation"):
        assert cfg[block] == archived[block], block


def test_recorder_marks_pass_and_fail():
    rec = RunRecorder(script="t.py")
    assert rec.check("ok", 1e-15, 1e-12)
    assert rec.all_passed
    assert not rec.check("bad", 1.0, 1e-12)
    assert not rec.all_passed
    assert not rec.check("nan", float("nan"), 1e12)


def test_recorder_metadata_is_complete():
    md = RunRecorder(script="t.py", seed=7).metadata()
    for k in ("timestamp_utc", "script", "config", "seed", "python", "packages",
              "dtype", "platform", "command_line", "wall_time_s"):
        assert k in md
    assert md["seed"] == 7
    assert "numpy" in md["packages"] and "sympy" in md["packages"]


def test_json_round_trip_handles_numpy_and_complex(tmp_path):
    p = save_json(tmp_path / "a.json",
                  {"arr": np.arange(3), "z": 1 + 2j,
                   "f": np.float64(0.5), "nested": {"i": np.int64(4)}})
    d = json.loads(p.read_text())
    assert d["arr"] == [0, 1, 2]
    assert d["z"] == {"re": 1.0, "im": 2.0}
    assert d["f"] == 0.5 and d["nested"]["i"] == 4


def test_hdf5_round_trip_keeps_metadata(tmp_path):
    import h5py
    p = save_arrays(tmp_path / "a.h5", {"x": np.arange(4.0)}, {"note": "hi"})
    with h5py.File(p, "r") as f:
        assert np.array_equal(f["x"][()], np.arange(4.0))
        assert json.loads(f.attrs["metadata_json"])["note"] == "hi"
