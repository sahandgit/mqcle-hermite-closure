"""The four numerical error layers must never be used as synonyms."""
from __future__ import annotations

import numpy as np

from mqcle.errors import population_error_decomposition


def test_named_errors_in_qcle_exact_limit():
    # An exactly tractable R-independent electronic problem has QCLE equal to
    # the full quantum evolution.  The independently sampled PBME estimators
    # may still carry operator and focused-estimator offsets.
    full = np.array([0.7, 0.3])
    qcle = full.copy()
    projected = np.array([0.68, 0.32])
    focused = np.array([0.65, 0.35])
    e = population_error_decomposition(full, qcle, projected, focused)
    assert e["qcle_model_error"] == 0.0
    assert np.isclose(e["discard_cost"], 0.02)
    assert np.isclose(e["focused_effect"], 0.03)
    assert np.isclose(e["focused_pbme_to_quantum_error"], 0.05)
    assert e["discard_cost"] != e["focused_pbme_to_quantum_error"]


def test_error_decomposition_rejects_shape_mismatch():
    try:
        population_error_decomposition([1.0, 0.0], [1.0], [1.0, 0.0])
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("shape mismatch was accepted")
