"""Named error layers used by the final MQCLE benchmarks."""
from __future__ import annotations

import numpy as np

__all__ = ["population_error_decomposition", "qcle_grid_error_metrics"]


def population_error_decomposition(full_quantum, coefficient_qcle,
                                   projected_pbme, focused_pbme=None) -> dict:
    """Return the four non-interchangeable max-norm population errors.

    ``E_discard`` is defined only by projected-density PBME minus coefficient
    QCLE.  The focused branch is an additional estimator/readout layer and is
    therefore reported separately.
    """
    q = np.asarray(full_quantum, dtype=float)
    c = np.asarray(coefficient_qcle, dtype=float)
    p = np.asarray(projected_pbme, dtype=float)
    if q.shape != c.shape or q.shape != p.shape:
        raise ValueError("full quantum, coefficient QCLE, and projected PBME "
                         "populations must have the same shape")
    out = {
        "qcle_model_error": float(np.max(np.abs(c - q))),
        "discard_cost": float(np.max(np.abs(p - c))),
        "projected_pbme_to_quantum_error": float(np.max(np.abs(p - q))),
    }
    if focused_pbme is not None:
        f = np.asarray(focused_pbme, dtype=float)
        if f.shape != q.shape:
            raise ValueError("focused PBME populations must have the same shape")
        out["focused_effect"] = float(np.max(np.abs(f - p)))
        out["focused_pbme_to_quantum_error"] = float(np.max(np.abs(f - q)))
    return out


def qcle_grid_error_metrics(reference_field, comparison_field,
                            cell_area: float) -> dict[str, float | list[float]]:
    """Compare two Hermitian QCLE fields on the same phase-space grid.

    The unnormalised Frobenius norm includes the phase-space quadrature weight.
    The relative norm uses the reference field as its denominator.  Integrated
    populations and coherence are reported separately because they are the
    observables used in the method-comparison tables.
    """
    ref = np.asarray(reference_field, dtype=complex)
    got = np.asarray(comparison_field, dtype=complex)
    if ref.shape != got.shape or ref.ndim < 4 or ref.shape[-2:] != (2, 2):
        raise ValueError("reference and comparison must have the same "
                         "(..., 2, 2) QCLE field shape")
    if not np.isfinite(cell_area) or cell_area <= 0.0:
        raise ValueError("cell_area must be finite and positive")

    delta = got - ref
    l2 = float(np.sqrt(cell_area * np.sum(np.abs(delta) ** 2)))
    ref_l2 = float(np.sqrt(cell_area * np.sum(np.abs(ref) ** 2)))
    trace_delta = np.trace(delta, axis1=-2, axis2=-1)
    trace_l1 = float(cell_area * np.sum(np.abs(trace_delta)))
    diagonal = np.diagonal(delta, axis1=-2, axis2=-1)
    pop_delta = np.real(np.sum(diagonal,
                               axis=tuple(range(diagonal.ndim - 1)))) * cell_area
    coh_delta = np.sum(delta[..., 0, 1]) * cell_area
    return {
        "field_frobenius_l2": l2,
        "field_relative_l2": l2 / max(ref_l2, np.finfo(float).tiny),
        "field_max_abs": float(np.max(np.abs(delta))),
        "trace_density_l1": trace_l1,
        "population_difference": pop_delta.tolist(),
        "population_linf": float(np.max(np.abs(pop_delta))),
        "coherence_integral_abs": float(abs(coh_delta)),
    }
