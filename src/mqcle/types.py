"""Shared dataclasses, enums and array-layout contracts.

Array layout contracts (fixed once, used everywhere)
----------------------------------------------------
mapping coordinate  x  : (..., 2N)  ordered (r_0..r_{N-1}, p_0..p_{N-1})
bath position       R  : (..., Nb)
bath momentum       P  : (..., Nb)
subsystem matrix    h  : (..., N, N)
dh/dR                  : (..., Nb, N, N)
full state vector      : (..., 2N + 2Nb) ordered (x, R, P)  -- see pbme.STATE_LAYOUT
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class TraceConvention(enum.Enum):
    """How the trace part of the subsystem Hamiltonian is assigned.

    The subsystem matrix splits as  h = hbar_traceless + (Tr h / N) * 1.

    TRACELESS
        The trace part is moved into the bath potential, so the mapping-
        dependent part of H_m and the excess operator both carry the
        traceless matrix.  This is the convention of Nassimi, Bonella &
        Kapral, J. Chem. Phys. 133, 134115 (2010), Eq. (10), and of this
        project's existing operator implementation.
    FULL
        The full matrix h appears both in H_m and in the excess operator.
        This is the convention of Kim, Nassimi & Kapral, J. Chem. Phys. 129,
        084102 (2008), Eqs. (23) and (30).

    Both are internally consistent and differ by a term proportional to the
    mapping identity, which acts as the identity on the physical singly
    excited subspace.  They MUST NOT be mixed: using the full matrix in H_m
    and the traceless matrix in the excess operator breaks the exact
    cancellation between the two Hermite-order-raising channels.  See
    tests/test_excess.py::test_mixed_convention_breaks_closure.
    """

    TRACELESS = "traceless"
    FULL = "full"


@dataclass(frozen=True)
class SubsystemSplit:
    """Result of applying a :class:`TraceConvention` at bath positions R."""

    h_map: np.ndarray        #: (..., N, N) matrix entering the mapping term of H_m
    dh_map_dR: np.ndarray    #: (..., Nb, N, N) its R-gradient (drives the excess term)
    v_eff: np.ndarray        #: (...,) effective bath potential (V_b plus any trace part)
    dv_eff_dR: np.ndarray    #: (..., Nb) its R-gradient
    convention: TraceConvention


@dataclass
class Ensemble:
    """A weighted ensemble of phase-space points."""

    R: np.ndarray            #: (n, Nb)
    P: np.ndarray            #: (n, Nb)
    x: np.ndarray            #: (n, 2N)
    weights: np.ndarray      #: (n,)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.R.shape[0]
        for name, arr in (("P", self.P), ("x", self.x), ("weights", self.weights)):
            if arr.shape[0] != n:
                raise ValueError(f"Ensemble: {name} has {arr.shape[0]} rows, expected {n}")

    @property
    def n_samples(self) -> int:
        return int(self.R.shape[0])

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"R": self.R, "P": self.P, "x": self.x, "weights": self.weights}


@dataclass(frozen=True)
class OperatorParts:
    """Separately reported contributions of the full generator (PART XII)."""

    pbme: np.ndarray
    excess: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.pbme + self.excess


__all__ = ["TraceConvention", "SubsystemSplit", "Ensemble", "OperatorParts"]
