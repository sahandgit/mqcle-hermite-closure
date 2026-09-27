"""Module 1 -- physical Hamiltonian and benchmark models.

Provides the diabatic subsystem matrix h(R), its R-gradient, the bath
potential and its gradient.  All quantities are vectorised over the leading
(sample) axis and returned in double precision.

Conventions
-----------
h(R)      : (..., N, N)   real symmetric
dh/dR     : (..., Nb, N, N)
V_b(R)    : (...,)
dV_b/dR   : (..., Nb)

The models here carry the *physical* subsystem matrix.  The split into the
mapping matrix and the effective bath potential (the trace convention) is the
job of :mod:`mqcle.mapping`, not of this module.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .constants import DEFAULT_MASS, DTYPE

__all__ = [
    "TwoStateModel",
    "TullyDualAvoidedCrossing",
    "TullySingleAvoidedCrossing",
    "ConstantCouplingModel",
    "build_model",
]


def _atleast_bath(R: np.ndarray) -> np.ndarray:
    """Return R with an explicit trailing bath axis, shape (..., Nb)."""
    R = np.asarray(R, dtype=DTYPE)
    if R.ndim == 0:
        return R.reshape(1)
    return R


class TwoStateModel(ABC):
    """Abstract two-state, ``nbath``-mode diabatic model."""

    nstates: int = 2
    nbath: int = 1
    mass: float = DEFAULT_MASS
    name: str = "abstract"

    # -- required interface (PART V) --------------------------------------
    @abstractmethod
    def h(self, R: np.ndarray) -> np.ndarray:
        """Diabatic subsystem matrix, shape ``(..., N, N)``, real symmetric."""

    @abstractmethod
    def dh_dR(self, R: np.ndarray) -> np.ndarray:
        """Gradient of :meth:`h`, shape ``(..., Nb, N, N)``."""

    @abstractmethod
    def bath_potential(self, R: np.ndarray) -> np.ndarray:
        """State-independent bath potential, shape ``(...,)``."""

    @abstractmethod
    def dbath_dR(self, R: np.ndarray) -> np.ndarray:
        """Gradient of :meth:`bath_potential`, shape ``(..., Nb)``."""

    # -- derived quantities ----------------------------------------------
    def adiabatic_energies(self, R: np.ndarray) -> np.ndarray:
        """Eigenvalues of h(R), ascending, shape ``(..., N)``."""
        return np.linalg.eigvalsh(self.h(R))

    def parameters(self) -> dict[str, Any]:
        """Every configured parameter, for experiment metadata (PART V test 4)."""
        return {"name": self.name, "nstates": self.nstates,
                "nbath": self.nbath, "mass": float(self.mass)}


@dataclass(frozen=True)
class _TullyDACParams:
    """Tully model II parameters, J. Chem. Phys. 93, 1061 (1990), Sec. IV.B."""

    A: float = 0.10
    B: float = 0.28
    C: float = 0.015
    D: float = 0.06
    E0: float = 0.05


class TullyDualAvoidedCrossing(TwoStateModel):
    r"""Tully model II (dual avoided crossing).

    .. math::
        V_{11}=0,\qquad
        V_{22}=-A e^{-B R^{2}}+E_{0},\qquad
        V_{12}=V_{21}=C e^{-D R^{2}} .

    The bath potential is zero: the whole nuclear potential is carried by the
    diabatic matrix, which is what makes the back-reaction state dependent.
    """

    name = "tully_dual_avoided_crossing"
    nbath = 1

    def __init__(self, mass: float = DEFAULT_MASS, **params: float) -> None:
        self.p = _TullyDACParams(**params)
        self.mass = float(mass)

    def h(self, R: np.ndarray) -> np.ndarray:
        R = _atleast_bath(R)
        r = R[..., 0]
        p = self.p
        out = np.zeros(r.shape + (2, 2), dtype=DTYPE)
        out[..., 0, 0] = 0.0
        out[..., 1, 1] = -p.A * np.exp(-p.B * r**2) + p.E0
        off = p.C * np.exp(-p.D * r**2)
        out[..., 0, 1] = off
        out[..., 1, 0] = off
        return out

    def dh_dR(self, R: np.ndarray) -> np.ndarray:
        R = _atleast_bath(R)
        r = R[..., 0]
        p = self.p
        out = np.zeros(r.shape + (1, 2, 2), dtype=DTYPE)
        out[..., 0, 1, 1] = 2.0 * p.A * p.B * r * np.exp(-p.B * r**2)
        doff = -2.0 * p.C * p.D * r * np.exp(-p.D * r**2)
        out[..., 0, 0, 1] = doff
        out[..., 0, 1, 0] = doff
        return out

    def bath_potential(self, R: np.ndarray) -> np.ndarray:
        R = _atleast_bath(R)
        return np.zeros(R.shape[:-1], dtype=DTYPE)

    def dbath_dR(self, R: np.ndarray) -> np.ndarray:
        R = _atleast_bath(R)
        return np.zeros(R.shape, dtype=DTYPE)

    def parameters(self) -> dict[str, Any]:
        out = super().parameters()
        out.update(asdict(self.p))
        out["reference"] = "Tully, J. Chem. Phys. 93, 1061 (1990), model II"
        return out


@dataclass(frozen=True)
class _TullySACParams:
    """Tully model I parameters (single avoided crossing)."""

    A: float = 0.01
    B: float = 1.6
    C: float = 0.005
    D: float = 1.0


class TullySingleAvoidedCrossing(TwoStateModel):
    r"""Tully model I.

    .. math::
        V_{11}=A\,\mathrm{sgn}(R)\left(1-e^{-B|R|}\right)=-V_{22},\qquad
        V_{12}=C e^{-D R^{2}} .

    Used only as a second, differently shaped model for the model-level tests;
    the production benchmark is :class:`TullyDualAvoidedCrossing`.
    """

    name = "tully_single_avoided_crossing"
    nbath = 1

    def __init__(self, mass: float = DEFAULT_MASS, **params: float) -> None:
        self.p = _TullySACParams(**params)
        self.mass = float(mass)

    def h(self, R: np.ndarray) -> np.ndarray:
        R = _atleast_bath(R)
        r = R[..., 0]
        p = self.p
        v11 = p.A * np.sign(r) * (1.0 - np.exp(-p.B * np.abs(r)))
        off = p.C * np.exp(-p.D * r**2)
        out = np.zeros(r.shape + (2, 2), dtype=DTYPE)
        out[..., 0, 0] = v11
        out[..., 1, 1] = -v11
        out[..., 0, 1] = off
        out[..., 1, 0] = off
        return out

    def dh_dR(self, R: np.ndarray) -> np.ndarray:
        R = _atleast_bath(R)
        r = R[..., 0]
        p = self.p
        # d/dR [ sgn(R)(1 - e^{-B|R|}) ] = B e^{-B|R|}, smooth away from R = 0.
        dv11 = p.A * p.B * np.exp(-p.B * np.abs(r))
        doff = -2.0 * p.C * p.D * r * np.exp(-p.D * r**2)
        out = np.zeros(r.shape + (1, 2, 2), dtype=DTYPE)
        out[..., 0, 0, 0] = dv11
        out[..., 0, 1, 1] = -dv11
        out[..., 0, 0, 1] = doff
        out[..., 0, 1, 0] = doff
        return out

    def bath_potential(self, R: np.ndarray) -> np.ndarray:
        R = _atleast_bath(R)
        return np.zeros(R.shape[:-1], dtype=DTYPE)

    def dbath_dR(self, R: np.ndarray) -> np.ndarray:
        R = _atleast_bath(R)
        return np.zeros(R.shape, dtype=DTYPE)

    def parameters(self) -> dict[str, Any]:
        out = super().parameters()
        out.update(asdict(self.p))
        out["reference"] = "Tully, J. Chem. Phys. 93, 1061 (1990), model I"
        return out


class ConstantCouplingModel(TwoStateModel):
    r"""Analytic test model with polynomial R dependence.

    .. math::
        h(R)=\begin{pmatrix} a_0+a_1R & c_0+c_1R\\ c_0+c_1R & b_0+b_1R\end{pmatrix},
        \qquad V_b(R)=\tfrac12 k R^{2}.

    Every derivative is elementary, so this model is the ground truth for the
    operator tests: with ``a1 = b1 = c1 = 0`` the subsystem matrix is
    R-independent, the excess term vanishes identically and the mapping flow is
    an exact rotation.
    """

    name = "constant_coupling"
    nbath = 1

    def __init__(self, a0=0.02, a1=0.0, b0=-0.01, b1=0.0,
                 c0=0.005, c1=0.0, k=0.0, mass: float = DEFAULT_MASS) -> None:
        self.coeff = dict(a0=a0, a1=a1, b0=b0, b1=b1, c0=c0, c1=c1, k=k)
        self.mass = float(mass)

    def h(self, R: np.ndarray) -> np.ndarray:
        R = _atleast_bath(R)
        r = R[..., 0]
        c = self.coeff
        out = np.zeros(r.shape + (2, 2), dtype=DTYPE)
        out[..., 0, 0] = c["a0"] + c["a1"] * r
        out[..., 1, 1] = c["b0"] + c["b1"] * r
        off = c["c0"] + c["c1"] * r
        out[..., 0, 1] = off
        out[..., 1, 0] = off
        return out

    def dh_dR(self, R: np.ndarray) -> np.ndarray:
        R = _atleast_bath(R)
        r = R[..., 0]
        c = self.coeff
        out = np.zeros(r.shape + (1, 2, 2), dtype=DTYPE)
        out[..., 0, 0, 0] = c["a1"]
        out[..., 0, 1, 1] = c["b1"]
        out[..., 0, 0, 1] = c["c1"]
        out[..., 0, 1, 0] = c["c1"]
        return out

    def bath_potential(self, R: np.ndarray) -> np.ndarray:
        R = _atleast_bath(R)
        return 0.5 * self.coeff["k"] * R[..., 0] ** 2

    def dbath_dR(self, R: np.ndarray) -> np.ndarray:
        R = _atleast_bath(R)
        out = np.zeros_like(R)
        out[..., 0] = self.coeff["k"] * R[..., 0]
        return out

    def parameters(self) -> dict[str, Any]:
        out = super().parameters()
        out.update(self.coeff)
        return out


_REGISTRY = {
    "tully_dual_avoided_crossing": TullyDualAvoidedCrossing,
    "tully_single_avoided_crossing": TullySingleAvoidedCrossing,
    "constant_coupling": ConstantCouplingModel,
}


def build_model(spec: dict[str, Any], mass: float = DEFAULT_MASS) -> TwoStateModel:
    """Instantiate a model from a config dict ``{'name': ..., <params>}``."""
    spec = dict(spec)
    name = spec.pop("name")
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](mass=mass, **spec)
