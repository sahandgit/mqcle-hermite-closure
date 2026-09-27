"""Physical constants, dtype policy and numerical tolerances.

Atomic units throughout: hbar = 1, electron mass = 1, energies in hartree,
lengths in bohr, times in hbar/hartree.
"""
from __future__ import annotations

import numpy as np

#: Reduced Planck constant in atomic units.
HBAR: float = 1.0

#: Default nuclear mass used by the Tully benchmark models (a.u.).
DEFAULT_MASS: float = 2000.0

#: All scientific arrays are double precision (PART III, rule 11).
DTYPE = np.float64
CDTYPE = np.complex128

# --- numerical tolerances used by the test suite -------------------------
#: Machine-precision identities (inverses, algebraic rearrangements).
TOL_EXACT: float = 1e-13
#: Analytic-vs-autodiff / analytic-vs-symbolic comparisons.
TOL_ANALYTIC: float = 1e-11
#: Central finite differences with an optimal step (~eps^(1/3) per order).
TOL_FD1: float = 1e-7
TOL_FD2: float = 1e-5
TOL_FD3: float = 1e-4
#: Gauss-Hermite quadrature of Gaussian-polynomial integrands.
TOL_QUAD: float = 1e-12

__all__ = [
    "HBAR", "DEFAULT_MASS", "DTYPE", "CDTYPE",
    "TOL_EXACT", "TOL_ANALYTIC", "TOL_FD1", "TOL_FD2", "TOL_FD3", "TOL_QUAD",
]
