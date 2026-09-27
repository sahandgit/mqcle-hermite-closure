r"""Gate L, part 1 -- the closed coefficient generator on a bath phase-space grid.

Theorem 2 states that the projected mapping density stays in the even part of
:math:`V_2` and that its coefficients obey the subsystem-basis quantum-classical
Liouville equation

.. math::
    \partial_t\rho_W = -\frac{i}{\hbar}[h,\rho_W]
                       -\frac{P}{M}\,\partial_R\rho_W
                       -\frac12\bigl(F\,\partial_P\rho_W
                                    +\partial_P\rho_W\,F\bigr),
    \qquad F=-\partial_R\bigl(V_b\mathbb 1+h\bigr).

This module evaluates that right-hand side on a uniform periodic
:math:`(R,P)` grid with Fourier spectral derivatives.

Design note (the trap in the directive, section 3.1)
----------------------------------------------------
These coefficient equations *are* the subsystem QCLE -- that is Gate J,
archived in ``results/raw/03_qcle_correspondence.json``.  A separately written
QCLE grid solver is therefore an independent *implementation* check, not an
independent physical theory; that check is implemented in
:mod:`mqcle.qcle_grid_reference`.  The physical reference used to quantify the
QCLE model error is exact Schroedinger propagation; see
:mod:`mqcle.schrodinger`.

Trace convention
----------------
The reference above uses the *physical* ``h`` and ``V_b``.  Rather than
hard-coding either convention, this module calls
:func:`mqcle.mapping.subsystem_split` and adds the trace part back, so the
assembled operator is identical under ``TRACELESS`` and ``FULL``:

.. math::
    h_{\text{phys}} = h_{\text{map}}
        + \bigl(v_{\text{eff}}-V_b\bigr)\mathbb 1 ,

which is an identity of the split, not an assumption about it.  Gate
``convention_independence`` in ``scripts/07`` checks it numerically.

Array layout
------------
``rho_W`` has shape ``(n_R, n_P, N, N)`` and dtype ``complex128``, with
``rho_W[i, j, a, b]`` the coefficient :math:`\rho_W^{ab}(R_i,P_j)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:                                   # scipy's pocketfft is ~2.5x numpy's here
    import scipy.fft as _sfft
    _FFT_WORKERS = -1
except Exception:                      # pragma: no cover - scipy is a hard dep
    _sfft = None
    _FFT_WORKERS = 1

from .constants import CDTYPE, DTYPE, HBAR
from .mapping import subsystem_split
from .models import TwoStateModel
from .types import TraceConvention

__all__ = [
    "Grid",
    "build_grid",
    "spectral_derivative",
    "coefficient_rhs",
    "trace",
    "hermiticity_error",
    "populations_from_rho",
    "physical_matrices",
]


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Grid:
    """A uniform periodic grid on the bath phase plane.

    The endpoint is excluded (``R[-1] + dR == R_max``) so that the grid is a
    genuine period of the box; the spectral derivatives below assume this.
    """

    R: np.ndarray            #: (n_R,) node positions
    P: np.ndarray            #: (n_P,) node momenta
    dR: float
    dP: float
    kR: np.ndarray           #: (n_R,) angular wavenumbers conjugate to R
    kP: np.ndarray           #: (n_P,) angular wavenumbers conjugate to P
    L_R: float
    L_P: float

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.R.size), int(self.P.size))

    @property
    def cell_area(self) -> float:
        return float(self.dR * self.dP)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """``(RR, PP)`` broadcast to ``(n_R, n_P)``."""
        return np.meshgrid(self.R, self.P, indexing="ij")

    def parameters(self) -> dict[str, Any]:
        return {"R_min": float(self.R[0]), "R_max": float(self.R[0] + self.L_R),
                "n_R": int(self.R.size), "P_min": float(self.P[0]),
                "P_max": float(self.P[0] + self.L_P), "n_P": int(self.P.size),
                "dR": float(self.dR), "dP": float(self.dP)}


def build_grid(R_min: float, R_max: float, n_R: int,
               P_min: float, P_max: float, n_P: int) -> Grid:
    """Build a periodic grid on ``[R_min, R_max) x [P_min, P_max)``."""
    if n_R < 4 or n_P < 4:
        raise ValueError("build_grid: need at least 4 nodes per axis")
    if not (R_max > R_min and P_max > P_min):
        raise ValueError("build_grid: empty box")
    L_R = float(R_max - R_min)
    L_P = float(P_max - P_min)
    R = R_min + L_R * np.arange(n_R, dtype=DTYPE) / n_R
    P = P_min + L_P * np.arange(n_P, dtype=DTYPE) / n_P
    dR, dP = L_R / n_R, L_P / n_P
    kR = 2.0 * np.pi * np.fft.fftfreq(n_R, d=dR)
    kP = 2.0 * np.pi * np.fft.fftfreq(n_P, d=dP)
    return Grid(R=R, P=P, dR=dR, dP=dP, kR=kR, kP=kP, L_R=L_R, L_P=L_P)


def spectral_derivative(f: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    r"""Fourier spectral derivative :math:`\partial f/\partial u` along ``axis``.

    ``k`` are the angular wavenumbers returned by :func:`build_grid`.  For an
    even number of nodes the Nyquist mode is discarded: it carries no
    well-defined derivative and keeping it injects a purely imaginary
    contribution that breaks Hermiticity of the right-hand side.
    """
    n = f.shape[axis]
    kk = np.asarray(k, dtype=DTYPE).copy()
    if n % 2 == 0:
        kk[n // 2] = 0.0
    shape = [1] * f.ndim
    shape[axis] = n
    fac = (1j * kk).reshape(shape)
    if _sfft is not None:
        g = _sfft.fft(f, axis=axis, workers=_FFT_WORKERS)
        g *= fac
        return _sfft.ifft(g, axis=axis, overwrite_x=True, workers=_FFT_WORKERS)
    return np.fft.ifft(fac * np.fft.fft(f, axis=axis), axis=axis)


# --------------------------------------------------------------------------
# physical matrices on the grid
# --------------------------------------------------------------------------
def physical_matrices(model: TwoStateModel, grid: Grid,
                      convention: TraceConvention = TraceConvention.TRACELESS
                      ) -> tuple[np.ndarray, np.ndarray]:
    r"""Return ``(h_phys, F)`` on the grid, shapes ``(n_R, N, N)``.

    ``h_phys`` is the physical subsystem matrix and
    :math:`F=-\partial_R(V_b\mathbb 1+h_{\text{phys}})` the QCLE force matrix.
    Both are reassembled from :func:`mqcle.mapping.subsystem_split` so that the
    result does not depend on which trace convention was requested.
    """
    if model.nbath != 1:
        raise ValueError("coefficient_operator: the (R,P) grid needs nbath == 1")
    Rcol = grid.R[:, None]                       # (n_R, 1)
    split = subsystem_split(model, Rcol, convention)
    n = model.nstates
    eye = np.eye(n, dtype=DTYPE)

    vb = model.bath_potential(Rcol)              # (n_R,)
    dvb = model.dbath_dR(Rcol)[..., 0]           # (n_R,)
    trace_back = split.v_eff - vb                # (n_R,)   == Tr h / N or 0
    dtrace_back = split.dv_eff_dR[..., 0] - dvb  # (n_R,)

    h_phys = split.h_map + trace_back[:, None, None] * eye
    dh_phys = split.dh_map_dR[:, 0] + dtrace_back[:, None, None] * eye
    F = -(dvb[:, None, None] * eye) - dh_phys
    return np.ascontiguousarray(h_phys), np.ascontiguousarray(F)


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------
def coefficient_rhs(rho_W: np.ndarray, grid: Grid, model: TwoStateModel,
                    hbar: float = HBAR,
                    convention: TraceConvention = TraceConvention.TRACELESS,
                    cache: tuple[np.ndarray, np.ndarray] | None = None
                    ) -> np.ndarray:
    r"""Right-hand side of the closed coefficient equations.

    Parameters
    ----------
    rho_W
        ``(n_R, n_P, N, N)`` complex coefficient field.
    cache
        Optional ``(h_phys, F)`` from :func:`physical_matrices`, to avoid
        rebuilding them at every Runge-Kutta stage.
    """
    rho = np.asarray(rho_W, dtype=CDTYPE)
    n_R, n_P = grid.shape
    n = model.nstates
    if rho.shape != (n_R, n_P, n, n):
        raise ValueError(f"coefficient_rhs: rho_W has shape {rho.shape}, "
                         f"expected {(n_R, n_P, n, n)}")

    h_phys, F = cache if cache is not None else physical_matrices(model, grid, convention)
    Pv = grid.P[None, :, None, None]             # (1, n_P, 1, 1)

    # ``einsum`` beats broadcast ``@`` by ~2x on these (n_R, n_P, 2, 2) stacks
    commutator = (np.einsum("rab,rpbc->rpac", h_phys, rho, optimize=True)
                  - np.einsum("rpab,rbc->rpac", rho, h_phys, optimize=True))
    d_rho_dR = spectral_derivative(rho, grid.kR, axis=0)
    d_rho_dP = spectral_derivative(rho, grid.kP, axis=1)
    anticommutator = (np.einsum("rab,rpbc->rpac", F, d_rho_dP, optimize=True)
                      + np.einsum("rpab,rbc->rpac", d_rho_dP, F, optimize=True))

    return (-1j / hbar) * commutator - (Pv / model.mass) * d_rho_dR \
        - 0.5 * anticommutator


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def spectral_radius_estimate(grid: Grid, model: TwoStateModel,
                             hbar: float = HBAR,
                             convention: TraceConvention = TraceConvention.TRACELESS
                             ) -> dict[str, float]:
    r"""Bound the spectral radius of the semi-discrete generator.

    The three channels contribute additively:

    * advection      :math:`\max_P|P|/M \cdot k_R^{\max}`,
    * force          :math:`\max_R\|F(R)\|_2 \cdot k_P^{\max}`,
    * electronic     :math:`\max_R\,(E_{\max}-E_{\min})/\hbar`.

    ``max_dt_rk4`` is :math:`2\sqrt2/\lambda`, the imaginary-axis stability
    limit of classical RK4.  Past it the solver does not degrade gracefully --
    it produces ``1e28``.  ``scripts/07`` refuses to start above it.
    """
    h_phys, F = physical_matrices(model, grid, convention)
    kR = np.abs(grid.kR).max()
    kP = np.abs(grid.kP).max()
    adv = float(np.abs(grid.P).max() / model.mass * kR)
    force = float(np.max(np.linalg.norm(F, ord=2, axis=(-2, -1))) * kP)
    w = np.linalg.eigvalsh(h_phys)
    elec = float(np.max(w[..., -1] - w[..., 0]) / hbar)
    lam = adv + force + elec
    return {"advection": adv, "force": force, "electronic": elec,
            "spectral_radius": lam, "max_dt_rk4": float(2.0 * np.sqrt(2.0) / lam)}


def trace(rho_W: np.ndarray, grid: Grid) -> complex:
    r""":math:`\int dR\,dP\ \mathrm{Tr}\,\rho_W`, the conserved phase-space norm."""
    diag = np.trace(np.asarray(rho_W), axis1=-2, axis2=-1)
    return complex(np.sum(diag) * grid.cell_area)


def hermiticity_error(rho_W: np.ndarray) -> float:
    r""":math:`\max|\rho_W-\rho_W^\dagger|`."""
    rho = np.asarray(rho_W)
    return float(np.max(np.abs(rho - np.conj(np.swapaxes(rho, -1, -2)))))


def populations_from_rho(rho_W: np.ndarray, grid: Grid) -> np.ndarray:
    r"""Diabatic populations :math:`\int dR\,dP\ \rho_W^{aa}`, shape ``(N,)``."""
    rho = np.asarray(rho_W)
    return np.real(np.einsum("ijaa->a", rho)) * grid.cell_area
