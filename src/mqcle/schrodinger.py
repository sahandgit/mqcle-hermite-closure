r"""The independent reference: exact Schroedinger propagation on a grid.

Why this module exists
----------------------
The closed coefficient equations of Theorem 2 *are* the subsystem-basis QCLE
(Gate J, ``results/raw/03_qcle_correspondence.json``).  A separately written
QCLE grid solver is useful for code verification but still solves the same
physical approximation.  The genuinely independent physical reference is the full time-dependent
Schroedinger equation for the coupled electronic-nuclear problem,

.. math::
    i\hbar\,\partial_t\psi_\lambda(R,t)
    =-\frac{\hbar^2}{2M}\partial_R^2\psi_\lambda
     +\sum_{\lambda'}\bigl[V_b(R)\delta_{\lambda\lambda'}
                            +h_{\lambda\lambda'}(R)\bigr]\psi_{\lambda'} ,

which makes no quantum-classical approximation at all.  Any residual gap
between the coefficient solver and this reference is QCLE's own approximation
error, not the solver's.

Method
------
Strang split-operator on a periodic Fourier grid:

.. math::
    U(\Delta t)=e^{-iT\Delta t/2\hbar}\,e^{-iV\Delta t/\hbar}\,
                e^{-iT\Delta t/2\hbar}+\mathcal O(\Delta t^3),

with the kinetic factor diagonal in :math:`k` and the potential factor the
exact :math:`2\times2` matrix exponential of the Hermitian :math:`V(R)`, built
from its eigendecomposition.  Every factor is unitary, so the norm is
conserved to machine precision independently of :math:`\Delta t` -- norm
conservation is therefore *not* evidence of accuracy here, and the convergence
tests in ``tests/test_schrodinger.py`` check the observables instead.

Provenance
----------
Written for this project.  If the ``qcle_grid_tully.py`` /
``Compare_gp_se_qcle.py`` reference solvers of the GP-RKHS-MInt project are
made available, run both and record the agreement; two independent
implementations agreeing is stronger evidence than either alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import CDTYPE, DTYPE, HBAR
from .models import TwoStateModel

__all__ = [
    "SchrodingerGrid",
    "build_schrodinger_grid",
    "gaussian_wavepacket",
    "SplitOperatorPropagator",
    "wavefunction_observables",
]


@dataclass(frozen=True)
class SchrodingerGrid:
    """Uniform periodic grid in ``R`` for the nuclear wavefunction."""

    R: np.ndarray
    dR: float
    k: np.ndarray
    L: float

    @property
    def n(self) -> int:
        return int(self.R.size)

    def parameters(self) -> dict[str, Any]:
        return {"R_min": float(self.R[0]), "R_max": float(self.R[0] + self.L),
                "n_R": self.n, "dR": float(self.dR)}


def build_schrodinger_grid(R_min: float, R_max: float, n_R: int) -> SchrodingerGrid:
    L = float(R_max - R_min)
    R = R_min + L * np.arange(n_R, dtype=DTYPE) / n_R
    dR = L / n_R
    k = 2.0 * np.pi * np.fft.fftfreq(n_R, d=dR)
    return SchrodingerGrid(R=R, dR=dR, k=k, L=L)


def gaussian_wavepacket(grid: SchrodingerGrid, R0: float, P0: float,
                        sigma_R: float, init_state: int = 0, nstates: int = 2,
                        hbar: float = HBAR) -> np.ndarray:
    r"""Minimum-uncertainty packet :math:`(2\pi\sigma_R^2)^{-1/4}
    e^{-(R-R_0)^2/4\sigma_R^2+iP_0R/\hbar}` in one diabatic state.

    Its Wigner transform is exactly the field built by
    :func:`mqcle.propagation.initial_coefficient_field`, so the two branches
    start from the same physical state.
    """
    R = grid.R
    env = np.exp(-((R - R0) ** 2) / (4.0 * sigma_R**2))
    psi1 = (2.0 * np.pi * sigma_R**2) ** (-0.25) * env * \
        np.exp(1j * P0 * (R - R0) / hbar)
    psi = np.zeros((grid.n, nstates), dtype=CDTYPE)
    psi[:, init_state] = psi1
    nrm = np.sqrt(np.sum(np.abs(psi) ** 2) * grid.dR)
    return psi / nrm


def wigner_transform_at(psi: np.ndarray, grid: "SchrodingerGrid",
                        R_out: np.ndarray, P_out: np.ndarray, *,
                        hbar: float = HBAR,
                        s_window: float | None = None) -> np.ndarray:
    r"""Partial Wigner transform of :math:`|\psi\rangle\langle\psi|`.

    .. math::
        \rho_W^{\lambda\lambda'}(R,P)=\frac{1}{\pi\hbar}\int ds\,
        \psi_\lambda(R+s)\,\psi_{\lambda'}^{*}(R-s)\,e^{-2iPs/\hbar},

    which is the object the quantum--classical Liouville equation approximates,
    so it is what the closed coefficient field and the grid field have to be
    compared against.  Returned on the requested nodes with shape
    ``(R_out.size, P_out.size, nstates, nstates)``, matching the layout of
    :func:`mqcle.propagation.initial_coefficient_field`.

    ``R_out`` must be a subset of the wavefunction's own nodes: the shift by
    :math:`\pm s` has to land on grid points, so :math:`s` runs over integer
    multiples of ``grid.dR`` and the transform is evaluated exactly, with no
    interpolation in either variable.  The :math:`P` integral is done as a
    direct sum at the requested momenta rather than by an FFT onto whatever
    grid the FFT would dictate, again to avoid interpolating.

    ``s_window`` truncates the :math:`s` sum to :math:`|s|\le` that value, and
    defaults to a quarter of the periodic domain.  This is not an
    approximation but the removal of one: on a periodic box, :math:`s` running
    over the whole period lets :math:`R+s` and :math:`R-s` wrap around and both
    land on the packet when :math:`R` is half a period away from it, which
    paints a full-amplitude ghost of the density at the antipode.  Capping
    :math:`|s|` at :math:`L/4` keeps every genuine correlation out to a
    separation of half the box---which is what a split packet on two surfaces
    needs---and removes the wrap.  Checked rather than argued: against the
    analytic Wigner transform of the initial Gaussian, the default reproduces
    :func:`mqcle.propagation.initial_coefficient_field` to
    :math:`2.2\times10^{-16}` and integrates to one, while the untruncated sum
    is wrong by the full peak height.
    """
    psi = np.asarray(psi, dtype=CDTYPE)
    if psi.ndim != 2 or psi.shape[0] != grid.n:
        raise ValueError(f"psi must be (n_R, nstates) on this grid, got "
                         f"{psi.shape} for n_R={grid.n}")
    nR_full, nstates = psi.shape
    R_out = np.asarray(R_out, dtype=DTYPE)
    P_out = np.asarray(P_out, dtype=DTYPE)

    # index of each requested R among the wavefunction's nodes
    idx = np.rint((R_out - grid.R[0]) / grid.dR).astype(np.int64)
    if np.any(np.abs(grid.R[np.mod(idx, nR_full)] - R_out) > 1e-9 * grid.dR):
        raise ValueError("R_out must fall on the wavefunction's own nodes; "
                         "the Wigner transform needs psi at R +/- s exactly")

    window = 0.25 * grid.L if s_window is None else float(s_window)
    M = min(nR_full // 2, int(np.floor(window / grid.dR)))
    m = np.arange(-M, M + 1, dtype=np.int64)

    jp = np.mod(idx[:, None] + m[None, :], nR_full)          # (nR, 2M+1)
    jm = np.mod(idx[:, None] - m[None, :], nR_full)
    # e^{-2 i P s / hbar} at the requested momenta, s = m dR
    kern = np.exp(-2j * np.outer(m.astype(DTYPE) * grid.dR, P_out) / hbar)

    out = np.empty((R_out.size, P_out.size, nstates, nstates), dtype=CDTYPE)
    pref = grid.dR / (np.pi * hbar)
    for a in range(nstates):
        pa = psi[:, a]
        for b in range(nstates):
            corr = pa[jp] * np.conj(psi[:, b])[jm]           # (nR, 2M+1)
            out[:, :, a, b] = pref * (corr @ kern)
    return out


class SplitOperatorPropagator:
    """Strang split-operator propagator for the coupled two-state problem."""

    def __init__(self, grid: SchrodingerGrid, model: TwoStateModel,
                 dt: float, hbar: float = HBAR) -> None:
        if model.nbath != 1:
            raise ValueError("schrodinger: this reference is 1D in the bath")
        self.grid, self.model, self.dt, self.hbar = grid, model, float(dt), float(hbar)
        Rcol = grid.R[:, None]
        n = model.nstates
        V = model.h(Rcol) + model.bath_potential(Rcol)[:, None, None] * np.eye(n)
        w, U = np.linalg.eigh(V)                                  # (n_R, N), (n_R, N, N)
        phase = np.exp(-1j * w * self.dt / self.hbar)             # (n_R, N)
        self._expV = np.einsum("rab,rb,rcb->rac", U, phase, np.conj(U))
        self._expT_half = np.exp(-1j * self.hbar * grid.k**2 * self.dt
                                 / (4.0 * model.mass))[:, None]
        self._V, self._w, self._U = V, w, U

    # -- one step ---------------------------------------------------------
    def _kinetic_half(self, psi: np.ndarray) -> np.ndarray:
        return np.fft.ifft(self._expT_half * np.fft.fft(psi, axis=0), axis=0)

    def step(self, psi: np.ndarray) -> np.ndarray:
        psi = self._kinetic_half(psi)
        psi = np.einsum("rab,rb->ra", self._expV, psi)
        return self._kinetic_half(psi)

    def propagate(self, psi: np.ndarray, n_steps: int, n_save: int = 10,
                  snapshot_times: tuple[float, ...] = ()
                  ) -> tuple[list[float], list[dict[str, Any]], dict[float, np.ndarray]]:
        times: list[float] = [0.0]
        recs = [wavefunction_observables(psi, self.grid, self.model, self.hbar)]
        snaps: dict[float, np.ndarray] = {}
        want = sorted(snapshot_times)
        if want and want[0] <= 0.0:
            snaps[0.0] = psi.copy()
            want.pop(0)
        for s in range(1, n_steps + 1):
            psi = self.step(psi)
            t = s * self.dt
            if s % n_save == 0 or s == n_steps:
                times.append(t)
                recs.append(wavefunction_observables(psi, self.grid, self.model,
                                                     self.hbar))
            while want and want[0] <= t + 0.5 * self.dt:
                snaps[float(want.pop(0))] = psi.copy()
        return times, recs, snaps


def wavefunction_observables(psi: np.ndarray, grid: SchrodingerGrid,
                             model: TwoStateModel, hbar: float = HBAR
                             ) -> dict[str, Any]:
    r"""Diabatic and adiabatic populations, coherence, ``<R>``, ``<P>``, norm."""
    dR = grid.dR
    dens = np.abs(psi) ** 2
    norm = float(np.sum(dens) * dR)
    pops = np.sum(dens, axis=0) * dR
    coh = np.sum(np.conj(psi[:, 0]) * psi[:, 1]) * dR

    _, U = np.linalg.eigh(model.h(grid.R[:, None]))               # (n_R, N, N)
    chi = np.einsum("rba,rb->ra", np.conj(U), psi)                # adiabatic amps
    ad = np.sum(np.abs(chi) ** 2, axis=0) * dR

    psik = np.fft.fft(psi, axis=0)
    kw = np.sum(np.abs(psik) ** 2 * grid.k[:, None]) / np.sum(np.abs(psik) ** 2)
    return {
        "norm": norm,
        "populations": np.real(pops).copy(),
        "adiabatic_populations": np.real(ad).copy(),
        "coherence_re": float(np.real(coh)),
        "coherence_im": float(np.imag(coh)),
        "mean_R": float(np.sum(dens * grid.R[:, None]) * dR / norm),
        "mean_P": float(hbar * kw),
        "boundary_mass": float((np.sum(dens[0]) + np.sum(dens[-1])) * dR
                               / max(norm, 1e-300)),
    }
