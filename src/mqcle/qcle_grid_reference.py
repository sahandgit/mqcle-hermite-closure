r"""Independent real-component grid solver for the two-state QCLE.

This module is adapted from the user's ``qcle_grid_tully.py`` reference
implementation.  It deliberately does *not* call
``mqcle.coefficient_operator.coefficient_rhs`` or the coefficient propagator.
Instead it evolves the four real fields

``A = rho_00, C = rho_11, bR = Re(rho_01), bI = Im(rho_01)``

with real Fourier pseudospectral derivatives and classical RK4.  The closed
Mapping-QCLE coefficient code evolves one complex matrix field with a Strang
split propagator.  Agreement of the two implementations is therefore an
independent numerical check that both integrate the subsystem QCLE.  It is not
a second physical theory and does not replace the full-quantum reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.fft as sfft

from .constants import CDTYPE, DTYPE, HBAR
from .models import TwoStateModel

__all__ = [
    "QCLEGridParams",
    "QCLEGridState",
    "QCLEGridHistory",
    "IndependentQCLEGridSolver",
]


@dataclass(frozen=True)
class QCLEGridParams:
    """Uniform periodic phase-space grid and physical constants."""

    R_min: float
    R_max: float
    n_R: int
    P_min: float
    P_max: float
    n_P: int
    mass: float = 2000.0
    hbar: float = HBAR
    fft_workers: int = -1

    def __post_init__(self) -> None:
        if not self.R_max > self.R_min:
            raise ValueError("R_max must exceed R_min")
        if not self.P_max > self.P_min:
            raise ValueError("P_max must exceed P_min")
        if self.n_R < 8 or self.n_P < 8:
            raise ValueError("n_R and n_P must be at least 8")
        if self.mass <= 0.0 or self.hbar <= 0.0:
            raise ValueError("mass and hbar must be positive")
        if self.fft_workers == 0:
            raise ValueError("fft_workers cannot be zero")


@dataclass
class QCLEGridState:
    """The four independent real components of a Hermitian two-state field."""

    A: np.ndarray
    C: np.ndarray
    bR: np.ndarray
    bI: np.ndarray

    def copy(self) -> "QCLEGridState":
        return QCLEGridState(*(np.asarray(x, dtype=DTYPE).copy()
                               for x in (self.A, self.C, self.bR, self.bI)))

    def as_matrix(self) -> np.ndarray:
        """Return the equivalent ``(..., 2, 2)`` complex density field."""
        shape = np.broadcast_shapes(self.A.shape, self.C.shape,
                                    self.bR.shape, self.bI.shape)
        out = np.empty(shape + (2, 2), dtype=CDTYPE)
        out[..., 0, 0] = self.A
        out[..., 1, 1] = self.C
        out[..., 0, 1] = self.bR + 1j * self.bI
        out[..., 1, 0] = self.bR - 1j * self.bI
        return out

    @classmethod
    def from_matrix(cls, rho: np.ndarray, hermiticity_tol: float = 1e-12
                    ) -> "QCLEGridState":
        """Convert a Hermitian complex field to the real-component form."""
        rho = np.asarray(rho, dtype=CDTYPE)
        if rho.ndim < 2 or rho.shape[-2:] != (2, 2):
            raise ValueError("rho must end in a 2x2 matrix shape")
        scale = max(float(np.max(np.abs(rho))), 1e-300)
        defect = float(np.max(np.abs(rho - np.swapaxes(rho.conj(), -1, -2))))
        if defect > hermiticity_tol * scale:
            raise ValueError(f"rho is not Hermitian: relative defect {defect/scale:.3e}")
        return cls(np.real(rho[..., 0, 0]).copy(),
                   np.real(rho[..., 1, 1]).copy(),
                   np.real(rho[..., 0, 1]).copy(),
                   np.imag(rho[..., 0, 1]).copy())


@dataclass
class QCLEGridHistory:
    """Saved scalar observables and the final real-component field."""

    times: list[float] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)
    final_state: QCLEGridState | None = None

    def column(self, key: str) -> np.ndarray:
        return np.asarray([r[key] for r in self.records])

    def as_arrays(self) -> dict[str, np.ndarray]:
        out = {"t": np.asarray(self.times, dtype=DTYPE)}
        if self.records:
            for key in self.records[0]:
                out[key] = self.column(key)
        return out


class IndependentQCLEGridSolver:
    r"""Real-FFT/RK4 solver for the diabatic two-state QCLE.

    The Hamiltonian used here is the full potential matrix
    ``V_b(R) I + h(R)``.  With ``rho_01 = bR + i bI``, the four equations are

    .. math::

       \dot A &= -P\partial_R A/M + V'_{11}\partial_P A
                 + V'_{12}\partial_P b_R - 2V_{12}b_I/\hbar,\\
       \dot C &= -P\partial_R C/M + V'_{22}\partial_P C
                 + V'_{12}\partial_P b_R + 2V_{12}b_I/\hbar,\\
       \dot b_R &= -P\partial_R b_R/M + V'_0\partial_P b_R
                 + V'_{12}(\partial_P A+\partial_P C)/2
                 +(V_{11}-V_{22})b_I/\hbar,\\
       \dot b_I &= -P\partial_R b_I/M + V'_0\partial_P b_I
                 +[V_{12}(A-C)-(V_{11}-V_{22})b_R]/\hbar.
    """

    def __init__(self, model: TwoStateModel, params: QCLEGridParams) -> None:
        if model.nstates != 2 or model.nbath != 1:
            raise ValueError("independent QCLE grid solver requires two states and one bath mode")
        self.model = model
        self.params = params
        p = params
        self.dR = (p.R_max - p.R_min) / p.n_R
        self.dP = (p.P_max - p.P_min) / p.n_P
        # Retain the cell-centred convention of the independent source code.
        self.R = p.R_min + self.dR * (0.5 + np.arange(p.n_R, dtype=DTYPE))
        self.P = p.P_min + self.dP * (0.5 + np.arange(p.n_P, dtype=DTYPE))
        self.kR = 2.0 * np.pi * np.fft.fftfreq(p.n_R, d=self.dR)
        self.kP = 2.0 * np.pi * np.fft.fftfreq(p.n_P, d=self.dP)
        self._ikR = (1j * 2.0 * np.pi
                     * sfft.rfftfreq(p.n_R, d=self.dR))[:, None]
        self._ikP = (1j * 2.0 * np.pi
                     * sfft.rfftfreq(p.n_P, d=self.dP))[None, :]

        Rarg = self.R[:, None]
        h = np.asarray(model.h(Rarg), dtype=DTYPE)
        dh = np.asarray(model.dh_dR(Rarg), dtype=DTYPE)[:, 0]
        vb = np.asarray(model.bath_potential(Rarg), dtype=DTYPE)
        dvb = np.asarray(model.dbath_dR(Rarg), dtype=DTYPE)[..., 0]
        eye = np.eye(2, dtype=DTYPE)
        self._V = h + vb[:, None, None] * eye
        self._dV = dh + dvb[:, None, None] * eye
        self._adiabatic_vectors = np.linalg.eigh(self._V)[1]

        self._V11 = self._V[:, 0, 0][:, None]
        self._V22 = self._V[:, 1, 1][:, None]
        self._V12 = self._V[:, 0, 1][:, None]
        self._V11p = self._dV[:, 0, 0][:, None]
        self._V22p = self._dV[:, 1, 1][:, None]
        self._V12p = self._dV[:, 0, 1][:, None]
        self._V0p = 0.5 * (self._V11p + self._V22p)
        self._gap = self._V11 - self._V22
        self._advection = -self.P[None, :] / p.mass
        #: RK4 stage buffers, reused between steps.  The solver is driven from
        #: one thread, so this is safe and it removes four large allocations
        #: per stage at production resolution.
        self._stage = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.params.n_R, self.params.n_P

    @property
    def cell_area(self) -> float:
        return float(self.dR * self.dP)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.R, self.P, indexing="ij")

    def _dR(self, f: np.ndarray) -> np.ndarray:
        g = sfft.rfft(f, axis=0, workers=self.params.fft_workers)
        g *= self._ikR
        return sfft.irfft(g, axis=0, n=self.params.n_R,
                          workers=self.params.fft_workers)

    def _dP(self, f: np.ndarray) -> np.ndarray:
        g = sfft.rfft(f, axis=1, workers=self.params.fft_workers)
        g *= self._ikP
        return sfft.irfft(g, axis=1, n=self.params.n_P,
                          workers=self.params.fft_workers)

    def rhs(self, A: np.ndarray, C: np.ndarray, bR: np.ndarray,
            bI: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the four real QCLE equations independently."""
        expected = self.shape
        if any(np.shape(x) != expected for x in (A, C, bR, bI)):
            raise ValueError(f"all fields must have shape {expected}")
        dA_R, dC_R = self._dR(A), self._dR(C)
        dbR_R, dbI_R = self._dR(bR), self._dR(bI)
        dA_P, dC_P = self._dP(A), self._dP(C)
        dbR_P, dbI_P = self._dP(bR), self._dP(bI)
        inv_h = 1.0 / self.params.hbar
        adv, V0p, V12, V12p, gap = (self._advection, self._V0p, self._V12,
                                    self._V12p, self._gap)

        # Written as accumulations into the derivative arrays, which were just
        # allocated here and are not visible to the caller.  At production
        # resolution each field is tens of megabytes and the expression form
        # of these four lines allocated a dozen more of them per evaluation,
        # four times per RK4 step; the traffic, not the arithmetic, was the
        # cost.  This is the same expression in the same order, and the
        # regression test asserts the result is bit-identical, not merely
        # close.
        shared = V12p * dbR_P
        mixing = bI * (2.0 * inv_h) * V12

        rhs_A = dA_R
        rhs_A *= adv
        rhs_A += self._V11p * dA_P
        rhs_A += shared
        rhs_A -= mixing

        rhs_C = dC_R
        rhs_C *= adv
        rhs_C += self._V22p * dC_P
        rhs_C += shared
        rhs_C += mixing

        rhs_bR = dbR_R
        rhs_bR *= adv
        rhs_bR += V0p * dbR_P
        dA_P += dC_P                      # dA_P and dC_P are dead after this
        dA_P *= 0.5 * V12p
        rhs_bR += dA_P
        rhs_bR += (inv_h * gap) * bI

        rhs_bI = dbI_R
        rhs_bI *= adv
        rhs_bI += V0p * dbI_P
        rhs_bI += (inv_h * V12) * (A - C)
        rhs_bI -= (inv_h * gap) * bR
        return rhs_A, rhs_C, rhs_bR, rhs_bI

    def rhs_state(self, state: QCLEGridState) -> QCLEGridState:
        return QCLEGridState(*self.rhs(state.A, state.C, state.bR, state.bI))

    def step(self, state: QCLEGridState, dt: float) -> QCLEGridState:
        """Advance one classical RK4 step using a memory-bounded accumulation."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        A, C, bR, bI = state.A, state.C, state.bR, state.bI
        h2, h6 = 0.5 * dt, dt / 6.0
        base = (A, C, bR, bI)
        if self._stage is None or self._stage[0].shape != A.shape:
            self._stage = tuple(np.empty_like(A) for _ in range(4))
        stage = self._stage

        def _stage_at(ks, h):
            """base + h*k, into the reused stage buffers."""
            for out, x, k in zip(stage, base, ks):
                np.multiply(k, h, out=out)
                out += x
            return stage

        k1 = list(self.rhs(A, C, bR, bI))
        acc = k1
        k2 = self.rhs(*_stage_at(k1, h2))
        for total, k in zip(acc, k2):
            total += 2.0 * k
        k3 = self.rhs(*_stage_at(k2, h2))
        for total, k in zip(acc, k3):
            total += 2.0 * k
        k4 = self.rhs(*_stage_at(k3, dt))
        for total, k in zip(acc, k4):
            total += k
        out = []
        for x, total in zip(base, acc):
            total *= h6
            total += x
            out.append(total)
        return QCLEGridState(*out)

    def initial_diabat_gaussian(self, R0: float, P0: float, sigma_R: float,
                                init_state: int = 0) -> QCLEGridState:
        if init_state not in (0, 1):
            raise ValueError("init_state must be 0 or 1")
        if sigma_R <= 0.0:
            raise ValueError("sigma_R must be positive")
        RR, PP = self.mesh()
        h = self.params.hbar
        W = np.exp(-((RR - R0) ** 2) / (2.0 * sigma_R**2)
                   - 2.0 * sigma_R**2 * (PP - P0) ** 2 / h**2) / (np.pi * h)
        zero = np.zeros_like(W)
        if init_state == 0:
            return QCLEGridState(W, zero.copy(), zero.copy(), zero.copy())
        return QCLEGridState(zero.copy(), W, zero.copy(), zero.copy())

    def observables(self, state: QCLEGridState) -> dict:
        """Return invariants, populations, moments, and edge sentinels."""
        area = self.cell_area
        density = state.A + state.C
        tr = float(np.sum(density) * area)
        safe = max(abs(tr), 1e-300)
        pops = np.array([np.sum(state.A), np.sum(state.C)], dtype=DTYPE) * area
        U = self._adiabatic_vectors
        adiabatic = np.empty(2, dtype=DTYPE)
        for a in range(2):
            density_a = (U[:, 0, a][:, None] ** 2 * state.A
                         + U[:, 1, a][:, None] ** 2 * state.C
                         + 2.0 * U[:, 0, a][:, None]
                         * U[:, 1, a][:, None] * state.bR)
            adiabatic[a] = np.sum(density_a) * area
        coherence = np.sum(state.bR + 1j * state.bI) * area
        RR, PP = self.mesh()
        energy_density = ((PP**2 / (2.0 * self.params.mass)) * density
                          + self._V11 * state.A + self._V22 * state.C
                          + 2.0 * self._V12 * state.bR)
        bR_edge = float(np.sum(np.abs(density[0]) + np.abs(density[-1]))
                         * area / safe)
        bP_edge = float(np.sum(np.abs(density[:, 0]) + np.abs(density[:, -1]))
                         * area / safe)
        return {
            "trace": tr,
            "populations": pops,
            "adiabatic_populations": adiabatic,
            "coherence_re": float(np.real(coherence)),
            "coherence_im": float(np.imag(coherence)),
            "mean_R": float(np.sum(RR * density) * area / safe),
            "mean_P": float(np.sum(PP * density) * area / safe),
            "energy": float(np.sum(energy_density) * area),
            "boundary_mass_R": bR_edge,
            "boundary_mass_P": bP_edge,
            "boundary_mass": bR_edge + bP_edge,
        }

    def stability(self) -> dict[str, float]:
        """Conservative RK4 imaginary-axis stability estimate."""
        kR = float(np.max(np.abs(self.kR)))
        kP = float(np.max(np.abs(self.kP)))
        advection = float(np.max(np.abs(self.P)) * kR / self.params.mass)
        force_eigs = np.linalg.eigvalsh(self._dV)
        force = float(np.max(np.abs(force_eigs)) * kP)
        eigs = np.linalg.eigvalsh(self._V)
        electronic = float(np.max(eigs[:, 1] - eigs[:, 0]) / self.params.hbar)
        radius = advection + force + electronic
        return {
            "advection": advection,
            "force": force,
            "electronic": electronic,
            "spectral_radius": radius,
            "max_dt_rk4": float(2.0 * np.sqrt(2.0) / max(radius, 1e-300)),
        }

    def propagate(self, state0: QCLEGridState, dt: float, n_steps: int,
                  n_save: int = 10,
                  progress: Callable[[int, int], None] | None = None
                  ) -> QCLEGridHistory:
        """Propagate while retaining only scalar records and the final field."""
        if n_steps < 1 or n_save < 1:
            raise ValueError("n_steps and n_save must be positive")
        limit = self.stability()["max_dt_rk4"]
        if dt > limit:
            raise ValueError(f"RK4 is unstable: dt={dt:g} exceeds {limit:.6g}")
        state = state0.copy()
        history = QCLEGridHistory(times=[0.0], records=[self.observables(state)])
        for step in range(1, n_steps + 1):
            state = self.step(state, dt)
            if step % n_save == 0 or step == n_steps:
                history.times.append(step * dt)
                history.records.append(self.observables(state))
            if progress is not None:
                progress(step, n_steps)
        history.final_state = state
        return history
