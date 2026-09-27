r"""Primary unsplit propagator for the Gaussian--Hermite coefficient system.

The evolved unknown is the Hermite coefficient field ``C[a, i_R, i_P]``.  The
semi-discrete generator of Eq. (L-H),

.. math::

    \mathbb L_H[C]=-\frac{P}{M}D_RC+\mathsf R(R)C+\mathsf G(R)D_PC,

is applied as a single unsplit operator.  Two routes are provided and both
call the same right-hand side:

``propagate_krylov``
    matrix-free Krylov exponential action, Eq. (Krylov-step).  Exact for a
    time-independent generator up to the Krylov tolerance, and the preferred
    route whenever the state fits in memory several times over.

``propagate_rk4``
    an independent method-of-lines path for convergence checks and for
    production sizes where the Krylov basis does not fit.  Conditionally
    stable, so the step is bounded by the estimated spectral radius.

Neither route reduces the state to :math:`\rho_W`, and neither calls
``StrangPropagator``.  The subsystem-QCLE Strang solver remains in
:mod:`mqcle.propagation` as an independent cross-check only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator, expm_multiply

from mqcle.constants import CDTYPE, HBAR
from mqcle.hermite.operator import (active_modes, d_real_hamiltonian_dR,
                                    force_matrix, hermite_rhs,
                                    hermite_T_matrices, real_hamiltonian,
                                    rotation_basis_matrices, rotation_matrix,
                                    sparse_blocks)

__all__ = ["HermiteGenerator", "propagate_krylov", "propagate_rk4"]


@dataclass
class HermiteGenerator:
    r"""Precomputed, ``R``-local matrices of the Hermite generator.

    ``Rmat`` and ``Gmat`` have shape ``(n_R, M, M)`` and depend on the bath
    position alone, so they are built once and broadcast over ``P`` and time.
    The state is **mode-major**, ``(M, n_R, n_P)``; see :func:`hermite_rhs`
    for the measurement behind that choice.
    """

    grid: object
    mass: float
    modes: list
    Rmat: np.ndarray
    Gmat: np.ndarray
    hbar: float = HBAR
    _shape: tuple = field(init=False)
    _R_blocks: list = field(init=False, default_factory=list)
    _G_blocks: list = field(init=False, default_factory=list)

    def __post_init__(self):
        n_R, n_P = self.grid.shape
        self._shape = (len(self.modes), n_R, n_P)
        self._R_blocks = sparse_blocks(self.Rmat)
        self._G_blocks = sparse_blocks(self.Gmat)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_model(cls, model, grid, K: int = 2, parity: str | None = "even",
                   hbar: float = HBAR, modes=None) -> "HermiteGenerator":
        r"""Build the generator for ``model`` on ``grid``.

        The full subsystem Hamiltonian enters :math:`\mathbb H`, and the bare
        bath potential supplies :math:`\partial_RV_b`.  That pairing is the
        convention of Eq. (Hm-general); mixing it with the traceless
        :math:`\bar h` breaks the force/excess cancellation.
        """
        if model.nbath != 1:
            raise ValueError("HermiteGenerator: the (R,P) grid needs nbath == 1")
        if modes is None:
            modes = active_modes(model.nstates, K, parity=parity)

        Rcol = grid.R[:, None]                          # (n_R, 1)
        h = model.h(Rcol)                               # (n_R, N, N)
        dh = model.dh_dR(Rcol)[:, 0]                    # (n_R, N, N)
        dvb = model.dbath_dR(Rcol)[..., 0]              # (n_R,)

        H_real = real_hamiltonian(h)
        dH_real = d_real_hamiltonian_dR(dh)

        T = hermite_T_matrices(modes)
        N_ij = rotation_basis_matrices(modes)
        Rmat = rotation_matrix(H_real, hbar, modes, N_ij=N_ij)
        Gmat = force_matrix(dvb, dH_real, T, modes)
        return cls(grid=grid, mass=model.mass, modes=list(modes),
                   Rmat=Rmat, Gmat=Gmat, hbar=hbar)

    # -- action -------------------------------------------------------------
    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def ndof(self) -> int:
        return int(np.prod(self._shape))

    def rhs(self, C: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Apply :math:`\\mathbb L_H` to a mode-major ``(M, n_R, n_P)`` state."""
        return hermite_rhs(C, self.grid, self._R_blocks, self._G_blocks,
                           self.mass, out=out)

    def rhs_adjoint(self, C: np.ndarray) -> np.ndarray:
        r"""Apply :math:`\mathbb L_H^\dagger`, needed by the Krylov norm estimate.

        The spectral derivatives are anti-Hermitian, :math:`D^\dagger=-D`, and
        both :math:`\mathsf R` and :math:`\mathsf G` are ``R``-local, so they
        commute with :math:`D_P` and with multiplication by :math:`P/M`.  Hence

        .. math::

            \mathbb L_H^\dagger C
            =+\frac{P}{M}D_RC+\mathsf R^\dagger C-\mathsf G^\dagger D_PC .
        """
        from mqcle.coefficient_operator import spectral_derivative
        C = np.asarray(C, dtype=CDTYPE)
        dC_dR = spectral_derivative(C, self.grid.kR, axis=1)
        dC_dP = spectral_derivative(C, self.grid.kP, axis=2)
        out = (self.grid.P[None, None, :] / self.mass) * dC_dR
        for m, n, coeff in self._R_blocks:
            out[n] += np.conj(coeff)[:, None] * C[m]
        for m, n, coeff in self._G_blocks:
            out[n] -= np.conj(coeff)[:, None] * dC_dP[m]
        return out

    @property
    def n_blocks(self) -> tuple:
        """Nonzero block counts of ``R`` and ``G``, for the cost record."""
        return len(self._R_blocks), len(self._G_blocks)

    # -- physical frame helpers (mode axis bookkeeping lives here) ----------
    @staticmethod
    def to_state(C_mode_minor: np.ndarray) -> np.ndarray:
        """``(n_R, n_P, M)`` from the frame helpers -> mode-major state."""
        return np.ascontiguousarray(np.moveaxis(C_mode_minor, -1, 0))

    @staticmethod
    def from_state(C: np.ndarray) -> np.ndarray:
        """Mode-major state -> ``(n_R, n_P, M)`` for the frame helpers."""
        return np.ascontiguousarray(np.moveaxis(C, 0, -1))

    def as_linear_operator(self) -> LinearOperator:
        """Matrix-free ``LinearOperator``; the global matrix is never formed."""
        shape = self._shape
        n = self.ndof

        def matvec(v):
            return self.rhs(v.reshape(shape)).ravel()

        def rmatvec(v):
            return self.rhs_adjoint(v.reshape(shape)).ravel()

        return LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec,
                              dtype=CDTYPE)

    # -- stability ----------------------------------------------------------
    def spectral_radius_estimate(self) -> dict:
        r"""Upper bound on :math:`\|\mathbb L_H\|`, split by channel.

        Advection contributes :math:`\max|P|/M\cdot\max|k_R|`, the rotation
        :math:`\max_R\|\mathsf R\|_2`, and the force
        :math:`\max_R\|\mathsf G\|_2\cdot\max|k_P|`.  The sum is a bound, not
        the true radius; it is used only to choose a defensible explicit step.
        """
        g = self.grid
        adv = float(np.max(np.abs(g.P)) / self.mass * np.max(np.abs(g.kR)))
        rot = float(np.max(np.linalg.norm(self.Rmat, ord=2, axis=(1, 2))))
        frc = float(np.max(np.linalg.norm(self.Gmat, ord=2, axis=(1, 2)))
                    * np.max(np.abs(g.kP)))
        total = adv + rot + frc
        return {"advection": adv, "rotation": rot, "force": frc,
                "total": total,
                "rk4_dt_limit": (2.0 * np.sqrt(2.0) / total) if total else np.inf}


# --------------------------------------------------------------------------
# propagation
# --------------------------------------------------------------------------
def propagate_krylov(gen: HermiteGenerator, C0: np.ndarray,
                     times, rtol: float = 1e-10) -> np.ndarray:
    r"""Krylov exponential action at the requested times.

    Advances :math:`C(t+\Delta t)=\exp(\Delta t\,\mathbb L_H)C(t)` one saved
    interval at a time through :func:`scipy.sparse.linalg.expm_multiply` on a
    matrix-free operator.

    The generator is exactly traceless: the two bath-derivative channels have
    zero diagonal on a symmetric spectral grid, and
    :math:`\mathrm{tr}\,\mathsf R=-\sum_iK_{ii}n_i=0` because ``K`` is
    antisymmetric.  ``traceA=0`` is therefore supplied exactly rather than
    estimated.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("propagate_krylov: times must be a 1-D array")
    A = gen.as_linear_operator()
    shape = gen.shape

    out = np.empty((times.size,) + shape, dtype=CDTYPE)
    v = np.asarray(C0, dtype=CDTYPE).ravel().copy()
    t_prev = 0.0
    for i, t in enumerate(times):
        dt = float(t) - t_prev
        if dt < 0:
            raise ValueError("propagate_krylov: times must be nondecreasing")
        if dt > 0:
            v = expm_multiply(A, v, start=0.0, stop=dt, num=2,
                              endpoint=True, traceA=0.0)[-1]
        out[i] = v.reshape(shape)
        t_prev = float(t)
    return out


def propagate_rk4(gen: HermiteGenerator, C0: np.ndarray, dt: float,
                  n_steps: int, save_every: int = 1, callback=None
                  ) -> tuple[np.ndarray, np.ndarray]:
    r"""Classical fourth-order Runge--Kutta on the same unsplit generator.

    Returns ``(times, states)``.  Conditionally stable: the caller should
    respect :meth:`HermiteGenerator.spectral_radius_estimate`, whose
    ``rk4_dt_limit`` is :math:`2\sqrt2/\lambda`.

    ``callback(step, t, C)`` runs after each saved step and may record
    diagnostics without retaining the whole trajectory.
    """
    C = np.asarray(C0, dtype=CDTYPE).copy()
    saved_t = [0.0]
    saved = [C.copy()]
    if callback is not None:
        callback(0, 0.0, C)

    for step in range(1, n_steps + 1):
        k1 = gen.rhs(C)
        k2 = gen.rhs(C + 0.5 * dt * k1)
        k3 = gen.rhs(C + 0.5 * dt * k2)
        k4 = gen.rhs(C + dt * k3)
        C = C + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if step % save_every == 0 or step == n_steps:
            t = step * dt
            saved_t.append(t)
            saved.append(C.copy())
            if callback is not None:
                callback(step, t, C)
    return np.asarray(saved_t), np.asarray(saved)
