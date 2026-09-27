r"""Gate L, part 2 -- time propagation of the closed coefficients, and of the
PBME trajectory ensemble, from a common initial condition.

Two branches, one initial state
-------------------------------
``propagate_coefficients``
    Classical Runge-Kutta 4 on the grid right-hand side of
    :mod:`mqcle.coefficient_operator`.  Because the closed coefficient
    equations *are* the subsystem QCLE (Gate J), this branch is a solver for
    the QCLE, and any residual disagreement with the exact quantum reference
    is QCLE's own approximation error, not the solver's.

``propagate_pbme_projected_ensemble``
    The PBME *operator* benchmark.  Mapping variables are drawn from an
    importance density and carry the signed weights of the projected density;
    the readout is :func:`mqcle.observables.rho_W_from_ensemble`.  Because the
    PBME flow is Liouvillian, this estimator is unbiased for the PBME density
    observable up to time-integration and finite-sampling error, both of which
    are measured rather than assumed.  This -- and only this -- is what may be
    differenced against the coefficient branch to measure the consequence of
    discarding :math:`\mathcal A'`.

``propagate_pbme_ensemble``
    The *practical focused trajectory* PBME: focused MMST sampling, uniform
    weights, and the focused second-moment readout
    :func:`mqcle.observables.rho_W_from_focused_ensemble`.  **Not**
    ``rho_W_from_ensemble``, which expects importance weights that focused
    sampling does not provide.  That readout is exact at ``t=0`` and under a
    common linear mapping evolution, and only approximate for ``R``-dependent
    dynamics, so its difference from the coefficient branch mixes the operator
    truncation with the sampling and readout approximation.  It is reported
    separately, never as the operator benchmark.

Four distinct layers, never to be conflated::

    full quantum  <->  QCLE  <->  PBME operator  <->  focused PBME
                 E_QCLE      E_discard          E_focused

All branches start from the Wigner transform of the same Gaussian nuclear
wave packet in the same diabatic state, so their observables are comparable
without any further normalisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .coefficient_operator import (Grid, coefficient_rhs, hermiticity_error,
                                   physical_matrices, populations_from_rho,
                                   spectral_radius_estimate, trace)

try:                                   # same fast path as coefficient_operator
    import scipy.fft as _sfft

    def _fft(a, axis, *, overwrite_x=False):
        return _sfft.fft(a, axis=axis, overwrite_x=overwrite_x, workers=-1)

    def _ifft(a, axis):
        return _sfft.ifft(a, axis=axis, overwrite_x=True, workers=-1)
except Exception:                      # pragma: no cover
    import numpy as _np

    def _fft(a, axis, *, overwrite_x=False):
        return _np.fft.fft(a, axis=axis)

    def _ifft(a, axis):
        return _np.fft.ifft(a, axis=axis)
from .constants import CDTYPE, DTYPE, HBAR
from .models import TwoStateModel
from .observables import rho_W_from_focused_ensemble
from .projector import wigner_aa_dagger_all
from .pbme import (hamiltonian_of_state, pack_state,
                    symmetric_rotation_step, unpack_state)
from .sampling import sample_initial_ensemble
from .types import Ensemble, TraceConvention

__all__ = [
    "rk4_step",
    "StrangPropagator",
    "spectral_radius_estimate",
    "initial_coefficient_field",
    "adiabatic_populations",
    "grid_observables",
    "History",
    "propagate_coefficients",
    "propagate_pbme_ensemble",
    "propagate_pbme_projected_ensemble",
    "sample_matched_ensemble",
    "sample_projected_ensemble",
    "block_statistics",
]


# --------------------------------------------------------------------------
# time stepping
# --------------------------------------------------------------------------
def rk4_step(rho_W: np.ndarray, dt: float, grid: Grid, model: TwoStateModel,
             hbar: float = HBAR,
             convention: TraceConvention = TraceConvention.TRACELESS,
             cache: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """One classical fourth-order Runge-Kutta step of the coefficient field."""
    if cache is None:
        cache = physical_matrices(model, grid, convention)
    f = lambda y: coefficient_rhs(y, grid, model, hbar, convention, cache)
    k1 = f(rho_W)
    k2 = f(rho_W + 0.5 * dt * k1)
    k3 = f(rho_W + 0.5 * dt * k2)
    k4 = f(rho_W + dt * k3)
    return rho_W + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class StrangPropagator:
    r"""Second-order Strang splitting with two *exactly solvable* sub-flows.

    The closed coefficient generator separates into

    .. math::
        \mathcal A_{\rm adv}\rho=-\frac{P}{M}\partial_R\rho ,
        \qquad
        \mathcal A_{\rm el}\rho=-\frac{i}{\hbar}[h,\rho]
            -\tfrac12\bigl(F\partial_P\rho+\partial_P\rho F\bigr),

    and each is integrated exactly rather than approximated:

    * :math:`\mathcal A_{\rm adv}` is diagonal in :math:`k_R` at fixed ``P``,
      so its flow is the phase :math:`e^{-ik_RP\Delta t/M}`;
    * :math:`\mathcal A_{\rm el}` is diagonal in :math:`k_P` at fixed ``R``.
      There, with :math:`A=h/\hbar+\tfrac{k_P}{2}F` and
      :math:`B=h/\hbar-\tfrac{k_P}{2}F` (both real symmetric), it reads
      :math:`\hat\rho\mapsto-i(A\hat\rho-\hat\rho B)`, whose exact flow is
      :math:`\hat\rho\mapsto e^{-iA\Delta t}\hat\rho\,e^{+iB\Delta t}` -- a
      product of two unitaries.

    Consequences, all exact and not merely accurate:

    * **unconditionally stable** -- every factor has unit modulus, so unlike
      RK4 there is no CFL restriction and ``dt`` is limited only by accuracy;
    * **trace conserving** -- advection leaves the :math:`k_R=0` mode
      untouched, and at :math:`k_P=0` one has :math:`B=A=h/\hbar`, so
      :math:`\mathrm{Tr}(e^{-iA\Delta t}\hat\rho\,e^{+iA\Delta t})
      =\mathrm{Tr}\,\hat\rho`;
    * **Hermiticity preserving** -- :math:`U_A(k)^\dagger=U_B(-k)`, which is
      exactly the relation the reality of :math:`\rho_W` requires.

    The Nyquist mode of each axis is zeroed, for the same reason as in
    :func:`mqcle.coefficient_operator.spectral_derivative`: at :math:`k=-k`
    the two relations above cannot both hold and Hermiticity would leak.

    This is a *different* integrator for the *same* spatial operator as
    :func:`rk4_step`.  ``tests/test_coefficient_operator.py`` checks that the
    two agree as ``dt`` shrinks; agreement of two independent time integrators
    on the same discretisation is a real check, and it is what makes the long
    production runs affordable.
    """

    def __init__(self, grid: Grid, model: TwoStateModel, dt: float,
                 hbar: float = HBAR,
                 convention: TraceConvention = TraceConvention.TRACELESS) -> None:
        self.grid, self.model, self.dt, self.hbar = grid, model, float(dt), float(hbar)
        h, F = physical_matrices(model, grid, convention)          # (n_R, N, N)
        n_R, n_P = grid.shape

        kR = np.array(grid.kR, dtype=DTYPE)
        kP = np.array(grid.kP, dtype=DTYPE)
        if n_R % 2 == 0:
            kR[n_R // 2] = 0.0
        if n_P % 2 == 0:
            kP[n_P // 2] = 0.0

        # half-step advection phase, indexed (k_R, P)
        self._adv_half = np.exp(-1j * kR[:, None] * grid.P[None, :]
                                * (0.5 * self.dt / model.mass))[:, :, None, None]
        # Between two Strang steps the trailing and leading half-advections
        # are adjacent exact flows, so A(dt/2) A(dt/2) = A(dt).  Keeping the
        # squared phase lets ``advance`` replace two FFT round trips by one
        # without changing the splitting or any physical parameter.
        self._adv_full = self._adv_half * self._adv_half

        A = h[:, None] / hbar + 0.5 * kP[None, :, None, None] * F[:, None]
        B = h[:, None] / hbar - 0.5 * kP[None, :, None, None] * F[:, None]
        self._UA = self._unitary(A, -self.dt)
        self._UB = self._unitary(B, +self.dt)

    @staticmethod
    def _unitary(M: np.ndarray, s: float) -> np.ndarray:
        """``exp(i s M)`` for a stack of real symmetric ``M``."""
        w, U = np.linalg.eigh(M)
        return np.einsum("...ab,...b,...cb->...ac", U, np.exp(1j * s * w),
                         np.conj(U))

    def _advect(self, rho: np.ndarray, phase: np.ndarray, *,
                consume: bool = False) -> np.ndarray:
        """Apply one exact advection flow.

        ``consume=True`` permits scipy.fft to reuse an intermediate buffer.
        Public stepping preserves its input; only temporaries owned by this
        propagator are consumed.
        """
        g = _fft(rho, axis=0, overwrite_x=consume)
        g *= phase
        return _ifft(g, axis=0)

    def _advect_half(self, rho: np.ndarray, *, consume: bool = False) -> np.ndarray:
        return self._advect(rho, self._adv_half, consume=consume)

    def _advect_full(self, rho: np.ndarray, *, consume: bool = False) -> np.ndarray:
        return self._advect(rho, self._adv_full, consume=consume)

    @staticmethod
    def _rotate_two_state(g: np.ndarray, UA: np.ndarray,
                          UB: np.ndarray) -> np.ndarray:
        """Batched ``UA @ g @ UB`` specialised to the production N=2 case.

        The explicit products perform the same sixteen scalar
        multiplications as the two generic matrix products, while avoiding
        dispatching millions of tiny 2x2 BLAS operations.
        """
        tmp = np.empty_like(g)
        out = np.empty_like(g)
        tmp[..., 0, 0] = UA[..., 0, 0] * g[..., 0, 0] \
            + UA[..., 0, 1] * g[..., 1, 0]
        tmp[..., 0, 1] = UA[..., 0, 0] * g[..., 0, 1] \
            + UA[..., 0, 1] * g[..., 1, 1]
        tmp[..., 1, 0] = UA[..., 1, 0] * g[..., 0, 0] \
            + UA[..., 1, 1] * g[..., 1, 0]
        tmp[..., 1, 1] = UA[..., 1, 0] * g[..., 0, 1] \
            + UA[..., 1, 1] * g[..., 1, 1]
        out[..., 0, 0] = tmp[..., 0, 0] * UB[..., 0, 0] \
            + tmp[..., 0, 1] * UB[..., 1, 0]
        out[..., 0, 1] = tmp[..., 0, 0] * UB[..., 0, 1] \
            + tmp[..., 0, 1] * UB[..., 1, 1]
        out[..., 1, 0] = tmp[..., 1, 0] * UB[..., 0, 0] \
            + tmp[..., 1, 1] * UB[..., 1, 0]
        out[..., 1, 1] = tmp[..., 1, 0] * UB[..., 0, 1] \
            + tmp[..., 1, 1] * UB[..., 1, 1]
        return out

    def _electronic(self, rho: np.ndarray, *, consume: bool = False) -> np.ndarray:
        g = _fft(rho, axis=1, overwrite_x=consume)
        if self.model.nstates == 2:
            g = self._rotate_two_state(g, self._UA, self._UB)
        else:  # generic fallback retained for non-production subsystem sizes
            g = np.einsum("rpab,rpbc->rpac", self._UA, g, optimize=True)
            g = np.einsum("rpab,rpbc->rpac", g, self._UB, optimize=True)
        return _ifft(g, axis=1)

    def step(self, rho: np.ndarray) -> np.ndarray:
        first = self._advect_half(rho)
        middle = self._electronic(first, consume=True)
        return self._advect_half(middle, consume=True)

    def advance(self, rho: np.ndarray, n_steps: int) -> np.ndarray:
        """Advance through ``n_steps`` with adjacent half-flows fused.

        This is algebraically identical to calling :meth:`step` repeatedly:
        ``(A/2 B A/2)^n = A/2 B A ... A B A/2``.  It saves ``n_steps - 1``
        full-grid FFT round trips and reduces accumulated FFT round-off.
        """
        if int(n_steps) != n_steps or n_steps < 0:
            raise ValueError("n_steps must be a non-negative integer")
        n_steps = int(n_steps)
        if n_steps == 0:
            return np.array(rho, dtype=CDTYPE, copy=True)
        state = self._advect_half(rho)
        for step in range(n_steps):
            state = self._electronic(state, consume=True)
            state = (self._advect_half(state, consume=True)
                     if step == n_steps - 1
                     else self._advect_full(state, consume=True))
        return state


# --------------------------------------------------------------------------
# initial condition
# --------------------------------------------------------------------------
def initial_coefficient_field(grid: Grid, R0: float, P0: float, sigma_R: float,
                              init_state: int = 0, nstates: int = 2,
                              hbar: float = HBAR) -> np.ndarray:
    r"""Wigner transform of a minimum-uncertainty Gaussian in one diabatic state.

    .. math::
        \rho_W^{\lambda\lambda'}(R,P)=\delta_{\lambda\lambda_0}
        \delta_{\lambda'\lambda_0}\,\frac{1}{\pi\hbar}
        \exp\!\Bigl[-\frac{(R-R_0)^2}{2\sigma_R^2}
        -\frac{2\sigma_R^2(P-P_0)^2}{\hbar^2}\Bigr],

    normalised to :math:`\int dR\,dP\,\mathrm{Tr}\,\rho_W=1`.  The momentum
    width is :math:`\sigma_P=\hbar/(2\sigma_R)`, the same relation used by
    :func:`mqcle.sampling.sample_bath_wigner_gaussian`, so the two branches
    start from the same physical state.
    """
    RR, PP = grid.mesh()
    sigma_P = hbar / (2.0 * sigma_R)
    W = np.exp(-((RR - R0) ** 2) / (2.0 * sigma_R**2)
               - ((PP - P0) ** 2) / (2.0 * sigma_P**2)) / (np.pi * hbar)
    rho = np.zeros(grid.shape + (nstates, nstates), dtype=CDTYPE)
    rho[:, :, init_state, init_state] = W
    return rho


# --------------------------------------------------------------------------
# observables
# --------------------------------------------------------------------------
def adiabatic_populations(rho_W: np.ndarray, grid: Grid,
                          model: TwoStateModel,
                          basis: np.ndarray | None = None) -> np.ndarray:
    r"""Populations in the *local* adiabatic basis, :math:`\int dR\,dP\,
    (U^\dagger\rho_W U)_{aa}` with :math:`U(R)` diagonalising :math:`h(R)`."""
    if basis is None:
        h = model.h(grid.R[:, None])                   # (n_R, N, N)
        _, basis = np.linalg.eigh(h)                   # ascending
    U = np.asarray(basis)
    # U depends on R but not P.  Sum over P first, then rotate only n_R
    # matrices instead of n_R*n_P matrices.  Linearity makes this exactly the
    # same observable, up to floating-point summation order.
    rho_R = np.sum(np.asarray(rho_W), axis=1)
    Ud = np.conj(np.swapaxes(U, -1, -2))
    rot = Ud @ rho_R @ U
    return np.real(np.einsum("iaa->a", rot)) * grid.cell_area


def grid_observables(rho_W: np.ndarray, grid: Grid, model: TwoStateModel,
                     adiabatic_basis: np.ndarray | None = None
                     ) -> dict[str, Any]:
    """Scalar diagnostics of one coefficient snapshot."""
    rho = np.asarray(rho_W, dtype=CDTYPE)
    diag = np.diagonal(rho, axis1=-2, axis2=-1)
    dens = np.real(np.sum(diag, axis=-1))
    norm = float(np.sum(dens) * grid.cell_area)
    integrated_diag = np.sum(diag, axis=(0, 1)) * grid.cell_area
    coh = np.sum(rho[:, :, 0, 1]) * grid.cell_area
    R_marginal = np.sum(dens, axis=1) * grid.cell_area
    P_marginal = np.sum(dens, axis=0) * grid.cell_area
    if rho.shape[-1] == 2:
        # Avoid allocating rho-rho^dagger, a second full complex field.
        herm = max(
            float(np.max(np.abs(2.0 * np.imag(rho[..., 0, 0])))),
            float(np.max(np.abs(2.0 * np.imag(rho[..., 1, 1])))),
            float(np.max(np.abs(rho[..., 0, 1]
                                - np.conj(rho[..., 1, 0])))),
        )
    else:
        herm = float(hermiticity_error(rho))
    return {
        "trace": complex(np.sum(integrated_diag)),
        "hermiticity": herm,
        "populations": np.real(integrated_diag),
        "adiabatic_populations": adiabatic_populations(
            rho, grid, model, basis=adiabatic_basis),
        "coherence_re": float(np.real(coh)),
        "coherence_im": float(np.imag(coh)),
        "mean_R": float(np.dot(grid.R, R_marginal) / norm),
        "mean_P": float(np.dot(grid.P, P_marginal) / norm),
        # mass fraction sitting in the outermost row/column of the periodic
        # box: the sentinel for "the density never reaches the boundary".
        # Split by axis, because the two failure modes are different: the R
        # boundary is reached when the box is too short, the P boundary when
        # phase-space shear has generated P-structure finer than dP and it
        # aliases.  A combined number cannot tell them apart.
        "boundary_mass_R": float(
            np.sum(np.abs(dens[0]) + np.abs(dens[-1]))
            * grid.cell_area / max(abs(norm), 1e-300)),
        "boundary_mass_P": float(
            np.sum(np.abs(dens[:, 0]) + np.abs(dens[:, -1]))
            * grid.cell_area / max(abs(norm), 1e-300)),
        "boundary_mass": float(
            (np.sum(np.abs(dens[0]) + np.abs(dens[-1]))
             + np.sum(np.abs(dens[:, 0]) + np.abs(dens[:, -1])))
            * grid.cell_area / max(abs(norm), 1e-300)),
        # The two edges of each axis separately.  A box that is simply too
        # narrow leaks at one edge -- the one the dynamics pushes density
        # towards -- while aliased fine structure appears at both.  The
        # combined number cannot distinguish "widen the box" from "refine dP",
        # which is exactly the question the P0 = 20 sentinel failure poses.
        "boundary_mass_P_low": float(
            np.sum(np.abs(dens[:, 0]))
            * grid.cell_area / max(abs(norm), 1e-300)),
        "boundary_mass_P_high": float(
            np.sum(np.abs(dens[:, -1]))
            * grid.cell_area / max(abs(norm), 1e-300)),
        "boundary_mass_R_low": float(
            np.sum(np.abs(dens[0]))
            * grid.cell_area / max(abs(norm), 1e-300)),
        "boundary_mass_R_high": float(
            np.sum(np.abs(dens[-1]))
            * grid.cell_area / max(abs(norm), 1e-300)),
        # The two marginals.  A scalar boundary mass says that something
        # reached the edge; only the marginal says whether it arrived as a
        # localised lobe, which means the box is too small, or as a floor
        # across the whole axis, which means the resolution is too coarse.
        "R_marginal": R_marginal,
        "P_marginal": P_marginal,
    }


@dataclass
class History:
    """Time series of scalar observables, plus optionally saved snapshots."""

    times: list[float] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    snapshots: dict[float, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    #: extra whole-run arrays that are not per-time scalars -- used by the
    #: projected PBME branch to carry block-resolved partial sums, from which
    #: every sample-size, seed and standard-error statistic is computed
    #: offline without re-propagating anything.
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def append(self, t: float, rec: dict[str, Any]) -> None:
        self.times.append(float(t))
        self.records.append(rec)

    def column(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.records])

    def as_arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {"t": np.array(self.times, dtype=DTYPE)}
        for key in self.records[0]:
            val = self.column(key)
            out[key] = np.real(val) if key == "trace" else val
        return out


# --------------------------------------------------------------------------
# branch 1: coefficient-level MQCLE
# --------------------------------------------------------------------------
def propagate_coefficients(rho_W0: np.ndarray, grid: Grid, model: TwoStateModel,
                           dt: float, n_steps: int, *, hbar: float = HBAR,
                           convention: TraceConvention = TraceConvention.TRACELESS,
                           n_save: int = 10,
                           snapshot_times: tuple[float, ...] = (),
                           integrator: str = "strang",
                           progress: Callable[[int, int], None] | None = None
                           ) -> History:
    """Propagate the closed coefficients; observables every ``n_save`` steps.

    ``integrator="strang"`` (default) uses :class:`StrangPropagator`:
    unconditionally stable, exactly trace- and Hermiticity-preserving, second
    order.  ``integrator="rk4"`` uses :func:`rk4_step`: fourth order, but only
    conditionally stable -- see
    :func:`mqcle.coefficient_operator.spectral_radius_estimate`.
    """
    integrator = integrator.lower()
    if integrator not in ("strang", "rk4"):
        raise ValueError(f"unknown integrator '{integrator}'")
    if n_save < 1:
        raise ValueError("n_save must be a positive integer")
    rad = spectral_radius_estimate(grid, model, hbar, convention)
    if integrator == "rk4" and dt > rad["max_dt_rk4"]:
        raise ValueError(
            f"RK4 is unstable here: dt={dt} exceeds the imaginary-axis limit "
            f"2*sqrt(2)/lambda = {rad['max_dt_rk4']:.4g} for this grid "
            f"(lambda = {rad['spectral_radius']:.4g}). Reduce dt, coarsen the "
            f"grid, or use integrator='strang'.")
    cache = (physical_matrices(model, grid, convention)
             if integrator == "rk4" else None)
    strang = (StrangPropagator(grid, model, dt, hbar, convention)
              if integrator == "strang" else None)
    rho = np.array(rho_W0, dtype=CDTYPE, copy=True)
    _, adiabatic_basis = np.linalg.eigh(model.h(grid.R[:, None]))
    hist = History(meta={"branch": "coefficient", "dt": float(dt),
                         "n_steps": int(n_steps), "n_save": int(n_save),
                         "convention": convention.value,
                         "grid": grid.parameters(),
                         "integrator": ("Strang splitting, exact sub-flows"
                                        if integrator == "strang"
                                        else "classical RK4"),
                         "integrator_key": integrator,
                         "stability": rad,
                         "cfl_number": float(dt * rad["spectral_radius"]),
                         "spatial_scheme": "Fourier spectral",
                         "strang_advection_schedule": (
                             "adjacent half-flows fused between saved outputs"
                             if integrator == "strang" else None)})
    want = sorted(snapshot_times)
    hist.append(0.0, grid_observables(
        rho, grid, model, adiabatic_basis=adiabatic_basis))
    if want and want[0] <= 0.0:
        hist.snapshots[0.0] = rho.copy()
        want.pop(0)

    # Production runs request no per-step callback, and when they ask for a
    # snapshot they ask for the terminal field.  Both cases can use the fast
    # path: work one saved-output block at a time, so that adjacent Strang
    # half-advections are evaluated as one exact full advection.  A snapshot
    # request that lands strictly inside the run needs the per-step loop below,
    # because only there is the field available between block boundaries.
    t_final = n_steps * dt
    terminal_only = all(w >= t_final - 0.5 * dt for w in want)
    if strang is not None and terminal_only:
        # A progress callback used to force the per-step path below, which
        # gave up the fused advection.  It is reported at block boundaries
        # instead, so a long production run says where it is without costing
        # anything.  Runs that ask for a snapshot inside the interval still
        # need the general path.
        step = 0
        while step < n_steps:
            block = min(n_save, n_steps - step)
            rho = strang.advance(rho, block)
            step += block
            hist.append(step * dt, grid_observables(
                rho, grid, model, adiabatic_basis=adiabatic_basis))
            if progress is not None:
                progress(step, n_steps)
        for w in want:
            hist.snapshots[float(w)] = rho.copy()
        return hist

    for step in range(1, n_steps + 1):
        rho = (strang.step(rho) if strang is not None
               else rk4_step(rho, dt, grid, model, hbar, convention, cache))
        t = step * dt
        if step % n_save == 0 or step == n_steps:
            hist.append(t, grid_observables(
                rho, grid, model, adiabatic_basis=adiabatic_basis))
        while want and want[0] <= t + 0.5 * dt:
            hist.snapshots[float(want.pop(0))] = rho.copy()
        if progress is not None:
            progress(step, n_steps)
    return hist


# --------------------------------------------------------------------------
# branch 2: PBME trajectory ensemble
# --------------------------------------------------------------------------
def propagate_pbme_ensemble(ensemble: Ensemble, model: TwoStateModel, dt: float,
                            n_steps: int, *, hbar: float = HBAR,
                            convention: TraceConvention = TraceConvention.TRACELESS,
                            n_save: int = 10) -> History:
    """Propagate a **focused** PBME ensemble and read out ``rho_W`` on the way.

    This is the *practical focused trajectory* PBME, not the PBME operator
    benchmark.  Its difference from the coefficient branch mixes the operator
    truncation with the focused sampling and readout approximation, so it must
    never be labelled the cost of discarding the excess term; use
    :func:`propagate_pbme_projected_ensemble` for that, and difference the two
    PBME branches against each other to isolate the focused-estimator effect.

    Uses :func:`mqcle.pbme.symmetric_rotation_step` and the *focused* readout.
    Steps in place rather than calling :func:`mqcle.pbme.propagate`, whose full
    trajectory would be gigabytes at production ensemble sizes.
    """
    if ensemble.meta.get("scheme") not in (None, "focused"):
        raise ValueError(
            "propagate_pbme_ensemble is the focused branch and expects "
            f"scheme='focused', got '{ensemble.meta.get('scheme')}'.  An "
            "importance-sampled ensemble carries signed weights and must go "
            "through propagate_pbme_projected_ensemble.")
    state = pack_state(ensemble.x, ensemble.R, ensemble.P)
    n = model.nstates
    hist = History(meta={"branch": "pbme_focused",
                         "dt": float(dt), "n_steps": int(n_steps),
                         "n_save": int(n_save), "convention": convention.value,
                         "n_samples": int(ensemble.n_samples),
                         "stepper": "symmetric frozen-R rotation splitting",
                         "sampling": "focused MMST action circles, uniform weights",
                         "readout": "rho_W_from_focused_ensemble",
                         "is_operator_benchmark": False})
    w = np.asarray(ensemble.weights, dtype=DTYPE)
    w = w / w.sum()          # uniform for focused sampling; a no-op by design
    for step in range(0, n_steps + 1):
        if step > 0:
            # step in place: storing the whole trajectory would be gigabytes
            state = symmetric_rotation_step(state, dt, model, hbar=hbar,
                                            convention=convention)
        if step % n_save and step != n_steps:
            continue
        x, R, P = unpack_state(state, n, model.nbath)
        rho = rho_W_from_focused_ensemble(x, ensemble.weights, hbar, n)
        # adiabatic readout: rotate each trajectory's own c-matrix by U(R_m)
        c = wigner_aa_dagger_all(x, hbar, n)                 # (m, N, N)
        rho_m = np.swapaxes(c, -1, -2)                       # per-trajectory rho
        _, U = np.linalg.eigh(model.h(R))                    # (m, N, N), ascending
        rot = np.conj(np.swapaxes(U, -1, -2)) @ rho_m @ U
        ad = np.real(np.einsum("m,maa->a", w.astype(CDTYPE), rot))
        hist.append(step * dt, {
            "trace": complex(np.trace(rho)),
            "hermiticity": float(np.max(np.abs(rho - rho.conj().T))),
            "populations": np.real(np.diagonal(rho)).copy(),
            "adiabatic_populations": np.asarray(ad, dtype=DTYPE).copy(),
            "coherence_re": float(np.real(rho[0, 1])),
            "coherence_im": float(np.imag(rho[0, 1])),
            "mean_R": float(np.mean(R)),
            "mean_P": float(np.mean(P)),
            "boundary_mass_R": 0.0,
            "boundary_mass_P": 0.0,
            "boundary_mass": 0.0,
        })
    return hist


def sample_matched_ensemble(rng: np.random.Generator, n: int, *, nstates: int,
                            init_state: int, R0: float, P0: float,
                            sigma_R: float, hbar: float = HBAR) -> Ensemble:
    """Focused ensemble drawn from the same Wigner packet as the grid field."""
    return sample_initial_ensemble(rng, n, nstates=nstates, init_state=init_state,
                                   R0=R0, P0=P0, sigma_R=sigma_R,
                                   scheme="focused", hbar=hbar)


def sample_projected_ensemble(rng: np.random.Generator, n: int, *, nstates: int,
                              init_state: int, R0: float, P0: float,
                              sigma_R: float, hbar: float = HBAR,
                              seed: int | None = None,
                              scheme: str = "sign_optimal") -> Ensemble:
    r"""Importance-sampled ensemble with signed projected-density weights.

    ``scheme='sign_optimal'`` draws the mapping variables from
    :math:`\phi|\Pi_{\lambda_0\lambda_0}|/Z`, which makes the weight bounded;
    ``scheme='wigner_seo'`` draws from :math:`\phi` itself.  Both are unbiased
    and both are run, so that their agreement is a check.  The bath is drawn
    from the same Wigner packet as the grid field, so
    :func:`mqcle.observables.rho_W_from_ensemble` on this ensemble is an
    unbiased estimator of :math:`\rho_W`.
    """
    return sample_initial_ensemble(rng, n, nstates=nstates, init_state=init_state,
                                   R0=R0, P0=P0, sigma_R=sigma_R,
                                   scheme=scheme, hbar=hbar, seed=seed)


# --------------------------------------------------------------------------
# branch 2b: projected-density PBME -- the operator benchmark
# --------------------------------------------------------------------------
def propagate_pbme_projected_ensemble(ensemble: Ensemble, model: TwoStateModel,
                                      dt: float, n_steps: int, *,
                                      hbar: float = HBAR,
                                      convention: TraceConvention
                                      = TraceConvention.TRACELESS,
                                      n_save: int = 10,
                                      n_blocks: int = 1,
                                      seed_of_block: tuple[int, ...] = ()
                                      ) -> History:
    r"""Unbiased projected-density PBME estimator.

    The mapping variables of ``ensemble`` must have been drawn from an
    importance density and carry the signed weights

    .. math::
        w_i=\frac{1}{N_s}\frac{\rho_m^{\mathcal P}(x_i,0)}{q(x_i)},

    i.e. ``scheme='sign_optimal'`` or ``'wigner_seo'`` -- see
    :func:`sample_projected_ensemble`.  Every sample is then propagated under
    the PBME Hamiltonian flow, which is Liouvillian, so

    .. math:: \rho_W^{\nu'\nu}(t)=\sum_i w_i\,c_{\nu\nu'}(x_i(t))

    is unbiased for the PBME density observable up to the time-integration
    error and the finite-sample error.  Both are *measured*: ``scripts/11c``
    refines ``dt`` and ``scripts/11`` refines :math:`N_s`.

    The weights are **never renormalised by their stochastic sum.**  Dividing
    by :math:`\sum_iw_i` would turn an unbiased estimator into a ratio
    estimator with an :math:`O(1/N_s)` bias, and with signed weights the
    denominator is itself a noisy quantity that can approach zero.  The trace
    is instead a *diagnostic*: :math:`\sum_\nu c_{\nu\nu}` is a function of the
    mapping radius alone, which the frozen-``R`` rotation preserves exactly, so
    ``trace`` may only drift by round-off, and its offset from one at ``t=0``
    is the finite-sample error of the estimator itself.

    ``n_blocks`` splits the ensemble into equal contiguous blocks and records
    the per-block partial sums in ``History.arrays``.  Because the blocks are
    independent and identically distributed, every sample-size, seed-replicate,
    standard-error and confidence-interval statistic is computed from those
    partial sums offline, without propagating anything twice.
    """
    n_total = int(ensemble.n_samples)
    if n_blocks < 1 or n_total % n_blocks:
        raise ValueError(f"n_blocks={n_blocks} must divide n_samples={n_total}")
    if ensemble.meta.get("scheme") not in ("wigner_seo", "sign_optimal"):
        raise ValueError(
            "propagate_pbme_projected_ensemble requires an importance-sampled "
            "ensemble (scheme='wigner_seo' or 'sign_optimal'), got "
            f"'{ensemble.meta.get('scheme')}'.  The focused ensemble has no "
            "importance weights and must go through "
            "propagate_pbme_ensemble instead.")
    block = n_total // n_blocks

    state = pack_state(ensemble.x, ensemble.R, ensemble.P)
    n = model.nstates
    w = np.asarray(ensemble.weights, dtype=CDTYPE)
    E0 = hamiltonian_of_state(state, model, hbar, model.mass, convention)
    rad0 = np.sum(np.asarray(ensemble.x, dtype=DTYPE) ** 2, axis=-1)

    hist = History(meta={
        "branch": "pbme_projected",
        "sampling": ensemble.meta.get("proposal", "importance sampling"),
        "scheme": ensemble.meta.get("scheme"),
        "readout": "rho_W_from_ensemble",
        "dt": float(dt), "n_steps": int(n_steps), "n_save": int(n_save),
        "convention": convention.value, "n_samples": n_total,
        "n_blocks": int(n_blocks), "block_size": int(block),
        "seed_of_block": list(seed_of_block),
        "stepper": "symmetric frozen-R rotation splitting",
        "weights_renormalised_by_their_sum": False,
        "weight_dtype": str(np.asarray(ensemble.weights).dtype),
        "signed_weight_fraction_negative":
            ensemble.meta.get("signed_weight_fraction_negative"),
        "is_operator_benchmark": True,
    })
    blk_rho: list[np.ndarray] = []
    blk_ad: list[np.ndarray] = []

    for step in range(0, n_steps + 1):
        if step > 0:
            state = symmetric_rotation_step(state, dt, model, hbar=hbar,
                                            convention=convention)
        if step % n_save and step != n_steps:
            continue
        x, R, P = unpack_state(state, n, model.nbath)
        c = wigner_aa_dagger_all(x, hbar, n)                        # (m, N, N)
        wc = w[:, None, None] * c
        # per-block partial sums; rho_W is the transpose of the c-contraction
        pb = np.swapaxes(wc.reshape(n_blocks, block, n, n).sum(axis=1), -1, -2)
        rho = pb.sum(axis=0)

        _, U = np.linalg.eigh(model.h(R))
        rot = np.conj(np.swapaxes(U, -1, -2)) @ np.swapaxes(c, -1, -2) @ U
        pa = np.real(np.einsum("m,maa->ma", w, rot)
                     .reshape(n_blocks, block, n).sum(axis=1))
        blk_rho.append(pb)
        blk_ad.append(pa)

        E = hamiltonian_of_state(state, model, hbar, model.mass, convention)
        hist.append(step * dt, {
            "trace": complex(np.trace(rho)),
            "hermiticity": float(np.max(np.abs(rho - rho.conj().T))),
            "populations": np.real(np.diagonal(rho)).copy(),
            "adiabatic_populations": pa.sum(axis=0).astype(DTYPE),
            "coherence_re": float(np.real(rho[0, 1])),
            "coherence_im": float(np.imag(rho[0, 1])),
            "mean_R": float(np.mean(R)),
            "mean_P": float(np.mean(P)),
            "energy_drift": float(np.max(np.abs(E - E0))),
            "mapping_radius_drift":
                float(np.max(np.abs(np.sum(x**2, axis=-1) - rad0))),
            "boundary_mass_R": 0.0,
            "boundary_mass_P": 0.0,
            "boundary_mass": 0.0,
        })

    hist.arrays["block_rho"] = np.asarray(blk_rho)          # (T, B, N, N) complex
    hist.arrays["block_adiabatic"] = np.asarray(blk_ad)     # (T, B, N)   real
    return hist


def block_statistics(blocks: np.ndarray, n_blocks_used: int | None = None
                     ) -> dict[str, np.ndarray]:
    r"""Mean, sd, SEM and 95% interval from independent block partial sums.

    ``blocks`` has shape ``(T, B, ...)`` with the block axis second.  Each
    block sum already carries the global :math:`1/N_s` of the weights, so the
    estimator over ``k`` of the ``B`` blocks is ``(B/k) sum(blocks[:, :k])``
    and the per-block estimators are ``B * blocks[:, b]``.

    The interval is a normal-theory ``mean +/- 1.96 SEM``; with ``B >= 5``
    blocks of :math:`\ge10^3` samples each the block means are close to normal
    by the central limit theorem, and the tests check that the SEM falls as
    :math:`N_s^{-1/2}`.
    """
    b = np.asarray(blocks)
    total = b.shape[1]                    # blocks the 1/N_s was shared over
    k = total if n_blocks_used is None else int(n_blocks_used)
    if not 1 <= k <= total:
        raise ValueError(f"n_blocks_used={k} outside 1..{total}")
    sub = b[:, :k]
    # Each block sum carries the GLOBAL 1/N_s, so a single block's own
    # estimator is `total * block`, not `k * block`.  Scaling by the subset
    # size instead makes every subset mean and standard error too small by
    # total/k -- which looks like a standard error that *grows* with sample
    # size, and is how this was caught.
    means = total * sub
    mean = means.mean(axis=1)
    sd = means.std(axis=1, ddof=1) if k > 1 else np.zeros_like(mean)
    sem = sd / np.sqrt(k)
    return {"mean": mean, "sd": sd, "sem": sem,
            "ci_lo": mean - 1.96 * sem, "ci_hi": mean + 1.96 * sem,
            "n_blocks": k}
