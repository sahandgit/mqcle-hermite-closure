r"""Module 6 -- original-space PBME.

State layout (fixed contract)
-----------------------------
``state = concatenate([x, R, P])`` with ``x`` of length ``2N`` ordered
``(r_0..r_{N-1}, p_0..p_{N-1})``, ``R`` and ``P`` of length ``Nb``.  Helper
functions :func:`pack_state` and :func:`unpack_state` are the only places that
know this layout.

Characteristics
---------------
.. math::
    \dot r_\lambda=\frac{\partial H_m}{\partial p_\lambda},\quad
    \dot p_\lambda=-\frac{\partial H_m}{\partial r_\lambda},\quad
    \dot R=\frac{P}{M},\quad
    \dot P=-\nabla_RH_m .

Density-side generator
----------------------
The Poisson part of the mapping QCLE acting on a density is

.. math::
    \mathcal A_{\rm PB}\rho=\{H_m,\rho\}_{x,X}
    =\nabla_RH_m\!\cdot\!\nabla_P\rho-\frac{P}{M}\!\cdot\!\nabla_R\rho
      +\sum_\lambda\Bigl(\frac{\partial H_m}{\partial r_\lambda}
      \frac{\partial\rho}{\partial p_\lambda}
      -\frac{\partial H_m}{\partial p_\lambda}
      \frac{\partial\rho}{\partial r_\lambda}\Bigr),

which equals :math:`-v\cdot\nabla\rho` because the Hamiltonian vector field is
divergence free.  Both routes are implemented and compared in the tests.

Integrator
----------
:func:`symmetric_rotation_step` is a palindromic splitting
``kick(dt/2) drift(dt/2) rotate(dt) drift(dt/2) kick(dt/2)`` in which the
mapping block is advanced by the *exact* frozen-``R`` rotation
:math:`e^{K(R)\,dt}` rather than by a Taylor step.

Demonstrated properties: it is symmetric, exactly time reversible, preserves
the mapping radius to round-off, and is exact whenever ``h`` is
``R``-independent.  **Symplecticity of the full extended step is not claimed**:
the momentum kick freezes ``x`` while the mapping rotation freezes ``R``, and
neither sub-map is the exact flow of a piece of ``H_m`` on its own, so the
composition is not a composition of Hamiltonian flows.  Use
:func:`symplecticity_defect` to measure
:math:`\epsilon_{\rm symp}=\|J^{\mathsf T}\Omega J-\Omega\|_F` rather than
assuming it vanishes.  The name ``strang_step`` is retained as an alias for
call compatibility.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .constants import DTYPE, HBAR
from .fields import ScalarField
from .mapping import (drift_matrix, grad_mapping_hamiltonian_R,
                      grad_mapping_hamiltonian_x, mapping_hamiltonian,
                      mapping_velocity, real_form, split_mapping,
                      subsystem_split)
from .models import TwoStateModel
from .types import TraceConvention

__all__ = ["pack_state", "unpack_state", "pbme_vector_field", "rk4_step",
           "symmetric_rotation_step", "strang_step", "propagate",
           "pbme_generator_on_field", "mapping_radius", "STATE_LAYOUT",
           "frozen_R_propagator", "apply_frozen_R_propagator",
           "symplectic_form", "symplecticity_defect",
           "hamiltonian_of_state"]

STATE_LAYOUT = "concatenate([x (2N: r_0..r_{N-1}, p_0..p_{N-1}), R (Nb), P (Nb)])"


def pack_state(x, R, P) -> np.ndarray:
    return np.concatenate([np.asarray(x, DTYPE), np.asarray(R, DTYPE),
                           np.asarray(P, DTYPE)], axis=-1)


def unpack_state(state, nstates: int, nbath: int):
    state = np.asarray(state, dtype=DTYPE)
    d = 2 * nstates
    return state[..., :d], state[..., d:d + nbath], state[..., d + nbath:]


def mapping_radius(x, nstates: int) -> np.ndarray:
    r"""Conserved-under-frozen-R quantity :math:`\sum_\lambda(r^2+p^2)`."""
    return np.sum(np.asarray(x, DTYPE) ** 2, axis=-1)


# --------------------------------------------------------------------------
# characteristics
# --------------------------------------------------------------------------
def pbme_vector_field(state, model: TwoStateModel, hbar: float = HBAR,
                      mass: float | None = None,
                      convention: TraceConvention = TraceConvention.TRACELESS
                      ) -> np.ndarray:
    """Time derivative of the packed state; same shape as ``state``."""
    mass = model.mass if mass is None else mass
    n, nb = model.nstates, model.nbath
    x, R, P = unpack_state(state, n, nb)
    xdot = mapping_velocity(x, R, model, hbar, convention)
    Rdot = P / mass
    Pdot = -grad_mapping_hamiltonian_R(x, R, model, hbar, convention)
    return pack_state(xdot, Rdot, Pdot)


def rk4_step(state, dt: float, model: TwoStateModel, **kw) -> np.ndarray:
    """Classical fourth-order Runge-Kutta reference step (not symplectic)."""
    f = lambda s: pbme_vector_field(s, model, **kw)
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def frozen_R_propagator(R, dt: float, model: TwoStateModel, hbar: float = HBAR,
                        convention: TraceConvention = TraceConvention.TRACELESS
                        ) -> np.ndarray:
    r"""Closed-form :math:`e^{K(R)\,dt}`, shape ``(..., 2N, 2N)``.

    With :math:`K=\hbar^{-1}\mathbb J\mathbb H` and
    :math:`[\mathbb H,\mathbb J]=0` (which holds for every Hermitian
    :math:`\mathsf h`; see :func:`mqcle.mapping.real_form`),

    .. math::
        e^{Kdt}=\cos\!\bigl(\mathbb H\,dt/\hbar\bigr)
                +\mathbb J\,\sin\!\bigl(\mathbb H\,dt/\hbar\bigr),

    evaluated from the eigendecomposition of the real symmetric
    :math:`2N\times2N` form.  This is exact, batched, and avoids a per-sample
    matrix exponential.  It is orthogonal by construction, so the mapping radius
    is preserved to round-off.  ``tests/test_pbme.py`` checks it against
    ``scipy.linalg.expm``.

    An earlier version built the propagator from the eigendecomposition of the
    :math:`N\times N` matrix as
    :math:`\bigl(\begin{smallmatrix}C&S\\-S&C\end{smallmatrix}\bigr)`.  That is
    the same thing when :math:`\mathsf h` is real symmetric, and silently wrong
    when it is not: the imaginary part is cast away.
    """
    n = model.nstates
    h = subsystem_split(model, R, convention).h_map
    big = real_form(h)                              # (..., 2N, 2N), real sym.
    w, Q = np.linalg.eigh(big)
    th = w * dt / hbar
    C = np.einsum("...ik,...k,...jk->...ij", Q, np.cos(th), Q)
    S = np.einsum("...ik,...k,...jk->...ij", Q, np.sin(th), Q)
    J = np.zeros((2 * n, 2 * n), dtype=DTYPE)
    J[:n, n:] = np.eye(n)
    J[n:, :n] = -np.eye(n)
    return C + np.einsum("ij,...jk->...ik", J, S)


def apply_frozen_R_propagator(x, R, dt: float, model: TwoStateModel,
                              hbar: float = HBAR,
                              convention: TraceConvention
                              = TraceConvention.TRACELESS) -> np.ndarray:
    r"""Return :math:`e^{K(R)dt}x` without forming the :math:`2N\times2N` matrix.

    For two states with a **real symmetric** :math:`\mathsf h` this is done in
    closed form.  Write :math:`\mathsf h_{\rm map}=\mu I+M` with :math:`M`
    traceless; then :math:`M^2=\omega^2I` with :math:`\omega^2=a^2+c^2`,
    :math:`a=\tfrac12(h_{00}-h_{11})`, :math:`c=h_{01}`, and with
    :math:`\tau=dt/\hbar`, :math:`\mathbb M=\operatorname{diag}(M,M)`,
    :math:`s=\sin(\omega\tau)/\omega`,

    .. math::
        e^{K dt}=\cos(\mathbb H\tau)+\mathbb J\sin(\mathbb H\tau)
        =A\,I-B\,\mathbb M+C\,\mathbb J+D\,\mathbb J\mathbb M,

    with :math:`A=\cos\mu\tau\cos\omega\tau`, :math:`B=s\sin\mu\tau`,
    :math:`C=\cos\omega\tau\sin\mu\tau`, :math:`D=s\cos\mu\tau`.  Acting on
    :math:`x=(r,p)` and using :math:`\mathbb J(r,p)=(p,-r)` this is four
    :math:`2\times2` mat-vecs and no eigendecomposition, which is roughly a
    factor of three on the whole PBME step -- the difference between
    :math:`10^5` and :math:`10^6` trajectories at fixed wall time, and
    therefore the difference between a Monte-Carlo error bar that is and is not
    smaller than the effect being measured.

    Every other case falls back to the general :func:`frozen_R_propagator`.
    ``tests/test_pbme.py`` checks that the two routes agree to round-off,
    including for complex Hermitian ``h``.
    """
    x = np.asarray(x, DTYPE)
    hm = subsystem_split(model, R, convention).h_map
    if model.nstates != 2 or np.iscomplexobj(hm):
        W = frozen_R_propagator(R, dt, model, hbar, convention)
        return np.einsum("...ij,...j->...i", W, x)
    tau = dt / hbar
    a = 0.5 * (hm[..., 0, 0] - hm[..., 1, 1])
    c = hm[..., 0, 1]
    mu = 0.5 * (hm[..., 0, 0] + hm[..., 1, 1])
    w = np.sqrt(a * a + c * c)
    cw = np.cos(w * tau)
    # sin(w tau)/w continued to w = 0; np.sinc(z) = sin(pi z)/(pi z)
    ssm = tau * np.sinc(w * tau / np.pi)
    cm, sm = np.cos(mu * tau), np.sin(mu * tau)
    A, B, C, D = cm * cw, ssm * sm, cw * sm, ssm * cm
    r, q = x[..., :2], x[..., 2:]
    Mr = np.stack([a * r[..., 0] + c * r[..., 1],
                   c * r[..., 0] - a * r[..., 1]], axis=-1)
    Mq = np.stack([a * q[..., 0] + c * q[..., 1],
                   c * q[..., 0] - a * q[..., 1]], axis=-1)
    A, B, C, D = (v[..., None] for v in (A, B, C, D))
    return np.concatenate([A * r - B * Mr + C * q + D * Mq,
                           A * q - B * Mq - C * r - D * Mr], axis=-1)


def _rotate_mapping(x, R, dt, model, hbar, convention):
    """Apply the exact frozen-R mapping propagator to ``x``."""
    return apply_frozen_R_propagator(x, R, dt, model, hbar, convention)


def symmetric_rotation_step(state, dt: float, model: TwoStateModel,
                            hbar: float = HBAR,
                            mass: float | None = None,
                            convention: TraceConvention = TraceConvention.TRACELESS
                            ) -> np.ndarray:
    """One palindromic, exactly reversible step (see module docstring).

    Symmetric and reversible by construction; not claimed symplectic.
    """
    mass = model.mass if mass is None else mass
    n, nb = model.nstates, model.nbath
    x, R, P = unpack_state(state, n, nb)

    P = P - 0.5 * dt * grad_mapping_hamiltonian_R(x, R, model, hbar, convention)
    R = R + 0.5 * dt * P / mass
    x = _rotate_mapping(x, R, dt, model, hbar, convention)
    R = R + 0.5 * dt * P / mass
    P = P - 0.5 * dt * grad_mapping_hamiltonian_R(x, R, model, hbar, convention)
    return pack_state(x, R, P)


#: Backwards-compatible alias.  The name says "Strang" but the scheme is a
#: symmetric frozen-R rotation splitting; see the module docstring.
strang_step = symmetric_rotation_step


def symplectic_form(nstates: int, nbath: int) -> np.ndarray:
    r"""Canonical :math:`\Omega` for the packed layout ``(r, p, R, P)``."""
    d = 2 * (nstates + nbath)
    O = np.zeros((d, d), dtype=DTYPE)
    n = nstates
    O[:n, n:2 * n] = np.eye(n)
    O[n:2 * n, :n] = -np.eye(n)
    b = 2 * n
    O[b:b + nbath, b + nbath:] = np.eye(nbath)
    O[b + nbath:, b:b + nbath] = -np.eye(nbath)
    return O


def symplecticity_defect(state, dt: float, model: TwoStateModel,
                         stepper=None, eps: float = 1e-6, **kw) -> float:
    r"""Measured :math:`\epsilon_{\rm symp}=\|J^{\mathsf T}\Omega J-\Omega\|_F`.

    ``J`` is the Jacobian of one step, obtained by central differences.  This
    is a *measurement*, not an assertion: the scheme is not claimed symplectic.
    """
    stepper = symmetric_rotation_step if stepper is None else stepper
    s0 = np.asarray(state, dtype=DTYPE).reshape(-1)
    d = s0.size
    J = np.empty((d, d), dtype=DTYPE)
    for i in range(d):
        e = np.zeros(d)
        e[i] = eps
        plus = stepper((s0 + e)[None, :], dt, model, **kw).reshape(-1)
        minus = stepper((s0 - e)[None, :], dt, model, **kw).reshape(-1)
        J[:, i] = (plus - minus) / (2 * eps)
    O = symplectic_form(model.nstates, model.nbath)
    return float(np.linalg.norm(J.T @ O @ J - O, "fro"))


def propagate(state, dt: float, n_steps: int, model: TwoStateModel,
              stepper=symmetric_rotation_step, **kw) -> np.ndarray:
    """Propagate and return the full trajectory, shape ``(n_steps + 1, ...)``."""
    traj = [np.asarray(state, DTYPE)]
    for _ in range(n_steps):
        traj.append(stepper(traj[-1], dt, model, **kw))
    return np.stack(traj, axis=0)


# --------------------------------------------------------------------------
# density-side generator
# --------------------------------------------------------------------------
def pbme_generator_on_field(field: ScalarField, x, R, P, model: TwoStateModel,
                            hbar: float = HBAR, mass: float | None = None,
                            convention: TraceConvention = TraceConvention.TRACELESS,
                            form: str = "poisson") -> np.ndarray:
    r"""Apply :math:`\mathcal A_{\rm PB}` to a scalar field.

    ``form='poisson'``   evaluates :math:`\{H_m,\rho\}` term by term;
    ``form='advection'`` evaluates :math:`-v\cdot\nabla\rho`.
    The two agree identically because the flow is divergence free; the tests
    check that, which catches a sign slip in either route.
    """
    mass = model.mass if mass is None else mass
    n = model.nstates
    gx = field.grad_x(x, R, P)
    gR = field.grad_R(x, R, P)
    gP = field.grad_P(x, R, P)
    dH_dx = grad_mapping_hamiltonian_x(x, R, model, hbar, convention)
    dH_dR = grad_mapping_hamiltonian_R(x, R, model, hbar, convention)

    if form == "poisson":
        hr, hp = split_mapping(dH_dx, n)
        fr, fp = split_mapping(gx, n)
        mapping_part = np.sum(hr * fp - hp * fr, axis=-1)
        bath_part = (np.sum(dH_dR * gP, axis=-1)
                     - np.sum(np.asarray(P, DTYPE) / mass * gR, axis=-1))
        return mapping_part + bath_part
    if form == "advection":
        v_x = mapping_velocity(x, R, model, hbar, convention)
        v_R = np.asarray(P, DTYPE) / mass
        v_P = -dH_dR
        return -(np.sum(v_x * gx, axis=-1) + np.sum(v_R * gR, axis=-1)
                 + np.sum(v_P * gP, axis=-1))
    raise ValueError(f"unknown form '{form}'")


def hamiltonian_of_state(state, model: TwoStateModel, hbar: float = HBAR,
                         mass: float | None = None,
                         convention: TraceConvention = TraceConvention.TRACELESS
                         ) -> np.ndarray:
    """Convenience wrapper: ``H_m`` evaluated on a packed state."""
    x, R, P = unpack_state(state, model.nstates, model.nbath)
    return mapping_hamiltonian(x, R, P, model, hbar, mass, convention)
