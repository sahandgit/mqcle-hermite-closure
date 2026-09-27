r"""Module 2 -- MMST mapping coordinates and the mapping Hamiltonian.

Mapping coordinate layout
-------------------------
``x = (r_0, ..., r_{N-1}, p_0, ..., p_{N-1})``, i.e. for ``N = 2``
``x = (r_0, r_1, p_0, p_1)``.

Mapping Hamiltonian
-------------------
.. math::
    H_m(x,R,P)=\frac{P^{2}}{2M}+V_{\rm eff}(R)
    +\frac{1}{2\hbar}\sum_{\lambda\lambda'}\mathsf h^{\lambda\lambda'}(R)
    \bigl(r_\lambda r_{\lambda'}+p_\lambda p_{\lambda'}
    -\hbar\delta_{\lambda\lambda'}\bigr),

where :math:`(\mathsf h, V_{\rm eff})` are produced by
:func:`subsystem_split` according to the selected
:class:`~mqcle.types.TraceConvention`:

``TRACELESS``  :math:`\mathsf h=\bar h=h-\tfrac1N(\mathrm{Tr}\,h)\mathbb 1`,
               :math:`V_{\rm eff}=V_b+\tfrac1N\mathrm{Tr}\,h`;
``FULL``       :math:`\mathsf h=h`, :math:`V_{\rm eff}=V_b`.

The same :math:`\mathsf h` must be used by :mod:`mqcle.excess`.  See
:class:`~mqcle.types.TraceConvention` for why mixing the two is invalid.

Frozen-R linearity
------------------
At fixed ``R`` the mapping sector is linear,

.. math:: \dot x=K(R)x,\qquad
    K(R)=\frac{1}{\hbar}\begin{pmatrix}0&\mathsf h\\-\mathsf h&0\end{pmatrix},

and :math:`K^{\mathsf T}=-K` because :math:`\mathsf h` is symmetric, so
:math:`e^{K t}\in SO(2N)`.  :func:`drift_matrix` builds ``K`` and
``tests/test_pbme.py`` verifies the identity numerically and symbolically.
This fact is recorded because it bears directly on whether nonlinear latent
maps can be justified (PART X).
"""
from __future__ import annotations

import numpy as np

from .constants import DTYPE, HBAR
from .models import TwoStateModel
from .types import SubsystemSplit, TraceConvention

__all__ = [
    "quadratic_form_real", "apply_real_form",
    "split_mapping", "combine_mapping", "subsystem_split",
    "mapping_bilinears", "mapping_hamiltonian",
    "grad_mapping_hamiltonian_x", "grad_mapping_hamiltonian_R",
    "drift_matrix", "mapping_velocity", "real_form",
]


# --------------------------------------------------------------------------
# coordinate helpers
# --------------------------------------------------------------------------
def split_mapping(x: np.ndarray, nstates: int) -> tuple[np.ndarray, np.ndarray]:
    """Split ``x`` of shape ``(..., 2N)`` into ``(r, p)``, each ``(..., N)``."""
    x = np.asarray(x, dtype=DTYPE)
    if x.shape[-1] != 2 * nstates:
        raise ValueError(f"x has last dimension {x.shape[-1]}, expected {2 * nstates}")
    return x[..., :nstates], x[..., nstates:]


def combine_mapping(r: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Inverse of :func:`split_mapping`."""
    r = np.asarray(r, dtype=DTYPE)
    p = np.asarray(p, dtype=DTYPE)
    if r.shape != p.shape:
        raise ValueError(f"r shape {r.shape} != p shape {p.shape}")
    return np.concatenate([r, p], axis=-1)


# --------------------------------------------------------------------------
# trace convention
# --------------------------------------------------------------------------
def subsystem_split(
    model: TwoStateModel,
    R: np.ndarray,
    convention: TraceConvention = TraceConvention.TRACELESS,
) -> SubsystemSplit:
    """Apply the trace convention at bath positions ``R``."""
    R = np.asarray(R, dtype=DTYPE)
    h = model.h(R)                    # (..., N, N)
    dh = model.dh_dR(R)               # (..., Nb, N, N)
    vb = model.bath_potential(R)      # (...,)
    dvb = model.dbath_dR(R)           # (..., Nb)
    n = model.nstates

    if convention is TraceConvention.FULL:
        return SubsystemSplit(h, dh, vb, dvb, convention)

    # Tr h is real for Hermitian h; taking the real part keeps v_eff a real
    # array, so the mapping Hamiltonian does not acquire a 1e-19 imaginary part
    # that then propagates into every downstream dtype.
    tr = np.real(np.trace(h, axis1=-2, axis2=-1)) / n      # (...,)
    dtr = np.real(np.trace(dh, axis1=-2, axis2=-1)) / n    # (..., Nb)
    eye = np.eye(n, dtype=DTYPE)
    h_bar = h - tr[..., None, None] * eye
    dh_bar = dh - dtr[..., None, None] * eye
    if np.iscomplexobj(h):
        # a Hermitian h has a real diagonal; a non-Hermitian one would show up
        # here rather than silently producing a complex "Hamiltonian"
        offd = np.max(np.abs(h - np.conj(np.swapaxes(h, -1, -2))))
        if offd > 1e-10:
            raise ValueError(f"subsystem matrix is not Hermitian: {offd:.3e}")
    return SubsystemSplit(h_bar, dh_bar, vb + tr, dvb + dtr, convention)


# --------------------------------------------------------------------------
# mapping Hamiltonian
# --------------------------------------------------------------------------
def mapping_bilinears(x: np.ndarray, nstates: int) -> np.ndarray:
    r"""Return :math:`u_{\lambda\lambda'}=r_\lambda r_{\lambda'}+p_\lambda p_{\lambda'}`,
    shape ``(..., N, N)``."""
    r, p = split_mapping(x, nstates)
    return r[..., :, None] * r[..., None, :] + p[..., :, None] * p[..., None, :]


def real_form(h: np.ndarray) -> np.ndarray:
    r"""Real symmetric :math:`2N\times2N` form of a Hermitian ``h``.

    With :math:`h=A+iB`, :math:`A=A^{\mathsf T}`, :math:`B=-B^{\mathsf T}`,

    .. math:: \mathbb H=\begin{pmatrix}A&-B\\B&A\end{pmatrix},

    in the ordering ``x = (r, p)``.  Every mapping-sector quantity below is a
    contraction against this one matrix:
    :math:`H_m\supset(2\hbar)^{-1}(x^{\mathsf T}\mathbb Hx-\hbar\,
    \mathrm{Tr}A)`, :math:`\partial H_m/\partial x_i=
    \hbar^{-1}\mathbb H_{ij}x_j`, and
    :math:`K=\hbar^{-1}\mathbb J\mathbb H`.

    For real symmetric ``h`` this is ``diag(h, h)`` and every formula reduces to
    the one written in terms of the symmetric bilinear
    :math:`u_{\lambda\lambda'}=r_\lambda r_{\lambda'}+p_\lambda p_{\lambda'}`.
    For complex Hermitian ``h`` it does not: a formula written in terms of ``u``
    alone contracts the antisymmetric ``B`` against a symmetric object and
    silently discards it.  Routing everything through this function is what
    makes the numerical mapping code agree with Corollary 1 rather than only
    with its real special case.
    """
    h = np.asarray(h)
    n = h.shape[-1]
    A, B = np.real(h), np.imag(h)
    out = np.zeros(h.shape[:-2] + (2 * n, 2 * n), dtype=DTYPE)
    out[..., :n, :n] = A
    out[..., n:, n:] = A
    out[..., :n, n:] = -B
    out[..., n:, :n] = B
    return out


def _split_AB(h: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """``(Re h, Im h)`` with ``None`` for a real ``h``; see :func:`real_form`."""
    h = np.asarray(h)
    if np.iscomplexobj(h):
        return np.real(h), np.imag(h)
    return h.astype(DTYPE), None


def quadratic_form_real(h: np.ndarray, x: np.ndarray, nstates: int) -> np.ndarray:
    r""":math:`\mathbb H_{ij}x_ix_j` without materialising :math:`\mathbb H`.

    With :math:`\mathbb H=\bigl(\begin{smallmatrix}A&-B\\B&A\end{smallmatrix}
    \bigr)`, :math:`x=(r,p)`, :math:`A^{\mathsf T}=A` and
    :math:`B^{\mathsf T}=-B`,

    .. math::
        \mathbb H_{ij}x_ix_j
        =r^{\mathsf T}Ar+p^{\mathsf T}Ap-2\,r^{\mathsf T}Bp .

    Identical to ``einsum`` against :func:`real_form`, at a quarter of the
    memory traffic: the production PBME step evaluates this twice per time step
    for every trajectory, so the :math:`2N\times2N` temporary is the single
    largest cost in the ensemble branch.
    """
    A, B = _split_AB(h)
    r, q = split_mapping(x, nstates)
    out = (np.einsum("...ab,...a,...b->...", A, r, r)
           + np.einsum("...ab,...a,...b->...", A, q, q))
    if B is not None:
        out = out - 2.0 * np.einsum("...ab,...a,...b->...", B, r, q)
    return out


def apply_real_form(h: np.ndarray, x: np.ndarray, nstates: int) -> np.ndarray:
    r""":math:`\mathbb H x` without materialising :math:`\mathbb H`:
    :math:`(Ar-Bp,\;Br+Ap)`."""
    A, B = _split_AB(h)
    r, q = split_mapping(x, nstates)
    top = np.einsum("...ab,...b->...a", A, r)
    bot = np.einsum("...ab,...b->...a", A, q)
    if B is not None:
        top = top - np.einsum("...ab,...b->...a", B, q)
        bot = bot + np.einsum("...ab,...b->...a", B, r)
    return np.concatenate([top, bot], axis=-1)


def mapping_hamiltonian(
    x: np.ndarray, R: np.ndarray, P: np.ndarray, model: TwoStateModel,
    hbar: float = HBAR, mass: float | None = None,
    convention: TraceConvention = TraceConvention.TRACELESS,
) -> np.ndarray:
    """Mapping Hamiltonian, shape ``(...,)``."""
    mass = model.mass if mass is None else mass
    P = np.asarray(P, dtype=DTYPE)
    n = model.nstates
    sp = subsystem_split(model, R, convention)
    xx = np.asarray(x, dtype=DTYPE)
    quad = (quadratic_form_real(sp.h_map, xx, n)
            - hbar * np.trace(np.real(sp.h_map), axis1=-2, axis2=-1)) / (2.0 * hbar)
    kinetic = np.sum(P**2, axis=-1) / (2.0 * mass)
    return kinetic + sp.v_eff + quad


def grad_mapping_hamiltonian_x(
    x: np.ndarray, R: np.ndarray, model: TwoStateModel,
    hbar: float = HBAR,
    convention: TraceConvention = TraceConvention.TRACELESS,
) -> np.ndarray:
    r"""Analytic :math:`\partial H_m/\partial x`, shape ``(..., 2N)``.

    :math:`\partial H_m/\partial r_\lambda=\frac1\hbar\sum_{\lambda'}
    \mathsf h^{\lambda\lambda'}r_{\lambda'}` and likewise for ``p``.
    """
    n = model.nstates
    sp = subsystem_split(model, R, convention)
    xx = np.asarray(x, dtype=DTYPE)
    return apply_real_form(sp.h_map, xx, n) / hbar


def grad_mapping_hamiltonian_R(
    x: np.ndarray, R: np.ndarray, model: TwoStateModel,
    hbar: float = HBAR,
    convention: TraceConvention = TraceConvention.TRACELESS,
) -> np.ndarray:
    r"""Analytic :math:`\nabla_R H_m`, shape ``(..., Nb)``.

    .. math:: \nabla_RH_m=\nabla_RV_{\rm eff}
        +\frac{1}{2\hbar}\sum_{\lambda\lambda'}\nabla_R\mathsf h^{\lambda\lambda'}
        (u_{\lambda\lambda'}-\hbar\delta_{\lambda\lambda'}).
    """
    n = model.nstates
    sp = subsystem_split(model, R, convention)
    xx = np.asarray(x, dtype=DTYPE)[..., None, :]     # broadcast over the bath
    quad = (quadratic_form_real(sp.dh_map_dR, xx, n)
            - hbar * np.trace(np.real(sp.dh_map_dR), axis1=-2, axis2=-1)
            ) / (2.0 * hbar)
    return sp.dv_eff_dR + quad


def drift_matrix(
    R: np.ndarray, model: TwoStateModel, hbar: float = HBAR,
    convention: TraceConvention = TraceConvention.TRACELESS,
) -> np.ndarray:
    r"""Frozen-R mapping drift matrix ``K(R)``, shape ``(..., 2N, 2N)``.

    ``K`` is antisymmetric, so ``expm(K t)`` is orthogonal and the mapping flow
    preserves the isotropic Gaussian envelope exactly.
    """
    n = model.nstates
    sp = subsystem_split(model, R, convention)
    big = real_form(sp.h_map)
    J = np.zeros((2 * n, 2 * n), dtype=DTYPE)
    J[:n, n:] = np.eye(n)
    J[n:, :n] = -np.eye(n)
    return (J @ big) / hbar


def mapping_velocity(
    x: np.ndarray, R: np.ndarray, model: TwoStateModel, hbar: float = HBAR,
    convention: TraceConvention = TraceConvention.TRACELESS,
) -> np.ndarray:
    r"""Hamiltonian mapping velocity :math:`v_x=(\partial_pH_m,-\partial_rH_m)`.

    Equal to ``K(R) @ x`` by construction; ``tests/test_pbme.py`` checks that
    the two independent routes agree to machine precision.
    """
    n = model.nstates
    g = grad_mapping_hamiltonian_x(x, R, model, hbar, convention)
    gr, gp = split_mapping(g, n)
    return combine_mapping(gp, -gr)
