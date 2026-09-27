r"""Physical frame inside the Gaussian--Hermite coefficient space.

The projected mapping density occupies an :math:`N^2`-dimensional subspace of
the :math:`M`-dimensional even :math:`K=2` Hermite sector -- four complex
directions inside eleven for two states.  The constant matrix :math:`B` of
Eq. (B-frame) maps between them,

.. math::

    \bm c=B\,\mathrm{vec}(\rho_W),
    \qquad
    B_{\bm n,\mu\nu}
    =\frac{1}{\bm n!}\int\mathrm dz\,\varphi(z)\He_{\bm n}(z)\Pi_{\nu\mu}(z).

This module supplies :math:`B`, the Hermite metric
:math:`W=\mathrm{diag}(\bm n!)`, the weighted projector and the readout.

The readout exists for observables and validation.  It must **not** be applied
to the state during production propagation: projecting would hide exactly the
implementation error that the residual of Eq. (physical-coeff-residual) is
there to expose.
"""
from __future__ import annotations

import numpy as np

from mqcle.constants import CDTYPE, DTYPE
from mqcle.hermite.operator import mode_index_map, multi_factorial_array

__all__ = [
    "physical_frame_matrix",
    "hermite_metric",
    "weighted_pseudoinverse",
    "physical_projector",
    "rho_to_hermite",
    "hermite_to_rho",
    "physical_frame_residual",
]


def physical_frame_matrix(nstates: int, modes) -> np.ndarray:
    r"""Constant frame matrix :math:`B`, shape ``(M, nstates**2)``.

    Built from the standardised projector polynomial of Eq. (Pi-z),

    .. math::

        \Pi_{\lambda\lambda'}(z)
        =z_\lambda z_{\lambda'}+z_{N+\lambda}z_{N+\lambda'}
        -i(z_\lambda z_{N+\lambda'}-z_{\lambda'}z_{N+\lambda})
        -\delta_{\lambda\lambda'},

    whose Hermite expansion is exact and finite.  The diagonal case is
    :math:`\Pi_{\lambda\lambda}=1+\He_2(z_\lambda)+\He_2(z_{N+\lambda})`; the
    off-diagonal case is Eq. (Pi-offdiag-Hermite).

    Column ``mu*N+nu`` holds the Hermite coefficients of
    :math:`\Pi_{\nu\mu}`, matching :math:`q_{\rm phys}
    =\sum_{\mu\nu}\Pi_{\nu\mu}\rho_W^{\mu\nu}`, so that the imaginary part of
    the coherence carries the orientation the manuscript fixes -- the pairing
    that the tests check, not merely the real populations.
    """
    modes = [tuple(n) for n in modes]
    index = mode_index_map(modes)
    N = nstates
    d = 2 * N
    M = len(modes)
    B = np.zeros((M, N * N), dtype=CDTYPE)

    def e(*axes) -> tuple[int, ...]:
        n = [0] * d
        for a in axes:
            n[a] += 1
        return tuple(n)

    zero = tuple([0] * d)

    for mu in range(N):
        for nu in range(N):
            col = mu * N + nu
            # the entry multiplying rho_W^{mu nu} is Pi_{nu mu}
            lam, lamp = nu, mu
            if lam == lamp:
                # Pi_aa = 1 + He_2(z_a) + He_2(z_{N+a})
                if zero in index:
                    B[index[zero], col] += 1.0
                for axis in (lam, N + lam):
                    b = index.get(e(axis, axis))
                    if b is not None:
                        B[b, col] += 1.0
            else:
                for m, coeff in (
                    (e(lam, lamp), 1.0),
                    (e(N + lam, N + lamp), 1.0),
                    (e(lam, N + lamp), -1j),
                    (e(lamp, N + lam), +1j),
                ):
                    b = index.get(m)
                    if b is not None:
                        B[b, col] += coeff
    return B


def hermite_metric(modes) -> np.ndarray:
    r"""Diagonal of the Hermite metric :math:`W=\mathrm{diag}(\bm n!)`."""
    return multi_factorial_array(modes).astype(DTYPE)


def weighted_pseudoinverse(B: np.ndarray, W: np.ndarray) -> np.ndarray:
    r"""Weighted pseudoinverse :math:`(B^\dagger WB)^{-1}B^\dagger W`.

    This is the readout of Eq. (rho-from-c).  ``W`` is passed as its diagonal.
    """
    B = np.asarray(B, dtype=CDTYPE)
    w = np.asarray(W, dtype=DTYPE)
    BhW = B.conj().T * w[None, :]
    gram = BhW @ B
    return np.linalg.solve(gram, BhW)


def physical_projector(B: np.ndarray, W: np.ndarray) -> np.ndarray:
    r"""Weighted coefficient-space projector
    :math:`\mathsf P^{(c)}_{\rm phys}=B(B^\dagger WB)^{-1}B^\dagger W`."""
    return np.asarray(B, dtype=CDTYPE) @ weighted_pseudoinverse(B, W)


def rho_to_hermite(rho_W: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""Initialise coefficients from a subsystem density, ``c = B vec(rho_W)``.

    ``rho_W`` has shape ``(..., N, N)`` and the result ``(..., M)``.
    """
    rho = np.asarray(rho_W, dtype=CDTYPE)
    N = rho.shape[-1]
    flat = rho.reshape(rho.shape[:-2] + (N * N,))
    return np.einsum("mk,...k->...m", B, flat, optimize=True)


def hermite_to_rho(C: np.ndarray, B: np.ndarray, W: np.ndarray,
                   Bplus: np.ndarray | None = None) -> np.ndarray:
    r"""Read out :math:`\rho_W` from coefficients, Eq. (rho-from-c).

    For observables and validation only.  ``C`` has shape ``(..., M)`` and the
    result ``(..., N, N)``.
    """
    C = np.asarray(C, dtype=CDTYPE)
    if Bplus is None:
        Bplus = weighted_pseudoinverse(B, W)
    flat = np.einsum("km,...m->...k", Bplus, C, optimize=True)
    N = int(round(np.sqrt(flat.shape[-1])))
    return flat.reshape(flat.shape[:-1] + (N, N))


def physical_frame_residual(C: np.ndarray, B: np.ndarray, W: np.ndarray,
                            Pphys: np.ndarray | None = None) -> float:
    r"""Relative weighted residual outside the physical frame.

    .. math::

        \epsilon^{(c)}_{\rm phys}
        =\frac{\|(I-\mathsf P^{(c)}_{\rm phys})\bm c\|_W}{\|\bm c\|_W}

    Eq. (physical-coeff-residual).  For exact arithmetic and a correct
    implementation this stays at round-off for all time.  It is a diagnostic:
    a growing residual is evidence of an error, and must never be suppressed
    by projecting the state.
    """
    C = np.asarray(C, dtype=CDTYPE)
    w = np.asarray(W, dtype=DTYPE)
    if Pphys is None:
        Pphys = physical_projector(B, W)
    resid = C - np.einsum("mn,...n->...m", Pphys, C, optimize=True)
    num = np.sqrt(np.real(np.sum(w * np.abs(resid) ** 2)))
    den = np.sqrt(np.real(np.sum(w * np.abs(C) ** 2)))
    return float(num / den) if den > 0 else float(num)
