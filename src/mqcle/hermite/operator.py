r"""Exact finite Gaussian--Hermite coefficient operator for the mapping QCLE.

This module implements the sparse analytic operator of
``GH_MQCLE_FORMULATION.tex``.  It is the *primary* numerical representation:
the evolved unknown is the Hermite coefficient field

.. math::

    q(z,X,t)=\sum_{|\bm n|\le K}c_{\bm n}(X,t)\,\He_{\bm n}(z),

and nothing here reduces the state to the four subsystem-QCLE fields
:math:`\rho_W^{\lambda\lambda'}`.  The reduction exists only in
:mod:`mqcle.hermite.frame`, and only for initialisation, readout and
validation.

The three objects built here are

``T^{ij}``
    constant sparse matrices realising
    :math:`\mathcal T_{ij}=z_i\partial_{z_j}+z_j\partial_{z_i}
    -\partial_{z_i}\partial_{z_j}` on the Hermite basis, Eq. (T-matrix);

``R(R)``
    the mapping-rotation matrix :math:`-K_{ij}z_j\partial_{z_i}`, already in
    its contracted form, Eq. (R-matrix);

``G_alpha(R)``
    the bath-force matrix
    :math:`\partial_{R_\alpha}V_b\,\mathsf I
    +\tfrac14\partial_{R_\alpha}\mathbb H_{ij}\mathsf T^{ij}`, Eq. (G-matrix).

Every matrix element comes from the Kronecker-delta recurrences.  Nothing is
fitted, sampled or quadratured; Gauss--Hermite quadrature appears only in the
test suite as an independent check.
"""
from __future__ import annotations

from itertools import product

import numpy as np

from mqcle.constants import CDTYPE, DTYPE, HBAR

__all__ = [
    "active_modes",
    "mode_index_map",
    "multi_factorial_array",
    "hermite_T_matrices",
    "rotation_basis_matrices",
    "real_hamiltonian",
    "d_real_hamiltonian_dR",
    "symplectic_J",
    "rotation_matrix",
    "force_matrix",
    "hermite_rhs",
]


# --------------------------------------------------------------------------
# mode set
# --------------------------------------------------------------------------
def active_modes(nstates: int, K: int, parity: str | None = None
                 ) -> list[tuple[int, ...]]:
    r"""Active Hermite multi-indices, deterministically ordered.

    The mapping vector is :math:`z=(r_1..r_N,p_1..p_N)`, so a multi-index has
    ``2*nstates`` components.  Ordering is by increasing total order
    :math:`|\bm n|` and then lexicographically, exactly as the implementation
    specification requires, so that the mode list is reproducible across runs
    and platforms.

    ``parity`` selects ``"even"`` or ``"odd"`` total order, or ``None`` for
    both.  The physical projected density needs only ``parity="even"`` with
    ``K=2``, which gives :math:`1+2N(2N+1)/2` modes -- eleven for two states.
    """
    if nstates < 1:
        raise ValueError("active_modes: nstates must be >= 1")
    if K < 0:
        raise ValueError("active_modes: K must be >= 0")
    if parity not in (None, "even", "odd"):
        raise ValueError("active_modes: parity must be None, 'even' or 'odd'")

    dim = 2 * nstates
    modes: list[tuple[int, ...]] = []
    for order in range(K + 1):
        if parity == "even" and order % 2:
            continue
        if parity == "odd" and not order % 2:
            continue
        block = [n for n in product(range(order + 1), repeat=dim)
                 if sum(n) == order]
        modes.extend(sorted(block))
    return modes


def mode_index_map(modes) -> dict[tuple[int, ...], int]:
    """Map each multi-index to its position in ``modes``."""
    return {tuple(n): a for a, n in enumerate(modes)}


def multi_factorial_array(modes) -> np.ndarray:
    r"""The Hermite metric weights :math:`\bm n!=\prod_a n_a!` as an array."""
    out = np.ones(len(modes), dtype=DTYPE)
    for a, n in enumerate(modes):
        v = 1.0
        for c in n:
            for k in range(2, c + 1):
                v *= k
        out[a] = v
    return out


# --------------------------------------------------------------------------
# constant Hermite algebra
# --------------------------------------------------------------------------
def hermite_T_matrices(modes) -> np.ndarray:
    r"""Constant matrices :math:`\mathsf T^{ij}`, shape ``(d, d, M, M)``.

    Implements Eq. (T-matrix),

    .. math::

        \mathsf T^{ij}_{\bm m\bm n}
        = n_j\delta_{\bm m,\bm n+\bm e_i-\bm e_j}
        + n_i\delta_{\bm m,\bm n-\bm e_i+\bm e_j}
        + n_i(n_j-\delta_{ij})\delta_{\bm m,\bm n-\bm e_i-\bm e_j}.

    The first two targets preserve total order and the third lowers it by two,
    so no entry ever raises the Hermite order.  Several ``(i, j)`` pairs can
    reach the same target, so contributions accumulate.  Targets with a
    negative component, or outside the active set, are dropped: a truncated
    ``K`` is exact for initial data inside :math:`\mathcal V_K` precisely
    because nothing leaves it upward.
    """
    modes = [tuple(n) for n in modes]
    index = mode_index_map(modes)
    d = len(modes[0])
    M = len(modes)
    T = np.zeros((d, d, M, M), dtype=DTYPE)

    for a, n in enumerate(modes):
        for i in range(d):
            for j in range(d):
                # m1 = n + e_i - e_j, coefficient n_j
                if n[j] > 0:
                    m = list(n)
                    m[i] += 1
                    m[j] -= 1
                    b = index.get(tuple(m))
                    if b is not None:
                        T[i, j, b, a] += n[j]
                # m2 = n - e_i + e_j, coefficient n_i
                if n[i] > 0:
                    m = list(n)
                    m[i] -= 1
                    m[j] += 1
                    b = index.get(tuple(m))
                    if b is not None:
                        T[i, j, b, a] += n[i]
                # m3 = n - e_i - e_j, coefficient n_i (n_j - delta_ij)
                coeff = n[i] * (n[j] - (1 if i == j else 0))
                if coeff:
                    m = list(n)
                    m[i] -= 1
                    m[j] -= 1
                    if min(m) >= 0:
                        b = index.get(tuple(m))
                        if b is not None:
                            T[i, j, b, a] += coeff
    return T


def rotation_basis_matrices(modes) -> np.ndarray:
    r"""Constant matrices :math:`\mathsf N^{ij}`, shape ``(d, d, M, M)``.

    Defined by :math:`\mathsf N^{ij}_{\bm m\bm n}
    =n_i\,\delta_{\bm m,\bm n-\bm e_i+\bm e_j}`, so that the rotation matrix of
    Eq. (R-matrix) is :math:`\mathsf R=-K_{ij}\mathsf N^{ij}`.

    This is the *contracted* form.  The apparent order-lowering remainder of
    :math:`-K_{ij}z_j\partial_{z_i}\He_{\bm n}` carries the coefficient
    :math:`n_i(n_j-\delta_{ij})`, which is symmetric in :math:`(i,j)` and
    therefore annihilated by the antisymmetric :math:`K`.  It is never
    assembled here; GH-03 verifies the cancellation independently.
    """
    modes = [tuple(n) for n in modes]
    index = mode_index_map(modes)
    d = len(modes[0])
    M = len(modes)
    N_ij = np.zeros((d, d, M, M), dtype=DTYPE)

    for a, n in enumerate(modes):
        for i in range(d):
            if n[i] == 0:
                continue
            for j in range(d):
                m = list(n)
                m[i] -= 1
                m[j] += 1
                b = index.get(tuple(m))
                if b is not None:
                    N_ij[i, j, b, a] += n[i]
    return N_ij


# --------------------------------------------------------------------------
# the real 2N x 2N form of a Hermitian subsystem Hamiltonian
# --------------------------------------------------------------------------
def symplectic_J(nstates: int) -> np.ndarray:
    r""":math:`\mathbb J=\begin{pmatrix}0&I\\-I&0\end{pmatrix}`."""
    n = nstates
    J = np.zeros((2 * n, 2 * n), dtype=DTYPE)
    J[:n, n:] = np.eye(n)
    J[n:, :n] = -np.eye(n)
    return J


def real_hamiltonian(h: np.ndarray) -> np.ndarray:
    r"""Real symmetric form :math:`\mathbb H` of a Hermitian ``h``.

    With :math:`h=A+iB`, :math:`A^{\mathsf T}=A`, :math:`B^{\mathsf T}=-B`,

    .. math::

        \mathbb H=\begin{pmatrix}A&-B\\B&A\end{pmatrix},
        \qquad \mathbb H^{\mathsf T}=\mathbb H .

    This is Eq. (realH).  It is the correct object after a generic unitary
    change of subsystem basis; the real-symmetric shortcut is not.  ``h`` may
    carry leading axes, e.g. ``(n_R, N, N)``.
    """
    h = np.asarray(h)
    A = h.real
    B = h.imag
    top = np.concatenate([A, -B], axis=-1)
    bot = np.concatenate([B, A], axis=-1)
    return np.concatenate([top, bot], axis=-2).astype(DTYPE)


def d_real_hamiltonian_dR(dh_dR: np.ndarray) -> np.ndarray:
    r""":math:`\partial_R\mathbb H` from :math:`\partial_R h`.

    The map ``h -> H`` of :func:`real_hamiltonian` is real-linear, so the
    derivative is obtained by applying the same block construction.
    """
    return real_hamiltonian(dh_dR)


# --------------------------------------------------------------------------
# R-dependent operator matrices
# --------------------------------------------------------------------------
def rotation_matrix(H_real: np.ndarray, hbar: float, modes,
                    N_ij: np.ndarray | None = None) -> np.ndarray:
    r"""Mapping-rotation matrix :math:`\mathsf R(\bm R)`, shape ``(n_R, M, M)``.

    ``H_real`` is :math:`\mathbb H(\bm R)` with shape ``(n_R, d, d)``.  With
    :math:`K=\hbar^{-1}\mathbb J\mathbb H` -- antisymmetric for Hermitian
    ``h`` -- this returns :math:`\mathsf R=-K_{ij}\mathsf N^{ij}`.

    The rotation preserves total Hermite order exactly, which is step (i) of
    the closure argument.
    """
    H_real = np.asarray(H_real, dtype=DTYPE)
    d = H_real.shape[-1]
    nstates = d // 2
    if N_ij is None:
        N_ij = rotation_basis_matrices(modes)
    J = symplectic_J(nstates)
    K = np.einsum("ab,rbc->rac", J, H_real, optimize=True) / hbar
    return -np.einsum("rij,ijmn->rmn", K, N_ij, optimize=True).astype(CDTYPE)


def force_matrix(dVb_dR: np.ndarray, dH_real_dR: np.ndarray,
                 T_mats: np.ndarray, modes) -> np.ndarray:
    r"""Bath-force matrix :math:`\mathsf G_\alpha(\bm R)`, shape ``(n_R, M, M)``.

    Eq. (G-matrix),

    .. math::

        \mathsf G_\alpha
        =\partial_{R_\alpha}V_b\,\mathsf I
        +\frac14\sum_{ij}\partial_{R_\alpha}\mathbb H_{ij}\,\mathsf T^{ij}.

    ``dVb_dR`` has shape ``(n_R,)`` and ``dH_real_dR`` shape ``(n_R, d, d)``,
    both for the single bath coordinate :math:`\alpha`.  The caller supplies
    one pair per bath coordinate.

    Convention: ``dH_real_dR`` must be built from the *full* subsystem
    Hamiltonian and ``dVb_dR`` from the bare bath potential.  Mixing a
    traceless :math:`\bar h` here with a full ``h`` in the rotation, or the
    reverse, breaks the cancellation; the manuscript documents that failure.
    """
    dVb_dR = np.asarray(dVb_dR, dtype=DTYPE)
    dH = np.asarray(dH_real_dR, dtype=DTYPE)
    M = T_mats.shape[-1]
    G = 0.25 * np.einsum("rij,ijmn->rmn", dH, T_mats, optimize=True)
    G = G.astype(CDTYPE)
    idx = np.arange(M)
    G[:, idx, idx] += dVb_dR[:, None]
    return G


# --------------------------------------------------------------------------
# the right-hand side
# --------------------------------------------------------------------------
def sparse_blocks(mat: np.ndarray, tol: float = 0.0) -> list:
    r"""Nonzero ``(m, n, coefficient(R))`` blocks of an ``(n_R, M, M)`` matrix.

    The Hermite matrices are structurally sparse: for the two-state even
    :math:`K=2` sector only 24 of the 121 blocks of :math:`\mathsf R` and 27 of
    :math:`\mathsf G` are nonzero anywhere on the grid.  Skipping the empty
    blocks is what makes the mode contraction bandwidth-bound rather than
    dominated by structural zeros.
    """
    mat = np.asarray(mat)
    M = mat.shape[-1]
    out = []
    for m in range(M):
        for n in range(M):
            col = mat[:, m, n]
            if np.max(np.abs(col)) > tol:
                out.append((m, n, np.ascontiguousarray(col)))
    return out


def hermite_rhs(C: np.ndarray, grid, R_blocks, G_blocks,
                mass: float, spectral_derivative=None,
                out: np.ndarray | None = None) -> np.ndarray:
    r"""Exact semi-discrete Hermite right-hand side, Eq. (semi-discrete).

    .. math::

        \dot C=-\frac{P}{M}D_RC+\mathsf R(R)C+\mathsf G(R)D_PC .

    ``C`` is **mode-major**, shape ``(M, n_R, n_P)``, so that each Hermite mode
    occupies a contiguous bath plane.  That layout was chosen by measurement,
    not by taste: at the :math:`P_0=70` production size it evaluates this
    right-hand side about eight times faster than the mode-minor alternative,
    because both bath transforms and the mode contraction then run on
    contiguous memory.

    The bath derivatives reuse the validated Fourier spectral routine of
    :mod:`mqcle.coefficient_operator`; the scientific change is the state and
    the mapping operator, not the bath discretisation.

    Nothing here touches :math:`\rho_W`, and nothing calls the subsystem-QCLE
    Strang propagator.
    """
    if spectral_derivative is None:
        from mqcle.coefficient_operator import spectral_derivative

    C = np.asarray(C, dtype=CDTYPE)
    n_R, n_P = grid.shape
    if C.ndim != 3 or C.shape[1:] != (n_R, n_P):
        raise ValueError(f"hermite_rhs: C has shape {C.shape}, "
                         f"expected (M, {n_R}, {n_P})")

    dC_dR = spectral_derivative(C, grid.kR, axis=1)
    dC_dP = spectral_derivative(C, grid.kP, axis=2)

    if out is None:
        out = np.empty_like(C)
    np.multiply(dC_dR, -(grid.P[None, None, :] / mass), out=out)

    for m, n, coeff in R_blocks:
        out[m] += coeff[:, None] * C[n]
    for m, n, coeff in G_blocks:
        out[m] += coeff[:, None] * dC_dP[n]
    return out
