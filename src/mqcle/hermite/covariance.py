r"""Covariance of the finite Hermite operator under a subsystem basis change.

Theorem 3 leaves one residual freedom: a constant unitary relabelling of the
subsystem basis, :math:`h\mapsto U^{\dagger}hU`.  On the reduced field that
freedom is easy to state and easy to check, because :math:`\rho_W` carries the
subsystem indices explicitly and simply conjugates.  On the Hermite
coefficients it is not: the coefficient vector has no subsystem indices at all,
and whether the *truncation* survives the relabelling is a question the reduced
form cannot even ask.

This module answers it.  The unitary acts on the mapping amplitude as
:math:`c\mapsto U^{\dagger}c`, hence on the standardised variables as the real
form

.. math:: \mathcal V=\mathrm{re}\bigl(U^{\dagger}\bigr)\in O(2N),

which is orthogonal because :math:`U` is unitary, and symplectic because the
real form preserves :math:`\mathbb J`.  Orthogonality is what matters here:
:math:`\varphi(z)=(2\pi)^{-N}e^{-z^{\mathsf T}z/2}` is invariant, so the
amplitude transforms by composition alone,

.. math:: q'(z)=q\bigl(\mathcal V^{\mathsf T}z\bigr),

and an orthogonal substitution cannot change the total degree of a polynomial.
The induced map on coefficients is therefore *block diagonal in total Hermite
order*: it acts within :math:`\mathcal V_K` and never leaves it.  That is the
statement that makes the truncation basis independent rather than an artefact
of the basis it was written in.

Writing the induced map out,

.. math::

    c'_{\bm m}=\sum_{\bm n}\mathsf D_{\bm m\bm n}c_{\bm n},\qquad
    \mathsf D_{\bm m\bm n}
      =\frac{1}{\bm m!}\int\!\mathrm d z\,\varphi(z)\,
        \mathrm{He}_{\bm m}(\mathcal Vz)\,\mathrm{He}_{\bm n}(z),

and covariance of Eq. (finite-Hermite) is then the pair of exact identities

.. math::

    \mathsf R\bigl(U^{\dagger}hU\bigr)=\mathsf D\,\mathsf R(h)\,\mathsf D^{-1},
    \qquad
    \mathsf G\bigl(U^{\dagger}hU\bigr)=\mathsf D\,\mathsf G(h)\,\mathsf D^{-1},

which hold matrix element by matrix element and need no propagation to check.
Gate GH-11 checks them, together with the group property
:math:`\mathsf D(\mathcal V)^{-1}=\mathsf D(\mathcal V^{\mathsf T})` and the
intertwining of the physical frame,
:math:`B\,\mathrm{vec}(U^{\dagger}\rho U)=\mathsf D\,B\,\mathrm{vec}(\rho)`.

The integral is evaluated by tensor-product Gauss--Hermite quadrature rather
than by expanding the substitution symbolically.  The integrand is a product of
two polynomials of total degree at most :math:`K`, so a rule exact to degree
:math:`2K+1` evaluates it exactly up to round-off, and the same routine then
serves any :math:`K` without a separate derivation per order.
"""
from __future__ import annotations

import math

import numpy as np

from ..constants import DTYPE
from ..mapping import real_form

__all__ = [
    "mapping_real_form",
    "hermite_rotation_matrix",
    "rotate_density",
    "rotation_O2",
    "unitary_U2",
]


def rotation_O2(theta: float, n: int = 2) -> np.ndarray:
    """A real orthogonal rotation in the :math:`(0,1)` plane.

    Real :math:`O` keeps a real symmetric ``h`` real symmetric, so this alone
    never leaves the case Theorem 1 was first stated for.  It is included as
    the easy half of the check, not as the whole of it.
    """
    O = np.eye(n, dtype=DTYPE)
    c, s = np.cos(theta), np.sin(theta)
    O[0, 0], O[0, 1], O[1, 0], O[1, 1] = c, -s, s, c
    return O


def unitary_U2(theta: float, phi: float, n: int = 2) -> np.ndarray:
    """A genuinely complex :math:`U(N)` element, reducing to :func:`rotation_O2`
    at ``phi = 0``.

    This is the half that matters.  A nonzero ``phi`` sends real symmetric
    ``h`` to a complex Hermitian one, so :math:`\\mathbb H` acquires its
    antisymmetric block and the check exercises Corollary 1 rather than its
    real special case.
    """
    U = np.eye(n, dtype=complex)
    c, s = np.cos(theta), np.sin(theta)
    U[0, 0], U[0, 1] = c, -s * np.exp(1j * phi)
    U[1, 0], U[1, 1] = s * np.exp(-1j * phi), c
    return U


def mapping_real_form(U: np.ndarray) -> np.ndarray:
    r"""The orthogonal :math:`\mathcal V\in O(2N)` that ``U`` induces on ``z``.

    The mapping amplitude transforms as :math:`c\mapsto U^{\dagger}c`, so the
    real variables transform by the real form of :math:`U^{\dagger}` -- not of
    ``U``.  Getting that conjugate wrong is invisible for real orthogonal
    ``U`` and wrong by the sign of the imaginary part for everything else,
    which is precisely the case the gate exists to cover.
    """
    U = np.asarray(U, dtype=complex)
    return real_form(np.conj(U).T[None])[0]


def _gauss_hermite_nodes(degree: int):
    r"""Nodes and weights for :math:`\int\varphi_1(t)f(t)\,\mathrm dt` in one
    dimension, exact for polynomials of degree at most ``degree``.

    ``numpy.polynomial.hermite_e`` uses the probabilists' weight
    :math:`e^{-t^{2}/2}`, which is the one the standardisation of this paper
    already carries, so the weights need only be normalised to sum to one and
    no :math:`\sqrt2` rescaling of the nodes is required.  A rule with ``k``
    nodes is exact to degree ``2k - 1``.
    """
    k = int(np.ceil((degree + 1) / 2))
    t, w = np.polynomial.hermite_e.hermegauss(max(k, 1))
    return t, w / w.sum()


def hermite_rotation_matrix(modes, V: np.ndarray, nstates: int | None = None
                            ) -> np.ndarray:
    r"""The matrix :math:`\mathsf D` of the basis change on Hermite coefficients.

    ``V`` is the orthogonal :math:`2N\times2N` action on ``z`` -- normally the
    output of :func:`mapping_real_form`.  ``modes`` is the ordered mode list
    the rest of the branch uses, so :math:`\mathsf D` is returned in exactly
    the layout of :math:`\mathsf R` and :math:`\mathsf G` and may be multiplied
    against them directly.

    Shape ``(M, M)``, real.  Because ``V`` is orthogonal the result is block
    diagonal in total order; that property is asserted by the gate rather than
    imposed here, so a fault in ``V`` shows up as a failure instead of being
    silently projected away.
    """
    V = np.asarray(V, dtype=DTYPE)
    d = V.shape[-1]
    if nstates is not None and d != 2 * nstates:
        raise ValueError(f"V is {d}x{d}, expected {2 * nstates}x{2 * nstates}")
    modes = [tuple(int(v) for v in n) for n in modes]
    K = max((sum(n) for n in modes), default=0)

    # exact for the product of two polynomials of total degree <= K
    t, w = _gauss_hermite_nodes(2 * K + 1)
    k = t.size
    grids = np.meshgrid(*([t] * d), indexing="ij")
    Z = np.stack([g.ravel() for g in grids], axis=1)          # (k**d, d)
    W = np.ones(k ** d, dtype=DTYPE)
    for g in np.meshgrid(*([w] * d), indexing="ij"):
        W = W * g.ravel()

    # He_n(z) and He_m(V z) on the same nodes
    ZV = Z @ np.asarray(V, dtype=DTYPE).T
    maxn = max((max(n) for n in modes), default=0)
    he = np.polynomial.hermite_e.hermeval(
        Z.T, np.eye(maxn + 1))                                # (maxn+1, d, npts)
    heV = np.polynomial.hermite_e.hermeval(ZV.T, np.eye(maxn + 1))

    def poly(table, n):
        out = np.ones(Z.shape[0], dtype=DTYPE)
        for i, ni in enumerate(n):
            if ni:
                out = out * table[ni, i]
        return out

    M = len(modes)
    fact = np.array([float(np.prod([math.factorial(v) for v in n]))
                     for n in modes], dtype=DTYPE)
    Hn = np.stack([poly(he, n) for n in modes])               # (M, npts)
    Hm = np.stack([poly(heV, n) for n in modes])              # (M, npts)
    D = (Hm * W) @ Hn.T
    return D / fact[:, None]


def rotate_density(rho: np.ndarray, U: np.ndarray) -> np.ndarray:
    r""":math:`\rho\mapsto U^{\dagger}\rho U`, batched over the bath grid.

    The same convention as ``scripts/07c_basis_covariance.py`` uses for the
    reduced branch, so the two checks are comparing like with like.
    """
    U = np.asarray(U)
    return np.einsum("ai,...ab,bj->...ij", np.conj(U), rho, U)
