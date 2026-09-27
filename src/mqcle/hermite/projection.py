r"""Exact Hermite projection :math:`\mathsf P_K` and the residual
:math:`(I-\mathsf P_K)`.

Why this module exists
----------------------
Ordinary polynomial degree and Hermite order are **not** the same bookkeeping.
For the standard Gaussian,

.. math:: z^{4}=\mathrm{He}_4+6\,\mathrm{He}_2+3,

so the highest-degree monomial part of a quartic,
:math:`\mathrm{Top}_4[z^4]=z^4`, differs from the orthogonal residual
:math:`(I-\mathsf P_2)z^4=\mathrm{He}_4=z^4-6z^2+3`.  An earlier version of
this pipeline computed the former and called it the latter.  Both quantities
are now available under separate names and are never conflated.

What *is* still true is the one-way implication used by the closure argument:
a polynomial amplitude of ordinary degree :math:`\le K` lies in
:math:`\mathrm{span}\{\mathrm{He}_{\mathbf n}:|\mathbf n|\le K\}`, because
:math:`\mathrm{He}_{\mathbf n}` is a triangular change of basis.  Hence
degree :math:`\le2` output proves :math:`V_2` membership; it does not tell you
how the result is distributed among Hermite orders :math:`0,2,\dots,K`.

Inner product
-------------
Everything here is orthogonality with respect to

.. math:: \langle f,g\rangle_\varphi=\int dz\,\varphi(z)f^{*}(z)g(z),

acting on the *amplitude* :math:`q=\rho/\varphi`.  Equivalently, on
Gaussian-weighted densities, :math:`\langle\rho_f,\rho_g\rangle_{\varphi^{-1}}
=\int dz\,\rho_f^{*}\rho_g/\varphi`.  The amplitude representation is the one
used throughout.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import sympy as sp

from .basis import (he_multi, he_numeric, multi_factorial, multi_indices,
                    standardise)

__all__ = [
    "hermite_coefficients_symbolic", "project_hermite_symbolic",
    "hermite_residual_symbolic", "hermite_orders", "top_degree_part",
    "polynomial_degrees", "hermite_coefficients_numeric",
    "amplitude_from_density_symbolic",
]


# --------------------------------------------------------------------------
# symbolic
# --------------------------------------------------------------------------
def hermite_coefficients_symbolic(poly, variables: Sequence,
                                  max_order: int | None = None
                                  ) -> dict[tuple, sp.Expr]:
    r"""Exact Hermite coefficients of a polynomial amplitude.

    Returns ``{multi_index: c_n}`` with
    :math:`q(z)=\sum_{\mathbf n}c_{\mathbf n}\mathrm{He}_{\mathbf n}(z)`.

    The expansion is exact and finite for any polynomial: peel the highest
    ordinary degree, whose Hermite coefficients equal its monomial
    coefficients (because :math:`\mathrm{He}_{\mathbf n}` has leading monomial
    :math:`z^{\mathbf n}`), subtract, and repeat.  No quadrature is involved,
    so there is no integration error.

    ``max_order`` truncates the *reported* dictionary only; the peeling always
    runs to completion so the result is exact.
    """
    q = sp.expand(sp.sympify(poly))
    coeffs: dict[tuple, sp.Expr] = {}
    guard = 0
    while q != 0:
        guard += 1
        if guard > 512:                     # pragma: no cover - safety net
            raise RuntimeError("Hermite peeling failed to terminate")
        P = sp.Poly(q, *variables)
        monoms = P.monoms()
        deg = max(sum(m) for m in monoms)
        top = [m for m in monoms if sum(m) == deg]
        for m in top:
            c = sp.expand(P.coeff_monomial(sp.prod(
                [v**e for v, e in zip(variables, m)])))
            if c == 0:
                continue
            coeffs[tuple(m)] = sp.simplify(c) if c.free_symbols else c
            q = sp.expand(q - c * he_multi(m, variables))
    if max_order is not None:
        coeffs = {k: v for k, v in coeffs.items() if sum(k) <= max_order}
    return coeffs


def project_hermite_symbolic(poly, variables: Sequence, max_order: int) -> sp.Expr:
    r""":math:`\mathsf P_K q=\sum_{|\mathbf n|\le K}c_{\mathbf n}
    \mathrm{He}_{\mathbf n}`."""
    coeffs = hermite_coefficients_symbolic(poly, variables)
    out = sp.Integer(0)
    for idx, c in coeffs.items():
        if sum(idx) <= max_order:
            out += c * he_multi(idx, variables)
    return sp.expand(out)


def hermite_residual_symbolic(poly, variables: Sequence, max_order: int) -> sp.Expr:
    r""":math:`(I-\mathsf P_K)q=\sum_{|\mathbf n|>K}c_{\mathbf n}
    \mathrm{He}_{\mathbf n}`."""
    coeffs = hermite_coefficients_symbolic(poly, variables)
    out = sp.Integer(0)
    for idx, c in coeffs.items():
        if sum(idx) > max_order:
            out += c * he_multi(idx, variables)
    return sp.expand(out)


def hermite_orders(poly, variables: Sequence) -> list[int]:
    """Hermite orders ``|n|`` actually present, from real Hermite coefficients.

    Distinct from :func:`polynomial_degrees`, which reports ordinary monomial
    degrees.  For a polynomial amplitude the two happen to have the same
    *maximum*, because the change of basis is triangular; they need not have
    the same *set*, and the associated residuals are different functions.
    """
    coeffs = hermite_coefficients_symbolic(poly, variables)
    return sorted({sum(k) for k, v in coeffs.items() if v != 0})


def polynomial_degrees(poly, variables: Sequence) -> list[int]:
    """Ordinary total monomial degrees present in ``poly``."""
    P = sp.Poly(sp.expand(sp.sympify(poly)), *variables)
    return sorted({int(sum(m)) for m in P.monoms()})


def top_degree_part(poly, variables: Sequence, above: int) -> sp.Expr:
    r""":math:`\mathrm{Top}_{>K}[q]`: the monomials of ordinary degree ``> above``.

    This is *not* :math:`(I-\mathsf P_K)q`.  It is reported under this name so
    the distinction is visible at every call site.
    """
    P = sp.Poly(sp.expand(sp.sympify(poly)), *variables)
    out = sp.Integer(0)
    for m, c in zip(P.monoms(), P.coeffs()):
        if sum(m) > above:
            out += c * sp.prod([v**e for v, e in zip(variables, m)])
    return sp.expand(out)


def amplitude_from_density_symbolic(density, phi, x_syms, z_syms, hbar):
    r"""Return the standardised amplitude :math:`q(z)=\rho/\phi` with
    :math:`x=\sqrt{\hbar/2}\,z` substituted."""
    q = sp.expand(sp.cancel(sp.expand(density) / phi))
    return standardise(q, x_syms, z_syms, hbar)


# --------------------------------------------------------------------------
# numeric
# --------------------------------------------------------------------------
def hermite_coefficients_numeric(fn, dim: int, max_order: int, npts: int = 16
                                 ) -> dict[tuple, complex]:
    r"""Gauss-Hermite quadrature of
    :math:`c_{\mathbf n}=\frac{1}{\mathbf n!}\int dz\,\varphi(z)
    \mathrm{He}_{\mathbf n}(z)q(z)`.

    ``fn(z)`` receives an ``(M, dim)`` array of standardised coordinates and
    returns the amplitude ``q`` at those points.  Provided as an independent
    numerical check on the symbolic route, not as its replacement.
    """
    t, w = np.polynomial.hermite.hermgauss(npts)
    nodes1d = t * np.sqrt(2.0)                  # e^{-t^2} -> e^{-z^2/2}
    wts1d = w / np.sqrt(np.pi)                  # weights sum to one
    grids = np.array(np.meshgrid(*([nodes1d] * dim), indexing="ij"))
    Z = grids.reshape(dim, -1).T
    W = np.ones(Z.shape[0])
    idx = np.array(np.meshgrid(*([np.arange(npts)] * dim),
                               indexing="ij")).reshape(dim, -1).T
    for a in range(dim):
        W = W * wts1d[idx[:, a]]
    q = np.asarray(fn(Z))
    out: dict[tuple, complex] = {}
    for n in multi_indices(dim, max_order):
        he = np.ones(Z.shape[0])
        for a, na in enumerate(n):
            he = he * he_numeric(int(na), Z[:, a])
        out[tuple(n)] = complex(np.sum(W * he * q) / multi_factorial(n))
    return out
