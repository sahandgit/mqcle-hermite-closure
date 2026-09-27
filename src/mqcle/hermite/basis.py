r"""Probabilists' Hermite polynomials and multi-indices.

Convention (fixed once, used everywhere)
----------------------------------------
.. math::
    \mathrm{He}_0=1,\qquad \mathrm{He}_1=z,\qquad
    \mathrm{He}_{n+1}(z)=z\,\mathrm{He}_n(z)-n\,\mathrm{He}_{n-1}(z),

orthogonal with respect to the **standard** Gaussian weight

.. math::
    \varphi(z)=(2\pi)^{-d/2}e^{-z^{\mathsf T}z/2},
    \qquad
    \langle f,g\rangle_\varphi=\int\!dz\,\varphi(z)f^{*}(z)g(z),
    \qquad
    \langle \mathrm{He}_{\mathbf m},\mathrm{He}_{\mathbf n}\rangle_\varphi
      =\mathbf n!\,\delta_{\mathbf m\mathbf n}.

Multivariate polynomials are products,
:math:`\mathrm{He}_{\mathbf n}(z)=\prod_a\mathrm{He}_{n_a}(z_a)`, and
:math:`\mathbf n!=\prod_an_a!`.

Standardised mapping variables
------------------------------
The mapping Gaussian of the physical problem is
:math:`\phi(x)\propto e^{-|x|^{2}/\hbar}`.  It becomes the standard Gaussian
under

.. math:: z=\sqrt{2/\hbar}\,x ,

so every Hermite statement in this codebase is a statement about ``z``, never
about ``x``.  :func:`standardise` and :func:`unstandardise` are the only places
that know the factor.
"""
from __future__ import annotations

from itertools import product as _iproduct
from math import factorial

import numpy as np
import sympy as sp

__all__ = ["he_poly", "he_multi", "multi_indices", "multi_factorial",
           "standardise", "unstandardise", "hermite_weight_symbolic",
           "STANDARD_GAUSSIAN_DOC"]

STANDARD_GAUSSIAN_DOC = "phi(z) = (2 pi)^{-d/2} exp(-z^T z / 2)"


def he_poly(n: int, z):
    r"""Univariate :math:`\mathrm{He}_n(z)` by the three-term recurrence."""
    if n < 0:
        raise ValueError("Hermite order must be non-negative")
    prev, cur = sp.Integer(1), sp.sympify(z)
    if n == 0:
        return prev
    for k in range(1, n):
        prev, cur = cur, sp.expand(z * cur - k * prev)
    return sp.expand(cur)


def he_multi(index, variables) -> sp.Expr:
    r"""Multivariate :math:`\mathrm{He}_{\mathbf n}(z)=\prod_a\mathrm{He}_{n_a}(z_a)`."""
    if len(index) != len(variables):
        raise ValueError("multi-index length must match the number of variables")
    out = sp.Integer(1)
    for n, v in zip(index, variables):
        out *= he_poly(int(n), v)
    return sp.expand(out)


def multi_indices(dim: int, order: int, exact: bool = False):
    """All multi-indices with ``|n| <= order`` (or ``== order`` if ``exact``)."""
    out = []
    for idx in _iproduct(range(order + 1), repeat=dim):
        s = sum(idx)
        if (s == order) if exact else (s <= order):
            out.append(tuple(idx))
    return sorted(out, key=lambda t: (sum(t), t))


def multi_factorial(index) -> int:
    r""":math:`\mathbf n!=\prod_a n_a!`."""
    out = 1
    for n in index:
        out *= factorial(int(n))
    return out


def standardise(expr, x_syms, z_syms, hbar):
    r"""Rewrite an expression in mapping variables ``x`` in terms of
    :math:`z=\sqrt{2/\hbar}\,x`, i.e. substitute :math:`x=\sqrt{\hbar/2}\,z`."""
    sub = {x: sp.sqrt(hbar / 2) * z for x, z in zip(x_syms, z_syms)}
    return sp.expand(sp.sympify(expr).subs(sub, simultaneous=True))


def unstandardise(expr, x_syms, z_syms, hbar):
    """Inverse of :func:`standardise`."""
    sub = {z: sp.sqrt(2 / hbar) * x for x, z in zip(x_syms, z_syms)}
    return sp.expand(sp.sympify(expr).subs(sub, simultaneous=True))


def hermite_weight_symbolic(z_syms, hbar=None) -> sp.Expr:
    r"""The standard Gaussian :math:`\varphi(z)` in ``len(z_syms)`` dimensions."""
    d = len(z_syms)
    return sp.exp(-sum(z**2 for z in z_syms) / 2) / (2 * sp.pi) ** sp.Rational(d, 2)


def he_numeric(n: int, z: np.ndarray) -> np.ndarray:
    """Numerical univariate ``He_n`` by the same recurrence (vectorised)."""
    z = np.asarray(z)
    prev = np.ones_like(z)
    if n == 0:
        return prev
    cur = z.copy()
    for k in range(1, n):
        prev, cur = cur, z * cur - k * prev
    return cur
