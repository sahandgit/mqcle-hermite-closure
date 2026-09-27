r"""Scalar fields on the extended phase space and their derivatives.

Addition to the mandated layout, justified as follows: both :mod:`mqcle.pbme`
and :mod:`mqcle.excess` need derivatives of a test function
:math:`f(x,R,P)`, and the operator tests require an *exact* reference for
those derivatives.  Putting the differentiation behind one interface keeps the
operator modules free of differentiation machinery and lets every operator be
evaluated twice -- once analytically and once by finite differences -- with no
duplicated code.

Two implementations
-------------------
``SymbolicField``
    Built from a SymPy expression; every derivative is differentiated
    symbolically and lambdified, so it is exact to round-off.  This is the
    ground truth for the operator gates.
``FiniteDifferenceField``
    Wraps any callable; central differences with per-order step sizes.  Used to
    cross-check the analytic path and, later, to act on numerically represented
    densities.

Required derivative set
-----------------------
``grad_x``          :math:`\partial f/\partial x`, shape ``(..., 2N)``
``grad_R``          :math:`\partial f/\partial R`, shape ``(..., Nb)``
``grad_P``          :math:`\partial f/\partial P`, shape ``(..., Nb)``
``excess_tensor``   :math:`D_{\lambda\lambda'}\partial_{P_k}f` with
                    :math:`D_{\lambda\lambda'}=\partial_{r_{\lambda'}}
                    \partial_{r_\lambda}+\partial_{p_{\lambda'}}
                    \partial_{p_\lambda}`, shape ``(..., Nb, N, N)``
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np
import sympy as sp

from .constants import DTYPE

__all__ = ["ScalarField", "SymbolicField", "FiniteDifferenceField",
           "symbols_for", "projected_density_expr", "result_dtype"]


def result_dtype(*values) -> type:
    """float64 unless any input is complex, in which case complex128.

    The off-diagonal projector functions carry an explicit ``-i L`` term, so
    forcing float64 on derivative results silently discards their imaginary
    part.  Every array allocated for derivative output goes through here.
    """
    return np.complex128 if any(np.iscomplexobj(v) for v in values) else DTYPE


class ScalarField(ABC):
    """A scalar function of ``(x, R, P)`` that can supply its derivatives."""

    nstates: int
    nbath: int

    @abstractmethod
    def value(self, x, R, P) -> np.ndarray: ...

    @abstractmethod
    def grad_x(self, x, R, P) -> np.ndarray: ...

    @abstractmethod
    def grad_R(self, x, R, P) -> np.ndarray: ...

    @abstractmethod
    def grad_P(self, x, R, P) -> np.ndarray: ...

    @abstractmethod
    def excess_tensor(self, x, R, P) -> np.ndarray: ...

    def __call__(self, x, R, P) -> np.ndarray:
        return self.value(x, R, P)


def symbols_for(nstates: int = 2, nbath: int = 1):
    """Return ``(r, p, R, P)`` tuples of SymPy symbols with the project layout."""
    r = sp.symbols(f"r0:{nstates}", real=True)
    p = sp.symbols(f"p0:{nstates}", real=True)
    R = sp.symbols(f"R0:{nbath}", real=True)
    P = sp.symbols(f"P0:{nbath}", real=True)
    return r, p, R, P


class SymbolicField(ScalarField):
    """Exact derivatives of a SymPy expression, lambdified to NumPy."""

    def __init__(self, expr, nstates: int = 2, nbath: int = 1,
                 syms: Sequence | None = None) -> None:
        self.nstates, self.nbath = nstates, nbath
        r, p, R, P = syms if syms is not None else symbols_for(nstates, nbath)
        self.syms = (r, p, R, P)
        self.expr = sp.sympify(expr)
        args = list(r) + list(p) + list(R) + list(P)
        self._args = args

        def lam(e):
            return sp.lambdify(args, e, "numpy")

        self._value = lam(self.expr)
        self._gx = [lam(sp.diff(self.expr, s)) for s in list(r) + list(p)]
        self._gR = [lam(sp.diff(self.expr, s)) for s in R]
        self._gP = [lam(sp.diff(self.expr, s)) for s in P]
        self._exc = [[[lam(sp.diff(self.expr, P[k], r[lp], r[lam_])
                           + sp.diff(self.expr, P[k], p[lp], p[lam_]))
                       for lp in range(nstates)] for lam_ in range(nstates)]
                     for k in range(nbath)]

    # -- helpers ---------------------------------------------------------
    def _unpack(self, x, R, P):
        x = np.asarray(x, dtype=DTYPE)
        R = np.asarray(R, dtype=DTYPE)
        P = np.asarray(P, dtype=DTYPE)
        cols = [x[..., i] for i in range(2 * self.nstates)]
        cols += [R[..., k] for k in range(self.nbath)]
        cols += [P[..., k] for k in range(self.nbath)]
        return cols, np.broadcast(*cols).shape if len(cols) > 1 else cols[0].shape

    def _eval(self, fn, cols, shape):
        out = np.asarray(fn(*cols))
        return np.broadcast_to(out.astype(result_dtype(out)), shape).copy()

    # -- interface -------------------------------------------------------
    def value(self, x, R, P):
        cols, shape = self._unpack(x, R, P)
        return self._eval(self._value, cols, shape)

    def grad_x(self, x, R, P):
        cols, shape = self._unpack(x, R, P)
        return np.stack([self._eval(f, cols, shape) for f in self._gx], axis=-1)

    def grad_R(self, x, R, P):
        cols, shape = self._unpack(x, R, P)
        return np.stack([self._eval(f, cols, shape) for f in self._gR], axis=-1)

    def grad_P(self, x, R, P):
        cols, shape = self._unpack(x, R, P)
        return np.stack([self._eval(f, cols, shape) for f in self._gP], axis=-1)

    def excess_tensor(self, x, R, P):
        cols, shape = self._unpack(x, R, P)
        probe = self._eval(self._exc[0][0][0], cols, shape)
        out = np.empty(shape + (self.nbath, self.nstates, self.nstates),
                       dtype=result_dtype(probe))
        for k in range(self.nbath):
            for a in range(self.nstates):
                for b in range(self.nstates):
                    out[..., k, a, b] = self._eval(self._exc[k][a][b], cols, shape)
        return out


class FiniteDifferenceField(ScalarField):
    """Central finite differences of an arbitrary callable ``fn(x, R, P)``.

    Step sizes follow the usual optimum ``h ~ eps**(1/(n+2))`` for an ``n``-th
    derivative: ``hx`` is used for the second mapping derivative and ``hP`` for
    the bath-momentum derivative of the excess tensor, so the mixed third
    derivative carries a truncation error ``O(hx^2 + hP^2)`` and a round-off
    error ``O(eps / (hx^2 hP))``.
    """

    def __init__(self, fn: Callable, nstates: int = 2, nbath: int = 1,
                 hx: float = 1e-3, hR: float = 1e-5, hP: float = 1e-3) -> None:
        self.fn, self.nstates, self.nbath = fn, nstates, nbath
        self.hx, self.hR, self.hP = hx, hR, hP

    @staticmethod
    def _shift(v, i, d):
        out = np.array(v, dtype=DTYPE, copy=True)
        out[..., i] += d
        return out

    def value(self, x, R, P):
        out = np.asarray(self.fn(x, R, P))
        return out.astype(result_dtype(out))

    def _grad(self, var, x, R, P, dim, h):
        cols = []
        for i in range(dim):
            def at(d, i=i):
                if var == "x":
                    return self.fn(self._shift(x, i, d), R, P)
                if var == "R":
                    return self.fn(x, self._shift(R, i, d), P)
                return self.fn(x, R, self._shift(P, i, d))
            cols.append((np.asarray(at(-2 * h)) - 8 * np.asarray(at(-h))
                         + 8 * np.asarray(at(h)) - np.asarray(at(2 * h)))
                        / (12.0 * h))
        return np.stack(cols, axis=-1)

    def grad_x(self, x, R, P):
        return self._grad("x", x, R, P, 2 * self.nstates, self.hx)

    def grad_R(self, x, R, P):
        return self._grad("R", x, R, P, self.nbath, self.hR)

    def grad_P(self, x, R, P):
        return self._grad("P", x, R, P, self.nbath, self.hP)

    def excess_tensor(self, x, R, P):
        n, hx, hP = self.nstates, self.hx, self.hP

        def d2(i, j, xx, PP):
            """Central second mapping derivative d^2 f / dx_i dx_j."""
            if i == j:
                return (np.asarray(self.fn(self._shift(xx, i, hx), R, PP))
                        - 2.0 * np.asarray(self.fn(xx, R, PP))
                        + np.asarray(self.fn(self._shift(xx, i, -hx), R, PP))) / hx**2
            pp = self._shift(self._shift(xx, i, hx), j, hx)
            pm = self._shift(self._shift(xx, i, hx), j, -hx)
            mp = self._shift(self._shift(xx, i, -hx), j, hx)
            mm = self._shift(self._shift(xx, i, -hx), j, -hx)
            return (np.asarray(self.fn(pp, R, PP)) - np.asarray(self.fn(pm, R, PP))
                    - np.asarray(self.fn(mp, R, PP)) + np.asarray(self.fn(mm, R, PP))
                    ) / (4.0 * hx**2)

        v0 = np.asarray(self.value(x, R, P))
        shape = v0.shape
        out = np.empty(shape + (self.nbath, n, n), dtype=result_dtype(v0))
        for k in range(self.nbath):
            Pp, Pm = self._shift(P, k, hP), self._shift(P, k, -hP)
            for a in range(n):
                for b in range(n):
                    dplus = d2(a, b, x, Pp) + d2(n + a, n + b, x, Pp)
                    dminus = d2(a, b, x, Pm) + d2(n + a, n + b, x, Pm)
                    out[..., k, a, b] = (dplus - dminus) / (2.0 * hP)
        return out


def projected_density_expr(rho_W, nstates: int = 2, nbath: int = 1,
                           hbar: float = 1.0, syms=None):
    r"""SymPy expression for :math:`\rho_m^{\mathcal P}=\sum g_{\lambda\lambda'}
    \rho_W^{\lambda'\lambda}`.

    ``rho_W[lam][lamp]`` are SymPy expressions in ``(R, P)``.  The result is the
    exact projected mapping density used as the reference field throughout the
    operator gates and by ``scripts/01_polynomial_closure.py``.
    """
    r, p, R, P = syms if syms is not None else symbols_for(nstates, nbath)
    phi = sp.exp(-sum(r[m]**2 + p[m]**2 for m in range(nstates)) / hbar) \
        / (sp.pi * hbar) ** nstates
    total = 0
    for a in range(nstates):
        for b in range(nstates):
            u = r[a] * r[b] + p[a] * p[b]
            L = r[a] * p[b] - r[b] * p[a]
            delta = 1 if a == b else 0
            pi_ab = (2 / hbar) * (u - sp.I * L - hbar * delta / 2)
            total += pi_ab * rho_W[b][a]
    return sp.expand(phi * total)
