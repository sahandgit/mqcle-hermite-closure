r"""Independent construction of the order-raising block, spec item 20.

The sparse recurrences of :func:`hermite_T_matrices` *encode* the no-raising
selection rule, so any test built on them assumes what Theorem 1 asserts.  The
test here rebuilds the same matrix elements a second time, from the
unconjugated density-side operator and nothing else:

.. math::

    \bigl(\mathcal T_{ij}\bigr)_{\bm m\bm n}
    =\frac{1}{\bm m!}\int\!\mathrm dz\,\varphi(z)\,He_{\bm m}(z)\,
     \bigl(z_i\partial_{z_j}+z_j\partial_{z_i}
           -\partial_{z_i}\partial_{z_j}\bigr)He_{\bm n}(z),

with the derivatives taken symbolically and the integral by tensor
Gauss--Hermite quadrature.  It shares no code path with the recurrence, so the
vanishing of the :math:`K=2\to K=4` block it produces is independent evidence
rather than a restatement.

Quadrature order: the integrand's degree *in a single coordinate* --- not its
total degree --- fixes ``nq``.  Here it is at most 4 (from :math:`He_{\bm m}`,
:math:`|\bm m|\le4`) plus 2 (from the operator) = 6, so a rule exact through
degree :math:`2n_q-1\ge6` is needed and ``nq = 4`` is the minimum.  ``nq = 5``
is used, one point clear of the boundary.

Measured when this test was written (all 6160 raising elements, N=2):
quadrature construction max ``9.715e-16``; recurrence construction exactly
``0.0``; the two agree on the non-raising blocks to ``5.329e-15``.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from mqcle.hermite.basis import he_multi, he_numeric
from mqcle.hermite.frame import hermite_metric
from mqcle.hermite.operator import (active_modes, hermite_T_matrices,
                                    rotation_basis_matrices)

NSTATES = 2
NQ = 5
ROUNDOFF = 1e-12


def _tensor_rule(nq, d):
    """Nodes, weights for the standardised Gaussian measure on R^d."""
    t, w = np.polynomial.hermite.hermgauss(nq)
    znodes, wstd = np.sqrt(2.0) * t, w / np.sqrt(np.pi)
    grids = np.meshgrid(*([znodes] * d), indexing="ij")
    Z = np.stack([g.ravel() for g in grids], axis=1)
    idx = np.stack([g.ravel() for g in
                    np.meshgrid(*([np.arange(nq)] * d), indexing="ij")], axis=1)
    Wq = np.ones(Z.shape[0])
    for a in range(d):
        Wq = Wq * wstd[idx[:, a]]
    return Z, Wq


def _he_table(modes, Z):
    """He_m evaluated on the nodes, one row per mode."""
    out = np.empty((len(modes), Z.shape[0]))
    for a, m in enumerate(modes):
        v = np.ones(Z.shape[0])
        for ax, ma in enumerate(m):
            v = v * he_numeric(int(ma), Z[:, ax])
        out[a] = v
    return out


def _quadrature_column(i, j, n, d, z, Z, Wq, HeM, fact):
    """One column of T^{ij}, built without touching the recurrence."""
    he_n = he_multi(n, z)
    op = sp.expand(z[i] * sp.diff(he_n, z[j])
                   + z[j] * sp.diff(he_n, z[i])
                   - sp.diff(he_n, z[i], z[j]))
    f = sp.lambdify(z, op, "numpy")
    val = np.asarray(f(*[Z[:, k] for k in range(d)]), dtype=float)
    if val.ndim == 0:
        val = np.full(Z.shape[0], float(val))
    return (HeM * (Wq * val)).sum(axis=1) / fact


@pytest.fixture(scope="module")
def k4():
    d = 2 * NSTATES
    modes = active_modes(NSTATES, 4, parity="even")
    Z, Wq = _tensor_rule(NQ, d)
    return d, modes, Z, Wq, _he_table(modes, Z), hermite_metric(modes)


def test_raising_block_vanishes_in_an_independent_construction(k4):
    """T^{ij} never carries |n| <= 2 into |m| = 4, built two different ways."""
    d, modes, Z, Wq, HeM, fact = k4
    z = sp.symbols(f"z0:{d}", real=True)
    T = hermite_T_matrices(modes)

    src = [b for b, m in enumerate(modes) if sum(m) <= 2]
    tgt = [a for a, m in enumerate(modes) if sum(m) == 4]
    assert (len(src), len(tgt)) == (11, 35), "K=4 even basis is not as expected"

    worst_quad = worst_code = worst_agree = 0.0
    for i in range(d):
        for j in range(d):
            for b in src:
                col = _quadrature_column(i, j, modes[b], d, z, Z, Wq, HeM, fact)
                for a in tgt:
                    worst_quad = max(worst_quad, abs(col[a]))
                    worst_code = max(worst_code, abs(float(T[i, j, a, b])))
                for a, m in enumerate(modes):
                    if sum(m) <= 2:
                        worst_agree = max(worst_agree,
                                          abs(col[a] - float(T[i, j, a, b])))

    assert worst_code == 0.0, (
        f"the recurrence itself carries a raising element, max {worst_code:.3e}")
    assert worst_quad < ROUNDOFF, (
        "the independent quadrature construction finds a nonzero K=2 -> K=4 "
        f"block, max {worst_quad:.3e}")
    assert worst_agree < ROUNDOFF, (
        f"the two constructions disagree off the raising block, {worst_agree:.3e}")


def test_rotation_block_is_exactly_order_preserving():
    """R carries no order-raising element at all; it is degree preserving."""
    modes = active_modes(NSTATES, 4, parity="even")
    Rb = rotation_basis_matrices(modes)
    src = [b for b, m in enumerate(modes) if sum(m) <= 2]
    tgt = [a for a, m in enumerate(modes) if sum(m) == 4]
    worst = max(float(np.max(np.abs(Rb[..., a, b]))) for b in src for a in tgt)
    assert worst == 0.0, f"rotation raises Hermite order, max {worst:.3e}"


@pytest.mark.parametrize("nq,exact", [(2, False), (3, True), (6, True)])
def test_quadrature_order_claim_for_the_K2_elements(nq, exact):
    """n_q = 3 is sufficient for the K=2 elements; n_q = 2 is not.

    Guards the statement the supplementary material makes about how many
    points the validation route needs.  Measured: n_q=2 errs by 4.0,
    n_q=3 by 1.78e-15, n_q=6 by 6.66e-15 over all 1936 elements.
    """
    d = 2 * NSTATES
    modes = active_modes(NSTATES, 2, parity="even")
    T = hermite_T_matrices(modes)
    fact = hermite_metric(modes)
    z = sp.symbols(f"z0:{d}", real=True)
    Z, Wq = _tensor_rule(nq, d)
    HeM = _he_table(modes, Z)

    worst = 0.0
    for i, j in ((0, 0), (0, 2), (1, 3), (2, 3)):
        for b, n in enumerate(modes):
            col = _quadrature_column(i, j, n, d, z, Z, Wq, HeM, fact)
            for a in range(len(modes)):
                worst = max(worst, abs(col[a] - float(T[i, j, a, b])))

    if exact:
        assert worst < ROUNDOFF, f"n_q = {nq} should be exact, got {worst:.3e}"
    else:
        assert worst > 1e-3, (
            f"n_q = {nq} is claimed insufficient but agreed to {worst:.3e}")
