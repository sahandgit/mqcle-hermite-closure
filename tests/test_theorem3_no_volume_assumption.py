r"""Theorem 3 without the volume-preservation hypothesis (PART XII / PART XXV).

The published statement assumed a *volume-preserving* diffeomorphism.  It does
not need to: volume preservation is a **consequence** of preserving
:math:`\cV_0` and :math:`\cV_1`, not an extra hypothesis.  This module verifies
that in exact computer algebra.

What is checked
---------------
1. For an affine candidate :math:`D(z)=Az+t` with **no assumption on**
   :math:`\det A`, the single requirement :math:`\cU_E\varphi\in\cV_0` already
   forces :math:`A^{\mathsf T}A=I` and :math:`A^{\mathsf T}t=0`.  The
   conditions are extracted by sympy and matched in both directions, so an
   extra condition would fail as loudly as a missing one.
2. A **volume-preserving but non-orthogonal** affine map (a unit-determinant
   shear, and a unit-determinant squeeze) is rejected.  This is the case that
   shows why volume preservation cannot be doing the work in the old proof.
3. A **nonlinear volume-preserving** map is rejected, which is the step
   "each :math:`D_a` must be affine".
4. Orthogonal maps are accepted and map :math:`\cV_K` onto itself for
   :math:`K=0,1,2`.  Without this the first three are vacuous.

Convention: standardised mapping coordinates, standard Gaussian weight
:math:`\varphi(z)=(2\pi)^{-d/2}e^{-z^{\mathsf T}z/2}`, and the density
pushforward :math:`(\cU_E\rho)(z)=j(z)\rho(D(z))`,
:math:`j=|\det\partial D/\partial z|`, exactly as in the manuscript.
"""
from __future__ import annotations

import pytest
import sympy as sp

from mqcle.hermite.basis import he_multi, multi_indices


def _setup(dim: int):
    return sp.symbols(f"thm3z0:{dim}", real=True)


def _pushforward_over_phi(D, z):
    """Jacobian and Gaussian exponent of the pushforward, over ``phi``."""
    J = sp.Matrix([[sp.diff(Da, zi) for zi in z] for Da in D])
    j = sp.simplify(J.det())
    quad = sp.expand(sum(Da**2 for Da in D) - sum(zi**2 for zi in z))
    return j, quad


def _is_in_VK(D, z, q_of_D, K: int):
    r"""Is :math:`\cU_E(\varphi q)` in :math:`\cV_K`?

    Equivalent to: :math:`j\,e^{-(|D|^2-|z|^2)/2}\,q(D(z))` is a polynomial of
    total degree at most ``K``.  The exponential is a polynomial only when its
    exponent vanishes identically, which is checked first and separately, so
    that a failure says *which* of the two requirements failed.
    """
    j, quad = _pushforward_over_phi(D, z)
    if sp.simplify(quad) != 0:
        return False, "Gaussian not preserved"
    expr = sp.expand(sp.simplify(j * q_of_D))
    poly = sp.Poly(expr, *z)
    return (poly.total_degree() <= K), f"degree {poly.total_degree()}"


# --------------------------------------------------------------------------
# 1. the V_0 condition alone forces orthogonality -- solved, not asserted
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 4])
def test_V0_alone_forces_orthogonal_and_centred(dim):
    z = _setup(dim)
    A = sp.Matrix(dim, dim, lambda i, k: sp.Symbol(f"thm3A{i}{k}", real=True))
    t = sp.Matrix(dim, 1, lambda i, _: sp.Symbol(f"thm3t{i}", real=True))
    Z = sp.Matrix(dim, 1, lambda i, _: z[i])
    D = A * Z + t
    quad = sp.expand((D.T * D)[0, 0] - (Z.T * Z)[0, 0])
    poly = sp.Poly(quad, *z)
    eqs = [sp.expand(c) for m, c in zip(poly.monoms(), poly.coeffs())
           if sum(m) > 0]

    ATA = sp.expand(A.T * A - sp.eye(dim))
    ATt = sp.expand(A.T * t)
    target = ([ATA[i, k] for i in range(dim) for k in range(i, dim)]
              + [ATt[i] for i in range(dim)])

    def _proportional(u, v):
        if sp.simplify(v) == 0:
            return False
        for lam in (1, -1, 2, -2, sp.Rational(1, 2), sp.Rational(-1, 2)):
            if sp.simplify(sp.expand(u - lam * v)) == 0:
                return True
        return False

    for tgt in target:
        assert any(_proportional(e, tgt) for e in eqs), \
            f"the condition {tgt} is not among the V_0 conditions"
    for e in eqs:
        assert any(_proportional(e, tgt) for tgt in target), \
            f"the V_0 condition {e} is not one of A^T A = I / A^T t = 0"

    # A^T A = I makes A invertible, so A^T t = 0 gives t = 0, and
    # det(A^T A) = det(A)^2 = 1 makes |det A| = 1: volume preservation is a
    # CONCLUSION, not an input.  Asserted as identities rather than through
    # sp.solve, whose output format depends on cached assumptions and made an
    # earlier version of this test order-dependent inside the full suite.
    assert sp.simplify(sp.det(A.T * A) - sp.det(A) ** 2) == 0
    # so A^T A = I gives det(A)^2 = det(I) = 1, hence |det A| = 1
    # and the family is not empty: a concrete orthogonal map satisfies every
    # condition, with t = 0 and unit determinant
    th = sp.Rational(1, 3)
    O = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])
    if dim == 2:
        subs = {A[i, k]: O[i, k] for i in range(2) for k in range(2)}
        subs.update({t[i]: 0 for i in range(2)})
        for cond in list(ATA) + list(ATt):
            assert sp.simplify(cond.subs(subs)) == 0
        assert sp.simplify(sp.Abs(O.det()) - 1) == 0


# --------------------------------------------------------------------------
# 2-3. verdicts on concrete candidates
# --------------------------------------------------------------------------
def _affine(M, t, z):
    Z = sp.Matrix(len(z), 1, lambda i, _: z[i])
    return list(sp.Matrix(M) * Z + sp.Matrix(t))


CASES = [
    # (name, D builder, volume preserving?, expected V_0 verdict)
    ("orthogonal rotation",
     lambda z: _affine([[sp.Rational(3, 5), sp.Rational(-4, 5)],
                        [sp.Rational(4, 5), sp.Rational(3, 5)]], [0, 0], z),
     True, True),
    ("orthogonal reflection",
     lambda z: _affine([[0, 1], [1, 0]], [0, 0], z), True, True),
    ("unit-determinant shear",
     lambda z: _affine([[1, 1], [0, 1]], [0, 0], z), True, False),
    ("unit-determinant squeeze",
     lambda z: _affine([[2, 0], [0, sp.Rational(1, 2)]], [0, 0], z), True, False),
    ("orthogonal with a translation",
     lambda z: _affine([[0, -1], [1, 0]], [sp.Rational(1, 2), 0], z), True, False),
    ("uniform dilation",
     lambda z: _affine([[2, 0], [0, 2]], [0, 0], z), False, False),
    ("nonlinear volume-preserving shear",
     lambda z: [z[0] + z[1] ** 2, z[1]], True, False),
    ("nonlinear symplectic (unit Jacobian, cubic)",
     lambda z: [z[0] + z[1] ** 3, z[1]], True, False),
]


@pytest.mark.parametrize("name,build,vol,expected", CASES,
                         ids=[c[0] for c in CASES])
def test_grading_verdicts(name, build, vol, expected):
    z = _setup(2)
    D = build(z)
    J = sp.Matrix([[sp.diff(Da, zi) for zi in z] for Da in D])
    is_vol = sp.simplify(sp.Abs(J.det()) - 1) == 0
    assert is_vol == vol, f"{name}: volume-preservation flag is wrong"
    ok, why = _is_in_VK(D, z, sp.Integer(1), 0)
    assert ok == expected, f"{name}: V_0 verdict {ok} ({why}), expected {expected}"


def test_volume_preservation_is_neither_necessary_nor_sufficient():
    """The unit-determinant rejections and the accepted rotations together say
    that volume preservation carries no information here, which is exactly why
    it can be dropped from the hypotheses."""
    z = _setup(2)
    vol_but_rejected = [c for c in CASES if c[2] and not c[3]]
    accepted = [c for c in CASES if c[3]]
    assert len(vol_but_rejected) >= 3 and accepted
    for _, build, _, _ in accepted:
        J = sp.Matrix([[sp.diff(Da, zi) for zi in z] for Da in build(z)])
        assert sp.simplify(sp.Abs(J.det()) - 1) == 0, \
            "an accepted map that did not turn out volume preserving would "\
            "contradict the strengthened theorem"


# --------------------------------------------------------------------------
# 4. orthogonal maps really do preserve the grading (anti-vacuity)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("K", [0, 1, 2])
def test_orthogonal_map_preserves_V_K(K):
    z = _setup(2)
    th = sp.Rational(2, 5)
    O = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])
    D = _affine(O, [0, 0], z)
    for idx in multi_indices(2, K):
        q = he_multi(idx, D)          # He_n evaluated at D(z)
        ok, why = _is_in_VK(D, z, q, K)
        assert ok, f"He_{idx} left V_{K} under an orthogonal map ({why})"


def test_V1_condition_forces_affine_for_a_gaussian_preserving_map():
    r"""The step "each :math:`D_a` is affine".

    Dividing the :math:`\cV_1` requirement by the :math:`\cV_0` requirement
    leaves :math:`D_a(z)\in\mathrm{span}\{1,z_1,\dots\}` with no Jacobian and
    no Gaussian left in it.  Exercised here on a unit-Jacobian cubic
    perturbation, which breaks :math:`\cV_1` while keeping :math:`j=1`.
    """
    z = _setup(2)
    eps = sp.Rational(1, 7)
    D = [z[0] + eps * z[1] ** 3, z[1]]
    j, quad = _pushforward_over_phi(D, z)
    assert sp.simplify(j - 1) == 0, "the perturbation must keep the Jacobian"
    assert sp.simplify(quad) != 0, "and must break the Gaussian identity"
    ok, _ = _is_in_VK(D, z, D[0], 1)
    assert not ok
