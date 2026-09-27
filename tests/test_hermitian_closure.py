r"""Corollary 1: the closure survives a complex Hermitian subsystem matrix.

The manuscript's Theorem 1 was stated for real symmetric :math:`h`.  Corollary
1 says the proof only ever uses two properties of the real symmetric
:math:`2N\times2N` form :math:`\mathbb H` that carries the quadratic part of
:math:`H_m` -- that it is symmetric, and that it commutes with the standard
complex structure :math:`\Jz` -- and that those two hold exactly for Hermitian
:math:`h`.  These tests check the algebra the corollary rests on and then check
its conclusion on the generator.

The expensive end-to-end audit (physical-frame residual, QCLE correspondence,
:math:`N=3`) lives in ``scripts/05_extra_cases.py --case Herm`` and is archived;
what is here is the part that is cheap enough to run on every commit.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from mqcle.validation import (apply_excess, apply_full, apply_pbme,
                              build_symbolic_setup, hermite_defect,
                              mapping_polynomial_degrees, real_form_2n)


def _J(n: int) -> sp.Matrix:
    """The standard complex structure on :math:`\\mathbb R^{2N}`."""
    return sp.Matrix(sp.BlockMatrix([[sp.zeros(n), sp.eye(n)],
                                     [-sp.eye(n), sp.zeros(n)]]))


@pytest.mark.parametrize("n", [2, 3])
def test_real_form_of_real_symmetric_h_is_block_diagonal(n):
    """The real case must be untouched: nothing about it may change."""
    h = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"a{min(i, j)}{max(i, j)}",
                                               real=True))
    big = real_form_2n(h)
    assert big == sp.Matrix(sp.BlockMatrix([[h, sp.zeros(n)],
                                            [sp.zeros(n), h]]))


@pytest.mark.parametrize("n", [2, 3])
def test_real_form_of_hermitian_h_is_symmetric_and_commutes_with_J(n):
    """The two properties Corollary 1's proof uses, and nothing more."""
    A = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"a{min(i, j)}{max(i, j)}",
                                               real=True))

    def bij(i, j):
        if i == j:
            return sp.Integer(0)
        g = sp.Symbol(f"g{min(i, j)}{max(i, j)}", real=True)
        return g if i < j else -g

    B = sp.Matrix(n, n, bij)
    h = A + sp.I * B
    assert sp.simplify(h - h.conjugate().T) == sp.zeros(n, n), "h not Hermitian"

    big = real_form_2n(h)
    assert sp.simplify(big - big.T) == sp.zeros(2 * n, 2 * n)
    J = _J(n)
    assert sp.simplify(big * J - J * big) == sp.zeros(2 * n, 2 * n)


def test_non_hermitian_h_breaks_one_of_the_two_properties():
    """The converse half of the statement, as an explicit counter-example.

    A symmetric *complex* matrix is not Hermitian; its real form still commutes
    with J but is no longer symmetric, so the cancellation has no reason to
    hold.  Without this the corollary would only be shown to be sufficient.
    """
    n = 2
    a, b = sp.symbols("a b", real=True)
    h = sp.Matrix([[a, sp.I * b], [sp.I * b, a]])       # symmetric, not Hermitian
    big = real_form_2n(h)
    assert sp.simplify(big - big.T) != sp.zeros(2 * n, 2 * n)


def _realise(expr, s):
    """Replace each undefined function of R by a real symbol.

    ``build_symbolic_setup`` deliberately leaves ``h00(R)`` an assumption-free
    undefined function -- declaring it real would change the function object's
    identity and break the substitutions ``mqcle.leakage`` keys on.  So a test
    that needs SymPy to *know* the entries are real has to say so here.
    """
    sub = {}
    for e in expr.atoms(sp.Function):
        name = getattr(e.func, "__name__", "")
        if name and name[0] in "hgV":
            sub[e] = sp.Symbol(name, real=True)
    return expr.subs(sub)


def test_setup_builds_a_genuinely_complex_hermitian_h():
    s = build_symbolic_setup(2, 1, "traceless", hermitian=True)
    h = sp.Matrix(2, 2, lambda i, j: _realise(s.h[i, j], s))
    assert sp.simplify(h - h.conjugate().T) == sp.zeros(2, 2)
    assert any(sp.simplify(sp.im(h[i, j])) != 0
               for i in range(2) for j in range(2)), "h is secretly real"
    assert sp.simplify(sp.im(_realise(sp.expand(s.H_m), s))) == 0, \
        "H_m must stay real"
    assert sp.simplify(s.bigH - s.bigH.T) == sp.zeros(4, 4)


def test_real_setup_is_unchanged_by_the_generalisation():
    """Regression: routing the excess operator through the 2N form must
    reproduce the block-diagonal contraction it replaced, exactly."""
    s = build_symbolic_setup(2, 1, "traceless")
    got = sp.expand(apply_excess(s, s.rho_m))
    want = 0
    for k in range(s.nbath):
        fk = sp.diff(s.rho_m, s.P[k])
        for a in range(s.nstates):
            for b in range(s.nstates):
                want += sp.diff(s.h_exc[a, b], s.R[k]) * (
                    sp.diff(fk, s.r[b], s.r[a]) + sp.diff(fk, s.p[b], s.p[a]))
    want = sp.expand(-(s.hbar / 8) * want)
    assert sp.simplify(got - want) == 0


@pytest.mark.parametrize("convention", ["traceless", "full"])
def test_closure_holds_for_complex_hermitian_h(convention):
    """The conclusion of Corollary 1, on the generator itself."""
    s = build_symbolic_setup(2, 1, convention, hermitian=True)
    full = sp.expand(apply_full(s, s.rho_m))
    assert mapping_polynomial_degrees(s, full) == [0, 2]
    assert sp.simplify(hermite_defect(s, full, 2)) == 0


def test_pbme_alone_still_leaves_V2_for_complex_hermitian_h():
    """The cancellation must be doing work, not holding vacuously."""
    s = build_symbolic_setup(2, 1, "traceless", hermitian=True)
    pbme = sp.expand(apply_pbme(s, s.rho_m))
    assert 4 in mapping_polynomial_degrees(s, pbme)
    assert sp.simplify(hermite_defect(s, pbme, 2)) != 0


def _J_of(s):
    n = s.nstates
    return sp.Matrix(sp.BlockMatrix([[sp.zeros(n), sp.eye(n)],
                                     [-sp.eye(n), sp.zeros(n)]]))


@pytest.mark.parametrize("hermitian", [False, True])
def test_J_transpose_H_J_equals_H(hermitian):
    r"""The identity that makes the second Moyal order reproduce :math:`\mathbb H`.

    :math:`J^{\mathsf T}\mathbb HJ=-J\mathbb HJ=\mathbb H` needs both that
    :math:`\mathbb H` commutes with :math:`J` and that :math:`J^2=-1`.  Without
    it the second-order term of the cosine series would carry a *rotated*
    matrix and could not cancel the force channel, which carries
    :math:`\mathbb H` itself.
    """
    s = build_symbolic_setup(2, 1, "traceless", hermitian=hermitian)
    J, H = _J_of(s), s.bigH
    assert sp.simplify(J.T * H * J - H) == sp.zeros(4, 4)


@pytest.mark.parametrize("hermitian", [False, True])
def test_second_moyal_order_is_the_excess_operator(hermitian):
    r"""Derive, rather than assert, the excess operator for Hermitian ``h``.

    The symmetrised (cosine) series is
    :math:`\cos(\hbar\Lambda_x/2)=1-\tfrac{\hbar^2}{8}\Lambda_x^2+\dots` with
    :math:`\Lambda_x=\overleftarrow\partial_iJ_{ij}\overrightarrow\partial_j`.
    Acting on the quadratic mapping part of :math:`H_m`, the second order gives

    .. math:: \Lambda_x^2(A,B)=\hbar^{-1}\mathbb H_{jl}
              \partial_j\partial_lB,

    so the term is exactly :math:`-\tfrac{\hbar}{8}\mathbb H_{jl}
    \partial_j\partial_l`.  For real symmetric ``h`` this is the familiar
    :math:`\nabla_R\mathsf h^{\lambda\lambda'}D_{\lambda\lambda'}` form; for
    complex Hermitian ``h`` it is not, and the extra antisymmetric piece is
    what makes the cancellation still work.
    """
    s = build_symbolic_setup(2, 1, "traceless", hermitian=hermitian)
    x, n2, J = s.x, 2 * s.nstates, _J_of(s)
    A = sum(s.bigH[i, j] * x[i] * x[j]
            for i in range(n2) for j in range(n2)) / (2 * s.hbar)
    B = sp.Function("g")(*x)
    lam2 = sum(J[i, j] * J[k, l] * sp.diff(A, x[i], x[k]) * sp.diff(B, x[j], x[l])
               for i in range(n2) for j in range(n2)
               for k in range(n2) for l in range(n2))
    want = sum(s.bigH[j, l] * sp.diff(B, x[j], x[l])
               for j in range(n2) for l in range(n2)) / s.hbar
    assert sp.simplify(lam2 - want) == 0


def test_complex_h_contributes_a_term_absent_from_the_real_form():
    r"""Eq. (11) as printed for real ``h`` is *not* valid for complex ``h``.

    Substituting a complex Hermitian ``h`` into
    :math:`\sum\nabla_R\mathsf h^{\lambda\lambda'}D_{\lambda\lambda'}` keeps
    only :math:`A`, because :math:`D_{\lambda\lambda'}` is symmetric under
    :math:`\lambda\leftrightarrow\lambda'` and :math:`B` is antisymmetric.  The
    real form supplies the missing mixed-derivative piece, and this test pins
    down that it is genuinely nonzero -- otherwise Corollary 1 would be about a
    distinction without a difference.
    """
    s = build_symbolic_setup(2, 1, "traceless", hermitian=True)
    x, n2 = s.x, 2 * s.nstates
    B = sp.Function("g")(*x)
    full = sum(s.bigH_exc[i, j] * sp.diff(B, x[i], x[j])
               for i in range(n2) for j in range(n2))
    # the naive Eq. (11) reading, which sees only the real part
    naive = sum(sp.re(s.h_exc[a, b]) * (sp.diff(B, s.r[a], s.r[b])
                                        + sp.diff(B, s.p[a], s.p[b]))
                for a in range(s.nstates) for b in range(s.nstates))
    diff = sp.simplify(sp.expand(full - naive))
    assert diff != 0, "the complex part would be silently discarded"


# ---------------------------------------------------------------------------
# the numerical mapping code, which used to assume h real symmetric silently
# ---------------------------------------------------------------------------
class _ComplexHermitianModel:
    """``base`` conjugated by a genuinely complex unitary: ``h -> U h U^dagger``."""

    nstates, nbath, name = 2, 1, "complex_hermitian"

    def __init__(self, base, U):
        self.base, self.U, self.mass = base, np.asarray(U), base.mass

    def h(self, R):
        return np.conj(self.U).T @ self.base.h(R) @ self.U

    def dh_dR(self, R):
        return np.conj(self.U).T @ self.base.dh_dR(R) @ self.U

    def bath_potential(self, R):
        return self.base.bath_potential(R)

    def dbath_dR(self, R):
        return self.base.dbath_dR(R)


def _complex_model():
    from mqcle.models import build_model
    base = build_model({"name": "tully_dual_avoided_crossing"}, mass=2000.0)
    th, ph = 0.6, 0.9
    U = np.array([[np.cos(th), -np.sin(th) * np.exp(1j * ph)],
                  [np.sin(th) * np.exp(-1j * ph), np.cos(th)]])
    assert np.allclose(np.conj(U).T @ U, np.eye(2))
    m = _ComplexHermitianModel(base, U)
    assert np.abs(np.imag(m.h(np.zeros((1, 1))))).max() > 1e-3, "U was not complex"
    return m


def test_drift_matrix_is_antisymmetric_for_complex_hermitian_h():
    """Step (i) of Corollary 1, in the numerical code rather than on paper.

    The old builder wrote ``K = [[0, h], [-h, 0]]``, which is antisymmetric only
    for real symmetric ``h``.  For complex Hermitian ``h`` it is not even real.
    """
    from mqcle.mapping import drift_matrix
    K = drift_matrix(np.linspace(-4, 4, 17)[:, None], _complex_model())
    assert np.isrealobj(K)
    assert np.max(np.abs(K + np.swapaxes(K, -1, -2))) < 1e-14


def test_mapping_flow_is_orthogonal_for_complex_hermitian_h():
    """``exp(K t)`` must stay in SO(2N), which is what preserves the Gaussian."""
    from scipy.linalg import expm
    from mqcle.mapping import drift_matrix
    K = drift_matrix(np.array([[0.4]]), _complex_model())[0]
    for t in (0.5, 3.0):
        E = expm(K * t)
        assert np.max(np.abs(E.T @ E - np.eye(4))) < 1e-12
        assert abs(np.linalg.det(E) - 1.0) < 1e-12


def test_grad_x_matches_finite_differences_for_complex_hermitian_h():
    """The analytic gradient must be the gradient of the Hamiltonian actually
    used -- the check that would have caught a dropped imaginary part."""
    from mqcle.mapping import grad_mapping_hamiltonian_x, mapping_hamiltonian
    m = _complex_model()
    rng = np.random.default_rng(3)
    x = rng.normal(size=4)
    R, P = np.array([0.4]), np.array([12.0])
    g = grad_mapping_hamiltonian_x(x, R, m)
    eps = 1e-6
    fd = np.empty(4)
    for i in range(4):
        d = np.zeros(4)
        d[i] = eps
        fd[i] = (mapping_hamiltonian(x + d, R, P, m)
                 - mapping_hamiltonian(x - d, R, P, m)) / (2 * eps)
    assert np.max(np.abs(g - fd)) < 1e-7
    H = np.asarray(mapping_hamiltonian(x, R, P, m))
    assert np.isrealobj(H), "the mapping Hamiltonian of a Hermitian h is real"


def test_non_hermitian_h_is_rejected_rather_than_silently_accepted():
    from mqcle.mapping import subsystem_split
    from mqcle.models import build_model
    base = build_model({"name": "tully_dual_avoided_crossing"}, mass=2000.0)

    class Bad(_ComplexHermitianModel):
        def h(self, R):
            out = super().h(R).copy()
            out[..., 0, 1] += 0.05j          # breaks Hermiticity
            return out

    th, ph = 0.6, 0.9
    U = np.array([[np.cos(th), -np.sin(th) * np.exp(1j * ph)],
                  [np.sin(th) * np.exp(-1j * ph), np.cos(th)]])
    with pytest.raises(ValueError, match="not Hermitian"):
        subsystem_split(Bad(base, U), np.zeros((3, 1)))
