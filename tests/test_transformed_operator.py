"""M10: the symmetry that licenses the compact form of the transformed
second-derivative operator, verified symbolically rather than assumed."""
from __future__ import annotations

import itertools

import sympy as sp


def _G(J, N, a, b, lam, lamp):
    r"""$G^{ab}_{\lambda\lambda'}=C^{ij}_{\lambda\lambda'}J_{ai}J_{bj}$ with
    $C^{ij}_{\lambda\lambda'}
    =\delta_{i\lambda}\delta_{j\lambda'}+\delta_{i,N+\lambda}\delta_{j,N+\lambda'}$."""
    return J[a, lam] * J[b, lamp] + J[a, N + lam] * J[b, N + lamp]


def test_G_is_not_symmetric_in_ab_at_fixed_state_pair():
    """The naive step -- replacing G[(d_a l) d_b + (d_b l) d_a] by 2G(d_a l) d_b
    -- would need this, and it is false."""
    N = 2
    J = sp.Matrix(2 * N, 2 * N,
                  lambda a, i: sp.Symbol(f"J{a}{i}", real=True))
    found = False
    for a, b, lam, lamp in itertools.product(range(2 * N), range(2 * N),
                                             range(N), range(N)):
        if sp.simplify(_G(J, N, a, b, lam, lamp)
                       - _G(J, N, b, a, lam, lamp)) != 0:
            found = True
            break
    assert found, "G would have to be symmetric for the naive step to be legal"


def test_G_is_symmetric_under_the_simultaneous_swap():
    r"""$G^{ab}_{\lambda\lambda'}=G^{ba}_{\lambda'\lambda}$ identically."""
    N = 2
    J = sp.Matrix(2 * N, 2 * N,
                  lambda a, i: sp.Symbol(f"J{a}{i}", real=True))
    for a, b, lam, lamp in itertools.product(range(2 * N), range(2 * N),
                                             range(N), range(N)):
        assert sp.simplify(_G(J, N, a, b, lam, lamp)
                           - _G(J, N, b, a, lamp, lam)) == 0


def test_the_contraction_that_actually_appears_is_symmetric():
    r"""$M^{ab}=\sum_{\lambda\lambda'}\nabla_Rh^{\lambda\lambda'}
    G^{ab}_{\lambda\lambda'}$ is symmetric in $a\leftrightarrow b$ *because*
    $h$ is symmetric -- this is what licenses the compact form in Sec. S8."""
    for N in (2, 3):
        J = sp.Matrix(2 * N, 2 * N,
                      lambda a, i: sp.Symbol(f"J{a}{i}", real=True))
        dh = sp.Matrix(N, N, lambda i, j:
                       sp.Symbol(f"dh{min(i,j)}{max(i,j)}", real=True))
        def M(a, b):
            return sp.expand(sum(dh[lam, lamp] * _G(J, N, a, b, lam, lamp)
                                 for lam in range(N) for lamp in range(N)))
        for a, b in itertools.combinations(range(2 * N), 2):
            assert sp.simplify(M(a, b) - M(b, a)) == 0


def test_the_contraction_is_not_symmetric_for_a_nonsymmetric_matrix():
    """The converse: symmetry of h is doing the work, not the structure of G."""
    N = 2
    J = sp.Matrix(2 * N, 2 * N,
                  lambda a, i: sp.Symbol(f"J{a}{i}", real=True))
    dh = sp.Matrix(N, N, lambda i, j: sp.Symbol(f"g{i}{j}", real=True))
    def M(a, b):
        return sp.expand(sum(dh[lam, lamp] * _G(J, N, a, b, lam, lamp)
                             for lam in range(N) for lamp in range(N)))
    assert sp.simplify(M(0, 1) - M(1, 0)) != 0
