"""Gate H prerequisites: the Hermite basis and the true projector."""
import numpy as np
import pytest
import sympy as sp

from mqcle.hermite.basis import (he_multi, he_poly, multi_factorial,
                                 multi_indices, standardise, unstandardise)
from mqcle.hermite.projection import (hermite_coefficients_numeric,
                                      hermite_coefficients_symbolic,
                                      hermite_orders, hermite_residual_symbolic,
                                      polynomial_degrees,
                                      project_hermite_symbolic, top_degree_part)

z = sp.symbols("z0:4", real=True)


def test_univariate_recurrence():
    zz = z[0]
    assert he_poly(0, zz) == 1
    assert he_poly(1, zz) == zz
    assert sp.expand(he_poly(2, zz) - (zz**2 - 1)) == 0
    assert sp.expand(he_poly(3, zz) - (zz**3 - 3 * zz)) == 0
    assert sp.expand(he_poly(4, zz) - (zz**4 - 6 * zz**2 + 3)) == 0


def test_orthogonality_under_the_stated_inner_product():
    """<He_m, He_n>_phi = n! delta_{mn} for the standard Gaussian."""
    zz = z[0]
    phi = sp.exp(-zz**2 / 2) / sp.sqrt(2 * sp.pi)
    for m in range(4):
        for n in range(4):
            val = sp.integrate(phi * he_poly(m, zz) * he_poly(n, zz),
                               (zz, -sp.oo, sp.oo))
            assert sp.simplify(val - (sp.factorial(n) if m == n else 0)) == 0


def test_the_headline_counterexample():
    """z^4: P_2 gives 6 z^2 - 3, and (I - P_2) gives He_4, not z^4."""
    zz = z[0]
    q = zz**4
    assert sp.expand(project_hermite_symbolic(q, [zz], 2) - (6 * zz**2 - 3)) == 0
    res = hermite_residual_symbolic(q, [zz], 2)
    assert sp.expand(res - (zz**4 - 6 * zz**2 + 3)) == 0
    # ... and the naive "top degree" answer is a different function
    assert sp.expand(top_degree_part(q, [zz], 2) - zz**4) == 0
    assert sp.expand(res - top_degree_part(q, [zz], 2)) != 0


def test_multivariate_mixed_monomial():
    """z0^2 z1^2 = (He_2 + 1)(He_2 + 1) expanded in the product basis."""
    q = z[0]**2 * z[1]**2
    c = hermite_coefficients_symbolic(q, z)
    assert c[(2, 2, 0, 0)] == 1
    assert c[(2, 0, 0, 0)] == 1
    assert c[(0, 2, 0, 0)] == 1
    assert c[(0, 0, 0, 0)] == 1
    rebuilt = sum(v * he_multi(k, z) for k, v in c.items())
    assert sp.expand(rebuilt - q) == 0


def test_reconstruction_is_exact_for_random_polynomials(rng):
    for _ in range(5):
        q = sum(int(rng.integers(-3, 4)) * sp.prod(
            [v**int(e) for v, e in zip(z, idx)])
            for idx in multi_indices(4, 3))
        c = hermite_coefficients_symbolic(q, z)
        rebuilt = sum(v * he_multi(k, z) for k, v in c.items())
        assert sp.expand(rebuilt - sp.expand(q)) == 0


def test_projection_is_idempotent_and_orthogonal():
    q = (z[0]**4 + z[0]**2 * z[1]**2 + z[2]**3 * z[3] + 5)
    p2 = project_hermite_symbolic(q, z, 2)
    assert sp.expand(project_hermite_symbolic(p2, z, 2) - p2) == 0
    res = hermite_residual_symbolic(q, z, 2)
    assert sp.expand(q - (p2 + res)) == 0
    # residual has no Hermite component of order <= 2
    assert all(sum(k) > 2 for k, v in
               hermite_coefficients_symbolic(res, z).items() if v != 0)


def test_polynomial_degree_bounds_hermite_order():
    """deg_poly <= K implies Hermite order <= K (triangular change of basis)."""
    q = 3 * z[0]**2 * z[1] + z[2] - 7
    assert max(polynomial_degrees(q, z)) == 3
    assert max(hermite_orders(q, z)) <= 3


def test_degree_and_order_sets_can_differ():
    """He_2 = z^2 - 1 has monomial degrees {0, 2} but Hermite orders {2}."""
    q = z[0]**2 - 1
    assert polynomial_degrees(q, z) == [0, 2]
    assert hermite_orders(q, z) == [2]


def test_numeric_projection_agrees_with_symbolic():
    q = z[0]**4 + 2 * z[0]**2 * z[1]**2 - z[1] + 3
    sym = hermite_coefficients_symbolic(q, z[:2])
    fn = sp.lambdify(z[:2], q, "numpy")
    num = hermite_coefficients_numeric(lambda Z: fn(Z[:, 0], Z[:, 1]), 2, 4, npts=10)
    for k, v in sym.items():
        assert abs(complex(v) - num[tuple(k)]) < 1e-10
    for k, v in num.items():
        if k not in sym:
            assert abs(v) < 1e-10


def test_standardisation_roundtrip():
    x = sp.symbols("x0:4", real=True)
    hbar = sp.Rational(1)
    expr = x[0]**2 + x[1] * x[2]
    back = unstandardise(standardise(expr, x, z, hbar), x, z, hbar)
    assert sp.expand(back - expr) == 0


def test_multi_index_helpers():
    assert multi_indices(2, 2) == [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0)]
    assert multi_indices(2, 2, exact=True) == [(0, 2), (1, 1), (2, 0)]
    assert multi_factorial((3, 2)) == 12
