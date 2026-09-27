"""Module 6: the excess coupling term, and the convention that must not be mixed.

``mqcle.types.TraceConvention`` names this file's
``test_mixed_convention_breaks_closure`` as the check that the mixed convention
is invalid.  That test exists here and it must keep that name.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from mqcle.constants import TOL_FD2
from mqcle.excess import (excess_coefficients, excess_operator_on_field,
                          excess_operator_on_function)
from mqcle.fields import FiniteDifferenceField, SymbolicField, symbols_for
from mqcle.mapping import subsystem_split
from mqcle.models import ConstantCouplingModel, build_model
from mqcle.types import TraceConvention
from mqcle.validation import (build_symbolic_setup, mapping_polynomial_degrees,
                              apply_full)

N, NB, HBAR, MASS = 2, 1, 1.0, 2000.0


@pytest.fixture(scope="module")
def dac():
    return build_model({"name": "tully_dual_avoided_crossing"}, mass=MASS)


def _field():
    r, p, R, P = symbols_for(N, NB)
    u01 = r[0] * r[1] + p[0] * p[1]
    phi = sp.exp(-sum(s**2 for s in list(r) + list(p)))
    return SymbolicField(phi * (1 + u01) * (P[0] ** 2 + R[0]), N, NB)


def _points(rng, n=10):
    return (rng.normal(0, 0.8, size=(n, 2 * N)),
            rng.uniform(-3, 3, size=(n, NB)),
            rng.normal(15, 3, size=(n, NB)))


def test_exact_and_finite_difference_excess_agree(dac):
    rng = np.random.default_rng(41)
    x, R, P = _points(rng)
    f = _field()
    fd = FiniteDifferenceField(lambda a, b, c: f.value(a, b, c), N, NB,
                               hx=1e-3, hP=1e-3)
    a = excess_operator_on_field(f, x, R, P, dac, HBAR)
    b = excess_operator_on_field(fd, x, R, P, dac, HBAR)
    scale = np.max(np.abs(a))
    assert scale > 1e-9
    assert np.max(np.abs(a - b)) / scale < TOL_FD2


def test_excess_vanishes_when_grad_R_h_is_zero():
    m = ConstantCouplingModel(a0=0.02, b0=-0.01, c0=0.005, k=0.3, mass=MASS)
    rng = np.random.default_rng(42)
    x, R, P = _points(rng)
    assert np.max(np.abs(excess_operator_on_field(_field(), x, R, P, m, HBAR))) \
        < 1e-14


def test_excess_coefficients_are_the_traceless_gradient(dac):
    R = np.linspace(-3, 3, 9)[:, None]
    c = excess_coefficients(dac, R, HBAR, TraceConvention.TRACELESS)
    s = subsystem_split(dac, R, TraceConvention.TRACELESS)
    assert c.shape[-2:] == (N, N)
    assert np.max(np.abs(np.trace(s.dh_map_dR[:, 0], axis1=-2, axis2=-1))) < 1e-14


def test_operator_on_function_matches_operator_on_field(dac):
    rng = np.random.default_rng(43)
    x, R, P = _points(rng)
    f = _field()
    a = excess_operator_on_field(f, x, R, P, dac, HBAR)
    b = excess_operator_on_function(lambda xx, RR, PP: f.value(xx, RR, PP),
                                    x, R, P, dac, HBAR)
    assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < TOL_FD2


def test_mixed_convention_breaks_closure():
    """The regression test named in ``mqcle.types.TraceConvention``.

    Full ``h`` in ``H_m`` with traceless ``hbar`` in the excess term destroys the
    exact cancellation between the two order-raising channels: the amplitude
    of the full generator acquires degree 4.  Both self-consistent conventions
    keep it at degree 2.
    """
    for conv in ("traceless", "full"):
        s = build_symbolic_setup(N, NB, conv)
        assert mapping_polynomial_degrees(s, sp.expand(apply_full(s, s.rho_m))) \
            == [0, 2], f"convention '{conv}' should close at degree 2"

    s = build_symbolic_setup(N, NB, "full", mixed_convention=True)
    deg = mapping_polynomial_degrees(s, sp.expand(apply_full(s, s.rho_m)))
    assert deg == [0, 2, 4], f"the mixed convention must FAIL, got {deg}"
