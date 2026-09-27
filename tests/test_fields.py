"""Gate E: complex-safe differentiation infrastructure."""
import numpy as np
import pytest
import sympy as sp

from mqcle.constants import HBAR
from mqcle.excess import excess_operator_on_field
from mqcle.fields import (FiniteDifferenceField, SymbolicField, result_dtype,
                          symbols_for)

N, NB = 2, 1


def _complex_field():
    """f = phi(x) (u_01 - i L_01) P0 -- the off-diagonal projector shape."""
    r, p, R, P = symbols_for(N, NB)
    u01 = r[0] * r[1] + p[0] * p[1]
    L01 = r[0] * p[1] - r[1] * p[0]
    phi = sp.exp(-sum(s**2 for s in list(r) + list(p)))
    return SymbolicField(phi * (u01 - sp.I * L01) * P[0], N, NB)


def test_result_dtype_helper():
    assert result_dtype(np.zeros(3)) is np.float64
    assert result_dtype(np.zeros(3), np.zeros(3, dtype=complex)) is np.complex128


def test_complex_value_and_derivatives_survive(rng):
    f = _complex_field()
    x = rng.normal(0, 0.8, size=(16, 4))
    R = rng.uniform(-2, 2, size=(16, 1))
    P = rng.normal(10, 2, size=(16, 1))
    for arr in (f.value(x, R, P), f.grad_x(x, R, P), f.grad_P(x, R, P),
                f.excess_tensor(x, R, P)):
        assert np.iscomplexobj(arr)
        assert np.max(np.abs(arr.imag)) > 1e-6


def test_real_and_imaginary_parts_agree_with_finite_differences(rng, dac):
    f = _complex_field()
    fd = FiniteDifferenceField(lambda xx, RR, PP: f.value(xx, RR, PP),
                               N, NB, hx=1e-3, hP=1e-3)
    x = rng.normal(0, 0.8, size=(12, 4))
    R = rng.uniform(-2, 2, size=(12, 1))
    P = rng.normal(10, 2, size=(12, 1))
    a = excess_operator_on_field(f, x, R, P, dac)
    b = excess_operator_on_field(fd, x, R, P, dac)
    assert np.iscomplexobj(a) and np.iscomplexobj(b)
    for part in (np.real, np.imag):
        scale = np.max(np.abs(part(a)))
        assert scale > 1e-9
        assert np.max(np.abs(part(a) - part(b))) / scale < 1e-4


def test_real_field_stays_real(rng):
    r, p, R, P = symbols_for(N, NB)
    f = SymbolicField(sp.exp(-r[0]**2) * P[0] * (r[0] * r[1] + p[0]**2), N, NB)
    x = rng.normal(0, 0.8, size=(8, 4))
    Rv = rng.uniform(-2, 2, size=(8, 1))
    Pv = rng.normal(10, 2, size=(8, 1))
    assert f.value(x, Rv, Pv).dtype == np.float64
    assert f.excess_tensor(x, Rv, Pv).dtype == np.float64
