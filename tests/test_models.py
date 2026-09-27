"""Module 1: the diabatic models and their gradients."""
from __future__ import annotations

import numpy as np
import pytest

from mqcle.constants import TOL_FD1
from mqcle.models import (ConstantCouplingModel, TullyDualAvoidedCrossing,
                          TullySingleAvoidedCrossing, build_model)

MODELS = [TullyDualAvoidedCrossing(), TullySingleAvoidedCrossing(),
          ConstantCouplingModel(a1=0.003, b1=-0.002, c1=0.001, k=0.01)]


def _fd(f, R, h=1e-5):
    return (f(R + h) - f(R - h)) / (2 * h)


@pytest.mark.parametrize("m", MODELS, ids=lambda m: m.name)
def test_h_is_real_symmetric_everywhere(m):
    R = np.linspace(-8.0, 8.0, 41)[:, None]
    h = m.h(R)
    assert h.dtype == np.float64
    assert np.max(np.abs(h - np.swapaxes(h, -1, -2))) == 0.0


@pytest.mark.parametrize("m", MODELS, ids=lambda m: m.name)
def test_dh_dR_matches_finite_differences(m):
    R = np.linspace(-6.0, 6.0, 25)[:, None]
    if m.name == "tully_single_avoided_crossing":
        # V_11 = A(1 - e^{-B|R|}) sgn(R) is only C^1 at the origin, so a
        # central difference straddling R = 0 is meaningless there.  This is
        # a property of Tully's model I, not a defect of the derivative.
        R = R[np.abs(R[:, 0]) > 0.5]
    ana = m.dh_dR(R)[:, 0]
    num = _fd(lambda r: m.h(r), R)
    assert np.max(np.abs(ana - num)) < TOL_FD1


@pytest.mark.parametrize("m", MODELS, ids=lambda m: m.name)
def test_dbath_dR_matches_finite_differences(m):
    R = np.linspace(-6.0, 6.0, 25)[:, None]
    ana = m.dbath_dR(R)[:, 0]
    num = _fd(lambda r: m.bath_potential(r), R)      # shape (n,), not (n, 1)
    assert np.max(np.abs(ana - num)) < TOL_FD1


def test_tully_dac_matches_its_published_definition():
    """Tully, JCP 93, 1061 (1990), model II."""
    m = TullyDualAvoidedCrossing()
    p = m.p
    R = np.array([[-1.7], [0.0], [2.3]])
    h = m.h(R)
    r = R[:, 0]
    assert np.allclose(h[:, 0, 0], 0.0)
    assert np.allclose(h[:, 1, 1], -p.A * np.exp(-p.B * r**2) + p.E0)
    assert np.allclose(h[:, 0, 1], p.C * np.exp(-p.D * r**2))
    assert np.allclose(m.bath_potential(R), 0.0)


def test_dual_avoided_crossing_has_two_symmetric_gap_minima():
    """The clue is in the name: the gap minima sit at the *two* diabatic
    crossings where V_11 = V_22, near R = +/- 1.57, not at the origin.  The
    archived run records min gap 0.02575 at R = 1.60 on its coarser scan."""
    m = TullyDualAvoidedCrossing()
    R = np.linspace(-4.0, 4.0, 8001)[:, None]
    e = m.adiabatic_energies(R)
    gap = e[:, 1] - e[:, 0]
    assert gap.min() > 0.0                      # avoided, not a true crossing

    right = R[:, 0] > 0.1
    r_min = R[right][np.argmin(gap[right]), 0]
    assert 1.4 < r_min < 1.8
    # symmetric, and the origin is a local MAXIMUM of the gap between them
    left = R[:, 0] < -0.1
    assert abs(R[left][np.argmin(gap[left]), 0] + r_min) < 2e-3
    i0 = int(np.argmin(np.abs(R[:, 0])))
    assert gap[i0] > gap.min()
    # The minimum sits near, but not exactly at, the diabatic crossing: the
    # coupling V_12(R) varies too, so min sqrt(dV^2 + 4 V_12^2) is displaced
    # from dV = 0.  What is true is that the diabatic splitting there is far
    # smaller than at the origin.
    h_min = m.h(np.array([[r_min]]))[0]
    h_0 = m.h(np.zeros((1, 1)))[0]
    dV_min = abs(h_min[0, 0] - h_min[1, 1])
    dV_0 = abs(h_0[0, 0] - h_0[1, 1])
    assert dV_min < 0.1 * dV_0


def test_build_model_round_trips_its_parameters():
    spec = {"name": "tully_dual_avoided_crossing", "A": 0.11, "B": 0.3}
    m = build_model(spec, mass=1234.0)
    par = m.parameters()
    assert par["A"] == 0.11 and par["B"] == 0.3
    assert par["mass"] == 1234.0 and par["nstates"] == 2
    with pytest.raises(KeyError):
        build_model({"name": "not_a_model"})


def test_broadcasting_over_leading_axes():
    m = TullyDualAvoidedCrossing()
    R = np.random.default_rng(0).uniform(-3, 3, size=(7, 5, 1))
    assert m.h(R).shape == (7, 5, 2, 2)
    assert m.dh_dR(R).shape == (7, 5, 1, 2, 2)
    assert m.bath_potential(R).shape == (7, 5)
