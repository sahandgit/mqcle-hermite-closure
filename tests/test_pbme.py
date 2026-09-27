"""Module 5: the trajectory integrator and its measured properties.

The integrator is a symmetric frozen-R rotation splitting.  It is symmetric,
exactly reversible, mapping-radius preserving, and exact for an R-independent
h -- and it is NOT symplectic.  Every one of those is asserted here, including
the last, because the manuscript reports the measured defect rather than
claiming symplecticity.
"""
from __future__ import annotations

import numpy as np
import pytest

from mqcle.models import ConstantCouplingModel, build_model
from mqcle.pbme import (frozen_R_propagator, hamiltonian_of_state,
                        mapping_radius, pack_state, pbme_vector_field,
                        propagate, rk4_step, strang_step,
                        symmetric_rotation_step, symplecticity_defect,
                        unpack_state)
from mqcle.sampling import sample_initial_ensemble
from mqcle.types import TraceConvention

N, NB, HBAR, MASS = 2, 1, 1.0, 2000.0
CONV = TraceConvention.TRACELESS


@pytest.fixture(scope="module")
def dac():
    return build_model({"name": "tully_dual_avoided_crossing"}, mass=MASS)


@pytest.fixture(scope="module")
def state0():
    e = sample_initial_ensemble(np.random.default_rng(31), 256, nstates=N,
                                init_state=0, R0=0.0, P0=20.0, sigma_R=0.5,
                                scheme="focused", hbar=HBAR)
    return pack_state(e.x, e.R, e.P)


def test_pack_unpack_round_trip(state0):
    x, R, P = unpack_state(state0, N, NB)
    assert np.array_equal(pack_state(x, R, P), state0)
    assert x.shape[1] == 2 * N and R.shape[1] == NB


def test_energy_is_conserved_in_the_crossing_region(dac, state0):
    traj = propagate(state0, 0.25, 400, dac, hbar=HBAR, convention=CONV)
    e0 = hamiltonian_of_state(state0, dac, HBAR, None, CONV)
    eT = hamiltonian_of_state(traj[-1], dac, HBAR, None, CONV)
    assert np.max(np.abs(eT - e0)) < 1e-6


def test_the_step_is_exactly_time_reversible(dac, state0):
    """Palindromic by construction: forward then backward returns the state."""
    s = state0.copy()
    for _ in range(50):
        s = symmetric_rotation_step(s, 0.5, dac, hbar=HBAR, convention=CONV)
    for _ in range(50):
        s = symmetric_rotation_step(s, -0.5, dac, hbar=HBAR, convention=CONV)
    assert np.max(np.abs(s - state0)) < 1e-12


def test_mapping_radius_is_preserved(dac, state0):
    traj = propagate(state0, 0.5, 200, dac, hbar=HBAR, convention=CONV)
    x0, _, _ = unpack_state(state0, N, NB)
    xT, _, _ = unpack_state(traj[-1], N, NB)
    r0 = mapping_radius(x0, N)
    rT = mapping_radius(xT, N)
    assert np.max(np.abs(rT - r0)) < 1e-12


def test_the_step_is_exact_for_an_R_independent_h(state0):
    """With grad_R h = 0 the mapping block is a constant rotation, which the
    frozen-R splitting integrates exactly -- at any step size."""
    m = ConstantCouplingModel(a0=0.02, b0=-0.01, c0=0.005, k=0.0, mass=MASS)
    big = state0.copy()
    for _ in range(4):
        big = symmetric_rotation_step(big, 2.0, m, hbar=HBAR, convention=CONV)
    small = state0.copy()
    for _ in range(64):
        small = symmetric_rotation_step(small, 0.125, m, hbar=HBAR, convention=CONV)
    assert np.max(np.abs(big - small)) < 1e-10


def test_frozen_R_propagator_is_orthogonal(dac):
    R = np.array([[-1.2], [0.0], [2.4]])
    B = frozen_R_propagator(R, 0.7, dac, HBAR, CONV)
    eye = np.eye(B.shape[-1])
    assert np.max(np.abs(np.einsum("mij,mkj->mik", B, B) - eye)) < 1e-13


def test_the_scheme_is_NOT_symplectic_and_the_defect_shrinks_with_dt(dac, state0):
    """Reported, not hidden.  eps_symplectic is nonzero on purpose."""
    one = state0[0]                     # the Jacobian is per trajectory
    dts = (0.5, 0.25, 0.125)
    eps = [symplecticity_defect(one, dt, dac, hbar=HBAR, convention=CONV)
           for dt in dts]
    assert all(e > 0 for e in eps), "the scheme is NOT symplectic"
    assert eps[0] > eps[1] > eps[2]
    order = float(np.polyfit(np.log(dts), np.log(eps), 1)[0])
    assert order > 1.5, f"defect should fall at least quadratically, got {order:.2f}"


def test_rk4_and_the_splitting_agree_as_dt_shrinks(dac, state0):
    """Two integrators of the same vector field share a dt -> 0 limit.  The
    statement to check is the RATE at which they converge to each other, not
    an absolute threshold at one arbitrary step size."""
    sub = state0[:32]
    dts = (0.5, 0.25, 0.125, 0.0625)
    diffs = []
    for dt in dts:
        n = int(round(4.0 / dt))
        a, b = sub.copy(), sub.copy()
        for _ in range(n):
            a = rk4_step(a, dt, dac, hbar=HBAR, convention=CONV)
            b = symmetric_rotation_step(b, dt, dac, hbar=HBAR, convention=CONV)
        diffs.append(float(np.max(np.abs(a - b))))
    assert all(diffs[i] > diffs[i + 1] for i in range(len(diffs) - 1))
    order = float(np.polyfit(np.log(dts), np.log(diffs), 1)[0])
    assert order > 1.5, f"difference should vanish at least quadratically, got {order:.2f}"


def test_vector_field_shapes(dac, state0):
    v = pbme_vector_field(state0, dac, HBAR, MASS, CONV)
    assert v.shape == state0.shape
    assert np.all(np.isfinite(v))
