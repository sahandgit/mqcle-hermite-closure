"""Module 2: mapping coordinates, the trace convention, and the exact flow."""
from __future__ import annotations

import numpy as np
import pytest

from mqcle.constants import TOL_ANALYTIC, TOL_EXACT, TOL_FD1
from mqcle.mapping import (combine_mapping, drift_matrix,
                           grad_mapping_hamiltonian_R,
                           grad_mapping_hamiltonian_x, mapping_bilinears,
                           mapping_hamiltonian, mapping_velocity,
                           split_mapping, subsystem_split)
from mqcle.models import build_model
from mqcle.types import TraceConvention

N, NB, HBAR, MASS = 2, 1, 1.0, 2000.0


@pytest.fixture(scope="module")
def dac():
    return build_model({"name": "tully_dual_avoided_crossing"}, mass=MASS)


def _sample(rng, n=12):
    x = rng.normal(0.0, 0.9, size=(n, 2 * N))
    R = rng.uniform(-4.0, 4.0, size=(n, NB))
    P = rng.normal(20.0, 4.0, size=(n, NB))
    return x, R, P


def test_split_and_combine_are_inverse():
    x = np.random.default_rng(1).normal(size=(9, 2 * N))
    r, p = split_mapping(x, N)
    assert np.array_equal(combine_mapping(r, p), x)


def test_bilinears_are_symmetric_and_reproduce_the_action():
    x = np.random.default_rng(2).normal(size=(6, 2 * N))
    u = mapping_bilinears(x, N)
    assert np.max(np.abs(u - np.swapaxes(u, -1, -2))) < TOL_EXACT
    r, p = split_mapping(x, N)
    assert np.allclose(np.diagonal(u, axis1=-2, axis2=-1), r**2 + p**2)


@pytest.mark.parametrize("conv", list(TraceConvention))
def test_subsystem_split_reassembles_the_physical_matrix(dac, conv):
    """h_map + (v_eff - V_b) 1 == h, whichever convention is used."""
    R = np.linspace(-5.0, 5.0, 21)[:, None]
    s = subsystem_split(dac, R, conv)
    back = s.h_map + (s.v_eff - dac.bath_potential(R))[:, None, None] * np.eye(N)
    assert np.max(np.abs(back - dac.h(R))) < TOL_EXACT
    dback = s.dh_map_dR[:, 0] + (s.dv_eff_dR[:, 0]
                                 - dac.dbath_dR(R)[:, 0])[:, None, None] * np.eye(N)
    assert np.max(np.abs(dback - dac.dh_dR(R)[:, 0])) < TOL_EXACT


def test_traceless_convention_really_is_traceless(dac):
    R = np.linspace(-5.0, 5.0, 21)[:, None]
    s = subsystem_split(dac, R, TraceConvention.TRACELESS)
    assert np.max(np.abs(np.trace(s.h_map, axis1=-2, axis2=-1))) < TOL_EXACT


def test_grad_x_and_grad_R_match_finite_differences(dac):
    rng = np.random.default_rng(3)
    x, R, _ = _sample(rng)
    gx = grad_mapping_hamiltonian_x(x, R, dac, HBAR)
    h = 1e-6
    for i in range(2 * N):
        e = np.zeros(2 * N)
        e[i] = h
        num = (mapping_hamiltonian(x + e, R, np.zeros_like(R), dac, HBAR, MASS)
               - mapping_hamiltonian(x - e, R, np.zeros_like(R), dac, HBAR, MASS)) / (2 * h)
        assert np.max(np.abs(gx[:, i] - num)) < TOL_FD1

    gR = grad_mapping_hamiltonian_R(x, R, dac, HBAR)
    hR = 1e-6
    num = (mapping_hamiltonian(x, R + hR, np.zeros_like(R), dac, HBAR, MASS)
           - mapping_hamiltonian(x, R - hR, np.zeros_like(R), dac, HBAR, MASS)) / (2 * hR)
    assert np.max(np.abs(gR[:, 0] - num)) < TOL_FD1


def test_drift_matrix_is_antisymmetric_and_generates_the_flow(dac):
    """xdot = K(R) x exactly -- the algebraic fact Theorem 1 step (i) uses."""
    rng = np.random.default_rng(4)
    x, R, _ = _sample(rng)
    K = drift_matrix(R, dac, HBAR)
    assert np.max(np.abs(K + np.swapaxes(K, -1, -2))) == 0.0
    v = mapping_velocity(x, R, dac, HBAR)
    assert np.max(np.abs(v - np.einsum("mij,mj->mi", K, x))) < TOL_ANALYTIC


def test_drift_matrix_block_structure(dac):
    """K = [[0, h],[-h, 0]]/hbar, so K^2 is block diagonal with -h^2/hbar^2."""
    R = np.array([[0.0], [1.3]])
    K = drift_matrix(R, dac, HBAR)
    s = subsystem_split(dac, R, TraceConvention.TRACELESS)
    assert np.max(np.abs(K[:, :N, :N])) < TOL_EXACT
    assert np.max(np.abs(K[:, N:, N:])) < TOL_EXACT
    assert np.max(np.abs(K[:, :N, N:] - s.h_map / HBAR)) < TOL_EXACT
    assert np.max(np.abs(K[:, N:, :N] + s.h_map / HBAR)) < TOL_EXACT


def test_hamiltonian_conventions_differ_by_the_mapping_identity(dac):
    """The two conventions differ by a multiple of sum(r^2 + p^2 - hbar)."""
    rng = np.random.default_rng(5)
    x, R, P = _sample(rng)
    a = mapping_hamiltonian(x, R, P, dac, HBAR, MASS, TraceConvention.TRACELESS)
    b = mapping_hamiltonian(x, R, P, dac, HBAR, MASS, TraceConvention.FULL)
    r, p = split_mapping(x, N)
    ident = np.sum(r**2 + p**2 - HBAR, axis=-1) / (2 * HBAR)
    tr = np.trace(dac.h(R), axis1=-2, axis2=-1) / N
    assert np.max(np.abs((b - a) - tr * (ident - 1.0))) < TOL_ANALYTIC
