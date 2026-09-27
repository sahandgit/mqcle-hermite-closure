r"""Result 9: the residual freedom of Theorem 3 acts by covariance.

A constant orthogonal change of subsystem basis, :math:`h\mapsto
O^{\mathsf T}hO`, lifts to the block map :math:`x\mapsto\mathrm{diag}(O,O)
^{\mathsf T}x` on the mapping variables, which is orthogonal *and* symplectic.
Under it the mapping Hamiltonian keeps its form, the Poisson bracket is
invariant, and the projected density transforms with :math:`\rho_W`.  Both the
closed generator and its Poisson-bracket truncation are therefore covariant,
which is why the freedom cannot be optimised to reduce the PBME error.

The manuscript previously said the opposite -- that because the defect is built
from :math:`\nabla_Rh`, whose entries the rotation changes, the basis could in
principle be tuned.  These tests are the regression that keeps that claim from
coming back.  The production version, with more angles and more steps, is
``scripts/07c_basis_covariance.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from mqcle.coefficient_operator import build_grid
from mqcle.models import build_model
from mqcle.pbme import pack_state, symmetric_rotation_step, unpack_state
from mqcle.propagation import StrangPropagator, initial_coefficient_field
from mqcle.types import TraceConvention

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "basis_covariance",
    Path(__file__).resolve().parents[1] / "scripts" / "07c_basis_covariance.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
RotatedModel = _mod.RotatedModel
rotation, unitary = _mod.rotation, _mod.unitary
rotate_mapping, rotate_density = _mod.rotate_mapping, _mod.rotate_density

#: real orthogonal AND genuinely complex; a real O never leaves the case
#: Theorem 1 was originally stated for, so it cannot test the U(N) claim
ROTS = [("O", rotation(0.3)), ("O", rotation(np.pi / 4)),
        ("U", unitary(0.6, 0.9)), ("U", unitary(np.pi / 4, np.pi / 3))]

#: at the crossing.  Away from it grad_R h is numerically zero and the coupled
#: channel -- the only part of the covariance claim anyone would doubt -- is
#: never exercised.  Same trap as CORRECTIONS.md L5.
R_CROSSING = 0.0
HBAR = 1.0


@pytest.fixture(scope="module")
def model():
    return build_model({"name": "tully_dual_avoided_crossing"},
                       mass=2000.0)


@pytest.mark.parametrize("kind,U", ROTS)
def test_rotated_model_is_a_similarity_transform(model, kind, U):
    """A basis change must not change the physics: eigenvalues are invariant."""
    R = np.linspace(-6.0, 6.0, 41)[:, None]
    w0 = np.linalg.eigvalsh(model.h(R))
    w1 = np.linalg.eigvalsh(RotatedModel(model, U).h(R))
    assert np.allclose(w0, w1, atol=1e-13)
    g0 = model.dh_dR(R)
    g1 = RotatedModel(model, U).dh_dR(R)
    assert np.allclose(g1, np.conj(U).T @ g0 @ U, atol=1e-13)
    if kind == "U":
        assert np.abs(np.imag(RotatedModel(model, U).h(R))).max() > 1e-3, \
            "the complex case must actually produce a complex h"


def test_rotation_actually_changes_grad_h(model):
    """Guard against a vacuous covariance test.

    If the basis change left the entries of grad_R h alone there would be
    nothing to be covariant about, and the tempting-but-wrong claim in the
    manuscript would not even have been tempting.
    """
    R = np.linspace(-6.0, 6.0, 41)[:, None]
    for _, U in ROTS:
        g0, g1 = model.dh_dR(R), RotatedModel(model, U).dh_dR(R)
        assert np.max(np.abs(g1 - g0)) > 1e-4


def test_the_test_region_carries_a_gradient(model):
    """The second half of the guard: the propagated region must not be flat."""
    R = np.linspace(R_CROSSING - 1.0, R_CROSSING + 1.0, 201)[:, None]
    g = model.dh_dR(R)[:, 0]
    everywhere = model.dh_dR(np.linspace(-6, 6, 2001)[:, None])[:, 0]
    local = np.linalg.norm(g, ord="fro", axis=(-2, -1)).max()
    glob = np.linalg.norm(everywhere, ord="fro", axis=(-2, -1)).max()
    assert local > 0.5 * glob, "the packet is not where the coupling is"


@pytest.mark.parametrize("kind,U", ROTS)
def test_pbme_generator_is_covariant(model, kind, U):
    n, dt, n_steps = 2, 0.5, 60
    rng = np.random.default_rng(7)
    m = 200
    x = rng.normal(scale=np.sqrt(HBAR / 2), size=(m, 2 * n))
    R = rng.normal(loc=R_CROSSING, scale=0.25, size=(m, 1))
    P = rng.normal(loc=40.0, scale=2.0, size=(m, 1))

    def run(x0, mod):
        st = pack_state(x0, R, P)
        for _ in range(n_steps):
            st = symmetric_rotation_step(st, dt, mod, hbar=HBAR,
                                         convention=TraceConvention.TRACELESS)
        return unpack_state(st, n, mod.nbath)

    xr, Rr, Pr = run(x, model)
    xs, Rs, Ps = run(rotate_mapping(x, U), RotatedModel(model, U))
    scale = float(np.max(np.abs(xr)))
    assert np.max(np.abs(xs - rotate_mapping(xr, U))) / scale < 1e-12
    # The transformation touches only the mapping sector, so the bath agrees to
    # round-off.  NOT bit-exactly: at the crossing the back-reaction force is
    # nonzero and is assembled from rotated matrices.  Bit equality here would
    # mean the packet was somewhere the force vanishes, which is what made the
    # first version of this check vacuous.
    assert np.max(np.abs(Rs - Rr)) < 1e-11
    assert np.max(np.abs(Ps - Pr)) < 1e-10


@pytest.mark.parametrize("kind,U", ROTS)
def test_closed_generator_is_covariant(model, kind, U):
    n, dt, n_steps = 2, 0.5, 40
    grid = build_grid(-45.0, 45.0, 256, 18.0, 62.0, 128)
    rho0 = initial_coefficient_field(grid, R_CROSSING, 40.0, 0.25, 0, n, HBAR)

    def run(rho, mod):
        prop = StrangPropagator(grid, mod, dt, HBAR, TraceConvention.TRACELESS)
        for _ in range(n_steps):
            rho = prop.step(rho)
        return rho

    ref = run(rho0, model)
    got = run(rotate_density(rho0, U), RotatedModel(model, U))
    want = rotate_density(ref, U)
    assert np.max(np.abs(got - want)) / np.max(np.abs(ref)) < 1e-12
