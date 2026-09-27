"""Gates H, I, J: true Hermite projection, physical frame, QCLE correspondence."""
import numpy as np
import pytest
import sympy as sp

from mqcle.hermite.basis import he_multi
from mqcle.validation import (apply_excess, apply_full, apply_pbme,
                              build_symbolic_setup, mapping_hermite_orders,
                              mapping_polynomial_degrees,
                              pbme_hermite_defect,
                              pbme_principal_polynomial_leakage,
                              physical_frame_projection,
                              physical_frame_residual,
                              verify_qcle_coefficient_correspondence)


@pytest.fixture(scope="module")
def setup():
    return build_symbolic_setup(2, 1, "traceless")


@pytest.fixture(scope="module")
def full_action(setup):
    return sp.expand(apply_full(setup, setup.rho_m))


@pytest.fixture(scope="module")
def pbme_action(setup):
    return sp.expand(apply_pbme(setup, setup.rho_m))


# -- Gate G: ordinary polynomial degrees ----------------------------------
def test_polynomial_degrees(setup, full_action, pbme_action):
    assert mapping_polynomial_degrees(setup, setup.rho_m) == [0, 2]
    assert mapping_polynomial_degrees(setup, full_action) == [0, 2]
    assert mapping_polynomial_degrees(setup, pbme_action) == [0, 2, 4]
    assert mapping_polynomial_degrees(
        setup, sp.expand(apply_excess(setup, setup.rho_m))) == [0, 2, 4]


# -- Gate H: true Hermite orders and residual -----------------------------
def test_hermite_orders_agree_on_membership(setup, full_action):
    """Degree <= 2 implies Hermite order <= 2; here both sets coincide."""
    assert mapping_hermite_orders(setup, full_action) == [0, 2]


def test_top_degree_is_not_the_hermite_residual(setup):
    """The central correction: Top_4 and (I - P_2) are different functions."""
    top = pbme_principal_polynomial_leakage(setup, above=2, standardised=True)
    her = pbme_hermite_defect(setup, max_order=2)
    diff = sp.expand(sp.simplify(top - her))
    assert diff != 0
    degrees = sorted({sum(m) for m in sp.Poly(diff, *setup.z).monoms()})
    assert degrees == [0, 2]        # they differ only below the top degree


def test_hermite_residual_is_wick_ordered_top_degree(setup):
    """(I - P_2) q equals Top_4 with each monomial z^a replaced by He_a(z)."""
    top = pbme_principal_polynomial_leakage(setup, above=2, standardised=True)
    her = pbme_hermite_defect(setup, max_order=2)
    P = sp.Poly(top, *setup.z)
    wick = sum(c * he_multi(m, setup.z) for m, c in zip(P.monoms(), P.coeffs()))
    assert sp.simplify(sp.expand(wick - her)) == 0


def test_full_generator_has_no_hermite_defect(setup, full_action):
    from mqcle.validation import hermite_defect
    assert sp.simplify(hermite_defect(setup, full_action, 2)) == 0


# -- Gate I: physical-frame invariance ------------------------------------
def test_physical_frame_residual_vanishes_for_full_generator(setup, full_action):
    """V_2 invariance does not imply this; it is a separate statement."""
    assert sp.simplify(physical_frame_residual(setup, full_action)) == 0


def test_physical_frame_residual_is_nonzero_for_pbme(setup, pbme_action):
    assert sp.simplify(physical_frame_residual(setup, pbme_action)) != 0


def test_frame_projection_reproduces_the_density(setup):
    """Projecting rho_m itself must return it unchanged with C = rho_W."""
    coeff, proj, residual = physical_frame_projection(setup, setup.rho_m)
    assert sp.simplify(residual) == 0
    for a in range(2):
        for b in range(2):
            assert sp.simplify(coeff[a][b] - setup.rho[b][a]) == 0


# -- Gate J: QCLE coefficient correspondence ------------------------------
def test_qcle_coefficient_correspondence(setup):
    out = verify_qcle_coefficient_correspondence(setup)
    assert out["frame_residual_is_zero"]
    assert out["all_zero"], out["residuals"]
    assert set(out["residuals"]) == {"residual_00", "residual_01",
                                     "residual_10", "residual_11"}
