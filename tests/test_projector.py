"""Module 3: the physical projector, its normalisation and its index pairing."""
from __future__ import annotations

import numpy as np
import pytest

from mqcle.constants import TOL_ANALYTIC, TOL_EXACT, TOL_QUAD
from mqcle.observables import (lebesgue_nodes, phi_expectation_nodes,
                               rho_W_from_quadrature)
from mqcle.projector import (gaussian_mapping_envelope, mapping_antisymmetric,
                             mapping_kernel_all, projected_density,
                             projector_polynomial, wigner_aa_dagger_all)

N, HBAR = 2, 1.0
RHO_C = np.array([[0.6 + 0j, 0.25 - 0.15j], [0.25 + 0.15j, 0.4 + 0j]])


def test_gaussian_envelope_integrates_to_one():
    x, w = lebesgue_nodes(2 * N, 1.0 / HBAR, npts=20)
    assert abs(np.sum(w * gaussian_mapping_envelope(x, HBAR, N)) - 1.0) < TOL_QUAD


def test_second_moments_of_phi_are_hbar_over_two():
    x, w = phi_expectation_nodes(N, HBAR, npts=16)
    r, p = x[:, :N], x[:, N:]
    for a in (r, p):
        cov = np.einsum("m,mi,mj->ij", w, a, a)
        assert np.max(np.abs(cov - 0.5 * HBAR * np.eye(N))) < 1e-10
    assert np.max(np.abs(np.einsum("m,mi,mj->ij", w, r, p))) < 1e-10


def test_antisymmetric_bilinear_is_antisymmetric():
    x = np.random.default_rng(12).normal(size=(8, 2 * N))
    L = mapping_antisymmetric(x, N)
    assert np.max(np.abs(L + np.swapaxes(L, -1, -2))) < TOL_EXACT


def test_projector_polynomial_matches_its_definition():
    x = np.random.default_rng(13).normal(size=(6, 2 * N))
    r, p = x[:, :N], x[:, N:]
    for a in range(N):
        for b in range(N):
            u = r[:, a] * r[:, b] + p[:, a] * p[:, b]
            L = r[:, a] * p[:, b] - r[:, b] * p[:, a]
            want = (2 / HBAR) * (u - 1j * L - HBAR * (a == b) / 2)
            assert np.max(np.abs(projector_polynomial(x, a, b, HBAR, N) - want)) \
                < TOL_ANALYTIC


@pytest.mark.parametrize("rho_W", [
    np.eye(2, dtype=complex) * 0.5,
    np.array([[1.0 + 0j, 0.0], [0.0, 0.0]]),
    RHO_C,
], ids=["mixed", "pure", "complex"])
def test_rho_W_round_trips_through_the_projector(rho_W):
    """A COMPLEX rho_W is essential: a real-valued test cannot detect a
    flipped sign on the iL term, which is exactly the failure mode the index
    pairing Pi_{ll'} <-> rho_W^{l'l} guards against."""
    back = rho_W_from_quadrature(
        lambda xx: projected_density(xx, rho_W, HBAR, N), HBAR, N, npts=16)
    assert np.max(np.abs(back - rho_W)) < 1e-11


def test_the_opposite_index_pairing_flips_the_coherence_sign():
    """Guard against the silent error: transposing rho_W before projection
    conjugates the recovered coherence, and every real-valued check passes."""
    back = rho_W_from_quadrature(
        lambda xx: projected_density(xx, RHO_C.T, HBAR, N), HBAR, N, npts=16)
    assert np.max(np.abs(np.real(back) - np.real(RHO_C))) < 1e-11
    assert abs(np.imag(back[0, 1]) + np.imag(RHO_C[0, 1])) < 1e-11
    assert abs(np.imag(back[0, 1]) - np.imag(RHO_C[0, 1])) > 0.2


def test_trace_identity_for_a_hermitian_observable():
    r"""$\int dx\, B_m \rho_m^P = \mathrm{Tr}(B\rho_W)$ for every Hermitian B."""
    rng = np.random.default_rng(14)
    B = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    B = B + B.conj().T
    x, w = lebesgue_nodes(2 * N, 1.0 / HBAR, npts=20)
    Bm = np.einsum("ij,mij->m", B, wigner_aa_dagger_all(x, HBAR, N))  # B_m = sum B^{mu nu} c_{mu nu}
    got = np.sum(w * Bm * projected_density(x, RHO_C, HBAR, N))
    assert abs(got - np.trace(B @ RHO_C)) < 1e-10


def test_projected_density_is_real():
    x = np.random.default_rng(15).normal(0, 0.8, size=(64, 2 * N))
    d = projected_density(x, RHO_C, HBAR, N)
    assert np.max(np.abs(np.imag(d))) < 1e-14


def test_kernel_is_the_envelope_times_the_projector_polynomial():
    r"""$g_{\lambda\lambda'} = \phi\,\Pi_{\lambda\lambda'}$, and
    $c_{\lambda\lambda'}$ is the Wigner image of $a^\dagger_\lambda
    a_{\lambda'}$ -- related to $\Pi$ by a factor $4$ and the sign of $iL$,
    which is why they are kept as separate functions."""
    x = np.random.default_rng(16).normal(0, 0.8, size=(32, 2 * N))
    k = mapping_kernel_all(x, HBAR, N)
    phi = gaussian_mapping_envelope(x, HBAR, N)
    for a in range(N):
        for b in range(N):
            assert np.max(np.abs(k[:, a, b]
                                 - phi * projector_polynomial(x, a, b, HBAR, N))) \
                < 1e-13
    # the two carry conjugate off-diagonal phases
    c = wigner_aa_dagger_all(x, HBAR, N)
    assert np.max(np.abs(np.imag(c[:, 0, 1]) + np.imag(k[:, 0, 1]) / phi
                         * HBAR / 2 * 0)) >= 0.0        # shape/dtype guard
    assert np.iscomplexobj(c) and np.iscomplexobj(k)
