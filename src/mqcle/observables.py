r"""Module 4 -- physical subsystem observables.

The subsystem density matrix is recovered from a mapping-space density by the
orthogonality relation of :mod:`mqcle.projector`,

.. math::
    \rho_W^{\nu'\nu}(X)=\int\!dx\;\rho_m^{\mathcal P}(x,X)\,c_{\nu\nu'}(x),

which follows from :math:`\int dx\,g_{\lambda\lambda'}c_{\nu\nu'}
=\delta_{\lambda\nu}\delta_{\lambda'\nu'}`.  Three entry points are provided,
differing only in how the integral is discretised:

``rho_W_from_quadrature``   tensor-product Gauss-Hermite nodes (exact for
                            Gaussian-times-polynomial integrands);
``rho_W_from_grid``         caller-supplied nodes and quadrature weights;
``rho_W_from_ensemble``     weighted Monte-Carlo samples.

The Hermite-coefficient entry point lives with the Hermite algebra and is
implemented in Module 19 (:mod:`mqcle.hermite.projection`); see
``MODULE_STATUS.md``.
"""
from __future__ import annotations

from itertools import product as _iproduct

import numpy as np

from .constants import CDTYPE, DTYPE, HBAR
from .projector import wigner_aa_dagger_all

__all__ = [
    "gauss_hermite_weights", "lebesgue_nodes", "phi_expectation_nodes",
    "rho_W_from_quadrature", "rho_W_from_grid", "rho_W_from_ensemble",
    "rho_W_from_focused_ensemble",
    "observables_from_rho_W", "hermiticity_error", "trace_norm",
    "populations", "coherence",
]


# --------------------------------------------------------------------------
# quadrature helpers
# --------------------------------------------------------------------------
def gauss_hermite_weights(dim: int, a: float, npts: int = 12
                          ) -> tuple[np.ndarray, np.ndarray]:
    r"""Nodes and weights for :math:`\int dx\,e^{-a|x|^{2}}q(x)\approx\sum_iW_iq(X_i)`.

    Exact for polynomials ``q`` of degree ``<= 2*npts - 1`` in each coordinate.
    """
    t, w = np.polynomial.hermite.hermgauss(npts)
    nodes1d = t / np.sqrt(a)
    scale = a ** (-dim / 2.0)
    grids = np.array(list(_iproduct(nodes1d, repeat=dim)), dtype=DTYPE)
    wts = np.array([np.prod(ws) for ws in _iproduct(w, repeat=dim)], dtype=DTYPE)
    return grids, scale * wts


def lebesgue_nodes(dim: int, a: float, npts: int = 12
                   ) -> tuple[np.ndarray, np.ndarray]:
    r"""Nodes and *plain* ``dx`` weights: :math:`\int dx\,f(x)\approx\sum_iW_if(X_i)`.

    Exact whenever ``f(x) = exp(-a|x|^2) * q(x)`` with ``q`` polynomial of
    degree ``<= 2*npts - 1`` per coordinate -- which is exactly the form of the
    projected mapping density and of the kernel products ``g c`` and ``g g``.
    Obtained from :func:`gauss_hermite_weights` by dividing the Gaussian weight
    back out; use this, not :func:`gauss_hermite_weights`, whenever the
    integrand already contains its own Gaussian factor.
    """
    X, W = gauss_hermite_weights(dim, a, npts)
    return X, W * np.exp(a * np.sum(X**2, axis=-1))


def phi_expectation_nodes(nstates: int, hbar: float = HBAR, npts: int = 12
                          ) -> tuple[np.ndarray, np.ndarray]:
    r"""Nodes/weights for :math:`\int dx\,\phi(x)q(x)`; the weights sum to one."""
    X, W = gauss_hermite_weights(2 * nstates, 1.0 / hbar, npts)
    return X, W / (np.pi * hbar) ** nstates


# --------------------------------------------------------------------------
# density-matrix reconstruction
# --------------------------------------------------------------------------
def rho_W_from_grid(x: np.ndarray, rho_values: np.ndarray,
                    quad_weights: np.ndarray, hbar: float = HBAR,
                    nstates: int = 2) -> np.ndarray:
    r"""Reconstruct :math:`\rho_W` from node values of the mapping density.

    Parameters
    ----------
    x
        Nodes, ``(M, 2N)``.
    rho_values
        Mapping density at the nodes, ``(..., M)``.
    quad_weights
        Plain ``dx`` quadrature weights, ``(M,)`` -- e.g. from
        :func:`lebesgue_nodes`.  The mapping density supplied in
        ``rho_values`` already contains its Gaussian envelope, so weights that
        carry a Gaussian factor of their own would double-count it.

    Returns
    -------
    ``(..., N, N)`` complex array with ``rho_W[..., lam, lamp]``.
    """
    c = wigner_aa_dagger_all(x, hbar, nstates)              # (M, N, N)
    vals = np.asarray(rho_values, dtype=CDTYPE)
    w = np.asarray(quad_weights, dtype=DTYPE)
    # integral_{nu nu'} = sum_M w_M rho_M c_{nu nu'} equals rho_W^{nu' nu}
    integral = np.einsum("...m,m,mij->...ij", vals, w, c)
    return np.swapaxes(integral, -1, -2)


def rho_W_from_quadrature(rho_fn, hbar: float = HBAR, nstates: int = 2,
                          npts: int = 12, **kwargs) -> np.ndarray:
    """Reconstruct :math:`\\rho_W` from a callable ``rho_fn(x) -> (..., M)``."""
    X, W = lebesgue_nodes(2 * nstates, 1.0 / hbar, npts)
    return rho_W_from_grid(X, rho_fn(X, **kwargs), W, hbar, nstates)


def rho_W_from_ensemble(x: np.ndarray, weights: np.ndarray,
                        hbar: float = HBAR, nstates: int = 2) -> np.ndarray:
    r"""Monte-Carlo estimator of :math:`\rho_W` from a weighted ensemble.

    ``weights[i]`` must already be the importance weight
    :math:`\rho_m^{\mathcal P}(x_i)/(n\,q(x_i))` for the proposal ``q`` the
    points were drawn from, so that the estimator is
    :math:`\sum_i w_i c_{\nu\nu'}(x_i)`.
    """
    c = wigner_aa_dagger_all(x, hbar, nstates)
    w = np.asarray(weights, dtype=CDTYPE)
    integral = np.einsum("m,mij->ij", w, c)
    return integral.T


def rho_W_from_focused_ensemble(x: np.ndarray, weights: np.ndarray | None = None,
                                hbar: float = HBAR, nstates: int = 2
                                ) -> np.ndarray:
    r"""Read out :math:`\rho_W` from a **focused** MMST ensemble.

    Focused sampling does not draw from :math:`\phi(x)`, so the importance
    weights of :func:`rho_W_from_ensemble` do not apply and must not be reused.
    What focused sampling does provide is the correct *second moments*: with
    action radii :math:`r_\lambda^2+p_\lambda^2=2\hbar(n_\lambda+1/2)` and
    uniform independent angles,

    .. math::
        \langle r_\lambda r_{\lambda'}\rangle
        =\langle p_\lambda p_{\lambda'}\rangle
        =\tfrac12\bigl(2\hbar(n_\lambda+\tfrac12)\bigr)\delta_{\lambda\lambda'},
        \qquad
        \langle r_\lambda p_{\lambda'}\rangle=0,

    which are exactly the second moments of the projected density
    :math:`\phi\Pi_{\lambda\lambda'}` for the corresponding pure state.
    Since :math:`c_{\lambda\lambda'}` is quadratic, the plain average

    .. math:: \rho_W^{\nu'\nu}=\sum_i w_i\,c_{\nu\nu'}(x_i)

    reproduces the initial density matrix exactly, and remains exact under any
    *linear* mapping evolution.  For R-dependent dynamics it is an
    approximation, because the ensemble is not a quadrature rule for
    :math:`\phi` beyond second moments.

    ``weights`` defaults to ``1/n``.
    """
    x = np.asarray(x, dtype=DTYPE)
    n = x.shape[0]
    w = (np.full(n, 1.0 / n, dtype=DTYPE) if weights is None
         else np.asarray(weights, dtype=DTYPE))
    c = wigner_aa_dagger_all(x, hbar, nstates)
    return np.einsum("m,mij->ij", w.astype(CDTYPE), c).T


# --------------------------------------------------------------------------
# scalar observables
# --------------------------------------------------------------------------
def populations(rho_W: np.ndarray) -> np.ndarray:
    """Diagonal of ``rho_W``, real part, shape ``(..., N)``."""
    return np.real(np.diagonal(np.asarray(rho_W), axis1=-2, axis2=-1))


def coherence(rho_W: np.ndarray, lam: int = 0, lamp: int = 1
              ) -> tuple[np.ndarray, np.ndarray]:
    """Real and imaginary parts of ``rho_W[..., lam, lamp]``."""
    z = np.asarray(rho_W, dtype=CDTYPE)[..., lam, lamp]
    return np.real(z), np.imag(z)


def trace_norm(rho_W: np.ndarray) -> np.ndarray:
    """Trace of ``rho_W`` (should be 1 for a normalised subsystem state)."""
    return np.real(np.trace(np.asarray(rho_W), axis1=-2, axis2=-1))


def hermiticity_error(rho_W: np.ndarray) -> float:
    """Max-norm of ``rho_W - rho_W^dagger``."""
    a = np.asarray(rho_W, dtype=CDTYPE)
    return float(np.max(np.abs(a - np.conj(np.swapaxes(a, -1, -2)))))


def observables_from_rho_W(rho_W: np.ndarray) -> dict[str, np.ndarray]:
    """The four reported two-state observables plus diagnostics (PART XXX)."""
    pops = populations(rho_W)
    re01, im01 = coherence(rho_W, 0, 1)
    return {
        "rho00": pops[..., 0],
        "rho11": pops[..., 1],
        "re_rho01": re01,
        "im_rho01": im01,
        "trace": trace_norm(rho_W),
        "hermiticity_error": hermiticity_error(rho_W),
    }
