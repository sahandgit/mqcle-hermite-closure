r"""Module 3 -- the exact projected mapping kernel.

Definitions (Kelly, van Zon, Schofield & Kapral, J. Chem. Phys. 136, 084101
(2012), Eqs. (11) and (13)):

.. math::
    \phi(x)=\frac{1}{(\pi\hbar)^{N}}
        \exp\Bigl[-\frac1\hbar\sum_\mu(r_\mu^{2}+p_\mu^{2})\Bigr],

.. math::
    \Pi_{\lambda\lambda'}(x)=\frac{2}{\hbar}
        \Bigl[u_{\lambda\lambda'}-iL_{\lambda\lambda'}
        -\frac{\hbar}{2}\delta_{\lambda\lambda'}\Bigr],
    \qquad g_{\lambda\lambda'}=\phi\,\Pi_{\lambda\lambda'},

.. math::
    c_{\lambda\lambda'}(x)=\bigl(\hat a^{\dagger}_\lambda\hat a_{\lambda'}\bigr)_W
      =\frac{1}{2\hbar}\Bigl[u_{\lambda\lambda'}+iL_{\lambda\lambda'}
      -\hbar\delta_{\lambda\lambda'}\Bigr],

with :math:`u_{\lambda\lambda'}=r_\lambda r_{\lambda'}+p_\lambda p_{\lambda'}`
and :math:`L_{\lambda\lambda'}=r_\lambda p_{\lambda'}-r_{\lambda'}p_\lambda`.

The normalisation is fixed by the two published orthogonality identities

.. math::
    \int\!dx\;g_{\lambda\lambda'}(x)\,c_{\nu\nu'}(x)
        =\delta_{\lambda\nu}\delta_{\lambda'\nu'},
.. math::
    (2\pi\hbar)^{N}\!\int\!dx\;g_{\lambda\lambda'}(x)g_{\nu'\nu}(x)
        =\delta_{\lambda\nu}\delta_{\lambda'\nu'},

both of which are checked to quadrature precision in
``tests/test_projector.py``.  The projected mapping density is

.. math::
    \rho_m^{\mathcal P}(x,X)=\sum_{\lambda\lambda'}
        g_{\lambda\lambda'}(x)\,\rho_W^{\lambda'\lambda}(X).
"""
from __future__ import annotations

import numpy as np

from .constants import CDTYPE, DTYPE, HBAR
from .mapping import mapping_bilinears, split_mapping

__all__ = [
    "gaussian_mapping_envelope", "mapping_antisymmetric",
    "projector_polynomial", "mapping_kernel", "mapping_kernel_all",
    "wigner_aa_dagger", "wigner_aa_dagger_all", "projected_density",
]


def gaussian_mapping_envelope(x: np.ndarray, hbar: float = HBAR,
                              nstates: int | None = None) -> np.ndarray:
    r"""Normalised Gaussian envelope :math:`\phi(x)`, shape ``(...,)``."""
    x = np.asarray(x, dtype=DTYPE)
    n = x.shape[-1] // 2 if nstates is None else nstates
    x2 = np.sum(x**2, axis=-1)
    return np.exp(-x2 / hbar) / (np.pi * hbar) ** n


def mapping_antisymmetric(x: np.ndarray, nstates: int) -> np.ndarray:
    r""":math:`L_{\lambda\lambda'}=r_\lambda p_{\lambda'}-r_{\lambda'}p_\lambda`,
    shape ``(..., N, N)``."""
    r, p = split_mapping(x, nstates)
    return r[..., :, None] * p[..., None, :] - r[..., None, :] * p[..., :, None]


def projector_polynomial(x: np.ndarray, lam: int, lamp: int,
                         hbar: float = HBAR, nstates: int = 2) -> np.ndarray:
    r"""Single element :math:`\Pi_{\lambda\lambda'}(x)`, complex, shape ``(...,)``."""
    return projector_polynomial_all(x, hbar, nstates)[..., lam, lamp]


def projector_polynomial_all(x: np.ndarray, hbar: float = HBAR,
                             nstates: int = 2) -> np.ndarray:
    r"""All of :math:`\Pi_{\lambda\lambda'}(x)`, shape ``(..., N, N)``, complex."""
    u = mapping_bilinears(x, nstates).astype(CDTYPE)
    L = mapping_antisymmetric(x, nstates).astype(CDTYPE)
    eye = np.eye(nstates, dtype=CDTYPE)
    return (2.0 / hbar) * (u - 1j * L - 0.5 * hbar * eye)


def mapping_kernel(x: np.ndarray, lam: int, lamp: int,
                   hbar: float = HBAR, nstates: int = 2) -> np.ndarray:
    r"""Single element :math:`g_{\lambda\lambda'}(x)=\phi\Pi_{\lambda\lambda'}`."""
    return mapping_kernel_all(x, hbar, nstates)[..., lam, lamp]


def mapping_kernel_all(x: np.ndarray, hbar: float = HBAR,
                       nstates: int = 2) -> np.ndarray:
    r"""All of :math:`g_{\lambda\lambda'}(x)`, shape ``(..., N, N)``, complex."""
    phi = gaussian_mapping_envelope(x, hbar, nstates)
    return phi[..., None, None] * projector_polynomial_all(x, hbar, nstates)


def wigner_aa_dagger_all(x: np.ndarray, hbar: float = HBAR,
                         nstates: int = 2) -> np.ndarray:
    r"""All of :math:`c_{\lambda\lambda'}(x)`, shape ``(..., N, N)``, complex."""
    u = mapping_bilinears(x, nstates).astype(CDTYPE)
    L = mapping_antisymmetric(x, nstates).astype(CDTYPE)
    eye = np.eye(nstates, dtype=CDTYPE)
    return (u + 1j * L - hbar * eye) / (2.0 * hbar)


def wigner_aa_dagger(x: np.ndarray, lam: int, lamp: int,
                     hbar: float = HBAR, nstates: int = 2) -> np.ndarray:
    r"""Single element :math:`c_{\lambda\lambda'}(x)`."""
    return wigner_aa_dagger_all(x, hbar, nstates)[..., lam, lamp]


def projected_density(x: np.ndarray, rho_W: np.ndarray,
                      hbar: float = HBAR, nstates: int = 2) -> np.ndarray:
    r"""Projected mapping density :math:`\rho_m^{\mathcal P}(x,X)`.

    Parameters
    ----------
    x
        Mapping coordinates, ``(..., 2N)``.
    rho_W
        Subsystem density matrix ``rho_W[..., lam, lamp]``, broadcastable
        against the leading axes of ``x``.

    Notes
    -----
    The contraction pairs :math:`g_{\lambda\lambda'}` with
    :math:`\rho_W^{\lambda'\lambda}`, i.e. it is a trace, not an
    element-by-element product.  Getting this index order wrong flips the sign
    of the imaginary part of the coherence.
    """
    g = mapping_kernel_all(x, hbar, nstates)
    rho_W = np.asarray(rho_W, dtype=CDTYPE)
    return np.einsum("...ij,...ji->...", g, rho_W)
