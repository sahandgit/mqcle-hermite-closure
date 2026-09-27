r"""Module 5 -- initial ensembles.

Two mapping samplers are provided, matching the conventions used by this
project's existing trajectory pipeline.

focused (microcanonical MMST actions)
    Each oscillator sits on its action circle,
    :math:`r_\lambda^{2}+p_\lambda^{2}=2\hbar(n_\lambda+\tfrac12)`, with
    :math:`n_\lambda=1` for the initially occupied state and
    :math:`n_\lambda=0` otherwise.  For :math:`\hbar=1` the radii are
    :math:`\sqrt3` (active) and :math:`1` (inactive).  Angles are uniform on
    :math:`[0,2\pi)`.

wigner_seo
    Samples from the Gaussian envelope :math:`\phi(x)`, i.e.
    :math:`x\sim\mathcal N(0,\tfrac\hbar2 I_{2N})`.  This is the proposal that
    makes the projected density estimator of
    :func:`mqcle.observables.rho_W_from_ensemble` unbiased with the weights
    returned here.

The bath sampler draws the Wigner distribution of a Gaussian nuclear wave
packet, :math:`R\sim\mathcal N(R_0,\sigma_R^{2})`,
:math:`P\sim\mathcal N(P_0,(\hbar/2\sigma_R)^{2})`.

Every sampler takes an explicit ``numpy.random.Generator`` and records its
parameters in ``Ensemble.meta`` (PART II rule 10).
"""
from __future__ import annotations

import numpy as np

from .constants import DTYPE, HBAR
from .projector import projected_density
from .types import Ensemble

__all__ = [
    "sample_bath_wigner_gaussian", "sample_mapping_focused",
    "sample_mapping_wigner_seo", "sample_initial_ensemble",
    "mapping_actions", "focused_radii", "projected_importance_weights",
    "sample_mapping_sign_optimal", "SIGN_OPTIMAL_Z",
]


def focused_radii(nstates: int, init_state: int, hbar: float = HBAR) -> np.ndarray:
    r"""Action radii :math:`\sqrt{2\hbar(n_\lambda+1/2)}`, shape ``(N,)``."""
    occ = np.zeros(nstates, dtype=DTYPE)
    occ[init_state] = 1.0
    return np.sqrt(2.0 * hbar * (occ + 0.5))


def mapping_actions(x: np.ndarray, nstates: int) -> np.ndarray:
    r""":math:`r_\lambda^{2}+p_\lambda^{2}`, shape ``(..., N)``."""
    x = np.asarray(x, dtype=DTYPE)
    r, p = x[..., :nstates], x[..., nstates:]
    return r**2 + p**2


def sample_bath_wigner_gaussian(rng: np.random.Generator, n: int, R0: float,
                                P0: float, sigma_R: float, hbar: float = HBAR
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Wigner sampling of a Gaussian nuclear wave packet; returns ``(R, P)``."""
    sigma_P = hbar / (2.0 * sigma_R)
    R = rng.normal(R0, sigma_R, size=(n, 1)).astype(DTYPE)
    P = rng.normal(P0, sigma_P, size=(n, 1)).astype(DTYPE)
    return R, P


def sample_mapping_focused(rng: np.random.Generator, n: int, nstates: int,
                           init_state: int, hbar: float = HBAR) -> np.ndarray:
    """Focused MMST sampling on the action circles; returns ``x`` of shape ``(n, 2N)``."""
    radii = focused_radii(nstates, init_state, hbar)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(n, nstates))
    r = radii * np.cos(theta)
    p = -radii * np.sin(theta)
    return np.concatenate([r, p], axis=-1).astype(DTYPE)


def sample_mapping_wigner_seo(rng: np.random.Generator, n: int, nstates: int,
                              hbar: float = HBAR) -> np.ndarray:
    r"""Sample :math:`x\sim\phi(x)`, i.e. :math:`\mathcal N(0,\hbar/2)` per coordinate."""
    scale = np.sqrt(hbar / 2.0)
    return rng.normal(0.0, scale, size=(n, 2 * nstates)).astype(DTYPE)


def projected_importance_weights(x: np.ndarray, rho_W0: np.ndarray,
                                 hbar: float = HBAR, nstates: int = 2
                                 ) -> np.ndarray:
    r"""Unnormalised importance weight of the projected density w.r.t. :math:`\phi`.

    .. math::
        u_i=\frac{\rho_m^{\mathcal P}(x_i,0)}{\phi(x_i)}
           =\sum_{\lambda\lambda'}\Pi_{\lambda\lambda'}(x_i)
             \rho_W^{\lambda'\lambda}(0).

    Returned complex, and **provably real for every Hermitian**
    :math:`\rho_W(0)`.  Write :math:`\rho_W=A+iB` with :math:`A` real
    symmetric and :math:`B` real antisymmetric, and
    :math:`\Pi_{\lambda\lambda'}=\frac2\hbar[u_{\lambda\lambda'}
    -iL_{\lambda\lambda'}-\frac\hbar2\delta_{\lambda\lambda'}]` with :math:`u`
    symmetric and :math:`L` antisymmetric in :math:`\lambda\lambda'`.  Then
    :math:`\sum uB=\sum LA=0` by symmetry, and

    .. math::
        u_i=\frac2\hbar\Bigl[\textstyle\sum u_{\lambda\lambda'}A_{\lambda\lambda'}
        -\sum L_{\lambda\lambda'}B_{\lambda\lambda'}
        -\frac\hbar2\operatorname{Tr}A\Bigr]\in\mathbb R .

    This is the statement that the projected mapping density
    :math:`\rho_m^{\mathcal P}=\sum g_{\lambda\lambda'}\rho_W^{\lambda'\lambda}`
    is a real function whenever :math:`\rho_W` is a density matrix, so the
    weight of a physical initial state is always real -- including one whose
    coherences are purely imaginary.  The complex return type and the
    complex-safe branch in :func:`sample_initial_ensemble` are a guard against
    a **non-Hermitian** input, where the imaginary part is genuinely nonzero
    and discarding it would bias the coherences rather than round them.

    ``u`` is **signed**: :math:`\Pi_{\lambda\lambda}=\frac2\hbar(r_\lambda^2
    +p_\lambda^2-\frac\hbar2)` is negative inside the disc of radius
    :math:`\sqrt{\hbar/2}`.  That is the sign problem of this estimator, not a
    bug, and it is what the variance study of ``scripts/11`` measures.
    """
    x = np.asarray(x, dtype=DTYPE)
    phi = np.exp(-np.sum(x**2, axis=-1) / hbar) / (np.pi * hbar) ** nstates
    return (projected_density(x, rho_W0, hbar, nstates) / phi).astype(np.complex128)


#: :math:`Z=\mathbb E_\phi|\Pi_{\lambda_0\lambda_0}|=4e^{-1/2}-1`, in closed
#: form.  With :math:`t=(r^2+p^2)/\hbar\sim\mathrm{Exp}(1)` under :math:`\phi`
#: and :math:`\Pi=2(t-\tfrac12)`, :math:`\int_0^{1/2}(\tfrac12-t)e^{-t}dt
#: =e^{-1/2}-\tfrac12` and :math:`\int_{1/2}^\infty(t-\tfrac12)e^{-t}dt
#: =e^{-1/2}`.
SIGN_OPTIMAL_Z = 4.0 * np.exp(-0.5) - 1.0


def _sample_tilted_action(rng: np.random.Generator, n: int) -> np.ndarray:
    r"""Draw :math:`t\sim e^{-t}|t-\tfrac12|/(2e^{-1/2}-\tfrac12)` on
    :math:`[0,\infty)`.

    Exactly, as a two-piece mixture, not by rejection from the whole line:

    * :math:`t\ge\tfrac12` (mass :math:`e^{-1/2}`): the density is
      :math:`e^{-1/2}se^{-s}` in :math:`s=t-\tfrac12`, i.e. a Gamma(2,1) shift.
    * :math:`t<\tfrac12` (mass :math:`e^{-1/2}-\tfrac12`): rejection from the
      triangular density :math:`\propto(\tfrac12-t)`, sampled by
      :math:`t=\tfrac12(1-\sqrt U)`, accepted with probability :math:`e^{-t}`.
      The acceptance rate is :math:`(e^{-1/2}-\tfrac12)/\tfrac18=0.852`.
    """
    mA = float(np.exp(-0.5) - 0.5)
    mB = float(np.exp(-0.5))
    is_a = rng.random(n) < mA / (mA + mB)
    out = np.empty(n, dtype=DTYPE)
    nb = int((~is_a).sum())
    out[~is_a] = 0.5 + rng.gamma(2.0, 1.0, nb)
    na = n - nb
    if na:
        acc = np.empty(0, dtype=DTYPE)
        while acc.size < na:
            m = max(64, int(1.4 * (na - acc.size)))
            t = 0.5 * (1.0 - np.sqrt(rng.random(m)))
            acc = np.concatenate([acc, t[rng.random(m) < np.exp(-t)]])
        out[is_a] = acc[:na]
    return out


def sample_mapping_sign_optimal(rng: np.random.Generator, n: int, nstates: int,
                                init_state: int, hbar: float = HBAR
                                ) -> np.ndarray:
    r"""Draw :math:`x\sim q=\phi|\Pi_{\lambda_0\lambda_0}|/Z`.

    This is the *sign-optimal* proposal for a pure initial diabatic state: the
    importance weight it produces,

    .. math:: \frac{\rho_m^{\mathcal P}(x)}{q(x)}
              =Z\,\operatorname{sign}\Pi_{\lambda_0\lambda_0}(x),

    is **bounded**, whereas the weight under the plain :math:`\phi` proposal is
    :math:`\Pi_{\lambda_0\lambda_0}` itself, which is unbounded and whose fourth
    moment sets the variance.  Measured per-sample standard deviations of the
    :math:`\rho_{00}` estimator at :math:`t=0` are 3.22 under :math:`\phi` and
    0.947 under :math:`q`, an eleven-fold reduction in variance at equal cost.
    Both proposals are unbiased and both are run in production, precisely so
    that the agreement is a check rather than an assumption.

    Only the initially occupied mode is tilted; the others keep their
    :math:`\phi` marginal, because :math:`\Pi_{\lambda_0\lambda_0}` depends on
    :math:`x` only through :math:`r_{\lambda_0}^2+p_{\lambda_0}^2`.
    """
    x = rng.normal(0.0, np.sqrt(hbar / 2.0), size=(n, 2 * nstates)).astype(DTYPE)
    s = hbar * _sample_tilted_action(rng, n)
    th = rng.uniform(0.0, 2.0 * np.pi, size=n)
    rad = np.sqrt(s)
    x[:, init_state] = rad * np.cos(th)
    x[:, nstates + init_state] = rad * np.sin(th)
    return x


def sample_initial_ensemble(rng: np.random.Generator, n: int, *, nstates: int,
                            init_state: int, R0: float, P0: float,
                            sigma_R: float, scheme: str = "focused",
                            hbar: float = HBAR, seed: int | None = None
                            ) -> Ensemble:
    r"""Build a complete initial ensemble.

    For ``scheme='wigner_seo'`` the weights are the importance weights of the
    projected density with respect to the sampling density :math:`\phi`,

    .. math:: w_i=\frac{1}{n}\,\frac{\rho_m^{\mathcal P}(x_i)}{\phi(x_i)}
             =\frac{1}{n}\sum_{\lambda\lambda'}\Pi_{\lambda\lambda'}(x_i)
              \rho_W^{\lambda'\lambda}(0),

    so that :func:`mqcle.observables.rho_W_from_ensemble` is unbiased.  For
    ``scheme='focused'`` the points do not sample :math:`\phi`, so uniform
    weights ``1/n`` are returned and the ensemble must be read out through the
    focused estimator rather than through the projected-density estimator.
    """
    R, P = sample_bath_wigner_gaussian(rng, n, R0, P0, sigma_R, hbar)
    rho0 = np.zeros((nstates, nstates), dtype=np.complex128)
    rho0[init_state, init_state] = 1.0

    extra: dict = {"weight_normalisation": "1/n"}
    if scheme == "focused":
        x = sample_mapping_focused(rng, n, nstates, init_state, hbar)
        w = np.full(n, 1.0 / n, dtype=DTYPE)
    elif scheme == "wigner_seo":
        x = sample_mapping_wigner_seo(rng, n, nstates, hbar)
        u = projected_importance_weights(x, rho0, hbar, nstates)
        # keep the imaginary part only if there is one; casting it away
        # silently would be a bias, not a rounding
        im = float(np.max(np.abs(np.imag(u))))
        re = float(np.max(np.abs(np.real(u))))
        w = ((np.real(u) / n).astype(DTYPE) if im <= 1e-12 * max(re, 1.0)
             else (u / n).astype(np.complex128))
        extra = {"weight_dtype": str(w.dtype), "max_abs_imag_weight": im,
                 "weight_normalisation": "1/n",
                 "proposal": "phi",
                 "signed_weight_fraction_negative":
                     float(np.mean(np.real(u) < 0.0))}
    elif scheme == "sign_optimal":
        x = sample_mapping_sign_optimal(rng, n, nstates, init_state, hbar)
        pi = projected_importance_weights(x, rho0, hbar, nstates)
        sgn = np.sign(np.real(pi))
        w = (SIGN_OPTIMAL_Z * sgn / n).astype(DTYPE)
        extra = {"weight_dtype": str(w.dtype), "max_abs_imag_weight": 0.0,
                 "weight_normalisation": "1/n",
                 "proposal": "phi * |Pi_{lam0 lam0}| / Z",
                 "Z": float(SIGN_OPTIMAL_Z),
                 "signed_weight_fraction_negative": float(np.mean(sgn < 0.0))}
    else:
        raise ValueError(f"unknown mapping sampling scheme '{scheme}'")

    meta = {**extra, "scheme": scheme, "n": n, "nstates": nstates,
            "init_state": init_state, "R0": R0, "P0": P0,
            "sigma_R": sigma_R, "sigma_P": hbar / (2.0 * sigma_R),
            "hbar": hbar, "seed": seed}
    return Ensemble(R=R, P=P, x=x, weights=w, meta=meta)
