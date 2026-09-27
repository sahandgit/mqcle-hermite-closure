r"""Gate M -- the PBME Hermite-leakage functional, and its estimability.

The quantity
-------------

.. math::
    \mathcal D_H(t)=\bigl\|(I-\mathsf P_2)\,\mathcal A_{\rm PB}
                     \rho_m^{\mathcal P}(t)\bigr\|_\varphi

measures how far one application of the PBME generator pushes the projected
mapping density out of the Gaussian-Hermite space :math:`V_2` that the *full*
generator preserves exactly.  Its closed form is verified symbolically in
``results/raw/02_hermite_projection.json``: for :math:`N=2` exactly 31
degree-4 multi-indices survive, and every one of their coefficients is a
bilinear form in :math:`\nabla_R h^{\lambda\lambda'}` and
:math:`\nabla_P\rho_W^{\nu\nu'}`.

The norm
--------

On amplitudes :math:`q=\rho/\varphi` the Hermite functions are orthogonal with
:math:`\langle\mathrm{He}_{\mathbf m},\mathrm{He}_{\mathbf n}\rangle_\varphi
=\mathbf n!\,\delta_{\mathbf{mn}}`, so for
:math:`q=\sum_{\mathbf n}c_{\mathbf n}(R,P)\,\mathrm{He}_{\mathbf n}(z)`

.. math::
    \|q\|^2=\int\!dR\,dP\sum_{\mathbf n}\mathbf n!\,|c_{\mathbf n}(R,P)|^2 .

The mapping norm is the :math:`\varphi`-weighted one; the bath directions
carry plain Lebesgue measure, because nothing Gaussian is assumed of them.
:func:`leakage_from_grid` returns both the raw norm and the ratio to
:math:`\|\rho_m^{\mathcal P}\|`, and the manuscript must state which is
plotted.

The obstruction this module exists to measure
---------------------------------------------

:math:`\mathcal D_H` needs :math:`\nabla_P\rho_W`.  A trajectory ensemble does
not supply it.  Two routes are therefore implemented and compared:

* **Route A**, :func:`leakage_from_grid` -- spectral :math:`\nabla_P` on the
  converged coefficient grid.  Ground truth.
* **Route B**, :func:`kde_grad_P_rho_W` -- a kernel estimate of
  :math:`\nabla_P\rho_W` from a finite PBME ensemble, which is what a
  trajectory method would actually have to do.

Route B is expected to be noisy: this project has already measured that
operator convergence does not follow from value convergence for third-order
operators.  The measurement is the result, whichever way it comes out.
"""
from __future__ import annotations

import functools
from typing import Any, Callable

import numpy as np
import sympy as sp

from .coefficient_operator import Grid, spectral_derivative
from .constants import CDTYPE, DTYPE, HBAR
from .hermite.basis import multi_factorial
from .hermite.projection import hermite_coefficients_symbolic
from .mapping import subsystem_split
from .models import TwoStateModel
from .projector import wigner_aa_dagger_all
from .types import TraceConvention

__all__ = [
    "leakage_coefficient_forms",
    "amplitude_coefficient_forms",
    "leakage_from_grid",
    "leakage_from_fields",
    "kde_grad_P_rho_W",
    "kde_rho_W",
    "silverman_bandwidth",
]


# --------------------------------------------------------------------------
# symbolic layer, compiled once
# --------------------------------------------------------------------------
def _sym_names(nstates: int) -> tuple[list[str], list[str]]:
    dh = [f"dh{min(a,b)}{max(a,b)}" for a in range(nstates) for b in range(a, nstates)]
    dr = [f"dr{a}{b}" for a in range(nstates) for b in range(nstates)]
    return dh, dr


def _compile(coeffs: dict[tuple, sp.Expr], s, nstates: int
             ) -> tuple[list[tuple], Callable]:
    """Lambdify ``{alpha: c_alpha}`` into one vectorised callable."""
    R0 = s.R[0]
    P0 = s.P[0]
    subs: dict[Any, Any] = {s.hbar: 1}
    dh_syms, dr_syms = [], []
    for a in range(nstates):
        for b in range(a, nstates):
            sym = sp.Symbol(f"dh{a}{b}", real=True)
            subs[sp.Derivative(sp.Function(f"h{a}{b}")(R0), R0)] = sym
            dh_syms.append(sym)
    for a in range(nstates):
        for b in range(nstates):
            sym = sp.Symbol(f"dr{a}{b}")
            subs[sp.Derivative(sp.Function(f"rho{a}{b}")(R0, P0), P0)] = sym
            dr_syms.append(sym)

    keys = sorted(coeffs)
    exprs = [sp.expand(coeffs[k].subs(subs)) for k in keys]
    leftover = set()
    for e in exprs:
        leftover |= e.atoms(sp.Derivative) | (e.free_symbols
                                              - set(dh_syms) - set(dr_syms))
    if leftover:
        raise RuntimeError(f"leakage: unsubstituted symbols {sorted(map(str, leftover))}")
    fn = sp.lambdify(tuple(dh_syms + dr_syms), exprs, "numpy")
    return keys, fn


@functools.lru_cache(maxsize=8)
def leakage_coefficient_forms(nstates: int = 2, nbath: int = 1,
                              convention: str = "traceless"):
    r"""Hermite coefficients of :math:`(I-\mathsf P_2)\mathcal A_{\rm PB}
    \rho_m^{\mathcal P}/\varphi`, compiled for numerical evaluation.

    Returns ``(multi_indices, callable)``; the callable takes
    ``(dh00, dh01, ..., dr00, dr01, ...)`` as broadcastable arrays and returns
    a list of coefficient arrays, one per multi-index.
    """
    from .validation import build_symbolic_setup, pbme_hermite_defect
    s = build_symbolic_setup(nstates, nbath, convention)
    q = sp.expand(pbme_hermite_defect(s, max_order=2))
    coeffs = hermite_coefficients_symbolic(q, s.z)
    return _compile(coeffs, s, nstates)


@functools.lru_cache(maxsize=8)
def amplitude_coefficient_forms(nstates: int = 2, nbath: int = 1,
                                convention: str = "traceless"):
    r"""Hermite coefficients of :math:`\rho_m^{\mathcal P}/\varphi` itself.

    Used as the normalising denominator.  These depend on ``rho_W``, not on its
    ``P``-derivative, so the compiled callable takes the ``rho`` entries in the
    ``dr`` slots.
    """
    from .validation import amplitude, build_symbolic_setup
    s = build_symbolic_setup(nstates, nbath, convention)
    q = sp.expand(amplitude(s, s.rho_m))
    coeffs = hermite_coefficients_symbolic(q, s.z)
    R0, P0 = s.R[0], s.P[0]
    subs: dict[Any, Any] = {s.hbar: 1}
    syms = []
    for a in range(nstates):
        for b in range(nstates):
            sym = sp.Symbol(f"dr{a}{b}")
            subs[sp.Function(f"rho{a}{b}")(R0, P0)] = sym
            syms.append(sym)
    keys = sorted(coeffs)
    exprs = [sp.expand(coeffs[k].subs(subs)) for k in keys]
    return keys, sp.lambdify(tuple(syms), exprs, "numpy")


# --------------------------------------------------------------------------
# numerical evaluation
# --------------------------------------------------------------------------
def _weighted_norm(keys, values, cell_area: float) -> float:
    r""":math:`\bigl(\int dR\,dP\sum_\alpha\alpha!\,|c_\alpha|^2\bigr)^{1/2}`."""
    total = 0.0
    for k, v in zip(keys, values):
        total += multi_factorial(k) * float(np.sum(np.abs(np.asarray(v)) ** 2))
    return float(np.sqrt(total * cell_area))


def leakage_from_fields(dh: np.ndarray, drho_dP: np.ndarray, cell_area: float,
                        nstates: int = 2, nbath: int = 1,
                        convention: str = "traceless") -> dict[str, Any]:
    r"""Leakage norm from explicit ``grad_R h`` and ``grad_P rho_W`` fields.

    Parameters
    ----------
    dh
        ``(n_R, N, N)`` gradient of the *mapping* matrix, i.e.
        ``subsystem_split(...).dh_map_dR[:, 0]``.
    drho_dP
        ``(n_R, n_P, N, N)`` gradient of the coefficients with respect to ``P``.
    """
    keys, fn = leakage_coefficient_forms(nstates, nbath, convention)
    dh_args = [dh[:, None, a, b] for a in range(nstates) for b in range(a, nstates)]
    dr_args = [drho_dP[..., a, b] for a in range(nstates) for b in range(nstates)]
    vals = fn(*dh_args, *dr_args)
    per_alpha = {tuple(int(i) for i in k): float(
        np.sqrt(multi_factorial(k) * np.sum(np.abs(np.asarray(v)) ** 2) * cell_area))
        for k, v in zip(keys, vals)}
    return {"norm": _weighted_norm(keys, vals, cell_area),
            "n_multi_indices": len(keys),
            "per_multi_index": per_alpha}


def leakage_from_grid(rho_W: np.ndarray, grid: Grid, model: TwoStateModel,
                      hbar: float = HBAR,
                      convention: TraceConvention = TraceConvention.TRACELESS,
                      drho_dP: np.ndarray | None = None) -> dict[str, Any]:
    r"""Route A: leakage from the coefficient grid, with spectral
    :math:`\nabla_P`.

    ``drho_dP`` may be supplied to substitute a Route-B estimate for the exact
    spectral derivative; everything else is then identical, which is what makes
    the two routes comparable.
    """
    rho = np.asarray(rho_W, dtype=CDTYPE)
    split = subsystem_split(model, grid.R[:, None], convention)
    dh = np.ascontiguousarray(split.dh_map_dR[:, 0])              # (n_R, N, N)
    if drho_dP is None:
        drho_dP = spectral_derivative(rho, grid.kP, axis=1)

    out = leakage_from_fields(dh, drho_dP, grid.cell_area, model.nstates,
                              model.nbath, convention.value)

    # normalising denominator: the norm of the projected density itself
    keys_a, fn_a = amplitude_coefficient_forms(model.nstates, model.nbath,
                                               convention.value)
    n = model.nstates
    vals_a = fn_a(*[rho[..., a, b] for a in range(n) for b in range(n)])
    out["rho_norm"] = _weighted_norm(keys_a, vals_a, grid.cell_area)
    out["relative"] = out["norm"] / out["rho_norm"] if out["rho_norm"] else float("nan")
    return out


# --------------------------------------------------------------------------
# Route B: sample-based estimation of grad_P rho_W
# --------------------------------------------------------------------------
def silverman_bandwidth(samples: np.ndarray) -> float:
    r"""Silverman's rule for a 1D sample, :math:`1.06\,\hat\sigma n^{-1/5}`.

    Reported, never silently applied: the bandwidth is the single most
    consequential choice in Route B and every run records the value used.
    """
    x = np.asarray(samples, dtype=DTYPE).ravel()
    n = x.size
    sd = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    scale = min(sd, iqr / 1.349) if iqr > 0 else sd
    return float(1.06 * scale * n ** (-0.2))


def _kde_weights(grid_axis: np.ndarray, samples: np.ndarray, h: float,
                 derivative: bool = False) -> np.ndarray:
    r"""Gaussian kernel matrix ``(n_grid, n_samples)``; the derivative variant
    returns :math:`\partial_u K_h(u-u_i)`."""
    u = (grid_axis[:, None] - samples[None, :]) / h
    K = np.exp(-0.5 * u * u) / (h * np.sqrt(2.0 * np.pi))
    return (-u / h) * K if derivative else K


def _deposit(grid: Grid, R: np.ndarray, P: np.ndarray, wc: np.ndarray
             ) -> np.ndarray:
    """Cloud-in-cell deposition of matrix-valued weights onto the grid.

    Bilinear (not nearest-neighbour) so that the deposited field is continuous
    in the sample positions; the residual binning smoothing is one cell wide
    and is dominated by the kernel bandwidth in every configuration used here.
    """
    n_R, n_P = grid.shape
    fi = (R - grid.R[0]) / grid.dR
    fj = (P - grid.P[0]) / grid.dP
    i0 = np.floor(fi).astype(np.int64)
    j0 = np.floor(fj).astype(np.int64)
    a, b = fi - i0, fj - j0
    out = np.zeros((n_R, n_P) + wc.shape[1:], dtype=wc.dtype)
    for di, wi in ((0, 1.0 - a), (1, a)):
        ii = np.mod(i0 + di, n_R)
        for dj, wj in ((0, 1.0 - b), (1, b)):
            jj = np.mod(j0 + dj, n_P)
            np.add.at(out, (ii, jj), (wi * wj)[:, None, None] * wc)
    return out / grid.cell_area


def _cic_window(k: np.ndarray, dx: float) -> np.ndarray:
    r"""Fourier transform of the cloud-in-cell deposition window.

    Linear (first-order B-spline) deposition convolves the point measure with a
    triangle of base :math:`2\Delta x`, whose transform is
    :math:`\mathrm{sinc}^2(k\Delta x/2)`.  Dividing it out removes the binning
    bias exactly; at the Nyquist wavenumber the factor is only
    :math:`(2/\pi)^2\approx0.405`, so the deconvolution never blows up.
    """
    x = 0.5 * np.asarray(k, dtype=DTYPE) * dx
    w = np.ones_like(x)
    nz = x != 0.0
    w[nz] = np.sin(x[nz]) / x[nz]
    return w ** 2


def _gaussian_kernel_fft(grid: Grid, h_R: float, h_P: float,
                         d_dP: bool, deconvolve: bool = True) -> np.ndarray:
    """Fourier transform of the (optionally P-differentiated) product kernel,
    divided by the deposition window so the binning introduces no extra bias."""
    kR, kP = grid.kR, grid.kP
    gR = np.exp(-0.5 * (kR * h_R) ** 2)
    gP = np.exp(-0.5 * (kP * h_P) ** 2)
    if d_dP:
        kk = np.array(kP, dtype=DTYPE).copy()
        if kk.size % 2 == 0:
            kk[kk.size // 2] = 0.0
        gP = 1j * kk * gP
    if deconvolve:
        gR = gR / _cic_window(kR, grid.dR)
        gP = gP / _cic_window(kP, grid.dP)
    return gR[:, None] * gP[None, :]


def _kde_binned(grid: Grid, R, P, wc, h_R: float, h_P: float,
                d_dP: bool) -> np.ndarray:
    """Binned KDE: deposit, then convolve on the periodic grid with FFT.

    Cost is O(n + n_R n_P log n_R n_P) instead of O(n_R n_P n), which is what
    makes the sample-size scaling study in ``scripts/08`` affordable at
    n = 10^5 on a 1024 x 256 grid.
    """
    dep = _deposit(grid, R, P, wc)
    G = _gaussian_kernel_fft(grid, h_R, h_P, d_dP)
    F = np.fft.fft2(dep, axes=(0, 1))
    return np.fft.ifft2(F * G[:, :, None, None], axes=(0, 1))


def kde_rho_W(ensemble_x: np.ndarray, R: np.ndarray, P: np.ndarray, grid: Grid,
              hbar: float = HBAR, nstates: int = 2,
              weights: np.ndarray | None = None,
              h_R: float | None = None, h_P: float | None = None,
              chunk: int = 4096, method: str = "binned"
              ) -> tuple[np.ndarray, dict[str, float]]:
    r"""Kernel estimate of :math:`\rho_W(R,P)` from a focused PBME ensemble.

    Each trajectory carries the matrix :math:`c_{\nu\nu'}(x_i)` -- the same
    quadratic readout as :func:`mqcle.observables.rho_W_from_focused_ensemble`
    -- smeared over the bath plane by a product Gaussian kernel.
    """
    x = np.asarray(ensemble_x, dtype=DTYPE)
    Rs = np.asarray(R, dtype=DTYPE).ravel()
    Ps = np.asarray(P, dtype=DTYPE).ravel()
    n = x.shape[0]
    w = (np.full(n, 1.0 / n) if weights is None
         else np.asarray(weights, dtype=DTYPE) / np.sum(weights))
    h_R = float(h_R or silverman_bandwidth(Rs))
    h_P = float(h_P or silverman_bandwidth(Ps))

    c = np.swapaxes(wigner_aa_dagger_all(x, hbar, nstates), -1, -2)   # per-traj rho
    if method == "binned":
        out = _kde_binned(grid, Rs, Ps, w[:, None, None] * c, h_R, h_P, False)
    elif method == "direct":
        out = np.zeros(grid.shape + (nstates, nstates), dtype=CDTYPE)
        for s0 in range(0, n, chunk):
            s1 = min(n, s0 + chunk)
            KR = _kde_weights(grid.R, Rs[s0:s1], h_R)
            KP = _kde_weights(grid.P, Ps[s0:s1], h_P)
            wc = w[s0:s1, None, None] * c[s0:s1]
            out += np.einsum("im,jm,mab->ijab", KR, KP, wc, optimize=True)
    else:
        raise ValueError(f"unknown KDE method '{method}'")
    return out, {"h_R": h_R, "h_P": h_P, "n_samples": int(n), "method": method}


def kde_grad_P_rho_W(ensemble_x: np.ndarray, R: np.ndarray, P: np.ndarray,
                     grid: Grid, hbar: float = HBAR, nstates: int = 2,
                     weights: np.ndarray | None = None,
                     h_R: float | None = None, h_P: float | None = None,
                     chunk: int = 4096, method: str = "binned"
                     ) -> tuple[np.ndarray, dict[str, float]]:
    r"""Route B: kernel estimate of :math:`\nabla_P\rho_W` from the ensemble.

    The derivative is taken analytically on the kernel, not by differencing the
    estimated density, which is the best case for a sample-based estimator.
    Even so the estimator inherits the bandwidth bias of the kernel; the study
    in ``scripts/08`` measures it rather than assuming it away.
    """
    x = np.asarray(ensemble_x, dtype=DTYPE)
    Rs = np.asarray(R, dtype=DTYPE).ravel()
    Ps = np.asarray(P, dtype=DTYPE).ravel()
    n = x.shape[0]
    w = (np.full(n, 1.0 / n) if weights is None
         else np.asarray(weights, dtype=DTYPE) / np.sum(weights))
    h_R = float(h_R or silverman_bandwidth(Rs))
    h_P = float(h_P or silverman_bandwidth(Ps))

    c = np.swapaxes(wigner_aa_dagger_all(x, hbar, nstates), -1, -2)
    if method == "binned":
        out = _kde_binned(grid, Rs, Ps, w[:, None, None] * c, h_R, h_P, True)
    elif method == "direct":
        out = np.zeros(grid.shape + (nstates, nstates), dtype=CDTYPE)
        for s0 in range(0, n, chunk):
            s1 = min(n, s0 + chunk)
            KR = _kde_weights(grid.R, Rs[s0:s1], h_R)
            dKP = _kde_weights(grid.P, Ps[s0:s1], h_P, derivative=True)
            wc = w[s0:s1, None, None] * c[s0:s1]
            out += np.einsum("im,jm,mab->ijab", KR, dKP, wc, optimize=True)
    else:
        raise ValueError(f"unknown KDE method '{method}'")
    return out, {"h_R": h_R, "h_P": h_P, "n_samples": int(n), "method": method}
