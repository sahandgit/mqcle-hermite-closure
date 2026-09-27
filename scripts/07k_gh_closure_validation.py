#!/usr/bin/env python3
r"""Numerical test of the closure theorem, in bases that could falsify it.

Two statements of the theorem are tested, and neither can be tested in the
production basis:

**Closure**, :math:`\mathcal V_2` invariant.  The exact generator couples
Hermite order :math:`K` to :math:`K` and to :math:`K-2` and never to
:math:`K+2`.  Measuring ``max_hermite_order`` in a basis truncated at
:math:`K=2` cannot establish that: the order-four modes are absent by
construction, so the answer is forced.  Here the basis carries the **full even
sector through degree four** -- 46 modes for two states -- only the physical
:math:`|\bm n|\le2` coefficients are initialised, the :math:`K=4` operator
propagates them, and

.. math::

    E_4(t)=\sum_{|\bm n|=4}\int\mathrm dR\,\mathrm dP\,|c_{\bm n}(R,P,t)|^2

is measured.  The theorem predicts :math:`E_4(t)=0`.

**Parity**, the even sector invariant.  Likewise a basis of even modes cannot
demonstrate parity preservation.  Here the basis carries **both parities**
through degree three -- 35 modes -- a purely even physical state is
initialised, and

.. math::

    E_{\rm odd}(t)=\sum_{|\bm n|\ \mathrm{odd}}\|c_{\bm n}(t)\|^2

is measured.

Each test ships with a **control** run that starts *inside* the sector the
test says is unreachable.  Without it, a zero would be equally consistent with
the theorem and with an order-four block that is accidentally all zeros, or
never applied.  The control establishes that the sector is dynamically live:
it must give :math:`E_4\ne0` and must feed order two, and it must give
:math:`E_{\rm odd}\ne0`.  A test whose control does not fire is not evidence.

The physical-frame residual of Eq. (physical-coeff-residual) is recorded on
the :math:`K=4` run as well.  It answers the *second* question, which is
distinct: :math:`\mathcal V_2` invariant is not the same statement as
:math:`\mathcal F_{\rm phys}` invariant inside :math:`\mathcal V_2`.

Writes ``results/raw/07k_gh_closure_validation.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mqcle.coefficient_operator import build_grid                  # noqa: E402
from mqcle.constants import CDTYPE, HBAR                           # noqa: E402
from mqcle.hermite.frame import (hermite_metric,                   # noqa: E402
                                 physical_frame_matrix,
                                 physical_frame_residual,
                                 physical_projector, rho_to_hermite,
                                 weighted_pseudoinverse)
from mqcle.hermite.kernel import (RealHermiteKernel,               # noqa: E402
                                  krylov_arnoldi_real)
from mqcle.hermite.operator import active_modes                    # noqa: E402
from mqcle.hermite.propagator import HermiteGenerator              # noqa: E402
from mqcle.models import TullyDualAvoidedCrossing                  # noqa: E402

METHOD = "gaussian_hermite_closure_validation"


# --------------------------------------------------------------------------
def _gaussian_rho(grid, R0, P0, sigma_R, hbar=HBAR):
    sigma_P = hbar / (2.0 * sigma_R)
    Rg, Pg = np.meshgrid(grid.R, grid.P, indexing="ij")
    w = (np.exp(-((Rg - R0) ** 2) / (2.0 * sigma_R ** 2)
                - ((Pg - P0) ** 2) / (2.0 * sigma_P ** 2)) / (np.pi * hbar))
    rho = np.zeros(grid.shape + (2, 2), dtype=CDTYPE)
    rho[..., 0, 0] = w
    return rho


def _order_energies(C, orders, grid, weights=None):
    r"""``{order: sum_{|n| = order} int dR dP |c_n|^2}``.

    ``weights`` applies the Hermite metric :math:`\bm n!`, giving the norm in
    which the projector is orthogonal.  The unweighted form is the one the
    specification writes; both are reported because they answer the same
    question and a reader may check either.
    """
    cell = grid.dR * grid.dP
    out = {}
    for o in sorted(set(orders)):
        idx = [a for a, oo in enumerate(orders) if oo == o]
        blk = C[idx]
        w = 1.0 if weights is None else np.asarray(weights)[idx][:, None, None]
        out[int(o)] = float(np.sum(w * np.abs(blk) ** 2) * cell)
    return out


def _run(tag, gen, kern, C0, grid, orders, weights, times, m, tol,
         B=None, W=None, Pphys=None, verbose=True):
    """Propagate and record the order-resolved energies at each saved time."""
    hist = []

    def record(i, t, state):
        e_raw = _order_energies(state, orders, grid)
        e_w = _order_energies(state, orders, grid, weights=weights)
        row = {"t": float(t),
               "E_raw": {str(k): v for k, v in e_raw.items()},
               "E_weighted": {str(k): v for k, v in e_w.items()}}
        if B is not None:
            Cm = gen.from_state(state.astype(CDTYPE))
            row["physical_frame_residual_W"] = physical_frame_residual(
                Cm, B, W, Pphys=Pphys)
            resid = Cm - np.einsum("mn,...n->...m", Pphys, Cm, optimize=True)
            den = float(np.linalg.norm(Cm))
            row["physical_frame_residual_euclid"] = (
                float(np.linalg.norm(resid) / den) if den else 0.0)
        hist.append(row)
        if verbose and (i % 5 == 0 or i == len(times)):
            tot = sum(e_raw.values())
            hi = max(orders)
            print(f"[07k]   {tag} t={t:8.2f}  "
                  + "  ".join(f"E{k}={v:.3e}" for k, v in sorted(e_raw.items()))
                  + f"  E{hi}/tot={e_raw[hi] / tot if tot else 0.0:.3e}",
                  flush=True)

    t0 = time.perf_counter()
    Cf, stats = krylov_arnoldi_real(kern, C0, times, m=m, tol=tol,
                                    callback=record)
    return Cf, hist, stats, time.perf_counter() - t0


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--P0", type=float, default=40.0)
    ap.add_argument("--R0", type=float, default=-15.0)
    ap.add_argument("--n-R", type=int, default=256)
    ap.add_argument("--n-P", type=int, default=128)
    ap.add_argument("--T", type=float, default=1500.0)
    ap.add_argument("--n-save", type=int, default=30)
    ap.add_argument("--krylov-m", type=int, default=20)
    ap.add_argument("--krylov-tol", type=float, default=1e-10)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--control-T", type=float, default=None,
                    help="length of the control runs; defaults to T/10")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.quick:
        args.n_R, args.n_P, args.T, args.n_save = 64, 32, 100.0, 5

    out = Path(args.out or ROOT / "results" / "raw"
               / "07k_gh_closure_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    model = TullyDualAvoidedCrossing(mass=2000.0)
    sigma_R = 10.0 / args.P0
    grid = build_grid(-45.0, 45.0, args.n_R,
                      args.P0 - 22.0, args.P0 + 22.0, args.n_P)
    rho0 = _gaussian_rho(grid, args.R0, args.P0, sigma_R)
    times = np.linspace(args.T / args.n_save, args.T, args.n_save)
    T_ctl = args.control_T if args.control_T else args.T / 10.0
    times_ctl = np.linspace(T_ctl / 5, T_ctl, 5)
    t_start = time.perf_counter()
    results = {}

    # ==================================================================
    # 1. closure: full even sector through degree four
    # ==================================================================
    modes4 = active_modes(model.nstates, 4, parity="even")
    orders4 = [sum(n) for n in modes4]
    gen4 = HermiteGenerator.from_model(model, grid, K=4, parity="even")
    assert [tuple(n) for n in gen4.modes] == [tuple(n) for n in modes4]
    W4 = hermite_metric(modes4)
    B4 = physical_frame_matrix(model.nstates, modes4)
    P4 = physical_projector(B4, W4)
    n_order4 = sum(1 for o in orders4 if o == 4)
    print(f"[07k] closure basis: K=4 even, {len(modes4)} modes "
          f"({orders4.count(0)} at order 0, {orders4.count(2)} at order 2, "
          f"{n_order4} at order 4)")
    print(f"[07k] grid {args.n_R}x{args.n_P}  T={args.T:g}  "
          f"Arnoldi m={args.krylov_m} tol={args.krylov_tol:g}")

    C0 = gen4.to_state(rho_to_hermite(rho0, B4))
    C0r = np.ascontiguousarray(np.real(C0))
    e0 = _order_energies(C0r, orders4, grid)
    print(f"[07k] initial order-4 content: {e0[4]:.3e} "
          f"(must be identically zero)")

    kern4 = RealHermiteKernel(gen4, nthread=args.threads)
    _, hist4, st4, wall4 = _run("closure", gen4, kern4, C0r, grid, orders4,
                                W4, times, args.krylov_m, args.krylov_tol,
                                B=B4, W=W4, Pphys=P4)

    E4_max = max(h["E_raw"]["4"] for h in hist4)
    E_phys0 = hist4[0]["E_raw"]["0"] + hist4[0]["E_raw"]["2"]
    results["closure"] = {
        "basis": {"K": 4, "parity": "even", "n_modes": len(modes4),
                  "n_modes_order4": n_order4},
        "initial_order4_energy": e0[4],
        "E4_max": E4_max,
        "E4_max_normalised": E4_max / E_phys0 if E_phys0 else float("nan"),
        "E_physical_initial": E_phys0,
        "history": hist4,
        "krylov": st4, "wall_time_s": wall4,
    }
    print(f"[07k] closure: max_t E4 = {E4_max:.3e}, normalised "
          f"{E4_max / E_phys0:.3e}")

    # -- control: the order-four sector must be dynamically live -------
    rng = np.random.default_rng(4)
    Cc = np.zeros_like(C0r)
    # the control packet sits ON the avoided crossing, where dH/dR is largest,
    # so that the forbidden sector is driven as hard as the model allows and a
    # null result cannot be blamed on a weak coupling
    env = np.real(_gaussian_rho(grid, 0.0, args.P0, sigma_R)[..., 0, 0])
    for a, o in enumerate(orders4):
        if o == 4:
            Cc[a] = env * rng.standard_normal()
    _, histc, stc, wallc = _run("control-K4", gen4, kern4, Cc, grid, orders4,
                                W4, times_ctl, args.krylov_m, args.krylov_tol,
                                verbose=False)
    ctl_E4 = histc[0]["E_raw"]["4"]
    ctl_E2_final = histc[-1]["E_raw"]["2"]
    results["closure_control"] = {
        "purpose": "order-4 initial data: the sector must evolve and must "
                   "feed order 2, otherwise E4 = 0 is not evidence",
        "E4_initial": ctl_E4,
        "E2_final": ctl_E2_final,
        "E2_initial": histc[0]["E_raw"]["2"],
        "T": T_ctl, "krylov": stc, "wall_time_s": wallc,
    }
    print(f"[07k] control: order-4 start E4={ctl_E4:.3e} -> order-2 content "
          f"after T={T_ctl:g} is {ctl_E2_final:.3e} "
          f"(started at {histc[0]['E_raw']['2']:.3e})")
    kern4.close()
    del kern4, gen4

    # ==================================================================
    # 2. parity: both parities through degree three
    # ==================================================================
    modes3 = active_modes(model.nstates, 3, parity=None)
    orders3 = [sum(n) for n in modes3]
    gen3 = HermiteGenerator.from_model(model, grid, K=3, parity=None)
    W3 = hermite_metric(modes3)
    B3 = physical_frame_matrix(model.nstates, modes3)
    n_odd = sum(1 for o in orders3 if o % 2)
    print(f"[07k] parity basis: K=3 both parities, {len(modes3)} modes "
          f"({n_odd} of odd order)")

    D0 = gen3.to_state(rho_to_hermite(rho0, B3))
    D0r = np.ascontiguousarray(np.real(D0))
    kern3 = RealHermiteKernel(gen3, nthread=args.threads)
    _, hist3, st3, wall3 = _run("parity", gen3, kern3, D0r, grid, orders3,
                                W3, times, args.krylov_m, args.krylov_tol,
                                verbose=False)

    def _odd_even(h):
        odd = sum(v for k, v in h["E_raw"].items() if int(k) % 2)
        even = sum(v for k, v in h["E_raw"].items() if not int(k) % 2)
        return odd, even

    odd0, even0 = _odd_even(hist3[0])
    Eodd_max = max(_odd_even(h)[0] for h in hist3)
    results["parity"] = {
        "basis": {"K": 3, "parity": "none", "n_modes": len(modes3),
                  "n_modes_odd": n_odd},
        "E_odd_initial": odd0,
        "E_odd_max": Eodd_max,
        "E_odd_max_normalised": Eodd_max / even0 if even0 else float("nan"),
        "E_even_initial": even0,
        "history": hist3, "krylov": st3, "wall_time_s": wall3,
    }
    print(f"[07k] parity: max_t E_odd = {Eodd_max:.3e}, normalised "
          f"{Eodd_max / even0:.3e}")

    Dc = np.zeros_like(D0r)
    for a, o in enumerate(orders3):
        if o % 2:
            Dc[a] = env * rng.standard_normal()
    _, histp, stp, wallp = _run("control-odd", gen3, kern3, Dc, grid, orders3,
                                W3, times_ctl, args.krylov_m, args.krylov_tol,
                                verbose=False)
    results["parity_control"] = {
        "purpose": "odd initial data: the odd sector must evolve, otherwise "
                   "E_odd = 0 is not evidence",
        "E_odd_initial": _odd_even(histp[0])[0],
        "E_odd_final": _odd_even(histp[-1])[0],
        "E_even_final": _odd_even(histp[-1])[1],
        "T": T_ctl, "krylov": stp, "wall_time_s": wallp,
    }
    print(f"[07k] control: odd start E_odd={_odd_even(histp[0])[0]:.3e} -> "
          f"{_odd_even(histp[-1])[0]:.3e} after T={T_ctl:g}")
    kern3.close()

    # ==================================================================
    resid_max = max(h["physical_frame_residual_W"] for h in hist4)
    resid_max_e = max(h["physical_frame_residual_euclid"] for h in hist4)
    results["physical_frame"] = {
        "max_residual_W_metric": resid_max,
        "max_residual_euclidean": resid_max_e,
        "note": "measured on the K=4 run, so this is F_phys invariance "
                "inside a space larger than V_2 -- a strictly stronger "
                "statement than the production diagnostic",
    }
    print(f"[07k] physical-frame residual on the K=4 run: "
          f"{resid_max:.3e} (W metric), {resid_max_e:.3e} (Euclidean)")

    checks = [
        {"name": "initial_order4_energy_is_zero",
         "value": results["closure"]["initial_order4_energy"],
         "tolerance": 0.0},
        {"name": "K4_leakage_normalised",
         "value": results["closure"]["E4_max_normalised"],
         "tolerance": 1e-20},
        {"name": "K4_control_sector_is_live",
         "value": -results["closure_control"]["E2_final"],
         "tolerance": -1e-12},
        {"name": "parity_leakage_normalised",
         "value": results["parity"]["E_odd_max_normalised"],
         "tolerance": 1e-20},
        {"name": "parity_control_sector_is_live",
         "value": -results["parity_control"]["E_odd_final"],
         "tolerance": -1e-12},
        {"name": "physical_frame_residual_K4",
         "value": resid_max, "tolerance": 1e-10},
    ]
    for c in checks:
        c["passed"] = bool(c["value"] <= c["tolerance"])

    payload = {
        "metadata": {
            "method": METHOD, "status": "completed",
            "purpose": "numerical test of the closure theorem in bases that "
                       "contain the sectors the theorem forbids",
            "timestamp_utc": datetime.now(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "script": Path(__file__).name,
            "source_modules": ["mqcle.hermite.operator", "mqcle.hermite.frame",
                               "mqcle.hermite.kernel"],
            "command_line": " ".join(sys.argv),
            "P0": args.P0, "R0": args.R0, "sigma_R": sigma_R,
            "model": "tully_dual_avoided_crossing", "mass": model.mass,
            "hbar": HBAR, "dtype": "float64",
            "grid": {"n_R": args.n_R, "n_P": args.n_P,
                     "R_box": [-45.0, 45.0],
                     "P_box": [args.P0 - 22.0, args.P0 + 22.0]},
            "T": args.T, "n_save": args.n_save,
            "krylov_m": args.krylov_m, "krylov_tol": args.krylov_tol,
            "integrator": "arnoldi_exponential",
            "wall_time_s": time.perf_counter() - t_start,
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__,
            "output_files": {"json": str(out)},
        },
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print("[07k] checks: " + ", ".join(
        f"{c['name']}={c['value']:.3e}{'' if c['passed'] else ' FAIL'}"
        for c in checks))
    print(f"[07k] wrote {out}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
