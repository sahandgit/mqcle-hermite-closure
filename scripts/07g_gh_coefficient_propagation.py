#!/usr/bin/env python3
r"""Primary branch: Gaussian--Hermite coefficient propagation.

This script propagates the finite Hermite coefficient system of
``GH_MQCLE_FORMULATION.tex``,

.. math::

    \partial_t\bm c
    =-\frac{P}{M}\partial_R\bm c+\mathsf R(R)\bm c+\mathsf G(R)\partial_P\bm c,

with the state ``C[a, i_R, i_P]`` carrying the eleven even ``K = 2`` Hermite
modes of the two-state problem.  The mapping sector is *propagated*, not
eliminated before numerics.

It is deliberately a separate script from ``07_coefficient_propagation.py``,
which propagates the already-reduced subsystem-QCLE fields
:math:`\rho_W^{\lambda\lambda'}` with a Strang split.  That script is retained
unchanged as an independent cross-check; this one never imports its
propagator.  Output files carry the method identifier
``gaussian_hermite_coefficient`` so that no downstream table can confuse the
two.

Run one regime at a time.  ``--smoke`` gives a small, fast grid for checking
the command line before committing to a production run.
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

# The kernel threads over modes and over R chunks; BLAS must not thread inside
# those workers as well.  Environment variables are the portable mechanism and
# only take effect before numpy is imported, so this must stay above the
# imports below.  See mqcle.hermite.kernel for the measurement behind it.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import h5py                                                       # noqa: E402

from mqcle.benchmark_regimes import REGIMES                       # noqa: E402
from mqcle.coefficient_operator import (build_grid,               # noqa: E402
                                        hermiticity_error,
                                        populations_from_rho, trace)
from mqcle.constants import CDTYPE, HBAR                          # noqa: E402
from mqcle.hermite.frame import (hermite_metric,                  # noqa: E402
                                 hermite_to_rho, physical_frame_matrix,
                                 physical_frame_residual,
                                 physical_projector, rho_to_hermite,
                                 weighted_pseudoinverse)
from mqcle.hermite.kernel import (RealHermiteKernel,              # noqa: E402
                                  generator_is_real, krylov_arnoldi_real,
                                  rk4_real, state_is_real)
from mqcle.hermite.propagator import (HermiteGenerator,           # noqa: E402
                                      propagate_krylov)
from mqcle.models import TullyDualAvoidedCrossing                 # noqa: E402
from mqcle.propagation import adiabatic_populations               # noqa: E402

METHOD = "gaussian_hermite_coefficient"


# --------------------------------------------------------------------------
def initial_rho_W(grid, R0: float, P0: float, sigma_R: float,
                  init_state: int = 0, hbar: float = HBAR) -> np.ndarray:
    r"""Minimum-uncertainty Gaussian in one diabat, as a subsystem field.

    This is the same physical initial state the other branches use.  Mapping
    variables are never sampled: the Hermite branch initialises through
    :math:`C=B\,\mathrm{vec}(\rho_W)`.
    """
    sigma_P = hbar / (2.0 * sigma_R)
    Rg, Pg = np.meshgrid(grid.R, grid.P, indexing="ij")
    w = (np.exp(-((Rg - R0) ** 2) / (2.0 * sigma_R ** 2)
                - ((Pg - P0) ** 2) / (2.0 * sigma_P ** 2))
         / (np.pi * hbar))
    rho = np.zeros(grid.shape + (2, 2), dtype=CDTYPE)
    rho[..., init_state, init_state] = w
    return rho


def _euclid_residual(Cm, Pphys) -> float:
    r""":math:`\|(I-\mathsf P_{\rm phys})C\|/\|C\|` in the Euclidean norm."""
    resid = Cm - np.einsum("mn,...n->...m", Pphys, Cm, optimize=True)
    den = float(np.linalg.norm(Cm))
    return float(np.linalg.norm(resid) / den) if den else 0.0


def diagnostics(C, gen, grid, model, B, W, Bplus, Pphys, mode_orders):
    """Every saved quantity the specification requires, in one place."""
    Cm = gen.from_state(C)
    rho = hermite_to_rho(Cm, B, W, Bplus=Bplus)
    pops = populations_from_rho(rho, grid)
    coh = np.sum(rho[..., 0, 1]) * grid.dR * grid.dP
    w = np.asarray(W)
    norm = float(np.real(np.sum(w[None, None, :] * np.abs(Cm) ** 2))
                 * grid.dR * grid.dP)
    # parity leakage: the active set is even, so any odd-order weight is a bug
    odd = [a for a, o in enumerate(mode_orders) if o % 2]
    parity = (float(np.max(np.abs(Cm[..., odd]))) if odd else 0.0)
    order_present = int(max((mode_orders[a] for a in range(Cm.shape[-1])
                             if np.max(np.abs(Cm[..., a])) > 1e-14),
                            default=0))
    return {
        "populations": [float(np.real(pops[0])), float(np.real(pops[1]))],
        "adiabatic_populations": [float(x) for x in
                                  np.real(adiabatic_populations(rho, grid, model))],
        "coherence_re": float(np.real(coh)),
        "coherence_im": float(np.imag(coh)),
        "physical_frame_residual": physical_frame_residual(Cm, B, W, Pphys=Pphys),
        # the same residual in the plain Euclidean norm, which is the form the
        # specification writes; the W-weighted one above is the norm in which
        # P_phys is an orthogonal projector, so both are reported
        "physical_frame_residual_euclid": _euclid_residual(Cm, Pphys),
        "hermite_norm": norm,
        "parity_leakage": parity,
        "max_hermite_order": order_present,
        "trace": float(np.real(trace(rho, grid))),
        "hermiticity_error": float(hermiticity_error(rho)),
    }


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--P0", type=float, required=True,
                    help="initial bath momentum; a registered regime supplies "
                         "the grid unless overridden")
    ap.add_argument("--integrator", choices=("krylov", "rk4"),
                    default="krylov",
                    help="both are unsplit. 'krylov' is the PRODUCTION route: "
                         "matrix-free Arnoldi exponential action of the same "
                         "generator, Eq. (Krylov-step). 'rk4' is the "
                         "independent temporal cross-check; at the stability-"
                         "limited step it is three orders less accurate than "
                         "the exponential and is not a production setting.")
    ap.add_argument("--krylov-m", type=int, default=20,
                    help="Arnoldi subspace dimension. Cost per substep is m "
                         "operator applications; storage is m+1 state "
                         "vectors, which is what bounds m at production sizes")
    ap.add_argument("--krylov-tol", type=float, default=1e-10,
                    help="requested local error per Arnoldi substep")
    ap.add_argument("--n-R", type=int, default=None)
    ap.add_argument("--n-P", type=int, default=None)
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--dt", type=float, default=None,
                    help="explicit step; default is --cfl times the RK4 limit")
    ap.add_argument("--cfl", type=float, default=0.5,
                    help="safety factor on the estimated RK4 step limit")
    ap.add_argument("--n-save", type=int, default=40,
                    help="number of saved times (excluding t=0)")
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--parity", default="even", choices=("even", "odd", "none"))
    ap.add_argument("--save-field", action="store_true",
                    help="store the terminal coefficient field in the HDF5 file")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-h5", default=None)
    ap.add_argument("--kernel", choices=("auto", "fast", "reference"),
                    default="auto",
                    help="arithmetic realisation of the SAME generator. "
                         "'fast' is the real, cache-blocked, threaded kernel "
                         "of mqcle.hermite.kernel; 'reference' is the "
                         "complex128 path of mqcle.hermite.operator. 'auto' "
                         "takes 'fast' when the generator and the initial "
                         "state are exactly real and 'reference' otherwise.")
    ap.add_argument("--threads", type=int, default=None,
                    help="worker threads for the fast kernel "
                         "(default: os.cpu_count())")
    ap.add_argument("--smoke", action="store_true",
                    help="small grid and short time, for checking the command")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if a valid artifact already exists")
    args = ap.parse_args(argv)

    P0 = args.P0
    reg = REGIMES.get(int(round(P0)))
    if reg is None and (args.n_R is None or args.n_P is None or args.T is None):
        ap.error(f"P0={P0} is not a registered regime; give --n-R --n-P --T")

    if args.smoke:
        n_R, n_P = args.n_R or 128, args.n_P or 64
        T = args.T or 20.0
        n_save = min(args.n_save, 4)
    else:
        n_R = args.n_R or reg.n_R
        n_P = args.n_P or reg.n_P
        T = args.T or reg.T
        n_save = args.n_save

    tag = f"P{int(round(P0))}"
    suffix = "_smoke" if args.smoke else ""
    out_json = Path(args.out_json or
                    ROOT / "results" / "raw" /
                    f"07g_gh_coefficient_propagation_{tag}{suffix}.json")
    out_h5 = Path(args.out_h5 or str(out_json).replace(".json", ".h5"))
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if out_json.exists() and not args.force:
        try:
            prev = json.loads(out_json.read_text())
            if prev.get("metadata", {}).get("status") == "completed":
                print(f"[07g] {out_json.name} already complete; "
                      f"pass --force to recompute")
                return 0
        except Exception:
            pass

    # ---- build ----------------------------------------------------------
    model = TullyDualAvoidedCrossing(mass=2000.0)
    R_box = reg.R_box if reg else (-45.0, 45.0)
    if reg is not None:
        P_min, P_max = reg.P_box
    else:
        P_min, P_max = P0 - 22.0, P0 + 22.0
    grid = build_grid(R_box[0], R_box[1], n_R, P_min, P_max, n_P)

    parity = None if args.parity == "none" else args.parity
    gen = HermiteGenerator.from_model(model, grid, K=args.K, parity=parity)
    modes = gen.modes
    mode_orders = [sum(n) for n in modes]
    M = len(modes)

    B = physical_frame_matrix(model.nstates, modes)
    W = hermite_metric(modes)
    Bplus = weighted_pseudoinverse(B, W)
    Pphys = physical_projector(B, W)

    sigma_R = 10.0 / P0
    rho0 = initial_rho_W(grid, -15.0, P0, sigma_R)
    C = gen.to_state(rho_to_hermite(rho0, B))

    lam = gen.spectral_radius_estimate()
    dt = args.dt if args.dt else args.cfl * lam["rk4_dt_limit"]
    n_steps = max(1, int(np.ceil(T / dt)))
    dt = T / n_steps
    save_every = max(1, n_steps // max(1, n_save))

    # ---- arithmetic realisation ----------------------------------------
    # The fast kernel applies the same operator with the same spectral
    # derivatives and the same RK4 tableau; it only drops an imaginary part
    # that is provably zero and reorders the mode contraction.  Its two
    # hypotheses are checked here, recorded, and published as gates.
    gen_real = generator_is_real(gen)
    state_real = state_is_real(C)
    imag_C0 = float(np.max(np.abs(np.imag(C))))
    if args.kernel == "fast" and not (gen_real and state_real):
        ap.error("--kernel fast requested but the generator or the initial "
                 f"state is not exactly real (generator_is_real={gen_real}, "
                 f"max|Im C0|={imag_C0:g})")
    use_fast = (gen_real and state_real and args.kernel in ("auto", "fast"))
    kernel_name = "real_blocked_threaded" if use_fast else "reference_complex"

    state_gb = C.nbytes / 1e9
    print(f"[07g] method={METHOD} P0={P0:g} grid={n_R}x{n_P} modes={M} "
          f"({gen.n_blocks[0]} R-blocks, {gen.n_blocks[1]} G-blocks)")
    print(f"[07g] state {state_gb:.3f} GB  dt={dt:.5g} "
          f"(limit {lam['rk4_dt_limit']:.5g})  steps={n_steps}  T={T:g}")
    print(f"[07g] kernel={kernel_name}  generator_is_real={gen_real}  "
          f"max|Im C0|={imag_C0:g}")

    # ---- propagate ------------------------------------------------------
    history = []
    t_start = time.perf_counter()
    rhs_calls = 0

    def record(step, t, state):
        history.append({"t": float(t), "step": int(step),
                        **diagnostics(state, gen, grid, model, B, W,
                                      Bplus, Pphys, mode_orders)})

    threads = None
    if use_fast:
        kern = RealHermiteKernel(gen, nthread=args.threads)
        threads = kern.nthread
        print(f"[07g] fast kernel: {threads} threads, real state "
              f"{kern.state_bytes / 1e9:.3f} GB, "
              f"r_slab={kern.r_slab} p_slab={kern.p_slab} chunk={kern.chunk}")

        def progress(step, t, state):
            record(step, t, state)
            print(f"[07g]   t={t:9.2f}  "
                  f"pops={history[-1]['populations'][0]:.6f},"
                  f"{history[-1]['populations'][1]:.6f}  "
                  f"resid={history[-1]['physical_frame_residual']:.2e}  "
                  f"trace={history[-1]['trace']:.10f}  "
                  f"[{(time.perf_counter() - t_start) / 3600:.2f} h]",
                  flush=True)

        C0r = np.ascontiguousarray(np.real(C))
        if args.integrator == "krylov":
            save_times = np.linspace(T / n_save, T, n_save)
            need = (args.krylov_m + 1) * C0r.nbytes / 1e9
            print(f"[07g] Arnoldi exponential: m={args.krylov_m} "
                  f"tol={args.krylov_tol:g}, basis {need:.2f} GB")

            def kprogress(i, t, state):
                progress(i, t, state)

            C, kstats = krylov_arnoldi_real(
                kern, C0r, save_times, m=args.krylov_m, tol=args.krylov_tol,
                callback=kprogress)
            rhs_calls = kstats["applications"]
        else:
            C = rk4_real(kern, C0r, dt, n_steps, save_every=save_every,
                         callback=progress)
            rhs_calls = 4 * n_steps
            kstats = None
        kern.close()
        del kern          # release the two derivative scratch arrays
    elif args.integrator == "rk4":
        record(0, 0.0, C)
        buf = np.empty_like(C)
        for step in range(1, n_steps + 1):
            k1 = gen.rhs(C)
            k2 = gen.rhs(C + 0.5 * dt * k1)
            k3 = gen.rhs(C + 0.5 * dt * k2)
            k4 = gen.rhs(C + dt * k3, out=buf)
            C = C + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            rhs_calls += 4
            del k1, k2, k3, k4
            if step % save_every == 0 or step == n_steps:
                record(step, step * dt, C)
                print(f"[07g]   t={step * dt:9.2f}  "
                      f"pops={history[-1]['populations'][0]:.6f},"
                      f"{history[-1]['populations'][1]:.6f}  "
                      f"resid={history[-1]['physical_frame_residual']:.2e}  "
                      f"trace={history[-1]['trace']:.10f}", flush=True)
    else:
        record(0, 0.0, C)
        times = np.linspace(T / n_save, T, n_save)
        states = propagate_krylov(gen, C, times)
        for i, t in enumerate(times):
            record(int(round(t / dt)), float(t), states[i])
        C = states[-1]
    wall = time.perf_counter() - t_start

    # ---- write ----------------------------------------------------------
    final = history[-1]
    checks = [
        {"name": "physical_frame_residual_max",
         "value": max(h["physical_frame_residual"] for h in history),
         "tolerance": 1e-8},
        {"name": "parity_leakage_max",
         "value": max(h["parity_leakage"] for h in history),
         "tolerance": 1e-12},
        {"name": "trace_drift",
         "value": abs(final["trace"] - history[0]["trace"]),
         "tolerance": 1e-8},
        {"name": "hermiticity_max",
         "value": max(h["hermiticity_error"] for h in history),
         "tolerance": 1e-8},
    ]
    if use_fast:
        # The real kernel is valid only because these two are exactly zero /
        # exactly true.  Publish them with the run rather than asserting them.
        checks.append({"name": "initial_state_imaginary_part",
                       "value": imag_C0, "tolerance": 0.0})
    for c in checks:
        c["passed"] = bool(c["value"] <= c["tolerance"])

    meta = {
        "method": METHOD,
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": Path(__file__).name,
        "source_modules": ["mqcle.hermite.operator", "mqcle.hermite.frame",
                           "mqcle.hermite.propagator"],
        "command_line": " ".join(sys.argv),
        "P0": P0, "model": "tully_dual_avoided_crossing", "mass": model.mass,
        "hbar": HBAR, "dtype": "complex128",
        "grid": {"R_box": list(R_box), "n_R": n_R,
                 "P_box": [P_min, P_max], "n_P": n_P},
        "time": {"T": T, "dt": dt, "n_steps": n_steps,
                 "save_every": save_every, "integrator": args.integrator,
                 "cfl": args.cfl, "rk4_dt_limit": lam["rk4_dt_limit"],
                 "krylov_m": args.krylov_m if args.integrator == "krylov"
                             else None,
                 "krylov_tol": args.krylov_tol if args.integrator == "krylov"
                               else None,
                 "krylov_stats": kstats},
        "hermite": {"K": args.K, "parity": args.parity, "n_modes": M,
                    "modes": [list(n) for n in modes],
                    "R_blocks": gen.n_blocks[0], "G_blocks": gen.n_blocks[1]},
        "spectral_radius": lam,
        "kernel": {"name": kernel_name,
                   "module": ("mqcle.hermite.kernel" if use_fast
                              else "mqcle.hermite.operator"),
                   "threads": threads,
                   "generator_is_real": bool(gen_real),
                   "initial_state_max_abs_imag": imag_C0,
                   "equivalence_gate": "GH-12",
                   "note": ("same operator, same spectral derivatives, same "
                            "RK4 tableau, same dt and grid; real arithmetic "
                            "and a reordered mode contraction")},
        "rhs_applications": rhs_calls,
        "wall_time_s": wall,
        "state_bytes": int(C.nbytes),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {"numpy": np.__version__, "h5py": h5py.__version__},
        "seed": None,
        "output_files": {"json": str(out_json), "h5": str(out_h5)},
    }
    payload = {"metadata": meta, "checks": checks,
               "all_passed": all(c["passed"] for c in checks),
               "results": {"final": final, "history": history}}
    out_json.write_text(json.dumps(payload, indent=2))

    with h5py.File(out_h5, "w") as f:
        f.attrs["method"] = METHOD
        f.attrs["P0"] = P0
        f.create_dataset("t", data=np.array([h["t"] for h in history]))
        for key in ("populations", "adiabatic_populations"):
            f.create_dataset(key, data=np.array([h[key] for h in history]))
        for key in ("coherence_re", "coherence_im", "physical_frame_residual",
                    "physical_frame_residual_euclid",
                    "hermite_norm", "parity_leakage", "trace",
                    "hermiticity_error"):
            f.create_dataset(key, data=np.array([h[key] for h in history]))
        f.create_dataset("modes", data=np.array([list(n) for n in modes]))
        f.create_dataset("R", data=grid.R)
        f.create_dataset("P", data=grid.P)
        if args.save_field:
            # archived in complex128 whichever kernel ran, so that the file
            # format does not depend on the arithmetic realisation
            f.create_dataset("C_final", data=np.asarray(C, dtype=CDTYPE),
                             compression="gzip", compression_opts=4)
            # The coefficient vector is the state; the four real components of
            # rho_W are what every other branch archives, so the weighted
            # readout is applied here rather than in the plotting stage.  A
            # figure that reduced the state itself would be performing part of
            # the calculation, and the two would then have to be kept in step.
            rho_fin = hermite_to_rho(gen.from_state(C), B, W, Bplus=Bplus)
            for key, arr in (("A", np.real(rho_fin[:, :, 0, 0])),
                             ("C", np.real(rho_fin[:, :, 1, 1])),
                             ("bR", np.real(rho_fin[:, :, 0, 1])),
                             ("bI", np.imag(rho_fin[:, :, 0, 1]))):
                f.create_dataset(f"gh_final_{key}", data=np.ascontiguousarray(arr),
                                 compression="gzip", compression_opts=4)

    print(f"[07g] wall {wall / 3600:.3f} h, {rhs_calls} RHS applications")
    print(f"[07g] checks: " + ", ".join(
        f"{c['name']}={c['value']:.2e}{'' if c['passed'] else ' FAIL'}"
        for c in checks))
    print(f"[07g] wrote {out_json}")
    print(f"[07g] wrote {out_h5}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
