#!/usr/bin/env python3
"""Gate L -- coefficient-level propagation against an exact quantum reference.
SUBSYSTEM-QCLE GRID REFERENCE.  This script propagates the already-reduced
fields rho_W(R,P,t) with a Fourier/Strang scheme.  It is retained as an
independent cross-check.  It is NOT the Gaussian-Hermite coefficient
algorithm; that is scripts/07g_gh_coefficient_propagation.py.


Runs the scientifically meaningful *triple* of section 3.1 of the directive:

1. coefficient-level MQCLE  (``mqcle.propagation.propagate_coefficients``);
2. PBME trajectories        (``mqcle.propagation.propagate_pbme_ensemble``);
3. exact Schroedinger       (``mqcle.schrodinger``) -- the independent physical leg.

Branch 1 solves the subsystem QCLE exactly (Gate J), so any residual gap
between branch 1 and branch 3 is QCLE's own approximation error, not the
solver's.  That statement is only meaningful once the solver has been shown
converged, which is what ``scripts/09_convergence.py`` measures.
``scripts/10b_qcle_grid_crosscheck.py`` adds a separately implemented solver
of the same QCLE as a numerical cross-code check; it is not another physical
leg of this comparison.

Usage
-----
    python scripts/07_coefficient_propagation.py --config tully_default.yaml --P0 40
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqcle.coefficient_operator import (build_grid,                  # noqa: E402
                                        spectral_radius_estimate)
from mqcle.benchmark_regimes import REGIMES, regime_for              # noqa: E402
from mqcle.io import RunRecorder, load_config, save_arrays           # noqa: E402
from mqcle.models import build_model                                 # noqa: E402
from mqcle.propagation import (History, initial_coefficient_field,   # noqa: E402
                               propagate_coefficients,
                               propagate_pbme_ensemble)
from mqcle.sampling import sample_initial_ensemble                   # noqa: E402
from mqcle.schrodinger import (SplitOperatorPropagator,              # noqa: E402
                               build_schrodinger_grid,
                               gaussian_wavepacket)
from mqcle.types import TraceConvention                              # noqa: E402

#: Per-momentum defaults.  ``sigma_R = hbar / (2 (P0/20))`` is the relation
#: used throughout this project's Tully benchmarks.
#:
#: Two independent resolution requirements, both of which must hold.
#:
#: ``dR <= 0.45 sigma_R``
#:     spectral accuracy on the initial packet in R.
#:
#: ``dP <= 0.3 sigma_R M / T``
#:     the one that is easy to miss.  Free bath advection maps
#:     ``rho(R, P) -> rho(R - P t / M, P)``, so a structure of R-scale
#:     ``sigma_R`` is sheared into a P-structure of scale ``sigma_R M / t``:
#:     the Wigner function filaments in momentum.  With ``T = 2 M |R0| / P0``
#:     and ``sigma_R = 10 / P0`` this scale is ``1/3`` for every benchmark
#:     here, independent of P0.  Under-resolving it does not degrade
#:     gracefully -- the fine structure aliases and reaches the periodic
#:     boundary.  This is what the split ``boundary_mass_P`` diagnostic
#:     detects; an earlier version of these defaults used n_P = 128 at
#:     P0 = 20 and failed it at 4.2e-4.
#:
#: A third requirement appears only at ``P0 = 20``: products such as
#: ``F(R) * d_P rho`` double the R-bandwidth, and at the crossing the aliased
#: part shows up as a uniform background across the periodic box.  Measured
#: peak ``boundary_mass_R`` through the crossing (T = 1700, n_P = 256):
#: n_R = 512 -> 8.98e-05, 1024 -> 1.86e-08, 2048 -> 1.65e-15, i.e. spectral
#: convergence.  The observable was already converged at every one of those
#: resolutions -- the final excited-state population is 0.225078 in all three
#: -- so this is a sentinel tightening, not an observable correction.
#:
#: The P half-width holds the momentum distribution: ``sigma_P = P0/20``,
#: eight standard deviations, plus the energy shift through the crossing.
DEFAULTS = {float(P0): regime.coefficient_defaults()
            for P0, regime in REGIMES.items()}


def _defaults_for(P0: float) -> dict:
    registered = regime_for(P0)
    if registered is not None:
        return registered.coefficient_defaults()
    sigma_R = 10.0 / P0
    sigma_P = 0.5 / sigma_R
    T = 2.0 * 2000.0 * 15.0 / P0
    P_half = max(20.0, 8.0 * sigma_P)
    dP_max = 0.3 * sigma_R * 2000.0 / T
    return dict(R_box=(-45.0, 45.0),
                n_R=int(2 ** np.ceil(np.log2(90.0 / (0.45 * sigma_R)))),
                P_center=P0, P_half=P_half,
                n_P=int(2 ** np.ceil(np.log2(2 * P_half / dP_max))),
                dt=0.5, T=T)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="tully_default.yaml")
    p.add_argument("--coupling", type=float, default=None,
                   help="override the Tully diabatic coupling amplitude C; "
                        "this is the parameter that sets the scale of "
                        "grad_R h, so it is the natural second axis for the "
                        "regime study of Gate M")
    p.add_argument("--tag", default="",
                   help="suffix appended to the output file name")
    p.add_argument("--P0", type=float, default=None,
                   help="initial mean bath momentum (default: from the config)")
    p.add_argument("--R0", type=float, default=None)
    p.add_argument("--sigma-R", type=float, default=None,
                   help="default: hbar / (2 (P0/20))")
    p.add_argument("--n-R", type=int, default=None)
    p.add_argument("--n-P", type=int, default=None)
    p.add_argument("--P-half", type=float, default=None,
                   help="half-width of the momentum box about --P-center")
    p.add_argument("--R-min", type=float, default=None,
                   help="lower edge of the periodic R box")
    p.add_argument("--R-max", type=float, default=None,
                   help="upper edge.  Shrinking the box at fixed n_R halves dR "
                        "and pushes the aliasing floor down spectrally, which "
                        "is what the R-boundary sentinel actually measures once "
                        "the physical tail has been removed")
    p.add_argument("--P-center", type=float, default=None,
                   help="centre of the momentum box (default: P0).  The box is "
                        "NOT required to be centred on P0: the accessible "
                        "momentum range is set by energy conservation, and the "
                        "dynamics need not be symmetric about the initial mean")
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--T", type=float, default=None,
                   help="default: 2 M |R0| / P0")
    p.add_argument("--n-save", type=int, default=10)
    p.add_argument("--integrator", choices=("strang", "rk4"), default="strang",
                   help="strang: unconditionally stable, exact invariants, 2nd "
                        "order (default).  rk4: 4th order but CFL limited.")
    p.add_argument("--cross-check-rk4", action="store_true",
                   help="also run a short RK4 leg and report the agreement")
    p.add_argument("--n-samples", type=int, default=None,
                   help="PBME ensemble size (default: from the config)")
    p.add_argument("--se-n-R", type=int, default=None,
                   help="default: >=8 grid points per de Broglie wavelength "
                        "at the largest momentum in the box")
    p.add_argument("--se-dt", type=float, default=None,
                   help="default: dt/4, so the reference is finer than the test")
    p.add_argument("--skip-pbme", action="store_true")
    p.add_argument("--skip-se", action="store_true")
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def _final(hist: History, key: str) -> np.ndarray:
    return np.asarray(hist.records[-1][key])


def _interp_series(hist: History, key: str, times: np.ndarray) -> np.ndarray:
    """Resample a (possibly vector) observable onto ``times``."""
    t = np.asarray(hist.times)
    col = np.stack([np.atleast_1d(np.asarray(r[key], dtype=float))
                    for r in hist.records])
    out = np.empty((times.size, col.shape[1]))
    for c in range(col.shape[1]):
        out[:, c] = np.interp(times, t, col[:, c])
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    ic = cfg["initial_conditions"]
    hbar, mass = float(cfg["hbar"]), float(cfg["mass"])
    convention = TraceConvention(cfg["trace_convention"])
    if args.coupling is not None:
        cfg["model"] = {**cfg["model"], "C": float(args.coupling)}
    model = build_model(cfg["model"], mass=mass)

    P0 = float(args.P0 if args.P0 is not None else ic["P0"])
    R0 = float(args.R0 if args.R0 is not None else ic["R0"])
    sigma_R = float(args.sigma_R if args.sigma_R is not None
                    else hbar / (2.0 * (P0 / 20.0)))
    init_state = int(ic["init_state"])
    d = _defaults_for(P0)
    if args.P_half is not None:
        d["P_half"] = float(args.P_half)
    if args.R_min is not None or args.R_max is not None:
        lo = float(args.R_min if args.R_min is not None else d["R_box"][0])
        hi = float(args.R_max if args.R_max is not None else d["R_box"][1])
        d["R_box"] = (lo, hi)
    d["P_center"] = (float(args.P_center) if args.P_center is not None
                     else float(d.get("P_center", P0)))
    n_R = int(args.n_R or d["n_R"])
    n_P = int(args.n_P or d["n_P"])
    dt = float(args.dt or d["dt"])
    T = float(args.T if args.T is not None
              else d.get("T", 2.0 * mass * abs(R0) / P0))
    n_steps = int(round(T / dt))
    n_samples = int(args.n_samples or cfg["mapping_sampling"]["n_samples"])
    se_dt = float(args.se_dt or d.get("se_dt", dt / 4.0))
    # >= 8 points per de Broglie wavelength at the fastest component present
    P_max = d["P_center"] + d["P_half"]
    L_R = d["R_box"][1] - d["R_box"][0]
    se_n_R = int(args.se_n_R or d.get("se_n_R") or 2 ** int(np.ceil(np.log2(
        max(1024.0, 8.0 * L_R * P_max / (2.0 * np.pi * hbar))))))

    rec = RunRecorder(script=Path(__file__).name, seed=int(cfg["seed"]), config={
        **{k: v for k, v in cfg.items() if not k.startswith("_")},
        "P0": P0, "R0": R0, "sigma_R": sigma_R, "sigma_P": hbar / (2 * sigma_R),
        "n_R": n_R, "n_P": n_P, "dt": dt, "T": T, "n_steps": n_steps,
        "n_save": args.n_save, "n_samples": n_samples,
        "se_n_R": se_n_R, "se_dt": se_dt,
        "se_points_per_de_broglie_wavelength":
            float(2.0 * np.pi * hbar * se_n_R / (L_R * P_max)),
        "R_box": list(d["R_box"]),
        "P_box": [d["P_center"] - d["P_half"], d["P_center"] + d["P_half"]],
    })
    print(f"[07] P0={P0}  R0={R0}  sigma_R={sigma_R}  T={T}  dt={dt}  "
          f"n_steps={n_steps}  grid={n_R}x{n_P}")

    # -- branch 1: coefficient-level MQCLE -------------------------------
    grid = build_grid(d["R_box"][0], d["R_box"][1], n_R,
                      d["P_center"] - d["P_half"], d["P_center"] + d["P_half"], n_P)
    rho0 = initial_coefficient_field(grid, R0, P0, sigma_R, init_state,
                                     model.nstates, hbar)
    t0 = time.perf_counter()
    rad = spectral_radius_estimate(grid, model, hbar, convention)
    print(f"[07] spectral radius {rad['spectral_radius']:.4g} "
          f"(adv {rad['advection']:.3g} + force {rad['force']:.3g} + "
          f"elec {rad['electronic']:.3g});  RK4 limit dt < {rad['max_dt_rk4']:.4g};  "
          f"integrator = {args.integrator}")
    coef = propagate_coefficients(rho0, grid, model, dt, n_steps, hbar=hbar,
                                  convention=convention, n_save=args.n_save,
                                  integrator=args.integrator)
    wall_coef = time.perf_counter() - t0
    print(f"[07] subsystem-QCLE grid reference branch done in {wall_coef:.1f} s")

    # Two different things, deliberately separated.  The semi-discrete
    # operator conserves the phase-space trace *identically* (the commutator
    # is traceless pointwise and the k=0 Fourier mode of every spectral
    # derivative is annihilated), so trace DRIFT measures the time integrator
    # and round-off only.  The offset of the initial trace from 1 measures
    # whether the grid resolves the initial Gaussian at all.
    tr_series = np.real(coef.column("trace"))
    drift = float(np.max(np.abs(tr_series - tr_series[0])))
    tr0_err = float(abs(tr_series[0] - 1.0))
    herm = float(np.max(coef.column("hermiticity")))
    bmass = float(np.max(coef.column("boundary_mass")))
    rec.check("gateL.trace_conservation", drift, 1e-10,
              "max |Tr rho_W(t) - Tr rho_W(0)| over the run")
    rec.check("gateL.initial_trace_normalisation", tr0_err, 1e-10,
              "|Tr rho_W(0) - 1|; a resolution check, not a conservation check")
    rec.check("gateL.hermiticity", herm, 1e-12, "max |rho_W - rho_W^dagger|")
    bmassR = float(np.max(coef.column("boundary_mass_R")))
    bmassP = float(np.max(coef.column("boundary_mass_P")))
    rec.check("gateL.boundary_mass_R", bmassR, 1e-9,
              "mass fraction in the outermost R rows; a short box fails this")
    rec.check("gateL.boundary_mass_P", bmassP, 1e-9,
              "mass fraction in the outermost P columns; unresolved "
              "phase-space shear fails this")
    bmassP_lo = float(np.max(coef.column("boundary_mass_P_low")))
    bmassP_hi = float(np.max(coef.column("boundary_mass_P_high")))
    bmassR_lo = float(np.max(coef.column("boundary_mass_R_low")))
    bmassR_hi = float(np.max(coef.column("boundary_mass_R_high")))
    print(f"[07] boundary mass   P_low={bmassP_lo:.3e}  P_high={bmassP_hi:.3e}  "
          f"R_low={bmassR_lo:.3e}  R_high={bmassR_hi:.3e}")

    # A second, independent time integrator on the same spatial operator.
    # The two schemes have different orders (Strang 2, RK4 4), so comparing
    # them at one dt measures their time-discretisation difference, not their
    # agreement.  The meaningful test is that the difference vanishes as
    # dt -> 0 at the rate the lower-order scheme sets, which is what is
    # measured here.
    if args.cross_check_rk4:
        from mqcle.propagation import StrangPropagator, rk4_step
        from mqcle.coefficient_operator import physical_matrices
        cache = physical_matrices(model, grid, convention)
        # A short leg: at production resolution an RK4 step costs ~4x a
        # Strang step and must be run at three refinements, so the leg length
        # is set by that budget.  Consistency of two integrators on the same
        # spatial operator is a local statement and does not need a long run.
        t_leg = min(20.0, T)
        # Start the leg IN the coupling region, not at R0.  The Strang error is
        # set by the commutator of the two sub-flows, which is proportional to
        # grad_R h; at the asymptotic launch point that is numerically zero
        # (e^{-B R0^2} with B R0^2 = 63), so the splitting is exact there and
        # the measured difference is RK4's own O(dt^4) error instead -- which
        # is exactly what a first version of this check saw (order 4.00).  A
        # cross-check has to be run where the operator is actually hard.
        rho_leg = initial_coefficient_field(grid, 0.0, P0, sigma_R, init_state,
                                            model.nstates, hbar)
        dt0 = min(dt, 0.4 * rad["max_dt_rk4"])
        diffs, dts = [], []
        for k in range(3):
            dtk = dt0 / (2 ** k)
            n_k = int(round(t_leg / dtk))
            a = rho_leg.copy()
            for _ in range(n_k):
                a = rk4_step(a, dtk, grid, model, hbar, convention, cache)
            sp_k = StrangPropagator(grid, model, dtk, hbar, convention)
            b = rho_leg.copy()
            for _ in range(n_k):
                b = sp_k.step(b)
            diffs.append(float(np.max(np.abs(a - b)) / np.max(np.abs(b))))
            dts.append(dtk)
        order = float(np.polyfit(np.log(dts), np.log(diffs), 1)[0])
        # One-sided on purpose.  The claim is that two integrators of the same
        # spatial operator converge to each other at least at the rate the
        # lower-order scheme sets.  Converging FASTER is not a failure -- it
        # happens wherever the two sub-flows nearly commute.
        rec.check("gateL.strang_vs_rk4_converges", max(0.0, 1.7 - order), 0.0,
                  f"the two integrators differ by O(dt^{order:.2f}) over "
                  f"t={t_leg:g} starting in the coupling region; at least "
                  f"O(dt^1.7) is required and faster is acceptable")
        rec.check("gateL.strang_vs_rk4_finest", diffs[-1], 1e-4,
                  f"relative field difference at the finest dt={dts[-1]:g}")
        results_cross = {"t": t_leg, "leg_R0": 0.0, "dt": dts,
                         "relative_difference": diffs, "measured_order": order}
        print(f"[07] strang vs rk4 over t={t_leg:g}: "
              + ", ".join(f"dt={a:g} -> {b:.2e}" for a, b in zip(dts, diffs))
              + f"   (order {order:.2f})")
    else:
        results_cross = None

    arrays = {f"coef_{k}": np.asarray(v) for k, v in coef.as_arrays().items()}
    # The dict key stays "coefficient" so that this script can still read
    # and regenerate the pre-existing archives.  The method identifier is
    # what downstream consumers must key on: this branch propagates the
    # reduced subsystem-QCLE fields, not Hermite coefficients.  The
    # Gaussian-Hermite branch is scripts/07g_gh_coefficient_propagation.py.
    results = {"method": "subsystem_qcle_grid_reference",
               "coefficient": {"wall_time_s": wall_coef,
                               "final_populations": _final(coef, "populations").tolist(),
                               "final_adiabatic_populations":
                                   _final(coef, "adiabatic_populations").tolist(),
                               "trace_drift": drift,
                               "initial_trace_error": tr0_err,
                               "hermiticity": herm,
                               "boundary_mass": bmass,
                               "boundary_mass_R": bmassR,
                               "boundary_mass_P": bmassP,
                               "boundary_mass_P_low": bmassP_lo,
                               "boundary_mass_P_high": bmassP_hi,
                               "boundary_mass_R_low": bmassR_lo,
                               "boundary_mass_R_high": bmassR_hi,
                               "P_box": [d["P_center"] - d["P_half"],
                                         d["P_center"] + d["P_half"]],
                               "dP": float(2 * d["P_half"] / n_P),
                               "shear_P_scale": float(cfg["mass"] * sigma_R / T),
                               "dP_over_shear_scale":
                                   float((2 * d["P_half"] / n_P)
                                         / (cfg["mass"] * sigma_R / T)),
                               "integrator": args.integrator,
                               "stability": rad,
                               "cfl_number": float(dt * rad["spectral_radius"]),
                               "strang_vs_rk4": results_cross,
                               "grid": grid.parameters()}}

    # -- branch 3: exact Schroedinger ------------------------------------
    se_hist = None
    if not args.skip_se:
        sg = build_schrodinger_grid(d["R_box"][0], d["R_box"][1], se_n_R)
        psi = gaussian_wavepacket(sg, R0, P0, sigma_R, init_state,
                                  model.nstates, hbar)
        prop = SplitOperatorPropagator(sg, model, dt=se_dt, hbar=hbar)
        t0 = time.perf_counter()
        times, recs, _ = prop.propagate(psi, int(round(T / se_dt)),
                                        n_save=max(1, int(args.n_save * dt / se_dt)))
        wall_se = time.perf_counter() - t0
        se_hist = History(times=list(times), records=list(recs),
                          meta={"branch": "schrodinger", "dt": se_dt,
                                "grid": sg.parameters(),
                                "provenance": "mqcle.schrodinger, written for "
                                              "this project; Strang split operator"})
        print(f"[07] Schroedinger reference done in {wall_se:.1f} s "
              f"(norm drift {abs(se_hist.records[-1]['norm'] - 1.0):.2e})")
        # Every factor of the split-operator reference is unitary, so the norm
        # is conserved identically and what is measured here is FFT round-off
        # accumulating as a random walk over the step count.  A fixed 1e-12 is
        # therefore the wrong tolerance for a 10^4-step run, not a strict one;
        # the same sqrt(n) scaling was measured for the coefficient solver.
        se_steps = int(round(T / se_dt))
        # The completed P10/P20/P40/P100 references give a worst measured
        # drift/sqrt(n_steps) of 1.17e-13 on this FFT runtime.  Keep margin for
        # platform variation while remaining orders below any population
        # difference used in the paper.
        se_tol = 1.5e-13 * np.sqrt(se_steps)
        rec.check("gateL.se_norm", abs(se_hist.records[-1]["norm"] - 1.0), se_tol,
                  f"unitarity of the split-operator reference over {se_steps} "
                  f"steps; tolerance 1.5e-13*sqrt(n_steps)")
        rec.check("gateL.se_boundary_mass",
                  float(max(r["boundary_mass"] for r in se_hist.records)), 1e-8,
                  "reference density reaching the periodic boundary")
        arrays.update({f"se_{k}": np.asarray(v)
                       for k, v in se_hist.as_arrays().items()})
        results["schrodinger"] = {
            "wall_time_s": wall_se,
            "final_populations": _final(se_hist, "populations").tolist(),
            "final_adiabatic_populations":
                _final(se_hist, "adiabatic_populations").tolist(),
            "norm_drift": float(abs(se_hist.records[-1]["norm"] - 1.0)),
            "grid": sg.parameters(), "dt": se_dt}

    # -- branch 2: PBME trajectories -------------------------------------
    pbme_hist = None
    if not args.skip_pbme:
        rng = np.random.default_rng(int(cfg["seed"]))
        ens = sample_initial_ensemble(rng, n_samples, nstates=model.nstates,
                                      init_state=init_state, R0=R0, P0=P0,
                                      sigma_R=sigma_R, scheme="focused", hbar=hbar)
        t0 = time.perf_counter()
        pbme_hist = propagate_pbme_ensemble(ens, model, dt, n_steps, hbar=hbar,
                                            convention=convention,
                                            n_save=args.n_save)
        wall_pbme = time.perf_counter() - t0
        print(f"[07] PBME branch done in {wall_pbme:.1f} s "
              f"({n_samples} trajectories)")
        arrays.update({f"pbme_{k}": np.asarray(v)
                       for k, v in pbme_hist.as_arrays().items()})
        results["pbme"] = {
            "wall_time_s": wall_pbme, "n_samples": n_samples,
            "final_populations": _final(pbme_hist, "populations").tolist(),
            "final_adiabatic_populations":
                _final(pbme_hist, "adiabatic_populations").tolist()}

    # -- errors against the independent reference ------------------------
    if se_hist is not None:
        tt = np.asarray(coef.times)
        ref = _interp_series(se_hist, "populations", tt)
        err_coef = np.max(np.abs(_interp_series(coef, "populations", tt) - ref), axis=1)
        results["errors"] = {
            "times": tt.tolist(),
            "coefficient_vs_schrodinger_max_pop_error": err_coef.tolist(),
            "coefficient_vs_schrodinger_final": float(err_coef[-1]),
        }
        print(f"[07] |subsystem-QCLE grid reference - Schroedinger| final population "
              f"error = {err_coef[-1]:.4e}   (physical QCLE approximation "
              "versus full quantum after numerical convergence; not the "
              "independent-grid cross-code error)")
        if pbme_hist is not None:
            err_pbme = np.max(np.abs(_interp_series(pbme_hist, "populations", tt)
                                     - ref), axis=1)
            results["errors"]["pbme_vs_schrodinger_max_pop_error"] = err_pbme.tolist()
            results["errors"]["pbme_vs_schrodinger_final"] = float(err_pbme[-1])
            print(f"[07] |PBME - Schroedinger|          final population error "
                  f"= {err_pbme[-1]:.4e}")

    rec.results = results
    tag = f"P{int(round(P0))}{args.tag}"
    out = Path(args.out or f"results/raw/07_coefficient_propagation_{tag}.json")
    save_arrays(out.with_suffix(".h5"), arrays,
                {"metadata": rec.metadata(), "results": results})
    rec.write(out)
    print(f"[07] checks: {len(rec.checks)}  all passed: {rec.all_passed}")
    for c in rec.checks:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['name']:38s} "
              f"{c['value']:.3e}  (tol {c['tolerance']:.1e})")
    print(f"[07] written to {out} and {out.with_suffix('.h5')}")
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
