#!/usr/bin/env python3
"""Gates 1-6: validate the original-space physics end to end.

Runs the checks that the unit tests assert, but on the *configured* production
model and initial conditions, and records every number in
``results/raw/00_validate_physics.json`` together with full run metadata.

    python scripts/00_validate_physics.py --config tully_default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sympy as sp

from mqcle.constants import HBAR
from mqcle.excess import excess_operator_on_field
from mqcle.fields import FiniteDifferenceField, SymbolicField, symbols_for
from mqcle.full_operator import full_generator_on_field
from mqcle.io import RunRecorder, load_config, save_arrays
from mqcle.mapping import (drift_matrix, grad_mapping_hamiltonian_R,
                           grad_mapping_hamiltonian_x, mapping_hamiltonian,
                           mapping_velocity)
from mqcle.models import build_model
from mqcle.observables import (lebesgue_nodes, observables_from_rho_W,
                               rho_W_from_quadrature)
from mqcle.pbme import (hamiltonian_of_state, mapping_radius, pack_state,
                        propagate, unpack_state)
from mqcle.projector import (mapping_kernel_all, projected_density,
                             wigner_aa_dagger_all)
from mqcle.sampling import mapping_actions, sample_initial_ensemble
from mqcle.types import TraceConvention


def fd1(fn, y, h=1e-6):
    return (fn(y - 2 * h) - 8 * fn(y - h) + 8 * fn(y + h) - fn(y + 2 * h)) / (12 * h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tully_default.yaml")
    ap.add_argument("--out", default="results/raw/00_validate_physics.json")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="overrides the config; the effective value is archived")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_samples = (args.n_samples if args.n_samples is not None
                 else int(cfg["mapping_sampling"]["n_samples"]))
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    rec = RunRecorder(script=Path(__file__).name, config=cfg, seed=seed)

    model = build_model(cfg["model"], mass=float(cfg["mass"]))
    conv = TraceConvention(cfg["trace_convention"])
    N, NB = model.nstates, model.nbath
    rec.results["model_parameters"] = model.parameters()
    rec.results["trace_convention"] = conv.value
    # The archived metadata must describe what actually ran, not what the YAML
    # said: the CLI can override the sample count.
    rec.results["effective_parameters"] = {
        "n_samples": n_samples,
        "n_samples_from_config": int(cfg["mapping_sampling"]["n_samples"]),
        "n_samples_overridden": args.n_samples is not None,
        "dt": float(cfg["propagation"]["dt"]),
        "n_steps": int(cfg["propagation"]["n_steps"]),
        "scheme": cfg["mapping_sampling"]["scheme"],
        "hbar": float(cfg["hbar"]), "mass": float(cfg["mass"]),
        "R0": float(cfg["initial_conditions"]["R0"]),
        "P0": float(cfg["initial_conditions"]["P0"]),
        "sigma_R": float(cfg["initial_conditions"]["sigma_R"]),
        "init_state": int(cfg["initial_conditions"]["init_state"]),
    }

    # ---- Gate 1: subsystem Hamiltonian ---------------------------------
    Rg = np.linspace(-10.0, 10.0, 201).reshape(-1, 1)
    h = model.h(Rg)
    rec.check("gate1.h_symmetry", np.max(np.abs(h - np.swapaxes(h, -1, -2))), 1e-14)
    rec.check("gate1.dh_dR_vs_fd",
              np.max(np.abs(model.dh_dR(Rg)[:, 0] - fd1(model.h, Rg))), 1e-7)
    gap = np.diff(model.adiabatic_energies(Rg), axis=-1)[:, 0]
    rec.results["min_adiabatic_gap"] = float(np.min(gap))
    rec.results["min_gap_position"] = float(Rg[np.argmin(gap), 0])

    # ---- Gate 2: mapping Hamiltonian -----------------------------------
    ens = sample_initial_ensemble(
        rng, n_samples, nstates=N, init_state=int(cfg["initial_conditions"]["init_state"]),
        R0=float(cfg["initial_conditions"]["R0"]), P0=float(cfg["initial_conditions"]["P0"]),
        sigma_R=float(cfg["initial_conditions"]["sigma_R"]),
        scheme=cfg["mapping_sampling"]["scheme"], hbar=HBAR, seed=seed)
    x, R, P = ens.x, ens.R, ens.P

    gx = grad_mapping_hamiltonian_x(x, R, model, HBAR, conv)
    err = 0.0
    for i in range(2 * N):
        e = np.zeros(2 * N)
        e[i] = 1.0
        fd = fd1(lambda d: mapping_hamiltonian(x + d * e, R, P, model, HBAR, None, conv), 0.0)
        err = max(err, float(np.max(np.abs(gx[:, i] - fd))))
    rec.check("gate2.grad_x_vs_fd", err, 1e-7)
    fdR = fd1(lambda d: mapping_hamiltonian(x, R + d, P, model, HBAR, None, conv), 0.0)
    rec.check("gate2.grad_R_vs_fd",
              np.max(np.abs(grad_mapping_hamiltonian_R(x, R, model, HBAR, conv)[:, 0] - fdR)),
              1e-7)
    K = drift_matrix(R, model, HBAR, conv)
    rec.check("gate2.K_antisymmetry", np.max(np.abs(K + np.swapaxes(K, -1, -2))), 1e-14)
    rec.check("gate2.xdot_equals_Kx",
              np.max(np.abs(mapping_velocity(x, R, model, HBAR, conv)
                            - np.einsum("nij,nj->ni", K, x))), 1e-14,
              note="PART X: frozen-R mapping sector is exactly linear")

    # ---- Gate 3: projector ---------------------------------------------
    Xq, Wq = lebesgue_nodes(2 * N, 1.0 / HBAR, 16)
    g = mapping_kernel_all(Xq, HBAR, N)
    c = wigner_aa_dagger_all(Xq, HBAR, N)
    ortho = np.einsum("m,mij,mkl->ijkl", Wq, g, c)
    ref = np.einsum("ik,jl->ijkl", np.eye(N), np.eye(N)).astype(complex)
    rec.check("gate3.orthogonality_g_c", np.max(np.abs(ortho - ref)), 1e-11,
              note="Kelly et al. (2012) Eq. (20)")
    rho0 = np.zeros((N, N), dtype=complex)
    rho0[int(cfg["initial_conditions"]["init_state"]),
         int(cfg["initial_conditions"]["init_state"])] = 1.0
    back = rho_W_from_quadrature(lambda xx: projected_density(xx, rho0, HBAR, N),
                                 HBAR, N, npts=16)
    rec.check("gate3.rho_W_roundtrip", np.max(np.abs(back - rho0)), 1e-11)
    rec.results["initial_observables"] = {
        k: (float(v) if np.ndim(v) == 0 else np.asarray(v).tolist())
        for k, v in observables_from_rho_W(back).items()}

    # ---- sampling diagnostics ------------------------------------------
    acts = mapping_actions(x, N)
    rec.results["mapping_actions_mean"] = acts.mean(axis=0).tolist()
    if cfg["mapping_sampling"]["scheme"] == "focused":
        target = np.array([3.0 if i == int(cfg["initial_conditions"]["init_state"])
                           else 1.0 for i in range(N)]) * HBAR
        rec.check("sampling.focused_actions", np.max(np.abs(acts - target)), 1e-12)
    if cfg["mapping_sampling"]["scheme"] == "focused":
        from mqcle.observables import rho_W_from_focused_ensemble
        rho_focus = rho_W_from_focused_ensemble(x, hbar=HBAR, nstates=N)
        init = int(cfg["initial_conditions"]["init_state"])
        rec.check("sampling.focused_readout_pure_state",
                  abs(np.real(rho_focus[init, init]) - 1.0), 5.0 / np.sqrt(n_samples),
                  note="dedicated focused estimator, not the phi-importance one")
        rec.results["focused_readout_rho00"] = float(np.real(rho_focus[0, 0]))
    rec.results["bath_moments"] = {
        "R_mean": float(R.mean()), "R_std": float(R.std()),
        "P_mean": float(P.mean()), "P_std": float(P.std())}

    # ---- Gate 4: PBME propagation --------------------------------------
    dt = float(cfg["propagation"]["dt"])
    n_steps = int(cfg["propagation"]["n_steps"])
    sub = slice(0, min(256, x.shape[0]))
    s0 = pack_state(x[sub], R[sub], P[sub])
    traj = propagate(s0, dt, n_steps, model, hbar=HBAR, convention=conv)
    e0 = hamiltonian_of_state(s0, model, HBAR, None, conv)
    eT = hamiltonian_of_state(traj[-1], model, HBAR, None, conv)
    rec.check("gate4.energy_drift", np.max(np.abs(eT - e0)), 1e-4)
    x0m = mapping_radius(unpack_state(s0, N, NB)[0], N)
    back_traj = propagate(traj[-1], -dt, n_steps, model, hbar=HBAR, convention=conv)
    rec.check("gate4.time_reversibility", np.max(np.abs(back_traj[-1] - s0)), 1e-9)
    rec.results["energy_drift_relative"] = float(
        np.max(np.abs(eT - e0)) / np.max(np.abs(e0)))
    rec.results["mapping_radius_drift"] = float(np.max(np.abs(
        mapping_radius(unpack_state(traj[-1], N, NB)[0], N) - x0m)))

    # ---- Gate D extension: interaction region and symplecticity ---------
    from mqcle.pbme import symplecticity_defect
    s_cross = pack_state(np.array([[1.0, 0.0, 0.0, 1.0]]), np.array([[0.0]]),
                         np.array([[float(cfg["initial_conditions"]["P0"])]]))
    e0c = hamiltonian_of_state(s_cross, model, HBAR, None, conv)
    conv_tab = {}
    for dt_i in (dt, dt / 2, dt / 4):
        n_i = int(round(80.0 / dt_i))
        tr_i = propagate(s_cross, dt_i, n_i, model, hbar=HBAR, convention=conv)
        back_i = propagate(tr_i[-1], -dt_i, n_i, model, hbar=HBAR, convention=conv)
        conv_tab[f"{dt_i:g}"] = {
            "energy_error": float(np.max(np.abs(
                hamiltonian_of_state(tr_i[-1], model, HBAR, None, conv) - e0c))),
            "reversibility": float(np.max(np.abs(back_i[-1] - s_cross))),
            "mapping_radius_drift": float(np.max(np.abs(
                mapping_radius(unpack_state(tr_i[-1], N, NB)[0], N)
                - mapping_radius(unpack_state(s_cross, N, NB)[0], N)))),
            "eps_symplectic": symplecticity_defect(
                s_cross.reshape(-1), dt_i, model, hbar=HBAR, convention=conv),
        }
    rec.results["interaction_region_convergence"] = conv_tab
    keys = list(conv_tab)
    rec.check("gateD.energy_error_decreases_with_dt",
              0.0 if (conv_tab[keys[0]]["energy_error"]
                      > conv_tab[keys[2]]["energy_error"]) else 1.0, 0.5,
              note="crossing-region initial condition, R = 0")
    rec.check("gateD.reversibility_interaction_region",
              max(v["reversibility"] for v in conv_tab.values()), 1e-9)
    rec.results["symplecticity_note"] = (
        "eps_symplectic is measured, not asserted to vanish: the scheme is a "
        "symmetric frozen-R rotation splitting, not a proven symplectic "
        "integrator.")

    # ---- Gates 5 and 6: excess and full generator -----------------------
    r_, p_, R_, P_ = symbols_for(N, NB)
    expr = sp.exp(-sum(s**2 for s in list(r_) + list(p_)) / 2) * (
        1 + r_[0] * p_[1] * P_[0] + P_[0] ** 2 * r_[1] * r_[0]) * sp.cos(R_[0])
    field = SymbolicField(expr, N, NB)
    fdf = FiniteDifferenceField(lambda xx, RR, PP: field.value(xx, RR, PP),
                                N, NB, hx=1e-3, hP=1e-3)
    xs, Rs, Ps = x[:200], R[:200], P[:200]
    ex_exact = excess_operator_on_field(field, xs, Rs, Ps, model, HBAR, conv)
    ex_fd = excess_operator_on_field(fdf, xs, Rs, Ps, model, HBAR, conv)
    scale = float(np.max(np.abs(ex_exact)))
    rec.results["excess_magnitude"] = scale
    rec.check("gate5.excess_exact_vs_fd", np.max(np.abs(ex_exact - ex_fd)) / scale,
              1e-4, note="symbolic derivatives versus central differences")
    parts = full_generator_on_field(field, xs, Rs, Ps, model, HBAR, None, conv)
    rec.check("gate6.total_equals_sum",
              np.max(np.abs(parts.total - (parts.pbme + parts.excess))), 1e-15)
    adv = full_generator_on_field(field, xs, Rs, Ps, model, HBAR, None, conv,
                                  form="advection")
    rec.check("gate6.poisson_equals_advection",
              np.max(np.abs(parts.total - adv.total)), 1e-10)
    # Diagnostic only: measured on an artificial analytic test field, so it
    # carries no physical meaning and must not be used in discussion.
    rec.results["diagnostic_testfield_excess_to_pbme_ratio"] = float(
        np.max(np.abs(parts.excess)) / np.max(np.abs(parts.pbme)))

    # complex off-diagonal field: the imaginary part must survive
    u01 = r_[0] * r_[1] + p_[0] * p_[1]
    L01 = r_[0] * p_[1] - r_[1] * p_[0]
    cfield = SymbolicField(sp.exp(-sum(s_**2 for s_ in list(r_) + list(p_)))
                           * (u01 - sp.I * L01) * P_[0], N, NB)
    cex = excess_operator_on_field(cfield, xs, Rs, Ps, model, HBAR, conv)
    cfd = excess_operator_on_field(
        FiniteDifferenceField(lambda xx, RR, PP: cfield.value(xx, RR, PP),
                              N, NB, hx=1e-3, hP=1e-3),
        xs, Rs, Ps, model, HBAR, conv)
    rec.check("gate5.complex_field_imaginary_part_survives",
              0.0 if np.max(np.abs(cex.imag)) > 1e-9 else 1.0, 0.5)
    rec.check("gate5.complex_excess_exact_vs_fd",
              np.max(np.abs(cex - cfd)) / np.max(np.abs(cex)), 1e-4)

    save_arrays("results/raw/00_trajectory_sample.h5",
                {"traj": traj[:, :32], "x0": x[:32], "R0": R[:32], "P0": P[:32]},
                attrs={"config": cfg, "seed": seed, "dt": dt, "n_steps": n_steps})
    rec.write(args.out)

    print(f"[00] gates run: {len(rec.checks)}   all passed: {rec.all_passed}")
    for c in rec.checks:
        flag = "PASS" if c["passed"] else "FAIL"
        print(f"  {flag}  {c['name']:34s} {c['value']:.3e}  (tol {c['tolerance']:.1e})")
    print("[00] diagnostic (test-field, not physical) excess/PBME ratio = "
          f"{rec.results['diagnostic_testfield_excess_to_pbme_ratio']:.3e}")
    print(f"[00] written to {args.out}")
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
