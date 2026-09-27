#!/usr/bin/env python3
r"""Basis covariance of the Gaussian--Hermite branch (gate GH-11, production).

Theorem 3 leaves a constant unitary relabelling of the subsystem basis,
:math:`h\mapsto U^{\dagger}hU`.  ``scripts/07c_basis_covariance.py`` measures
what that freedom does to the trajectory branch and to the reduced field.  This
script measures what it does to the object the closure is actually about: the
Hermite coefficient vector.

The two are not the same question.  The reduced field carries the subsystem
indices explicitly, so its covariance is a conjugation and nothing more.  The
coefficient vector carries no subsystem indices at all; a unitary acts on it
through the orthogonal :math:`\mathcal V=\mathrm{re}(U^{\dagger})` on the
standardised variables, and the question is whether that action *preserves
total Hermite order*.  If it did not, :math:`\mathcal V_K` would be a
basis-dependent subspace and the truncation would be an artefact of the basis
it was written in rather than a property of the generator.  It does, and this
script measures by how much it does, in three independent ways:

1. **structure** -- the induced matrix :math:`\mathsf D` is block diagonal in
   total order, and represents the group;
2. **operator** -- :math:`\mathsf R` and :math:`\mathsf G` conjugate by
   :math:`\mathsf D` exactly, which is covariance of Eq. (finite-Hermite)
   itself and needs no propagation;
3. **propagation** -- the rotated problem, propagated and rotated back,
   reproduces the unrotated one.

The third is a corollary of the second, but it is the form in which a fault in
the *assembly* would show: the wrong :math:`\mathbb H` fed to one channel, a
stale mode list, a transposed frame.  It is kept for that reason.

The packet starts at the crossing.  Away from it :math:`\nabla_Rh` is
numerically zero, both channels of the coupled term vanish, and the check
degenerates into free advection plus a rotation.  The script refuses to run if
the propagated region carries no gradient.

Writes ``results/raw/07cg_gh_basis_covariance.json``.  Minutes, not hours.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mqcle.coefficient_operator import build_grid                    # noqa: E402
from mqcle.constants import CDTYPE, HBAR                             # noqa: E402
from mqcle.hermite.covariance import (hermite_rotation_matrix,       # noqa: E402
                                      mapping_real_form, rotate_density,
                                      rotation_O2, unitary_U2)
from mqcle.hermite.frame import (hermite_metric, hermite_to_rho,     # noqa: E402
                                 physical_frame_matrix,
                                 physical_frame_residual,
                                 rho_to_hermite, weighted_pseudoinverse)
from mqcle.hermite.operator import (active_modes,                    # noqa: E402
                                    d_real_hamiltonian_dR,
                                    force_matrix, hermite_T_matrices,
                                    real_hamiltonian, rotation_matrix)
from mqcle.hermite.propagator import HermiteGenerator, propagate_rk4  # noqa: E402
from mqcle.models import TullyDualAvoidedCrossing                    # noqa: E402

METHOD = "gaussian_hermite_coefficient"

#: One real orthogonal element and two genuinely complex ones.  A real ``O``
#: keeps real symmetric ``h`` real symmetric and never leaves the case
#: Theorem 1 was first stated for; only a nonzero phase sends ``h`` to a
#: complex Hermitian matrix and so exercises Corollary 1.
ROTATIONS = [
    ("O(2), theta=0.30", rotation_O2(0.30)),
    ("O(2), theta=pi/4", rotation_O2(np.pi / 4)),
    ("U(2), theta=0.60 phi=0.90", unitary_U2(0.60, 0.90)),
    ("U(2), theta=pi/4 phi=pi/3", unitary_U2(np.pi / 4, np.pi / 3)),
]


class RotatedModel(TullyDualAvoidedCrossing):
    r"""``base`` in a rotated subsystem basis: :math:`h\mapsto U^{\dagger}hU`.

    Only the subsystem matrix rotates.  The bath potential is a multiple of the
    identity in subsystem space and is untouched, which is exactly why the
    rotation is a symmetry of the problem rather than a different model.
    """

    def __init__(self, base, U):
        self.base, self.U = base, np.asarray(U, dtype=complex)
        self.nstates, self.nbath = base.nstates, base.nbath
        self.mass = base.mass
        self.name = f"{base.name}_rotated"

    def h(self, R):
        return np.conj(self.U).T @ self.base.h(R) @ self.U

    def dh_dR(self, R):
        return np.conj(self.U).T @ self.base.dh_dR(R) @ self.U

    def bath_potential(self, R):
        return self.base.bath_potential(R)

    def dbath_dR(self, R):
        return self.base.dbath_dR(R)


def _random_hermitian(rng, shape):
    a = (rng.standard_normal(shape + (2, 2))
         + 1j * rng.standard_normal(shape + (2, 2)))
    return a + np.conj(np.swapaxes(a, -1, -2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--P0", type=float, default=40.0)
    ap.add_argument("--R0", type=float, default=0.0,
                    help="start AT the crossing; away from it grad_R h is "
                         "numerically zero and the check is vacuous")
    ap.add_argument("--n-R", type=int, default=512)
    ap.add_argument("--n-P", type=int, default=256)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--n-random", type=int, default=16,
                    help="random Hermitian (h, dh/dR) pairs per rotation for "
                         "the operator-conjugation identity")
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    out = Path(args.out or ROOT / "results" / "raw"
               / "07cg_gh_basis_covariance.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    model = TullyDualAvoidedCrossing(mass=2000.0)
    modes = active_modes(model.nstates, 2, parity="even")
    M = len(modes)
    B = physical_frame_matrix(model.nstates, modes)
    W = hermite_metric(modes)
    Bplus = weighted_pseudoinverse(B, W)
    T_mats = hermite_T_matrices(modes)
    order = np.array([sum(n) for n in modes])
    off_order = order[:, None] != order[None, :]

    print(f"[07cg] {M} modes, {len(ROTATIONS)} rotations, "
          f"{args.n_random} random Hamiltonians each", flush=True)

    # ---- 1 & 2. structure and operator conjugation ----------------------
    struct_rows = []
    for name, U in ROTATIONS:
        V = mapping_real_form(U)
        D = hermite_rotation_matrix(modes, V)
        D_inv = hermite_rotation_matrix(modes, V.T)
        dR_worst = dG_worst = 0.0
        for _ in range(args.n_random):
            h = _random_hermitian(rng, (1,))
            dh = _random_hermitian(rng, (1,))
            hU = np.conj(U).T @ h @ U
            dhU = np.conj(U).T @ dh @ U
            R0 = rotation_matrix(real_hamiltonian(h), HBAR, modes)[0]
            RU = rotation_matrix(real_hamiltonian(hU), HBAR, modes)[0]
            G0 = force_matrix(np.ones(1), d_real_hamiltonian_dR(dh),
                              T_mats, modes)[0]
            GU = force_matrix(np.ones(1), d_real_hamiltonian_dR(dhU),
                              T_mats, modes)[0]
            dR_worst = max(dR_worst, float(
                np.max(np.abs(RU - D @ R0 @ D_inv))
                / max(float(np.max(np.abs(R0))), 1.0)))
            dG_worst = max(dG_worst, float(
                np.max(np.abs(GU - D @ G0 @ D_inv))
                / max(float(np.max(np.abs(G0))), 1.0)))
        rho = _random_hermitian(rng, (3, 2))
        lhs = rho_to_hermite(rotate_density(rho, U), B)
        rhs = np.einsum("mn,rpn->rpm", D, rho_to_hermite(rho, B))
        row = {
            "rotation": name,
            "complex": bool(np.abs(np.imag(np.asarray(U, dtype=complex))).max()
                            > 0),
            "V_orthogonality": float(np.max(np.abs(V @ V.T - np.eye(2 * model.nstates)))),
            "order_mixing": float(np.max(np.abs(D[off_order]))),
            "group_property": float(np.max(np.abs(D @ D_inv - np.eye(M)))),
            "frame_intertwining": float(np.max(np.abs(lhs - rhs))
                                        / max(float(np.max(np.abs(lhs))), 1.0)),
            "rotation_matrix_conjugation": dR_worst,
            "force_matrix_conjugation": dG_worst,
        }
        struct_rows.append(row)
        print(f"[07cg] {name:<28s} order-mix {row['order_mixing']:.2e}  "
              f"R {dR_worst:.2e}  G {dG_worst:.2e}  "
              f"frame {row['frame_intertwining']:.2e}", flush=True)

    # ---- guard: the coupled channel must be active ----------------------
    R_half = 6.0
    grid = build_grid(args.R0 - R_half, args.R0 + R_half, args.n_R,
                      args.P0 - 20.0, args.P0 + 20.0, args.n_P)
    gmax = float(np.max(np.abs(model.dh_dR(grid.R[:, None]))))
    gref = float(np.max(np.abs(model.dh_dR(
        np.linspace(-6.0, 6.0, 2001)[:, None]))))
    print(f"[07cg] |grad_R h| over the propagated region {gmax:.3e} "
          f"(global max {gref:.3e})", flush=True)
    if gmax < 0.1 * gref:
        print("[07cg] REFUSING: the propagated region carries no gradient, "
              "so the coupled channel would not be exercised", flush=True)
        return 1

    # ---- 3. covariance after propagation --------------------------------
    gen = HermiteGenerator.from_model(model, grid, K=2, parity="even")
    dt = 0.25 * gen.spectral_radius_estimate()["rk4_dt_limit"]
    sigma_R = 10.0 / args.P0
    sigma_P = HBAR / (2.0 * sigma_R)
    Rg, Pg = np.meshgrid(grid.R, grid.P, indexing="ij")
    rho0 = np.zeros(grid.shape + (2, 2), dtype=CDTYPE)
    rho0[..., 0, 0] = (np.exp(-((Rg - args.R0) ** 2) / (2.0 * sigma_R ** 2)
                              - ((Pg - args.P0) ** 2) / (2.0 * sigma_P ** 2))
                       / (np.pi * HBAR))

    def run(g, r0):
        C0 = g.to_state(rho_to_hermite(r0, B))
        _, states = propagate_rk4(g, C0, dt, args.n_steps,
                                  save_every=args.n_steps)
        C = g.from_state(states[-1])
        return hermite_to_rho(C, B, W, Bplus=Bplus), C

    t0 = time.perf_counter()
    rho_ref, C_ref = run(gen, rho0)
    scale = float(np.max(np.abs(rho_ref)))
    prop_rows = []
    for name, U in ROTATIONS:
        genU = HermiteGenerator.from_model(RotatedModel(model, U), grid,
                                           K=2, parity="even")
        got, C_U = run(genU, rotate_density(rho0, U))
        d = float(np.max(np.abs(got - rotate_density(rho_ref, U))))
        # how different the rotated problem actually was, so that a check
        # which passed by doing nothing is visible in the record
        sep = float(np.max(np.abs(got - rho_ref)))
        prop_rows.append({
            "rotation": name,
            "complex": bool(np.abs(np.imag(np.asarray(U, dtype=complex))).max()
                            > 0),
            "max_abs_difference": d, "relative": d / scale,
            "separation_before_rotating_back": sep / scale,
            "physical_frame_residual": float(
                physical_frame_residual(C_U, B, W)),
        })
        print(f"[07cg] propagated {name:<28s} |drho|/scale = {d / scale:.3e}  "
              f"(rotated problem differs by {sep / scale:.3f})", flush=True)
        del genU, C_U
    wall_prop = time.perf_counter() - t0

    worst = {
        "order_mixing": max(r["order_mixing"] for r in struct_rows),
        "group_property": max(r["group_property"] for r in struct_rows),
        "frame_intertwining": max(r["frame_intertwining"] for r in struct_rows),
        "operator_conjugation": max(
            max(r["rotation_matrix_conjugation"],
                r["force_matrix_conjugation"]) for r in struct_rows),
        "propagated": max(r["relative"] for r in prop_rows),
        "least_separation": min(r["separation_before_rotating_back"]
                                for r in prop_rows),
        "frame_residual": max(r["physical_frame_residual"] for r in prop_rows),
    }

    checks = [
        {"name": "order_mixing", "value": worst["order_mixing"],
         "tolerance": 1e-12},
        {"name": "group_property", "value": worst["group_property"],
         "tolerance": 1e-12},
        {"name": "frame_intertwining", "value": worst["frame_intertwining"],
         "tolerance": 1e-12},
        {"name": "operator_conjugation", "value": worst["operator_conjugation"],
         "tolerance": 1e-12},
        {"name": "covariance_after_propagation", "value": worst["propagated"],
         "tolerance": 1e-9},
        {"name": "physical_frame_residual", "value": worst["frame_residual"],
         "tolerance": 1e-10},
    ]
    for c in checks:
        c["passed"] = bool(c["value"] <= c["tolerance"])
    # a check that passed because the rotated problem was identical would be
    # worthless; require that it was genuinely different
    checks.append({"name": "rotated_problem_is_different",
                   "value": worst["least_separation"], "tolerance": 0.01,
                   "passed": bool(worst["least_separation"] > 0.01)})

    payload = {
        "metadata": {
            "method": METHOD, "status": "completed", "gate": "GH-11",
            "timestamp_utc": datetime.now(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "script": Path(__file__).name,
            "source_modules": ["mqcle.hermite.covariance",
                               "mqcle.hermite.operator",
                               "mqcle.hermite.frame",
                               "mqcle.hermite.propagator"],
            "command_line": " ".join(sys.argv),
            "model": "tully_dual_avoided_crossing", "mass": model.mass,
            "hbar": HBAR, "P0": args.P0, "R0": args.R0,
            "grid": {"n_R": args.n_R, "n_P": args.n_P},
            "time": {"dt": dt, "n_steps": args.n_steps,
                     "T": dt * args.n_steps},
            "hermite": {"K": 2, "parity": "even", "n_modes": M},
            "rotations": [n for n, _ in ROTATIONS],
            "n_complex_rotations": sum(1 for r in struct_rows if r["complex"]),
            "n_random_hamiltonians": args.n_random,
            "grad_h_over_region": gmax, "grad_h_global_max": gref,
            "seed": args.seed,
            "wall_time_s": time.perf_counter() - t_start,
            "wall_time_propagation_s": wall_prop,
            "platform": platform.platform(),
            "python": sys.version, "numpy": np.__version__,
            "output_files": {"json": str(out)},
        },
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
        "results": {"structure": struct_rows, "propagation": prop_rows,
                    "worst": worst},
    }
    out.write_text(json.dumps(payload, indent=2))
    print("[07cg] checks: " + ", ".join(
        f"{c['name']}={c['value']:.2e}{'' if c['passed'] else ' FAIL'}"
        for c in checks))
    print(f"[07cg] wrote {out}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
