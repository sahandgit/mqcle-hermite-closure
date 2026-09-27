#!/usr/bin/env python3
r"""Is the residual freedom of Theorem 3 worth optimising?  Measured, not argued.

Theorem 3 leaves a constant unitary change of subsystem basis
:math:`h\mapsto U^{\dagger}hU`.  An earlier draft of the manuscript said that
because this changes :math:`\nabla_Rh^{\lambda\lambda'}`, and the closed form of
the PBME defect is built from :math:`\nabla_Rh`, the rotation "could in
principle be optimised to reduce the PBME defect".  That is wrong, and the error
is worth stating precisely because it is the kind of statement that looks safe.

The defect is not a scalar.  On the mapping variables the same transformation
acts as :math:`x\mapsto\mathcal U x` with :math:`\mathcal U` the real form of
:math:`V=U^{\dagger}`, which is orthogonal *and* symplectic.  Under it

* the mapping Hamiltonian keeps its form with :math:`h\to U^{\dagger}hU`,
* the Poisson bracket is invariant, hence the PBME generator is covariant,
* the projected density transforms with :math:`\rho_W\to U^{\dagger}\rho_WU`,
* the closed generator is covariant for the same reason.

So the transformation carries a physical problem to *the same* physical problem
in a relabelled basis.  Every basis-independent number -- the exact populations,
the closed-dynamics error, the PBME error -- is unchanged, and there is nothing
to optimise.  What genuinely is basis dependent in practice is the *sampling
scheme*, which is a property of the estimator and not of the equation.

Two design points, both learned the hard way:

* **The packet starts at the crossing.**  Away from it :math:`\nabla_Rh` is
  numerically zero, both channels of the coupled term vanish, and the check
  measures free advection plus a rotation -- quantities nobody doubts are
  covariant.  This is the same trap ``CORRECTIONS.md`` L5 records for the
  integrator cross-check.  The script refuses to run if the propagated region
  does not actually carry a gradient.
* **At least one rotation is genuinely complex.**  A real orthogonal :math:`O`
  keeps a real symmetric :math:`h` real symmetric, so it never leaves the case
  Theorem 1 was originally stated for.  Testing the :math:`U(N)` claim requires
  a :math:`U` with a nonzero imaginary part, which sends :math:`h` to a complex
  Hermitian matrix and exercises Corollary 1.

Writes ``results/raw/07c_basis_covariance.json``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqcle.coefficient_operator import build_grid                    # noqa: E402
from mqcle.io import RunRecorder, load_config                        # noqa: E402
from mqcle.mapping import real_form, subsystem_split                 # noqa: E402
from mqcle.models import TwoStateModel, build_model                  # noqa: E402
from mqcle.pbme import (pack_state, symmetric_rotation_step,         # noqa: E402
                        unpack_state)
from mqcle.propagation import (StrangPropagator,                     # noqa: E402
                               initial_coefficient_field)
from mqcle.sampling import sample_initial_ensemble                   # noqa: E402
from mqcle.types import TraceConvention                              # noqa: E402


class RotatedModel(TwoStateModel):
    """``base`` in a rotated subsystem basis: :math:`h\\mapsto U^{\\dagger}hU`.

    Only the subsystem matrix rotates.  The bath potential is a multiple of the
    identity in subsystem space and is therefore untouched, which is exactly
    why the rotation is a symmetry of the problem rather than a new model.
    """

    def __init__(self, base: TwoStateModel, U: np.ndarray):
        self.base = base
        self.U = np.asarray(U)
        self.nstates = base.nstates
        self.nbath = base.nbath
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

    def parameters(self):
        return {**self.base.parameters(), "name": self.name,
                "rotation_real": np.real(self.U).tolist(),
                "rotation_imag": np.imag(self.U).tolist()}


def rotation(theta: float, n: int = 2) -> np.ndarray:
    """A real orthogonal rotation in the (0,1) plane: the ``O(N)`` subgroup."""
    O = np.eye(n)
    c, s = np.cos(theta), np.sin(theta)
    O[0, 0], O[0, 1], O[1, 0], O[1, 1] = c, -s, s, c
    return O


def unitary(theta: float, phi: float, n: int = 2) -> np.ndarray:
    """A genuinely complex ``U(N)`` element, reducing to :func:`rotation` at
    ``phi = 0``.  Only this exercises the part of Theorem 3's residual freedom
    that Corollary 1 is needed to license."""
    U = np.eye(n, dtype=complex)
    c, s = np.cos(theta), np.sin(theta)
    U[0, 0], U[0, 1] = c, -s * np.exp(1j * phi)
    U[1, 0], U[1, 1] = s * np.exp(-1j * phi), c
    return U


def rotate_mapping(x: np.ndarray, U: np.ndarray) -> np.ndarray:
    r"""``x = (r, p)`` under the real form of :math:`V=U^{\dagger}`.

    The mapping amplitude is :math:`c=(r+ip)/\sqrt{2\hbar}` and transforms as
    :math:`c\mapsto U^{\dagger}c`, so ``x`` transforms by the real form of
    :math:`U^{\dagger}`, which is orthogonal because :math:`U` is unitary.
    """
    V = real_form(np.conj(np.asarray(U)).T[None])[0]
    return np.asarray(x) @ V.T


def rotate_density(rho: np.ndarray, U: np.ndarray) -> np.ndarray:
    r""":math:`\rho\mapsto U^{\dagger}\rho U`, batched over the grid."""
    U = np.asarray(U)
    return np.einsum("ai,...ab,bj->...ij", np.conj(U), rho, U)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tully_default.yaml")
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--n-R", type=int, default=1024)
    ap.add_argument("--n-P", type=int, default=512)
    ap.add_argument("--R0", type=float, default=0.0,
                    help="start AT the crossing: away from it grad_R h is "
                         "numerically zero and the check is vacuous")
    ap.add_argument("--out", default="results/raw/07c_basis_covariance.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ic = cfg["initial_conditions"]
    hbar, mass = float(cfg["hbar"]), float(cfg["mass"])
    convention = TraceConvention(cfg["trace_convention"])
    base = build_model(cfg["model"], mass=mass)
    n = base.nstates
    P0, R0, dt = 40.0, float(args.R0), float(args.dt)
    sigma_R = hbar / (2.0 * (P0 / 20.0))

    #: real orthogonal first, then two genuinely complex U(2) elements
    ROTS = [("O(2), theta=0.30", rotation(0.30)),
            ("O(2), theta=pi/4", rotation(np.pi / 4)),
            ("U(2), theta=0.60 phi=0.90", unitary(0.60, 0.90)),
            ("U(2), theta=pi/4 phi=pi/3", unitary(np.pi / 4, np.pi / 3))]

    rec = RunRecorder(script=Path(__file__).name, seed=int(cfg["seed"]), config={
        "rotations": [name for name, _ in ROTS],
        "n_samples": args.n_samples, "n_steps": args.n_steps, "dt": dt,
        "n_R": args.n_R, "n_P": args.n_P, "P0": P0, "R0": R0,
        "sigma_R": sigma_R, "convention": convention.value})

    # ---- guard: the coupled channel must actually be active ---------------
    T = args.n_steps * dt
    span = abs(P0) / mass * T + 4.0 * sigma_R
    Rg = np.linspace(R0 - span, R0 + span, 2001)[:, None]
    g = subsystem_split(base, Rg, convention).dh_map_dR[:, 0]
    gmax = float(np.linalg.norm(g, ord="fro", axis=(-2, -1)).max())
    gref = float(np.linalg.norm(
        subsystem_split(base, np.linspace(-6, 6, 2001)[:, None],
                        convention).dh_map_dR[:, 0],
        ord="fro", axis=(-2, -1)).max())
    rec.results["grad_h_over_propagated_region"] = gmax
    rec.results["grad_h_global_max"] = gref
    rec.results["region_spans_R"] = [float(Rg.min()), float(Rg.max())]
    rec.check("covariance.coupled_channel_is_active",
              gref / max(gmax, 1e-300), 10.0)
    print(f"[07c] |grad_R h| over R in [{Rg.min():.1f}, {Rg.max():.1f}]: "
          f"{gmax:.3e}  (global max {gref:.3e})", flush=True)
    if gmax < 0.1 * gref:
        print("[07c] REFUSING: the propagated region carries no gradient, so "
              "the coupled channel would not be exercised", flush=True)
        rec.write(args.out)
        return 1

    # ---- 1. PBME generator covariance, trajectory by trajectory ----------
    rng = np.random.default_rng(int(cfg["seed"]))
    ens = sample_initial_ensemble(rng, args.n_samples, nstates=n,
                                  init_state=int(ic["init_state"]),
                                  R0=R0, P0=P0, sigma_R=sigma_R,
                                  scheme="focused", hbar=hbar)

    def run_pbme(x0, model):
        st = pack_state(x0, ens.R, ens.P)
        for _ in range(args.n_steps):
            st = symmetric_rotation_step(st, dt, model, hbar=hbar,
                                         convention=convention)
        return unpack_state(st, n, model.nbath)

    t0 = time.perf_counter()
    x_ref, R_ref, P_ref = run_pbme(ens.x, base)
    scale = float(np.max(np.abs(x_ref)))
    pbme_rows = []
    for name, U in ROTS:
        xr, Rr, Pr = run_pbme(rotate_mapping(ens.x, U), RotatedModel(base, U))
        dx = float(np.max(np.abs(xr - rotate_mapping(x_ref, U))))
        pbme_rows.append({
            "rotation": name,
            "complex": bool(np.abs(np.imag(np.asarray(U, dtype=complex))).max() > 0),
            "max_dx": dx, "relative_dx": dx / scale,
            "max_dR": float(np.max(np.abs(Rr - R_ref))),
            "max_dP": float(np.max(np.abs(Pr - P_ref)))})
        print(f"[07c] PBME   {name:<28s} |dx|/scale = {dx / scale:.3e}",
              flush=True)
    wall_pbme = time.perf_counter() - t0

    # ---- 2. closed coefficient generator covariance, on the grid ---------
    grid = build_grid(-45.0, 45.0, args.n_R, P0 - 22.0, P0 + 22.0, args.n_P)
    rho0 = initial_coefficient_field(grid, R0, P0, sigma_R,
                                     int(ic["init_state"]), n, hbar)

    def run_closed(rho, model):
        prop = StrangPropagator(grid, model, dt, hbar, convention)
        for _ in range(args.n_steps):
            rho = prop.step(rho)
        return rho

    t0 = time.perf_counter()
    rho_ref = run_closed(rho0, base)
    rscale = float(np.max(np.abs(rho_ref)))
    closed_rows = []
    for name, U in ROTS:
        out = run_closed(rotate_density(rho0, U), RotatedModel(base, U))
        d = float(np.max(np.abs(out - rotate_density(rho_ref, U))))
        closed_rows.append({"rotation": name, "max_abs_difference": d,
                            "relative": d / rscale})
        print(f"[07c] closed {name:<28s} |drho|/scale = {d / rscale:.3e}",
              flush=True)
    wall_closed = time.perf_counter() - t0

    rec.results["pbme_generator"] = {"rows": pbme_rows,
                                     "wall_time_s": round(wall_pbme, 2)}
    rec.results["closed_generator"] = {"rows": closed_rows,
                                       "wall_time_s": round(wall_closed, 2)}
    worst_pbme = max(r["relative_dx"] for r in pbme_rows)
    worst_closed = max(r["relative"] for r in closed_rows)
    rec.results["worst_relative_pbme"] = worst_pbme
    rec.results["worst_relative_closed"] = worst_closed
    rec.results["n_complex_rotations"] = sum(
        1 for r in pbme_rows if r["complex"])

    # round-off accumulates over n_steps; 1e-9 is far above the observed level
    # and far below anything that would let an optimisation exist
    worst_bath = max(max(r["max_dR"], r["max_dP"]) for r in pbme_rows)
    rec.results["worst_bath_deviation"] = worst_bath
    rec.check("covariance.pbme_generator", worst_pbme, 1e-9)
    # The bath coordinates are not bit-identical here and should not be: at the
    # crossing the back-reaction force is nonzero and is assembled from rotated
    # matrices, so round-off differs.  They WERE bit-identical in the earlier
    # version of this study, and that was a symptom of its vacuousness -- the
    # packet sat where the force was numerically zero.
    rec.check("covariance.bath_coordinates", worst_bath, 1e-10)
    rec.check("covariance.closed_generator", worst_closed, 1e-9)
    rec.check("covariance.at_least_one_complex_rotation",
              0.0 if rec.results["n_complex_rotations"] else 1.0, 0.5)

    rec.write(args.out)
    print(f"\n[07c] worst relative deviation: PBME {worst_pbme:.3e}, "
          f"closed {worst_closed:.3e}")
    print(f"[07c] checks all passed: {rec.all_passed}")
    print(f"[07c] written to {args.out}")
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
