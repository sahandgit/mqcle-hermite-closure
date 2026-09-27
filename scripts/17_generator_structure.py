#!/usr/bin/env python3
r"""Exact computer-algebra checks of the operator statements of the paper.

N = 2 subsystem states, one bath coordinate, arbitrary functions h(R), V_b(R)
and an arbitrary Hermitian rho_W(R, P).  The operators are built here from
their definitions -- the Poisson bracket in (r, p, R, P) and the excess term
-(hbar/8) sum_ij d_R H_ij d_xi d_xj d_P -- without using src/mqcle, whose only
role is the archive writer.  Densities are handled as phi(x) times a
polynomial amplitude, with d_xi (phi q) = phi (d_xi q - 2 x_i q / hbar), so
every step is polynomial algebra and every comparison is an exact zero test.

  C0  normalisation: <Pi_ll'>_phi = delta_ll'; the phase-space average of the
      mapping symbol of |k><k'| over rho^phys is rho_W^{k'k}; the quadratic
      form (x^T H x - hbar Tr h) / (2 hbar) is sum_ll' h_ll' (a+_l a_l')_W.
  C1  (A_PB + A') rho^phys = phi sum_ll' Pi_ll' Q^{l'l}, Q the right-hand side
      of the QCLE, for real symmetric and complex Hermitian h and in both trace
      conventions; the real-symmetric form of A' fails for complex h.
  C2  Hermite identities for every |n| <= 6 (four mapping variables):
      d_i d_j (varphi He_n) = varphi He_{n+e_i+e_j} and
      (z_i z_j - delta_ij) He_n = He_{n+e_i+e_j} + T_ij He_n.
  C3  the generator applied to c(R,P) varphi He_n, for every |n| <= 4, is the
      coefficient operator -(P/M) d_R + R + G d_P, with R = -sum K_ij N^ij and
      G = d_R V_b + (1/4) sum d_R H_ij T^ij; the Poisson bracket alone is the
      same plus (1/4) sum d_R H_ij d_P c He_{n+e_i+e_j}.  Real symmetric and
      complex Hermitian h.
  C4  the PBME deviation on a physical density, -A' rho^phys / varphi: its
      Hermite orders; its order-2 part in both trace conventions; its order-4
      part against the Hermite ordering of q^(4); the number of order-4
      functions it occupies.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mqcle.io import RunRecorder  # noqa: E402

N = 2
hb, M = sp.symbols("hbar M", positive=True)
R, P = sp.symbols("R P", real=True)
r = sp.symbols("r0:2", real=True)
p = sp.symbols("p0:2", real=True)
X = list(r) + list(p)                         # x = (r_1, r_2, p_1, p_2)
Z = sp.symbols("z0:4", real=True)             # z = (2/hbar)^{1/2} x
TO_Z = {X[i]: sp.sqrt(hb / 2) * Z[i] for i in range(2 * N)}
J = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]])


def ex(e):
    return sp.expand(e)


def is_zero(e) -> bool:
    e = sp.expand(e)
    return e == 0 or sp.simplify(e) == 0


# ------------------------------------------------------------------ model
def hamiltonian(kind: str, convention: str):
    """(h used in H_m and A', V_b used in H_m, physical h, physical V_b)."""
    Vb = sp.Function("Vb", real=True)(R)
    if kind == "real":
        h00, h01, h11 = (sp.Function(n, real=True)(R) for n in ("h00", "h01", "h11"))
        h = sp.Matrix([[h00, h01], [h01, h11]])
    else:
        a00, a11, a01, b01 = (sp.Function(n, real=True)(R) for n in ("a00", "a11", "a01", "b01"))
        h = sp.Matrix([[a00, a01 + sp.I * b01], [a01 - sp.I * b01, a11]])
    if convention == "traceless":
        t = h.trace() / N
        return h - t * sp.eye(N), Vb + t, h, Vb
    return h, Vb, h, Vb


def big_H(h):
    """Real symmetric 2N x 2N form [[A, -B], [B, A]] of h = A + iB."""
    A = h.applyfunc(lambda e: sp.expand(sp.re(e)))
    B = h.applyfunc(lambda e: sp.expand(sp.im(e)))
    if any(e.has(sp.re, sp.im) for e in list(A) + list(B)):
        raise SystemExit("[17] real and imaginary parts of h not resolved")
    return sp.BlockMatrix([[A, -B], [B, A]]).as_explicit()


def rho_matrix():
    A00, A11, Cr, Ci = (sp.Function(n, real=True)(R, P) for n in ("A00", "A11", "Cr", "Ci"))
    return sp.Matrix([[A00, Cr + sp.I * Ci], [Cr - sp.I * Ci, A11]])


def Pi(l, lp):
    """Projector polynomial of the physical mapping density (x variables)."""
    return (2 / hb) * (r[l] * r[lp] + p[l] * p[lp]
                       - sp.I * (r[l] * p[lp] - r[lp] * p[l])
                       - hb / 2 * sp.KroneckerDelta(l, lp))


def adag_a(l, lp):
    """Weyl symbol of a+_l a_l' in the mapping representation."""
    return (r[l] * r[lp] + p[l] * p[lp] + sp.I * (r[l] * p[lp] - p[l] * r[lp])
            - hb * sp.KroneckerDelta(l, lp)) / (2 * hb)


def Hm_of(h, Vb):
    H = big_H(h)
    xv = sp.Matrix(X)
    return P**2 / (2 * M) + Vb + ex(((xv.T * H * xv)[0] - hb * h.trace()) / (2 * hb))


# ------------------------------------------- operators on amplitudes q = f/phi
def D(i, q):
    """phi^{-1} d_{x_i} (phi q)."""
    return ex(sp.diff(q, X[i]) - 2 * X[i] * q / hb)


def pb_amp(Hm, q):
    """phi^{-1} {H_m, phi q} over (r, p, R, P); phi depends on x only."""
    out = 0
    for l in range(N):
        out += sp.diff(Hm, r[l]) * D(N + l, q) - sp.diff(Hm, p[l]) * D(l, q)
    out += sp.diff(Hm, R) * sp.diff(q, P) - sp.diff(Hm, P) * sp.diff(q, R)
    return ex(out)


def excess_amp(h, q, real_symmetric_form=False):
    """phi^{-1} A'(phi q) with A' = -(hbar/8) sum_ij d_R H_ij d_i d_j d_P."""
    qP = sp.diff(q, P)
    out = 0
    if real_symmetric_form:        # the form written for real symmetric h
        for l in range(N):
            for lp in range(N):
                out += sp.diff(h[l, lp], R) * (D(l, D(lp, qP)) + D(N + l, D(N + lp, qP)))
    else:
        dH = sp.diff(big_H(h), R)
        for i in range(2 * N):
            for j in range(2 * N):
                if dH[i, j] != 0:
                    out += dH[i, j] * D(i, D(j, qP))
    return ex(-hb / 8 * out)


def gauss_mean_x(q):
    """<q>_phi for a polynomial q(x), phi = (pi hbar)^{-N} exp(-x.x/hbar)."""
    poly = sp.Poly(ex(q), *X)
    out = 0
    for m, c in zip(poly.monoms(), poly.coeffs()):
        v = c
        for k in m:
            v *= 0 if k % 2 else sp.factorial2(k - 1) * (hb / 2) ** sp.Rational(k, 2)
        out += v
    return sp.simplify(out)


# -------------------------------------------------------------- Hermite algebra
def He(n, v):
    return sp.hermite_prob(n, v)


def He_multi(nvec):
    return sp.Mul(*[He(k, Z[i]) for i, k in enumerate(nvec)])


def hermite_coefficients(q_z):
    """Exact Hermite coefficients of a polynomial in z, by peeling the top order."""
    rest = ex(q_z)
    out: dict[tuple, sp.Expr] = {}
    while rest != 0:
        poly = sp.Poly(rest, *Z)
        top = max(sum(m) for m in poly.monoms())
        for m, c in zip(poly.monoms(), poly.coeffs()):
            if sum(m) == top:
                out[m] = out.get(m, 0) + c
                rest -= ex(c * He_multi(m))
        rest = ex(rest)
    return {m: sp.simplify(c) for m, c in out.items() if not is_zero(c)}


def shifted(n, *moves):
    m = list(n)
    for i, s in moves:
        m[i] += s
    return None if min(m) < 0 else tuple(m)


def He_or_zero(n):
    return 0 if n is None else He_multi(n)


def T_action(n, i, j):
    """T_ij He_n from the recurrence of the main text."""
    d = 1 if i == j else 0
    return (n[j] * He_or_zero(shifted(n, (i, 1), (j, -1)))
            + n[i] * He_or_zero(shifted(n, (i, -1), (j, 1)))
            + n[i] * (n[j] - d) * He_or_zero(shifted(n, (i, -1), (j, -1))))


def multi_indices(order_max):
    return [n for n in itertools.product(range(order_max + 1), repeat=2 * N)
            if sum(n) <= order_max]


# ======================================================================= C0
def check_normalisation(rec):
    rho = rho_matrix()
    means = [[gauss_mean_x(Pi(l, lp)) for lp in range(N)] for l in range(N)]
    ok_mean = all(is_zero(means[l][lp] - (1 if l == lp else 0))
                  for l in range(N) for lp in range(N))
    qphys = ex(sum(Pi(l, lp) * rho[lp, l] for l in range(N) for lp in range(N)))
    read = sp.Matrix(N, N, lambda k, kp: gauss_mean_x(ex(adag_a(k, kp) * qphys)))
    ok_read = all(is_zero(read[k, kp] - rho[kp, k]) for k in range(N) for kp in range(N))
    ok_form = True
    for kind in ("real", "complex"):
        h, Vb, _, _ = hamiltonian(kind, "untraced")
        lhs = Hm_of(h, Vb) - P**2 / (2 * M) - Vb
        rhs = ex(sum(h[l, lp] * adag_a(l, lp) for l in range(N) for lp in range(N)))
        ok_form &= is_zero(lhs - rhs)
    rec.check("C0.gaussian_mean_of_Pi_is_delta", 0.0 if ok_mean else 1.0, 0.5)
    rec.check("C0.readout_of_symbol_of_ket_k_bra_kprime_is_rho_kprime_k",
              0.0 if ok_read else 1.0, 0.5)
    rec.check("C0.quadratic_form_equals_sum_h_adag_a", 0.0 if ok_form else 1.0, 0.5)
    return {"mean_Pi_is_delta": ok_mean, "readout_is_rho_transpose": ok_read,
            "H_form_equals_adag_a_form": ok_form}


# ======================================================================= C1
def check_qcle(rec):
    rows = []
    for kind in ("real", "complex"):
        for conv in ("traceless", "untraced"):
            h, Vb, hphys, Vbphys = hamiltonian(kind, conv)
            Hm = Hm_of(h, Vb)
            rho = rho_matrix()
            qphys = ex(sum(Pi(l, lp) * rho[lp, l] for l in range(N) for lp in range(N)))
            F = -(sp.diff(hphys, R) + sp.diff(Vbphys, R) * sp.eye(N))
            dP = sp.diff(rho, P)
            Q = (-sp.I / hb * (hphys * rho - rho * hphys) - P / M * sp.diff(rho, R)
                 - (F * dP + dP * F) / 2)
            target = ex(sum(Pi(l, lp) * Q[lp, l] for l in range(N) for lp in range(N)))
            res = ex(pb_amp(Hm, qphys) + excess_amp(h, qphys) - target)
            ok = is_zero(res)
            row = {"h": kind, "convention": conv, "residual_is_zero": ok}
            rec.check(f"C1.{kind}.{conv}.exact_generator_reproduces_qcle", 0.0 if ok else 1.0, 0.5)
            if kind == "complex" and conv == "untraced":
                bad = ex(pb_amp(Hm, qphys) + excess_amp(h, qphys, real_symmetric_form=True) - target)
                row["real_symmetric_form_residual_is_zero"] = is_zero(bad)
                rec.check("C1.complex.real_symmetric_form_of_excess_fails",
                          1.0 if row["real_symmetric_form_residual_is_zero"] else 0.0, 0.5,
                          "the D_ll' form discards Im h and must not reproduce the QCLE")
            rows.append(row)
            print(f"[17] C1 {kind:7s} {conv:9s} residual zero: {ok}", flush=True)
    return rows


# ======================================================================= C2
def Dz(i, q):
    """varphi^{-1} d_{z_i} (varphi q), varphi the unit Gaussian."""
    return ex(sp.diff(q, Z[i]) - Z[i] * q)


def check_identities(rec, order_max):
    n_cases = n_bad_raise = n_bad_force = 0
    for n in multi_indices(order_max):
        Hn = He_multi(n)
        for i in range(2 * N):
            for j in range(2 * N):
                n_cases += 1
                up = He_multi(shifted(n, (i, 1), (j, 1)))
                lhs = Dz(i, Dz(j, Hn))
                n_bad_raise += not is_zero(lhs - up)
                d = 1 if i == j else 0
                n_bad_force += not is_zero(ex((Z[i] * Z[j] - d) * Hn) - ex(up + T_action(n, i, j)))
    rec.check("C2.second_derivative_is_pure_raising", n_bad_raise, 0.5)
    rec.check("C2.force_split_raising_plus_T", n_bad_force, 0.5)
    print(f"[17] C2 {n_cases} (n, i, j) cases, |n| <= {order_max}: failures "
          f"{n_bad_raise}, {n_bad_force}", flush=True)
    return {"order_max": order_max, "n_multi_indices": len(multi_indices(order_max)),
            "n_cases": n_cases, "raising_failures": n_bad_raise, "force_split_failures": n_bad_force}


# ======================================================================= C3
def check_coefficient_operator(rec, order_max):
    rows = []
    c = sp.Function("c")(R, P)
    for kind in ("real", "complex"):
        h, Vb, _, _ = hamiltonian(kind, "traceless")
        Hm = Hm_of(h, Vb)
        H = big_H(h)
        K = (J * H) / hb
        dH = sp.diff(H, R)
        anti = all(is_zero(K[i, j] + K[j, i]) for i in range(2 * N) for j in range(2 * N))
        commute = all(is_zero(e) for e in (H * J - J * H))
        bad_exact = bad_pb = 0
        for n in multi_indices(order_max):
            Hn_x = ex(He_multi(n).subs({Z[i]: sp.sqrt(2 / hb) * X[i] for i in range(2 * N)}))
            q = ex(c * Hn_x)
            exact = ex((pb_amp(Hm, q) + excess_amp(h, q)).subs(TO_Z))
            pbonly = ex(pb_amp(Hm, q).subs(TO_Z))
            rot = sum(-K[i, j] * n[i] * He_or_zero(shifted(n, (i, -1), (j, 1)))
                      for i in range(2 * N) for j in range(2 * N) if K[i, j] != 0)
            force = sum(dH[i, j] * T_action(n, i, j)
                        for i in range(2 * N) for j in range(2 * N) if dH[i, j] != 0)
            pred = ex(-(P / M) * sp.diff(c, R) * He_multi(n) + c * rot
                      + sp.diff(Vb, R) * sp.diff(c, P) * He_multi(n)
                      + sp.Rational(1, 4) * sp.diff(c, P) * force)
            raise_ = sum(dH[i, j] * He_multi(shifted(n, (i, 1), (j, 1)))
                         for i in range(2 * N) for j in range(2 * N) if dH[i, j] != 0)
            bad_exact += not is_zero(exact - pred)
            bad_pb += not is_zero(pbonly - pred - sp.Rational(1, 4) * sp.diff(c, P) * raise_)
        rows.append({"h": kind, "order_max": order_max,
                     "n_basis_functions": len(multi_indices(order_max)),
                     "K_antisymmetric": anti, "H_commutes_with_J": commute,
                     "exact_failures": bad_exact, "poisson_bracket_failures": bad_pb})
        rec.check(f"C3.{kind}.K_antisymmetric", 0.0 if anti else 1.0, 0.5)
        rec.check(f"C3.{kind}.H_commutes_with_J", 0.0 if commute else 1.0, 0.5)
        rec.check(f"C3.{kind}.generator_is_coefficient_operator", bad_exact, 0.5)
        rec.check(f"C3.{kind}.poisson_bracket_is_operator_plus_raising", bad_pb, 0.5)
        print(f"[17] C3 {kind:7s} {len(multi_indices(order_max))} basis functions: "
              f"failures exact {bad_exact}, Poisson bracket {bad_pb}", flush=True)
    return rows


# ======================================================================= C4
def check_pbme_deviation(rec):
    rows = []
    for kind in ("real", "complex"):
        for conv in ("traceless", "untraced"):
            h, Vb, _, _ = hamiltonian(kind, conv)
            rho = rho_matrix()
            qphys = ex(sum(Pi(l, lp) * rho[lp, l] for l in range(N) for lp in range(N)))
            delta = ex((-excess_amp(h, qphys)).subs(TO_Z))          # PBME minus exact
            cz = hermite_coefficients(delta)
            orders = sorted({sum(m) for m in cz})
            low = ex(sum(cf * He_multi(m) for m, cf in cz.items() if sum(m) <= 2))
            Piz = [[ex(Pi(l, lp).subs(TO_Z)) for lp in range(N)] for l in range(N)]
            dh = sp.diff(h, R)
            trP = sp.diff(rho.trace(), P)
            phys_part = ex(sp.Rational(1, 4) * trP *
                           sum(Piz[l][lp] * dh[lp, l] for l in range(N) for lp in range(N)))
            gauss_part = ex(-sp.Rational(1, 4) * trP * dh.trace())
            in_sector = is_zero(low - phys_part)
            with_gauss = is_zero(low - phys_part - gauss_part)
            # order 4 against the Hermite ordering of q4 = (1/4) sum dH_ij z_i z_j sum Pi2 dP rho
            dH = sp.diff(big_H(h), R)
            Pi2 = [[ex(Piz[l][lp] + (1 if l == lp else 0)) for lp in range(N)] for l in range(N)]
            q4 = ex(sp.Rational(1, 4) * sum(dH[i, j] * Z[i] * Z[j] for i in range(4) for j in range(4))
                    * sum(Pi2[l][lp] * sp.diff(rho[lp, l], P) for l in range(N) for lp in range(N)))
            poly4 = sp.Poly(q4, *Z)
            wick = ex(sum(cf * He_multi(m) for m, cf in zip(poly4.monoms(), poly4.coeffs())))
            top = ex(sum(cf * He_multi(m) for m, cf in cz.items() if sum(m) == 4))
            wick_ok = is_zero(top - wick)
            product_differs = not is_zero(top - q4)
            n4 = sum(1 for m in cz if sum(m) == 4)
            n4_total = sum(1 for m in multi_indices(4) if sum(m) == 4)
            row = {"h": kind, "convention": conv, "hermite_orders": orders,
                   "order2_is_physical_density_of_quarter_dRh_dPtr": in_sector,
                   "order2_is_that_minus_quarter_trace_dRh_dPtr_times_gaussian": with_gauss,
                   "order4_is_hermite_ordered_q4": wick_ok,
                   "q4_product_differs_from_order4_part": product_differs,
                   "n_order4_functions_occupied": n4, "n_order4_functions": n4_total}
            rows.append(row)
            tag = f"C4.{kind}.{conv}"
            rec.check(f"{tag}.orders_are_2_and_4", 0.0 if orders == [2, 4] else 1.0, 0.5)
            if conv == "traceless":
                rec.check(f"{tag}.order2_part_in_physical_sector", 0.0 if in_sector else 1.0, 0.5)
            else:
                rec.check(f"{tag}.order2_part_is_physical_plus_gaussian", 0.0 if with_gauss else 1.0, 0.5)
            rec.check(f"{tag}.order4_is_hermite_ordered_q4", 0.0 if wick_ok else 1.0, 0.5)
            rec.check(f"{tag}.plain_product_is_not_the_order4_part", 0.0 if product_differs else 1.0, 0.5)
            print(f"[17] C4 {kind:7s} {conv:9s} orders {orders}; order-2 in sector {in_sector}, "
                  f"= physical + Gaussian {with_gauss}; order 4 = :q4: {wick_ok}; "
                  f"{n4}/{n4_total} order-4 functions", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity-order", type=int, default=6)
    ap.add_argument("--operator-order", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "raw" / "17_generator_structure.json"))
    args = ap.parse_args()
    rec = RunRecorder(script=Path(__file__).name, config=vars(args), seed=None)
    t0 = time.perf_counter()
    rec.results["C0_normalisation"] = check_normalisation(rec)
    rec.results["C1_qcle"] = check_qcle(rec)
    rec.results["C2_identities"] = check_identities(rec, args.identity_order)
    rec.results["C3_coefficient_operator"] = check_coefficient_operator(rec, args.operator_order)
    rec.results["C4_pbme_deviation"] = check_pbme_deviation(rec)
    rec.results["seconds"] = round(time.perf_counter() - t0, 1)
    rec.write(args.out)
    print(f"[17] checks: {len(rec.checks)}  all passed: {rec.all_passed}  "
          f"({rec.results['seconds']} s) -> {args.out}")
    return 0 if rec.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
