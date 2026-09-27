r"""Exact symbolic machinery for the closure analysis.

Everything here is exact computer algebra with arbitrary symbolic
:math:`h^{\lambda\lambda'}(R)`, :math:`V_b(R)` and
:math:`\rho_W^{\lambda\lambda'}(R,P)`, so a result obtained for a given
``(nstates, nbath)`` is a proof for that case rather than a numerical spot
check.

Three distinct leakage notions
------------------------------
They are related but not equal, and this module keeps them apart by name.

``mapping_polynomial_degrees``
    Ordinary total monomial degrees of the Gaussian-divided amplitude
    :math:`q=\rho/\phi`.  Cheap, and enough to *bound* Hermite order from
    above, because the Hermite change of basis is triangular.

``mapping_hermite_orders`` / :func:`hermite_defect`
    Genuine Hermite orders and the orthogonal residual
    :math:`(I-\mathsf P_K)q` in the inner product
    :math:`\langle f,g\rangle_\varphi=\int dz\,\varphi f^{*}g`.

``physical_frame_residual``
    Departure from :math:`\mathcal F_{\rm phys}=
    \mathrm{span}\{\phi\Pi_{\lambda\lambda'}\}`, of complex dimension
    :math:`N^{2}`, which is a *proper* subspace of :math:`V_2`.  For
    :math:`N=2` the even part of :math:`V_2` alone has dimension 11, so
    :math:`V_2` invariance does not imply physical-frame invariance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp

from .hermite.basis import he_multi, multi_indices, standardise
from .hermite.projection import (hermite_orders, hermite_residual_symbolic,
                                 polynomial_degrees, top_degree_part)

__all__ = [
    "SymbolicSetup", "build_symbolic_setup", "projector_polynomial_sym",
    "real_form_2n", "real_form_parts",
    "apply_pbme", "apply_excess", "apply_full", "amplitude",
    "mapping_polynomial_degrees", "mapping_hermite_orders", "hermite_defect",
    "pbme_principal_polynomial_leakage", "pbme_hermite_defect",
    "physical_frame_projection", "physical_frame_residual",
    "qcle_reference_rhs", "verify_qcle_coefficient_correspondence", "vk_audit", "degree_report",
    "gaussian_expectation",
]


@dataclass
class SymbolicSetup:
    """Symbols and expressions for one ``(nstates, nbath)`` closure analysis."""

    nstates: int
    nbath: int
    hbar: Any
    mass: Any
    r: tuple
    p: tuple
    R: tuple
    P: tuple
    z: tuple                     #: standardised mapping variables
    h: sp.Matrix                 #: physical subsystem matrix h(R)
    h_map: sp.Matrix             #: matrix entering H_m under the convention
    h_exc: sp.Matrix             #: matrix entering the excess operator
    Vb: Any
    v_eff: Any
    rho: list                    #: rho[lam][lamp](R, P)
    phi: Any
    H_m: Any
    rho_m: Any                   #: projected mapping density
    bigH: sp.Matrix = None       #: real symmetric 2N x 2N form in H_m
    bigH_exc: sp.Matrix = None   #: real symmetric 2N x 2N form in the excess
    hermitian: bool = False      #: True when h carries an imaginary part

    @property
    def x(self) -> tuple:
        return tuple(self.r) + tuple(self.p)


def projector_polynomial_sym(a: int, b: int, r, p, hbar):
    r""":math:`\Pi_{\lambda\lambda'}=\frac{2}{\hbar}
    [u_{\lambda\lambda'}-iL_{\lambda\lambda'}-\frac{\hbar}{2}\delta]`."""
    u = r[a] * r[b] + p[a] * p[b]
    L = r[a] * p[b] - r[b] * p[a]
    return (2 / hbar) * (u - sp.I * L - hbar * int(a == b) / 2)


def real_form_2n(hmat: sp.Matrix) -> sp.Matrix:
    r"""The real symmetric :math:`2N\times2N` form of a Hermitian ``hmat``.

    Writing :math:`h=A+iB` with :math:`A=A^{\mathsf T}` and
    :math:`B=-B^{\mathsf T}` real, the MMST mapping symbol of
    :math:`\hat h=\sum h^{\lambda\lambda'}|\lambda\rangle\langle\lambda'|`
    is :math:`\frac1{2\hbar}(x^{\mathsf T}\mathbb Hx-\hbar\,\mathrm{Tr}A)` with

    .. math:: \mathbb H=\begin{pmatrix}A&-B\\B&A\end{pmatrix}
              =\mathbb H^{\mathsf T},

    in the ordering :math:`x=(r_0\ldots r_{N-1},p_0\ldots p_{N-1})`.  The
    imaginary parts of :math:`\hat h` cancel in the symbol because
    :math:`A` contracts with the symmetric bilinear
    :math:`r_\lambda r_{\lambda'}+p_\lambda p_{\lambda'}` and :math:`B` with
    the antisymmetric one, so the classical Hamiltonian stays real.

    For real symmetric ``hmat`` (:math:`B=0`) this reduces to
    :math:`\mathrm{diag}(h,h)`, which is the block structure the ``N``-state
    real analysis uses, so nothing changes in that case.

    The point of routing everything through this function is that Theorem 1
    depends on :math:`\mathbb H` only through *symmetry* -- the force channel
    contributes :math:`+\tfrac14\partial_R\mathbb H_{ij}z_iz_j` and the excess
    channel :math:`-\tfrac14\partial_R\mathbb H_{ij}\partial_{z_i}
    \partial_{z_j}`, whose top Hermite parts are equal for any symmetric
    :math:`\mathbb H`.  Complex Hermitian ``h`` is therefore covered by the
    same proof, which ``scripts/05_extra_cases.py --case Herm`` checks.
    """
    n = hmat.shape[0]
    A = sp.Matrix(n, n, lambda i, j: sp.re(hmat[i, j]))
    B = sp.Matrix(n, n, lambda i, j: sp.im(hmat[i, j]))
    return real_form_parts(A, B)


def real_form_parts(A: sp.Matrix, B: sp.Matrix) -> sp.Matrix:
    """:func:`real_form_2n` when the parts are already separate.

    Used by :func:`build_symbolic_setup`, which must not call ``sp.re`` on an
    undefined function: the assumption needed to make ``re(h00(R))`` collapse
    to ``h00(R)`` would change the function object's identity and break every
    substitution keyed on ``sp.Function("h00")`` elsewhere in the package.
    """
    return sp.Matrix(sp.BlockMatrix([[A, -B], [B, A]]))


def build_symbolic_setup(nstates: int = 2, nbath: int = 1,
                         convention: str = "traceless",
                         mixed_convention: bool = False,
                         hermitian: bool = False) -> SymbolicSetup:
    """Assemble the symbolic projected density and generator ingredients.

    ``mixed_convention=True`` deliberately uses the full ``h`` in ``H_m`` and
    the traceless matrix in the excess operator.  That combination is invalid
    and exists only so a regression test can assert that it *fails*.

    ``hermitian=True`` gives ``h`` a general antisymmetric imaginary part, so
    the subsystem matrix is complex Hermitian rather than real symmetric.  The
    mapping Hamiltonian and the excess operator are then both built from the
    real symmetric :math:`2N\\times2N` form of Eq. :func:`real_form_2n`, which
    is what they are built from in the real case too -- there the form happens
    to be block diagonal.  Nothing else in the analysis changes, which is the
    content of the extended Theorem 1.
    """
    hbar, mass = sp.symbols("hbar M", positive=True)
    r = sp.symbols(f"r0:{nstates}", real=True)
    p = sp.symbols(f"p0:{nstates}", real=True)
    R = sp.symbols(f"R0:{nbath}", real=True)
    P = sp.symbols(f"P0:{nbath}", real=True)
    z = sp.symbols(f"z0:{2 * nstates}", real=True)

    # NOTE: these must stay plain undefined functions.  Declaring them
    # ``real=True`` makes a DIFFERENT UndefinedFunction object, so
    # ``sp.Function("h00")(R0)`` built anywhere else -- mqcle.leakage does
    # exactly that when it lambdifies the defect -- no longer matches and the
    # substitution silently leaves derivatives behind.  The real and imaginary
    # parts are therefore carried as separate matrices instead of being
    # recovered with sp.re/sp.im.
    A = sp.Matrix(nstates, nstates,
                  lambda i, j: sp.Function(f"h{min(i, j)}{max(i, j)}")(*R))
    if hermitian:
        # antisymmetric imaginary part: g_{ij} = -g_{ji}, g_{ii} = 0
        def _b(i, j):
            if i == j:
                return sp.Integer(0)
            g = sp.Function(f"g{min(i, j)}{max(i, j)}")(*R)
            return g if i < j else -g
        B = sp.Matrix(nstates, nstates, _b)
    else:
        B = sp.zeros(nstates, nstates)
    h = A + sp.I * B
    Vb = sp.Function("Vb")(*R)
    rho = [[sp.Function(f"rho{a}{b}")(*R, *P) for b in range(nstates)]
           for a in range(nstates)]

    tr = A.trace() / nstates      # Hermitian => B has a zero diagonal
    h_traceless = h - tr * sp.eye(nstates)
    A_traceless = A - tr * sp.eye(nstates)
    if convention == "traceless":
        h_map, A_map, v_eff = h_traceless, A_traceless, Vb + tr
    elif convention == "full":
        h_map, A_map, v_eff = h, A, Vb
    else:
        raise ValueError(f"unknown convention '{convention}'")
    if mixed_convention:
        h_exc, A_exc = h_traceless, A_traceless
    else:
        h_exc, A_exc = h_map, A_map

    bigH = real_form_parts(A_map, B)
    bigH_exc = real_form_parts(A_exc, B)
    x = tuple(r) + tuple(p)

    H_m = sum(P[k]**2 / (2 * mass) for k in range(nbath)) + v_eff
    for i in range(2 * nstates):
        for j in range(2 * nstates):
            H_m += bigH[i, j] * x[i] * x[j] / (2 * hbar)
    H_m -= A_map.trace() / 2

    phi = sp.exp(-sum(r[m]**2 + p[m]**2 for m in range(nstates)) / hbar) \
        / (sp.pi * hbar) ** nstates
    rho_m = sp.expand(phi * sum(
        projector_polynomial_sym(a, b, r, p, hbar) * rho[b][a]
        for a in range(nstates) for b in range(nstates)))

    return SymbolicSetup(nstates, nbath, hbar, mass, r, p, R, P, z, h, h_map,
                         h_exc, Vb, v_eff, rho, phi, sp.expand(H_m), rho_m,
                         bigH, bigH_exc, hermitian)


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------
def apply_pbme(s: SymbolicSetup, f):
    r"""Density-side Poisson part :math:`\{H_m,f\}_{x,X}`."""
    out = 0
    for a in range(s.nstates):
        out += sp.diff(s.H_m, s.r[a]) * sp.diff(f, s.p[a])
        out -= sp.diff(s.H_m, s.p[a]) * sp.diff(f, s.r[a])
    for k in range(s.nbath):
        out += sp.diff(s.H_m, s.R[k]) * sp.diff(f, s.P[k])
        out -= sp.diff(s.H_m, s.P[k]) * sp.diff(f, s.R[k])
    return out


def apply_excess(s: SymbolicSetup, f):
    r"""Density-side excess part
    :math:`-\frac{\hbar}{8}\sum_k\partial_{R_k}\mathbb H_{ij}
    \partial_{x_i}\partial_{x_j}\partial_{P_k}f`.

    The contraction runs over the full :math:`2N` mapping index set against
    the real symmetric form :func:`real_form_2n` of the subsystem matrix --
    the *same* matrix that carries the quadratic part of ``H_m``.  For real
    symmetric ``h`` that form is block diagonal and this reduces to the
    familiar :math:`\nabla_R\mathsf h^{\lambda\lambda'}
    (\partial_{r_\lambda}\partial_{r_{\lambda'}}
    +\partial_{p_\lambda}\partial_{p_{\lambda'}})`.
    """
    out = 0
    x = s.x
    n2 = 2 * s.nstates
    bigH = s.bigH_exc if s.bigH_exc is not None else real_form_2n(s.h_exc)
    for k in range(s.nbath):
        fk = sp.diff(f, s.P[k])
        for i in range(n2):
            for j in range(n2):
                g = sp.diff(bigH[i, j], s.R[k])
                if g == 0:
                    continue
                out += g * sp.diff(fk, x[i], x[j])
    return -(s.hbar / 8) * out


def apply_full(s: SymbolicSetup, f):
    """Full generator, as the explicit sum of the two independent parts."""
    return apply_pbme(s, f) + apply_excess(s, f)


# --------------------------------------------------------------------------
# amplitudes, degrees, orders
# --------------------------------------------------------------------------
def amplitude(s: SymbolicSetup, expr, standardised: bool = True):
    r"""Return :math:`q=\rho/\phi`, in ``z`` variables when ``standardised``."""
    q = sp.expand(sp.cancel(sp.expand(expr) / s.phi))
    return standardise(q, s.x, s.z, s.hbar) if standardised else q


def mapping_polynomial_degrees(s: SymbolicSetup, expr) -> list[int]:
    """Ordinary total polynomial degrees of the Gaussian-divided amplitude.

    Not Hermite orders.  ``deg <= K`` implies membership in ``V_K``; it does
    not say how the amplitude is distributed over Hermite orders ``0..K``.
    """
    return polynomial_degrees(amplitude(s, expr), s.z)


def mapping_hermite_orders(s: SymbolicSetup, expr) -> list[int]:
    """Hermite orders present, computed from actual Hermite coefficients."""
    return hermite_orders(amplitude(s, expr), s.z)


def hermite_defect(s: SymbolicSetup, expr, max_order: int = 2):
    r""":math:`(I-\mathsf P_K)q` for the amplitude of ``expr``."""
    return hermite_residual_symbolic(amplitude(s, expr), s.z, max_order)


def pbme_principal_polynomial_leakage(s: SymbolicSetup, above: int = 2,
                                      standardised: bool = False):
    r"""Monomials of ordinary degree ``> above`` in
    :math:`\mathcal A_{\rm PB}\rho_m^{\mathcal P}/\phi`.

    This is :math:`\mathrm{Top}_{>K}`, **not** the orthogonal Hermite residual.
    Default output is in the original ``x`` variables, the form in which the
    closed expression was first written.
    """
    q = amplitude(s, sp.expand(apply_pbme(s, s.rho_m)), standardised=standardised)
    return top_degree_part(q, s.z if standardised else s.x, above)


def pbme_hermite_defect(s: SymbolicSetup, max_order: int = 2):
    r"""True :math:`(I-\mathsf P_K)` residual of
    :math:`\mathcal A_{\rm PB}\rho_m^{\mathcal P}/\phi`, in ``z``."""
    return hermite_defect(s, sp.expand(apply_pbme(s, s.rho_m)), max_order)


# --------------------------------------------------------------------------
# physical frame
# --------------------------------------------------------------------------
def gaussian_expectation(poly, z_syms) -> sp.Expr:
    r""":math:`\mathbb E_\varphi[\,\cdot\,]` for the standard Gaussian:
    odd monomials vanish and :math:`\mathbb E[z^{2k}]=(2k-1)!!`."""
    P = sp.Poly(sp.expand(sp.sympify(poly)), *z_syms)
    total = sp.Integer(0)
    for monom, coeff in zip(P.monoms(), P.coeffs()):
        if any(e % 2 for e in monom):
            continue
        m = sp.Integer(1)
        for e in monom:
            if e > 0:
                m *= sp.factorial2(e - 1)
        total += coeff * m
    return sp.expand(total)


def _physical_frame(s: SymbolicSetup) -> list[sp.Expr]:
    r"""The :math:`N^{2}` frame polynomials :math:`\Pi_{\lambda\lambda'}(z)`.

    The Gaussian factor is common to the frame and to the amplitude, so only
    the polynomial factors are carried.
    """
    return [standardise(projector_polynomial_sym(a, b, s.r, s.p, s.hbar),
                        s.x, s.z, s.hbar)
            for a in range(s.nstates) for b in range(s.nstates)]


def _conj_poly(expr, z_syms):
    """Complex conjugate of a polynomial with real variables."""
    return sp.expand(sp.conjugate(expr).replace(
        lambda e: isinstance(e, sp.conjugate), lambda e: e.args[0]))


def physical_frame_projection(s: SymbolicSetup, expr):
    r"""Orthogonally project the amplitude of ``expr`` onto
    :math:`\mathcal F_{\rm phys}` in :math:`\langle\cdot,\cdot\rangle_\varphi`.

    Conjugation touches only the frame functions, whose coefficients are
    explicit complex numbers; the symbolic bath fields are never conjugated.

    Returns ``(coefficients, projection, residual)``.  ``coefficients[a][b]``
    multiplies :math:`\Pi_{ab}`.
    """
    q = amplitude(s, expr)
    frame = _physical_frame(s)
    n2 = len(frame)
    cframe = [_conj_poly(f, s.z) for f in frame]
    G = sp.Matrix(n2, n2, lambda i, j: gaussian_expectation(
        sp.expand(cframe[i] * frame[j]), s.z))
    b = sp.Matrix(n2, 1, lambda i, _: gaussian_expectation(
        sp.expand(cframe[i] * q), s.z))
    C = G.LUsolve(b)
    proj = sp.expand(sum(C[i] * frame[i] for i in range(n2)))
    residual = sp.expand(sp.simplify(q - proj))
    coeff = [[sp.simplify(C[a * s.nstates + b]) for b in range(s.nstates)]
             for a in range(s.nstates)]
    return coeff, proj, residual


def physical_frame_residual(s: SymbolicSetup, expr):
    r""":math:`\epsilon_{\rm phys}=q-\mathcal P_{\rm phys}q`."""
    return physical_frame_projection(s, expr)[2]


# --------------------------------------------------------------------------
# QCLE correspondence
# --------------------------------------------------------------------------
def qcle_reference_rhs(s: SymbolicSetup) -> sp.Matrix:
    r"""The subsystem-basis QCLE right-hand side, as a symbolic matrix.

    .. math::
        -\frac{i}{\hbar}[h,\rho_W]-\frac{P}{M}\partial_R\rho_W
        -\frac12\bigl(F\partial_P\rho_W+\partial_P\rho_W F\bigr),
        \qquad F=-\partial_R(V_b\mathbb 1+h).

    Built from the *physical* ``h`` and ``V_b``, so it is independent of the
    trace convention chosen for the mapping side.  This is the single
    definition used both by Gate J (:func:`verify_qcle_coefficient_
    correspondence`) and by the Gate L numerical cross-check in
    ``tests/test_coefficient_operator.py`` -- the two must not drift apart.
    """
    n, nb = s.nstates, s.nbath
    rhoM = sp.Matrix(n, n, lambda a, b: s.rho[a][b])
    rhs = -sp.I / s.hbar * (s.h * rhoM - rhoM * s.h)
    for k in range(nb):
        F = -sp.diff(s.Vb, s.R[k]) * sp.eye(n) - sp.diff(s.h, s.R[k])
        dP = sp.Matrix(n, n, lambda a, b: sp.diff(s.rho[a][b], s.P[k]))
        dR = sp.Matrix(n, n, lambda a, b: sp.diff(s.rho[a][b], s.R[k]))
        rhs = rhs - s.P[k] / s.mass * dR - sp.Rational(1, 2) * (F * dP + dP * F)
    return rhs


def verify_qcle_coefficient_correspondence(s: SymbolicSetup) -> dict[str, Any]:
    r"""Gate J: the closed coefficients must obey the subsystem-basis QCLE.

    Solves :math:`\mathcal A_m\rho_m^{\mathcal P}
    =\phi\sum\Pi_{\lambda\lambda'}C_{\lambda'\lambda}` and compares against

    .. math::
        \partial_t\rho_W=-\frac{i}{\hbar}[h,\rho_W]-\frac{P}{M}\!\cdot\!
        \nabla_R\rho_W-\frac12\bigl(F\nabla_P\rho_W+\nabla_P\rho_W F\bigr),
        \qquad F=-\nabla_R(V_b\mathbb 1+h).

    The reference uses the *physical* ``h`` and ``V_b``, so the gate also
    checks that the trace convention was applied consistently.
    """
    n = s.nstates
    coeff, _, residual_frame = physical_frame_projection(
        s, sp.expand(apply_full(s, s.rho_m)))

    rhs = qcle_reference_rhs(s)

    residuals = {}
    for a in range(n):
        for b in range(n):
            # coeff[a][b] multiplies Pi_{ab}, which pairs with rho_W^{ba}
            residuals[f"residual_{b}{a}"] = sp.simplify(
                sp.expand(coeff[a][b] - rhs[b, a]))
    return {"residuals": {k: str(v) for k, v in residuals.items()},
            "all_zero": all(v == 0 for v in residuals.values()),
            "frame_residual_is_zero": bool(sp.simplify(residual_frame) == 0)}


# --------------------------------------------------------------------------
# V_K audit
# --------------------------------------------------------------------------
def vk_audit(s: SymbolicSetup, orders, max_indices: int | None = None
             ) -> dict[str, Any]:
    r"""Independent regression test of :math:`\mathcal A_mV_K\subseteq V_K`.

    Applies each generator to basis densities
    :math:`\phi(x)\mathrm{He}_{\mathbf n}(z(x))f(R,P)` with :math:`|n|=K` and a
    generic symbolic bath field ``f``, then reports the highest **Hermite**
    order in the output.  Expected: ``full_max <= K`` and, on suitable basis
    functions, ``pbme_max = K + 2``.

    This does not replace the analytical theorem; it checks the theorem's
    conclusion on explicit basis elements.
    """
    f = sp.Function("f")(*s.R, *s.P)
    dim = 2 * s.nstates
    results: dict[str, Any] = {}
    for K in orders:
        idxs = multi_indices(dim, K, exact=True)
        n_total = len(idxs)
        if max_indices is not None:
            idxs = idxs[:max_indices]
        rec = {"n_basis_functions_total": n_total,
               "n_basis_functions_checked": len(idxs),
               "full_max": -1, "pbme_max": -1, "excess_max": -1}
        for idx in idxs:
            he_x = he_multi(idx, [sp.sqrt(2 / s.hbar) * v for v in s.x])
            dens = sp.expand(s.phi * he_x * f)
            for name, op in (("full", apply_full), ("pbme", apply_pbme),
                             ("excess", apply_excess)):
                out = sp.expand(op(s, dens))
                got = mapping_hermite_orders(s, out) or [0]
                rec[f"{name}_max"] = max(rec[f"{name}_max"], max(got))
        rec["full_within_VK"] = bool(rec["full_max"] <= K)
        rec["pbme_raises_by_two"] = bool(rec["pbme_max"] == K + 2)
        results[str(K)] = rec
    return results


# --------------------------------------------------------------------------
# convenience report
# --------------------------------------------------------------------------
def degree_report(nstates: int = 2, nbath: int = 1,
                  convention: str = "traceless",
                  mixed_convention: bool = False,
                  powers: int = 1) -> dict[str, Any]:
    """Ordinary polynomial degrees generated by each operator."""
    s = build_symbolic_setup(nstates, nbath, convention, mixed_convention)
    ops = {"pbme": apply_pbme, "excess": apply_excess, "full": apply_full}
    out: dict[str, Any] = {
        "nstates": nstates, "nbath": nbath, "convention": convention,
        "mixed_convention": mixed_convention,
        "initial_polynomial_degrees": mapping_polynomial_degrees(s, s.rho_m),
        "initial_hermite_orders": mapping_hermite_orders(s, s.rho_m),
        "powers": {},
    }
    for name, op in ops.items():
        expr = s.rho_m
        seq = []
        for _ in range(powers):
            expr = sp.expand(op(s, expr))
            seq.append(mapping_polynomial_degrees(s, expr))
        out["powers"][name] = seq
    return out
