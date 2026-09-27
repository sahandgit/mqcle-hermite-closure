#!/usr/bin/env python3
"""Stage 5 -- the final acceptance audit, checked rather than asserted.

Each line of the directive's acceptance list becomes a programmatic check
against the repository and the archive.  The script prints PASS/FAIL with the
evidence for each, and exits non-zero if anything fails, so "the manuscript is
ready" is a computed statement.
"""
from __future__ import annotations

import json
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
MS = ROOT / "manuscript"
sys.path.insert(0, str(ROOT / "src"))

from mqcle.benchmark_regimes import MOMENTA  # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def load(name):
    p = RAW / name
    return json.loads(p.read_text()) if p.is_file() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def p10_refinement_is_current(doc: dict | None) -> bool:
    """The derived acceptance must be bound to the exact four raw inputs."""
    if not doc or doc.get("all_passed") is not True:
        return False
    expected = doc.get("results", {}).get("input_sha256", {})
    inputs = {
        "fine_json": RAW / "07_coefficient_propagation_P10.json",
        "fine_h5": RAW / "07_coefficient_propagation_P10.h5",
        "coarse_json": (ROOT / "results" / "failed" /
                        "P10_2048x1024_20260829" /
                        "07_coefficient_propagation_P10.json"),
        "coarse_h5": (ROOT / "results" / "failed" /
                      "P10_2048x1024_20260829" /
                      "07_coefficient_propagation_P10.h5"),
    }
    return all(path.is_file() and expected.get(name) == _sha256(path)
               for name, path in inputs.items())


def primary_hermite_propagation_name(P0: int) -> str:
    r"""PRIMARY result: the Gaussian--Hermite coefficient propagation.

    This is the archive the manuscript's computational claims rest on.  It is
    produced by ``scripts/07g_gh_coefficient_propagation.py``, which advances
    the eleven-component coefficient field through the exact finite Hermite
    operator and never forms :math:`\rho_W` while propagating.
    """
    return f"07g_gh_coefficient_propagation_P{P0}.json"


def primary_hermite_propagation_h5_name(P0: int) -> str:
    return f"07g_gh_coefficient_propagation_P{P0}.h5"


def qcle_reference_propagation_name(P0: int) -> str:
    r"""INDEPENDENT REFERENCE: the subsystem-QCLE grid/Strang propagation.

    Produced by ``scripts/07_coefficient_propagation.py``, which advances the
    four already-reduced fields :math:`\rho_W^{\lambda\lambda'}` with a Strang
    split.  It is a verification comparison, not the primary method, and its
    outputs must never be relabelled as Gaussian--Hermite output.
    """
    return ("13_P20_wide_momentum_box.json" if P0 == 20
            else f"07_coefficient_propagation_P{P0}.json")


def qcle_reference_propagation_h5_name(P0: int) -> str:
    return ("13_P20_wide_momentum_box.h5" if P0 == 20
            else f"07_coefficient_propagation_P{P0}.h5")


def main() -> int:
    tex = (MS / "mqcle_hermite_closure.tex").read_text()
    sup = (MS / "supplementary_material.tex").read_text()
    def strip_comments(t: str) -> str:
        """LaTeX comments are not part of the document and must not be audited.

        An earlier version flagged a \\needs{} that appears only inside the
        file's own header comment."""
        out = []
        for line in t.splitlines():
            i, esc = None, False
            for k, ch in enumerate(line):
                if ch == "\\":
                    esc = not esc
                elif ch == "%" and not esc:
                    i = k
                    break
                else:
                    esc = False
            out.append(line if i is None else line[:i])
        return "\n".join(out)

    tex, sup = (strip_comments(t) for t in (tex, sup))
    body = tex + sup

    # 1. the projector onto V_2 is defined together with its inner product and
    #    the norm in which the physical-sector residual is measured
    has_pk = (r"\mathsf P_2" in tex and "orthogonal projector" in " ".join(tex.split())
              and "Gaussian-weighted inner product" in " ".join(tex.split())
              and r"\label{eq:eps}" in tex)
    check(has_pk, "P_2 defined with its inner product and norm",
          "orthogonal projector in the Gaussian-weighted inner product; norm of eq:eps")

    # 2. every number of the text is a macro written from results/raw, with
    #    its provenance recorded under the same name
    pv_tex, pv_json = MS / "paper_values.tex", ROOT / "results" / "tables" / "paper_values.json"
    ok_pv, ev_pv = False, "paper_values.tex or its ledger missing"
    if pv_tex.is_file() and pv_json.is_file():
        names_tex = set(re.findall(r"\\newcommand\{\\(\w+)\}", pv_tex.read_text()))
        ledger = json.loads(pv_json.read_text()).get("macros", {})
        no_src = sorted(n for n, v in ledger.items() if not v.get("source"))
        ok_pv = names_tex == set(ledger) and not no_src and bool(names_tex)
        ev_pv = (f"{len(names_tex)} macros, each with a recorded source"
                 if ok_pv else f"tex/ledger mismatch or missing sources: {no_src[:5]}")
    check(ok_pv, "every quoted number is a generated macro with provenance", ev_pv)
    used = set(re.findall(r"\\([A-Za-z]+)", tex + sup))
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", pv_tex.read_text())) \
        if pv_tex.is_file() else set()
    unused = sorted(n for n in defined if n not in used)
    check(True, "generated macros not quoted anywhere (informational)",
          f"{len(unused)} of {len(defined)} unused")

    # 2b. the computer-algebra checks of the generator, Table SI
    d17 = load("17_generator_structure.json")
    check(bool(d17) and d17.get("all_passed") is True,
          "computer-algebra checks of the generator pass",
          "17_generator_structure.json" if d17 else "archive missing")

    # 2c. the terminal phase-space comparison: the PBME field re-propagates the
    #     projected-density ensemble trajectory for trajectory, and every
    #     comparison passed its own checks
    ps = {P: (load(f"18_pbme_phase_space_P{P}.json"),
              load(f"19_phase_space_comparison_P{P}.json")) for P in (20, 30, 70, 100)}
    have = [P for P, (a, b) in ps.items() if a and b]
    ok = len(have) == 4 and all(ps[P][0].get("all_passed") is True
                                and ps[P][1].get("all_passed") is True for P in have)
    check(ok, "phase-space comparison archives pass their checks",
          f"{len(have)}/4 regimes present, all passed={ok}")

    # 3. physical-frame residual zero for the exact generator
    d04 = load("04_physical_frame.json")
    check(bool(d04) and d04["results"]["cases"]["full"]["residual_is_zero"]
          and not d04["results"]["cases"]["pbme"]["residual_is_zero"],
          "physical-frame residual exactly zero for A_m, nonzero for A_PB",
          "04_physical_frame.json")

    # 4. QCLE coefficient residuals zero
    d03 = load("03_qcle_correspondence.json")
    check(bool(d03) and d03["all_passed"],
          "QCLE coefficient residuals all zero, both trace conventions",
          "03_qcle_correspondence.json")

    # 5. N = 3 and multiple bath modes point at real files
    ok = all((RAW / f).is_file() for f in
             ("05_extra_cases_N3.json", "06_extra_cases_Nb2.json"))
    check(ok, "N=3 and two-bath-mode claims point at archived runs",
          "05_extra_cases_N3.json, 06_extra_cases_Nb2.json")

    # 5b. Corollary 1: complex Hermitian h, both conventions and N = 2, 3
    herm = {n: load(n) for n in ("06b_extra_cases_hermitian.json",
                                 "06b_extra_cases_hermitian_full.json",
                                 "06c_extra_cases_hermitian_N3.json")}
    have = {k: v for k, v in herm.items() if v}
    ok = bool(have) and all(v["all_passed"] for v in have.values())
    really_complex = all(
        v["results"].get("h_has_nonzero_imaginary_part") for v in have.values())
    check(ok and really_complex and len(have) == 3,
          "Corollary 1 verified for complex Hermitian h, N=2 and N=3",
          f"{len(have)}/3 runs present, all passed={ok}, "
          f"Im h nonzero in each={really_complex}")

    # 5c. the residual freedom acts by covariance, checked where it can act
    dcov = load("07c_basis_covariance.json")
    if dcov:
        r = dcov["results"]
        active = (r["grad_h_over_propagated_region"]
                  >= 0.5 * r["grad_h_global_max"])
        check(dcov["all_passed"] and active
              and r.get("n_complex_rotations", 0) >= 1,
              "basis covariance measured where grad_R h is large, with a "
              "genuinely complex U",
              f"|grad_R h| = {r['grad_h_over_propagated_region']:.3e} "
              f"({r['grad_h_over_propagated_region'] / r['grad_h_global_max']:.2f}"
              f" of its global max), {r.get('n_complex_rotations', 0)} complex "
              f"rotations, worst deviation "
              f"{max(r['worst_relative_pbme'], r['worst_relative_closed']):.2e}")

    # 5d. Gate M has a second axis that actually varies the predictor
    dcorr = load("12_leakage_reanalysis.json")
    if dcorr:
        lr = dcorr["results"]
        cs = lr.get("coupling_scan", {})
        check(cs.get("n_points", 0) >= 5,
              "Gate M carries a coupling axis, not only a momentum axis",
              f"{cs.get('n_points', 0)} coupling points; predictor range "
              f"{cs.get('predictor_dynamic_range', float('nan')):.2f} against "
              f"response range {cs.get('response_dynamic_range', float('nan')):.0f}")
        rows = lr["rows"]
        response = lr.get("response_variable", "")
        check(all("discard_cost" in r and "discard_cost_sem" in r for r in rows)
              and "projected" in response.lower(),
              "the leakage is correlated against the discard cost, not against "
              "PBME's error versus the exact reference",
              response or "no projected response recorded")
    else:
        check(False, "Gate M carries a coupling axis, not only a momentum axis",
              "12_leakage_reanalysis.json missing")
        check(False, "the leakage is correlated against the discard cost, not "
                     "against PBME's error versus the exact reference",
              "12_leakage_reanalysis.json missing")

    # 6. propagation passes time-step AND grid convergence, or is absent
    d09 = load("09_convergence.json")
    props = sorted(RAW.glob("07_coefficient_propagation_P*.json"))
    wide20 = RAW / "13_P20_wide_momentum_box.json"
    if wide20.is_file():
        props.append(wide20)
    if props and d09 is None:
        check(False, "propagation results require a convergence study",
              "07_* present but 09_convergence.json missing")
    elif d09 is not None:
        ts = d09["results"]["time_step"]
        order_ok = abs(ts["fitted_order"] - ts["expected_order"]) < 0.35
        mono = all(ts["rows"][i]["error"] > ts["rows"][i + 1]["error"]
                   for i in range(len(ts["rows"]) - 1))
        check(order_ok and mono,
              "measured time-step convergence at the expected order",
              f"fitted {ts['fitted_order']:.3f} vs expected "
              f"{ts['expected_order']:g}, monotone={mono}")
        gr = max(d09["results"]["n_R"]["rows"][-1]["error"],
                 d09["results"]["n_P"]["rows"][-1]["error"])
        check(gr < 1e-5, "grid convergence in n_R and n_P", f"max error {gr:.3e}")
        bx = max(r["error_vs_smallest_box"] for r in d09["results"]["box"]["rows"])
        check(bx < 1e-7, "box independence", f"max change {bx:.3e}")
    else:
        check(True, "no propagation results and none claimed", "vacuous")

    # documented exceptions: a recorded check that does not pass and was NOT
    # tuned to pass.  Each names the CORRECTIONS.md entry that justifies it.
    p10_refinement = load("07c_p10_refinement_acceptance.json")
    p10_refinement_ok = p10_refinement_is_current(p10_refinement)
    EXCEPTIONS = ({
        ("07_coefficient_propagation_P10.json", "gateL.boundary_mass_R"):
            "CORRECTIONS.md ISSUE R21; 07c two-grid observable acceptance",
        ("07_coefficient_propagation_P10.json", "gateL.boundary_mass_P"):
            "CORRECTIONS.md ISSUE R21; 07c two-grid observable acceptance",
    } if p10_refinement_ok else {})
    for p in props:
        d = json.loads(p.read_text())
        failed = [c for c in d["checks"] if not c["passed"]]
        excused = [c for c in failed if (p.name, c["name"]) in EXCEPTIONS]
        real = [c for c in failed if (p.name, c["name"]) not in EXCEPTIONS]
        note = ("all pass" if not failed else
                "; ".join(f"{c['name']}={c['value']:.2e} [documented: "
                          f"{EXCEPTIONS[(p.name, c['name'])]}]"
                          if (p.name, c['name']) in EXCEPTIONS
                          else f"{c['name']}={c['value']:.2e}" for c in failed))
        check(not real, f"{p.name}: every recorded check passes or is documented",
              note)
        for c in excused:
            check(True, f"{p.name}: {c['name']} is a documented exception",
                  EXCEPTIONS[(p.name, c["name"])])

    dgi = load("07b_grid_independence_P20.json")
    if dgi:
        check(dgi["all_passed"],
              "the P0=20 observable is grid independent at the level reported",
              f"motion is {dgi['results']['spread_over_reported_error']:.2e} "
              f"of the reported error")

    # 7. polynomial degree, Hermite order and the physical sector kept apart
    flat = " ".join(body.lower().split())
    has_three = ("polynomial degree" in flat and "hermite order" in flat
                 and "physical sector" in flat
                 and "hermite-ordered form, not the product itself" in flat)
    check(has_three,
          "polynomial degree, Hermite order and physical sector kept distinct",
          "the defect is identified as the Hermite-ordered product, not the product")

    # 8. recorded run parameters equal the parameters used
    ok, why = True, []
    for p in props:
        d = json.loads(p.read_text())
        cfg = d["metadata"]["config"]
        cmd = d["metadata"]["command_line"]
        m = re.search(r"--P0\s+(\S+)", cmd)
        if m and abs(float(m.group(1)) - cfg["P0"]) > 1e-12:
            ok = False
            why.append(f"{p.name}: command line P0 != recorded P0")
    check(ok, "recorded parameters match the command line",
          "; ".join(why) or "every 07 run's config echoes its command line")

    # 9. no placeholders, no unresolved references
    placeholders = [w for w in ("RESULT REQUIRED", "NUMERICAL VALIDATION REQUIRED",
                                "PENDING", "Eq.~(??)", "Sec.~??")
                    if w in body]
    check(not placeholders, "no result placeholders in the manuscript",
          ", ".join(placeholders) or "none")

    outstanding = re.findall(r"\\needs\{([^}]*)\}", body)
    # match on the leading keyword: the markers now carry instructions to the
    # authors, so an exact-string allow-list rotted the moment one was reworded
    ALLOWED_KEYWORDS = ("ORCID", "CRediT", "DATA REPOSITORY",
                        "CORRESPONDING AUTHOR", "AI TOOL DISCLOSURE")
    def _head(s):
        return " ".join(s.split())[:60]
    unexpected = sorted({_head(s) for s in outstanding
                         if not any(k in s for k in ALLOWED_KEYWORDS)})
    check(not unexpected,
          "the only outstanding markers are front matter that cannot be invented",
          "; ".join(unexpected) if unexpected else
          f"{len(outstanding)} markers, all front matter: "
          + "; ".join(sorted({s.split(":")[0].split(".")[0].strip()
                              for s in outstanding})))

    for stem in ("mqcle_hermite_closure", "supplementary_material"):
        log = MS / f"{stem}.log"
        if not log.is_file():
            check(False, f"{stem}: compiled", "no log")
            continue
        txt = log.read_text(errors="ignore")
        undef = len(re.findall(r"LaTeX Warning: (?:Reference|Citation) ", txt))
        check(undef == 0, f"{stem}: no undefined references",
              f"{undef} undefined reference/citation warnings")

    # 10. CLAIM_TABLE has a verdict for every claim
    ct = (ROOT / "CLAIM_TABLE.md")
    check(ct.is_file() and "Verdict" in ct.read_text(),
          "CLAIM_TABLE.md carries a verdict and a category per claim",
          "CLAIM_TABLE.md")
    corr = (ROOT / "CORRECTIONS.md")
    check(corr.is_file() and corr.read_text().count("**SEVERITY**") >= 5,
          "CORRECTIONS.md holds a report per fix in the required format",
          f"{corr.read_text().count('**SEVERITY**')} reports")

    # =====================================================================
    # the final revision's gates (PART XXVI).  Each one exists because the
    # previous revision would have passed without it.
    # =====================================================================

    # F1. the full quantum reference has its own convergence file, and its
    #     uncertainty is below the smallest physical difference it supports
    ref_ok, ref_ev = True, []
    for P0 in MOMENTA:
        c = load(f"10_schrodinger_convergence_P{P0}.json")
        g = load(primary_hermite_propagation_name(P0))
        if not c or not g:
            ref_ok = False
            ref_ev.append(f"P{P0}: no convergence or GH file")
            continue
        q = c["results"]
        # the reference is the most refined run; its uncertainty is the largest
        # of the Richardson estimate and the grid and box changes
        u = max(q["temporal"]["richardson_remaining_at_dt_over_4"],
                q["comparisons_against_production"]["grid_x2"]["final_population_difference"],
                q["comparisons_against_production"]["box_enlarged"]["final_population_difference"])
        ref = q["reference_uncertainty"]["converged_value_final_populations"][1]
        e = abs(g["results"]["final"]["populations"][1] - ref)
        # an order of magnitude quieter than the difference it resolves
        good = u < 0.1 * e
        ref_ok &= bool(good)
        ref_ev.append(f"P{P0}: u={u:.1e} vs E_QCLE={e:.1e}")
    check(ref_ok, "the full quantum reference is converged below the "
                  "differences it resolves", "; ".join(ref_ev))

    # F1b. A separately implemented real-component solver must agree with the
    # closed complex coefficient solver, while being labelled as the same
    # equation rather than misrepresented as an independent physical theory.
    grid_qcle = load("10b_qcle_grid_crosscheck_P40.json")
    if grid_qcle:
        gr = grid_qcle["results"]
        interpretation = gr.get("interpretation", {})
        comparison = gr.get("coefficient_vs_independent_grid_qcle", {})
        budget = gr.get("cross_code_population_error_budget")
        observed = comparison.get("max_population_linf_over_time")
        h5_exists = (RAW / "10b_qcle_grid_crosscheck_P40.h5").is_file()
        good = (grid_qcle.get("all_passed") is True and h5_exists
                and interpretation.get("same_equation") is True
                and interpretation.get("independent_physical_reference") is False
                and budget is not None and observed is not None
                and observed <= budget)
        evidence = (f"max population difference {observed:.2e} <= measured "
                    f"numerical budget {budget:.2e}; HDF5={h5_exists}; "
                    f"same QCLE, independent code"
                    if observed is not None and budget is not None
                    else "cross-check fields are incomplete")
    else:
        good, evidence = False, "10b_qcle_grid_crosscheck_P40.json missing"
    check(good, "independent grid-QCLE implementation agrees with the closed "
                "coefficient solver", evidence)

    # F2. the PBME operator benchmark exists, with seeds and a standard error
    pb_ok, pb_ev = True, []
    for P0 in MOMENTA:
        d = load(f"11_pbme_projected_P{P0}.json")
        stats = load(f"11b_pbme_seed_convergence_P{P0}.json")
        if not d or not stats:
            pb_ok = False
            pb_ev.append(f"P{P0}: projected or 11b statistics file missing")
            continue
        r = d["results"]
        ok = (r["n_seeds"] >= 5
              and r["weights_renormalised_by_their_sum"] is False
              and "sem" in r["final"]["cv"]
              and r["branch"] == "pbme_projected"
              and stats.get("all_passed") is True)
        pb_ok &= bool(ok)
        pb_ev.append(f"P{P0}: N={r['n_samples']:,}, {r['n_seeds']} seeds, "
                     f"sem={max(r['final']['cv']['sem']):.2e}")
    check(pb_ok, "projected-density PBME with >=5 seeds and a recorded SEM",
          "; ".join(pb_ev))

    # F3. the trajectory branch's time step is not what limits it
    tc = load("11c_pbme_timestep_convergence.json")
    tcc = None
    if tc:
        tcc = next((c for c in tc.get("checks", [])
                    if c["name"] == "pbme.integration_error_below_sampling_error"),
                   None)
    check(bool(tcc) and tcc["passed"],
          "projected-PBME time-step convergence passes",
          f"largest step-to-step change is {tcc['value']:.3f} of the run's own "
          f"standard error" if tcc else "no time-step study archived")

    # F4. the leakage study uses the projected response, not the focused one
    lk = load("12_leakage_reanalysis.json")
    resp = (lk or {}).get("results", {}).get("response_variable", "")
    check("projected" in resp.lower(),
          "the leakage response is the projected PBME minus the coefficient QCLE",
          resp or "no re-analysis archived")

    # F5. the P0 = 20 momentum boundary passes the production criterion
    wb = load(qcle_reference_propagation_name(20))
    bm = (wb or {}).get("results", {}).get("coefficient", {}).get(
        "boundary_mass_P")
    check(bm is not None and bm <= 1e-9,
          "the P0 = 20 momentum-boundary sentinel passes",
          f"boundary_mass_P = {bm:.2e}" if bm is not None
          else "no P0 = 20 run archived")

    # F6. every requested momentum is present in every physical comparison
    # branch; otherwise a six-column table could silently mix real and missing
    # data through generated placeholders.
    missing_regimes = []
    for P0 in MOMENTA:
        required = (
            qcle_reference_propagation_name(P0),
            qcle_reference_propagation_h5_name(P0),
            f"08_leakage_P{P0}.json",
            f"10_schrodinger_convergence_P{P0}.json",
            f"11_pbme_projected_P{P0}.json",
            f"11b_pbme_seed_convergence_P{P0}.json",
        )
        absent = [name for name in required if not (RAW / name).is_file()]
        invalid = []
        for name in required:
            if not name.endswith(".json") or name in absent:
                continue
            doc = load(name)
            p10_documented = (
                P0 == 10
                and name == "07_coefficient_propagation_P10.json"
                and p10_refinement_ok
                and {c["name"] for c in doc.get("checks", [])
                     if not c.get("passed", False)} <= {
                         "gateL.boundary_mass_R", "gateL.boundary_mass_P"})
            if doc.get("all_passed") is not True and not p10_documented:
                invalid.append(f"{name} (all_passed != true)")
        if absent or invalid:
            details = absent + invalid
            missing_regimes.append(f"P{P0}: {', '.join(details)}")
    check(not missing_regimes,
          "all six momenta are present in every physical comparison branch",
          "; ".join(missing_regimes) if missing_regimes else
          ", ".join(f"P{p}" for p in MOMENTA))

    aggregate = load("12_leakage_reanalysis.json")
    n_regimes = (aggregate or {}).get("results", {}).get("n_regimes", 0)
    check(n_regimes == len(MOMENTA),
          "the leakage momentum-axis aggregate contains all six regimes",
          f"n_regimes={n_regimes}, expected={len(MOMENTA)}")

    # 10b. the central comparison: the Poisson bracket error exceeds the
    #      quantum-classical error in every regime, both resolved
    if pv_json.is_file():
        rows = json.loads(pv_json.read_text())["rows"]
        bad_rows = [P for P, r in rows.items()
                    if not (r["e_pbme"] > r["e_qcle"] and r["e_pbme"] > 2 * r["pb_sem"]
                            and r["e_qcle"] > 5 * r["q_unc"])]
        check(not bad_rows and len(rows) == len(MOMENTA),
              "E_PBME > E_QCLE in every regime, both resolved",
              ", ".join(f"P{P}: {r['e_pbme'] / r['e_qcle']:.0f}x"
                        for P, r in sorted(rows.items(), key=lambda kv: int(kv[0])))
              if not bad_rows else f"fails at {bad_rows}")

    # 11. the journal-format audit passes
    fa = subprocess.run([sys.executable, "scripts/14_format_audit.py"],
                        cwd=ROOT, capture_output=True, text=True, timeout=3600)
    tail = [l for l in fa.stdout.strip().splitlines() if l.strip()]
    check(fa.returncode == 0, "the JCP format audit passes",
          tail[-1] if tail else "no output")

    # 12. the test suite passes
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q",
                        "--no-header", "-x"], cwd=ROOT,
                       capture_output=True, text=True, timeout=3600)
    last = [l for l in r.stdout.strip().splitlines() if l.strip()][-1]
    check(r.returncode == 0, "the test suite passes", last)

    # =====================================================================
    # 13. the PRIMARY branch.  The manuscript's computational evidence is the
    #     Gaussian--Hermite coefficient propagation, so its absence is a
    #     failure of acceptance and not a missing optional extra.  The
    #     subsystem-QCLE archives checked above are the independent
    #     reference; they cannot stand in for these.
    # =====================================================================
    # A check whose loop body never executes must NOT report PASS: an
    # acceptance audit that is satisfied by the absence of its own subject is
    # worse than no audit.  Every one of these is therefore conditioned on the
    # regime having produced a result at all, and says so in its evidence.
    gh_missing, gh_h5_missing, gh_bad, wrong_integrator = [], [], [], []
    gh_present = []
    for P0 in MOMENTA:
        doc = load(primary_hermite_propagation_name(P0))
        if doc is None:
            gh_missing.append(f"P{P0}")
        else:
            gh_present.append(P0)
            if doc.get("all_passed") is not True:
                gh_bad.append(f"P{P0}")
            if doc.get("metadata", {}).get("method") \
                    != "gaussian_hermite_coefficient":
                gh_bad.append(f"P{P0}(method)")
            if doc.get("metadata", {}).get("time", {}).get("integrator") \
                    != "krylov":
                wrong_integrator.append(f"P{P0}")
        if not (RAW / primary_hermite_propagation_h5_name(P0)).is_file():
            gh_h5_missing.append(f"P{P0}")

    n_all = len(MOMENTA)
    check(not gh_missing,
          "Gaussian-Hermite production result exists for every regime",
          "missing: " + ", ".join(gh_missing) if gh_missing else
          ", ".join(f"P{p}" for p in MOMENTA))
    check(not gh_h5_missing,
          "Gaussian-Hermite field archive exists for every regime",
          "missing: " + ", ".join(gh_h5_missing) if gh_h5_missing
          else f"all {n_all} present")
    check(bool(gh_present) and not gh_bad,
          "every Gaussian-Hermite production result passed its own checks",
          "no result to check" if not gh_present else
          ("failed: " + ", ".join(gh_bad) if gh_bad
           else f"{len(gh_present)} of {n_all} regimes passed"))
    check(bool(gh_present) and not wrong_integrator,
          "the primary runs used the exponential propagator",
          "no result to check" if not gh_present else
          ("used a non-production integrator: " + ", ".join(wrong_integrator)
           if wrong_integrator else f"{len(gh_present)} of {n_all} krylov"))

    for name, label in (
            ("07k_gh_closure_validation.json",
             "the K=4 closure and parity validation exists and passes"),
            ("09g_gh_convergence_P40.json",
             "the Krylov/RK4 temporal convergence study exists and passes"),
            ("07g_gh_crosscheck_P40_convergence.json",
             "the Hermite vs subsystem-QCLE cross-check exists and passes"),
            ("07cg_gh_basis_covariance.json",
             "the basis-covariance study exists and passes")):
        doc = load(name)
        check(doc is not None and doc.get("all_passed") is True, label,
              "absent" if doc is None else
              f"all_passed={doc.get('all_passed')}")

    # the closure test must be non-vacuous: its control has to have fired
    ck = load("07k_gh_closure_validation.json")
    if ck is not None:
        r = ck.get("results", {})
        live = (r.get("closure_control", {}).get("E2_final", 0.0) > 0.0
                and r.get("parity_control", {}).get("E_odd_final", 0.0) > 0.0)
        check(live,
              "the closure and parity controls fired, so the nulls are evidence",
              f"control E2_final={r.get('closure_control', {}).get('E2_final')}, "
              f"E_odd_final={r.get('parity_control', {}).get('E_odd_final')}")

    # -- submission mode --------------------------------------------------
    # A development run may carry \needs{} markers and \ghAbsent regimes; a
    # submission run may carry neither.  The two are different questions and
    # are answered separately, so that the audit cannot pass "ready" while the
    # front matter is a placeholder or a scientific claim rests on a
    # calculation that was never made.
    if "--submission" in sys.argv:
        needs = re.findall(r"\\needs\{([^}]*)\}", body)
        check(not needs, "submission mode: no front-matter placeholders left",
              f"{len(needs)} \\needs{{}} markers remain: "
              + "; ".join(sorted({n[:44] for n in needs})[:5])
              if needs else "none")

        gen_tex = (MS / "paper_values.tex").read_text()
        absent = re.findall(r"\\newcommand\{\\(\w+)\}\{\\ghAbsent\}", gen_tex)
        check(not absent,
              "submission mode: no macro is still \\ghAbsent",
              f"{len(absent)} absent macros: "
              + ", ".join(sorted(absent)[:8]) if absent else "none")

        pending = [w for w in ("RESULT_PENDING", "RESULT REQUIRED")
                   if w in body or w in gen_tex]
        check(not pending,
              "submission mode: no RESULT_PENDING placeholder remains",
              ", ".join(pending) if pending else "none")

    # -- report ---------------------------------------------------------
    width = max(len(n) for _, n, _ in RESULTS)
    print("\n=== Stage 5 acceptance audit ===\n")
    for ok, name, ev in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {ev}")
    n_fail = sum(1 for ok, _, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} criteria met.")
    if n_fail:
        print("NOT READY.")
    else:
        print("All acceptance criteria met, subject to the front-matter items "
              "above, which cannot be invented.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
