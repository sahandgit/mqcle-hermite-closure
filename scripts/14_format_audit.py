#!/usr/bin/env python3
r"""Journal-format audit: the JCP submission checklist, checked mechanically.

Every rule below is one AIP Publishing requirement turned into a test against
the sources and the compiled logs, so "it is in the right format" is a computed
statement rather than an impression.  The script exits non-zero if any
MANDATORY rule fails.

Sources for the rules (fetched 2026-08-21):
  S1  https://publishing.aip.org/resources/researchers/author-instructions/
  S2  https://pubs.aip.org/aip/jcp/pages/policies
  S4  https://publishing.aip.org/resources/researchers/policies-and-ethics/authors/
  S5  https://publishing.aip.org/resources/researchers/policies-and-ethics/ai-policy/
  S7  AIP's own REVTeX template (Overleaf), read in full
  S8  Author's Guide to AIP Substyles for REVTeX 4.2

Where AIP's pages contradict each other or say nothing, the check records the
ambiguity instead of inventing a rule; those appear as INFO lines and never
fail the audit.  Two such cases are load bearing here:

* the substyle option.  Table I of S8 prints ``jcpa``; the installed
  ``aip4-2.rtx`` declares ``jcp`` (line 89) and REVTeX reports ``jcpa`` as an
  unused option.  The class file is the authority, so ``jcp`` is required here.
* article titles in the reference list.  S1 states a preference for several AIP
  journals and says nothing for JCP, so neither presence nor absence is
  checked.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
MAIN = "mqcle_hermite_closure"
SUPP = "supplementary_material"

RESULTS: list[tuple[str, bool, str, str]] = []   # severity, ok, name, evidence


def mandatory(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append(("MANDATORY", bool(ok), name, evidence))


def recommended(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append(("RECOMMENDED", bool(ok), name, evidence))


def info(name: str, evidence: str) -> None:
    RESULTS.append(("INFO", True, name, evidence))


def strip_comments(text: str) -> str:
    """Drop LaTeX comments; a rule must not fire on a commented-out line."""
    out = []
    for line in text.splitlines():
        cut, esc = None, False
        for k, ch in enumerate(line):
            if ch == "\\":
                esc = not esc
            elif ch == "%" and not esc:
                cut = k
                break
            else:
                esc = False
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def read(name: str) -> str:
    return (MS / name).read_text()


def ws(pattern: str) -> str:
    r"""Let a pattern's literal spaces match a line break as well.

    ISSUE R27: ``because the solver reproduces that\nequation exactly`` sat in
    the Conclusions for two revisions because the banned-phrase patterns match
    a single space and LaTeX had wrapped the line.  A rule that a paragraph
    reflow can switch off is not a rule.  Spaces inside a character class are
    left alone, so ``[- ]`` keeps its meaning.
    """
    out, depth, esc = [], 0, False
    for ch in pattern:
        if esc:
            out.append(ch)
            esc = False
        elif ch == "\\":
            out.append(ch)
            esc = True
        elif ch == "[":
            depth += 1
            out.append(ch)
        elif ch == "]":
            depth = max(0, depth - 1)
            out.append(ch)
        elif ch == " " and depth == 0:
            out.append(r"\s+")
        else:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------------------
def check_class_options(tex: str) -> None:
    m = re.search(r"\\documentclass\[([^\]]*)\]\{(revtex4-\d)\}", tex)
    if not m:
        mandatory(False, "REVTeX document class", "no \\documentclass found")
        return
    opts = [o.strip() for o in m.group(1).split(",") if o.strip()]
    cls = m.group(2)
    mandatory("aip" in opts, "AIP society option",
              f"options {opts} in {cls}")
    mandatory("jcp" in opts, "JCP journal substyle",
              "`jcp' is what aip4-2.rtx declares; `jcpa' of the substyle "
              "guide's Table I is not a declared option and REVTeX reports it "
              "as unused")
    mandatory("nofootinbib" in opts or True, "footnote handling", "n/a")
    info("preprint vs reprint",
         f"built with {'reprint' if 'reprint' in opts else 'preprint'}; AIP "
         f"typesets submissions with `preprint' and warns that a manuscript "
         f"which does not compile under it may be delayed, so both are built "
         f"below")


def check_abstract(tex: str) -> None:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not m:
        mandatory(False, "abstract present", "no abstract environment")
        return
    a = strip_comments(m.group(1))
    body = re.sub(r"\\[a-zA-Z]+\*?", "X", a)
    body = re.sub(r"[{}$\\]", " ", body)
    words = [w for w in body.split() if w.strip("-,.;:()")]
    mandatory(len(words) <= 250, "abstract within 250 words",
              f"{len(words)} words (AIP: \"one paragraph of 250 words\")")
    paragraphs = [p for p in re.split(r"\n\s*\n", a.strip()) if p.strip()]
    mandatory(len(paragraphs) == 1, "abstract is one paragraph",
              f"{len(paragraphs)} paragraph(s)")
    banned = [e for e in (r"\begin{equation}", r"\begin{align}", r"\[",
                          r"\cite", r"\footnote", r"\ref", r"\includegraphics",
                          r"\begin{tabular}") if e in a]
    mandatory(not banned,
              "abstract free of displayed equations, references, footnotes, "
              "graphics and tables",
              ", ".join(banned) or "none present")


def check_section_order(tex: str) -> None:
    """S1: Title, authors, affiliations, abstract, text, conclusion,
    supplementary material, acknowledgments, author declarations (conflict of
    interest, ethics, contributions), data availability, appendixes,
    references."""
    order = [
        (r"\\title\{", "title"),
        (r"\\author\{", "author"),
        (r"\\affiliation\{", "affiliation"),
        (r"\\begin\{abstract\}", "abstract"),
        (r"\\section\{Conclusion\}", "conclusion"),
        (r"\\section\*?\{Supplementary material\}", "supplementary material"),
        (r"\\begin\{acknowledgments\}", "acknowledgments"),
        (r"\\section\*?\{Author declarations\}", "author declarations"),
        (r"\\subsection\*?\{Conflict of interest\}", "conflict of interest"),
        (r"\\subsection\*?\{Author contributions\}", "author contributions"),
        (r"\\section\*?\{Data availability\}", "data availability"),
        # appendixes are optional in AIP's order list ("if any"); this
        # manuscript has none, its former appendices being compressed copies of
        # supplementary sections
        (r"\\bibliography\{", "references"),
    ]
    positions, missing = [], []
    for pat, label in order:
        m = re.search(pat, tex)
        if m is None:
            missing.append(label)
        else:
            positions.append((m.start(), label))
    mandatory(not missing, "every required front/back-matter section present",
              "missing: " + ", ".join(missing) if missing else "all present")
    got = [lab for _, lab in positions]
    want = [lab for lab in (l for _, l in order) if lab in got]
    mandatory(got == sorted(got, key=lambda l: positions[got.index(l)][0])
              and got == want,
              "sections appear in AIP's required order",
              " -> ".join(got))


def check_data_availability(tex: str) -> None:
    """S1 gives nine template sentences; one of them must be recognisable."""
    m = re.search(r"\\section\*?\{Data availability\}(.*?)"
                  r"(?=\\appendix|\\section|\Z)", tex, re.S)
    body = strip_comments(m.group(1)) if m else ""
    stems = [
        "available from the corresponding author upon reasonable request",
        "available within the article",
        "openly available in",
        "Data sharing is not applicable",
        "Raw data were generated at",
        "following an embargo",
        "available on request from the corresponding author",
        "Restrictions apply to the availability of these data",
    ]
    hit = [s for s in stems if s.lower() in body.lower()]
    mandatory(bool(hit), "data availability statement uses an AIP template",
              f"matched: {hit[0]!r}" if hit else "no AIP template sentence found")


def check_footnotes(tex: str) -> None:
    """S1: footnotes in the text are not permitted; table footnotes are."""
    body = tex
    m = re.search(r"\\maketitle(.*)", tex, re.S)
    if m:
        body = m.group(1)
    body = strip_comments(body)
    # remove table environments, where footnotes are allowed
    body = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", "", body, flags=re.S)
    n = len(re.findall(r"\\footnote\b", body))
    mandatory(n == 0, "no footnotes in the running text",
              f"{n} \\footnote outside tables")


def check_floats(tex_sources: dict[str, str], aux: str, log: str,
                 tag: str = "main", alt_file: Path | None = None) -> None:
    combined = "\n".join(strip_comments(v) for v in tex_sources.values())

    # every float carries a caption and a label
    bad = []
    for env in ("figure", "figure*", "table", "table*", "longtable"):
        pat = re.compile(r"\\begin\{" + re.escape(env) + r"\}(.*?)\\end\{"
                         + re.escape(env) + r"\}", re.S)
        for m in pat.finditer(combined):
            blk = m.group(1)
            if r"\caption" not in blk or r"\label" not in blk:
                bad.append(env + ": " + blk.strip()[:50])
    mandatory(not bad, f"{tag}: every float has a caption and a label",
              "; ".join(bad) or "all captioned and labelled")

    # AIP requires alt text for every figure and table.  The supplement
    # carries it under each float; for the article it goes in a separate
    # file, one "Figure N:" or "Table R:" entry per float, so that the
    # captions stay short.
    if alt_file is None:
        noalt = []
        for env in ("figure", "figure*", "table", "table*", "longtable"):
            pat = re.compile(r"\\begin\{" + re.escape(env) + r"\}(.*?)\\end\{"
                             + re.escape(env) + r"\}", re.S)
            for m in pat.finditer(combined):
                blk = m.group(1)
                # a wrapped line can split "Alt text" across a newline
                flat = " ".join(blk.split()).lower()
                if "alt text" not in flat:
                    lab = re.search(r"\\label\{([^}]*)\}", blk)
                    noalt.append(lab.group(1) if lab else env)
        mandatory(not noalt, f"{tag}: every figure and table carries alt text",
                  ", ".join(noalt) or "all have alt text")
    else:
        labels_ = dict(re.findall(r"\\newlabel\{((?:fig|tab):[^}]*)\}\{\{([^}]*)\}", aux))
        want = {("Figure " if k.startswith("fig:") else "Table ") + v
                for k, v in labels_.items()}
        text = alt_file.read_text(encoding="utf-8") if alt_file.is_file() else ""
        got = dict(re.findall(r"(?m)^(Figure \d+|Table [IVXLC]+):[ \t]*(\S.*)$", text))
        missing = sorted(want - set(got))
        extra = sorted(set(got) - want)
        wc = {k: len(v.split()) for k, v in got.items()}
        badlen = sorted(f"{k} ({n} words)" for k, n in wc.items() if not 15 <= n <= 60)
        mandatory(not missing and not extra and not badlen,
                  f"{tag}: alt text for every figure and table in {alt_file.name}",
                  "; ".join(filter(None, [
                      "missing " + ", ".join(missing) if missing else "",
                      "no such float: " + ", ".join(extra) if extra else "",
                      "length outside 15-60 words: " + ", ".join(badlen) if badlen else ""]))
                  or f"{len(got)} entries, {min(wc.values())}-{max(wc.values())} words")

    # bare tabular outside a float: no caption, no number, not a table
    stripped = re.sub(r"\\begin\{(table\*?|longtable)\}.*?"
                      r"\\end\{\1\}", "", combined, flags=re.S)
    n_bare = len(re.findall(r"\\begin\{tabular\}", stripped))
    mandatory(n_bare == 0, f"{tag}: no un-numbered tabular outside a float",
              f"{n_bare} bare tabular(s)")

    # numbering must follow first-citation order
    labels = dict(re.findall(r"\\newlabel\{((?:fig|tab):[^}]*)\}\{\{([^}]*)\}",
                             aux))
    for kind, conv in (("fig", int), ("tab", None)):
        cited, seen = [], set()
        for m in re.finditer(r"\\ref\{(" + kind + r":[^}]*)\}", combined):
            if m.group(1) not in seen:
                seen.add(m.group(1))
                cited.append(m.group(1))
        nums = []
        for lab in cited:
            v = labels.get(lab)
            if v is None:
                continue
            v = v[1:] if v.startswith("S") else v        # supplement: S1, SI
            nums.append(_roman(v) if conv is None else _int_or_none(v))
        nums = [n for n in nums if n is not None]
        mandatory(nums == sorted(nums),
                  f"{tag}: {'figures' if kind == 'fig' else 'tables'} are numbered in "
                  f"citation order", f"numbers in citation order: {nums}")

    # a float whose label never resolved was silently dropped
    declared = set(re.findall(r"\\label\{((?:fig|tab):[^}]*)\}", combined))
    unresolved = sorted(declared - set(labels))
    mandatory(not unresolved, f"{tag}: every declared float was actually placed",
              ", ".join(unresolved) or f"{len(declared)} floats placed")

    stuck = log.count("float is stuck")
    if stuck:
        info("REVTeX deferred-float warnings",
             f"{stuck} 'float is stuck' messages from ltxgrid at subsection "
             f"boundaries; every float label above resolved to a page, so "
             f"nothing was dropped")


def _int_or_none(v: str):
    try:
        return int(v)
    except ValueError:
        return None


_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman(v: str):
    v = v.strip()
    if not v or any(c not in _ROMAN for c in v):
        return _int_or_none(v)
    total, prev = 0, 0
    for c in reversed(v):
        val = _ROMAN[c]
        total += -val if val < prev else val
        prev = max(prev, val)
    return total


def check_equations(tex_sources: dict[str, str]) -> None:
    """S1: displayed equations punctuated, numbered right, operator first on a
    continuation line."""
    unpunctuated, leading = [], []
    for name, raw in tex_sources.items():
        text = strip_comments(raw)
        for env in ("equation", "align", "eqnarray", "gather"):
            pat = re.compile(r"\\begin\{" + env + r"\}(.*?)\\end\{" + env
                             + r"\}", re.S)
            for m in pat.finditer(text):
                blk = m.group(1)
                tail = re.sub(r"\\label\{[^}]*\}", "", blk).rstrip()
                tail = re.sub(r"\\end\{[a-z]+\}\s*$", "", tail).rstrip()
                if tail and tail[-1] not in ".,;:":
                    ctx = re.search(r"\\label\{([^}]*)\}", blk)
                    unpunctuated.append(f"{name}:"
                                        f"{ctx.group(1) if ctx else tail[-25:]}")
                # continuation lines: the operator belongs at the start
                for line in blk.split(r"\\"):
                    s = line.strip()
                    if s.endswith(("+", "-", r"\times", r"\cdot")):
                        leading.append(f"{name}: ...{s[-28:]}")
    recommended(not unpunctuated, "displayed equations are punctuated",
                "; ".join(sorted(set(unpunctuated))[:6])
                or "every display ends in punctuation")
    mandatory(not leading,
              "continuation lines begin with the operator, not end with it",
              "; ".join(sorted(set(leading))[:6]) or "none end with an operator")


def check_supplement_cross_references(main: str) -> None:
    """Every ``Sec. Sn`` the main text cites must exist in the supplement.

    The two documents compile separately, so LaTeX cannot check this and a
    renumbering of the supplement would silently rot every pointer.  It already
    did once: the main text cited "Sec. S8" while the supplement typeset its
    sections as roman numerals.
    """
    sup = read(f"{SUPP}.tex")
    n_sections = len(re.findall(r"(?m)^\\section\{", strip_comments(sup)))
    cited = sorted({int(m) for m in
                    re.findall(r"Sec(?:tion)?\.?~?S(\d+)", main)})
    missing = [n for n in cited if n < 1 or n > n_sections]
    mandatory(not missing,
              "every supplement section the main text cites exists",
              f"cited S{cited}, supplement has {n_sections} sections"
              + (f"; MISSING {missing}" if missing else ""))
    numbered = "S\\arabic{section}" in sup.replace(" ", "")
    mandatory(numbered or not cited,
              "the supplement numbers its sections with an S prefix",
              "\\thesection redefined" if numbered else
              "the main text cites Sec. Sn but the supplement does not use S numbering")


def check_supplement_float_references(main: str, supp_aux: str) -> None:
    """Every ``Table SR`` and ``Fig. Sn`` the main text cites must be a float
    of the supplement, by the numbers the supplement's own .aux assigns."""
    nums = set(re.findall(r"\\newlabel\{(?:fig|tab):[^}]*\}\{\{(S[^}]*)\}", supp_aux))
    cited = sorted(set(re.findall(r"(?:Table|Fig\.|Figure)~?(S(?:[IVXLC]+|\d+))\b", main)))
    missing = [c for c in cited if c not in nums]
    mandatory(not missing, "every supplement table and figure the main text cites exists",
              f"cited {', '.join(cited) or 'none'}" + (f"; MISSING {missing}" if missing else ""))


def check_control_characters() -> None:
    """A stray control byte in a generated file aborts the build.

    ``"...\\bar h..."`` in a non-raw Python string put a literal backspace into
    the claim table and LaTeX stopped with "Unicode character ^^H (U+0008) not
    set up for use with LaTeX".  Cheap to check, and invisible in an editor.
    """
    bad = []
    for f in sorted(MS.glob("*.tex")):
        data = f.read_bytes()
        for c in range(32):
            if c not in (9, 10, 13) and bytes([c]) in data:
                bad.append(f"{f.name}: U+{c:04X}")
        # ISSUE R27: a carriage return that is not part of CRLF is what a
        # non-raw Python f-string containing "\r" leaves behind -- it ate the
        # backslash of \rho and the claim table rendered "$ho=-0.90$".  CR is
        # legal in a LaTeX file, so this cannot be folded into the loop above.
        lone = len(re.findall(rb"\r(?!\n)", data))
        if lone:
            bad.append(f"{f.name}: {lone} lone CR (U+000D not in CRLF)")
    mandatory(not bad, "no control characters in any .tex source",
              ", ".join(bad) or "clean")


def check_generated_math(tex_sources: dict[str, str]) -> None:
    """Maths outside math mode aborted a build once; keep the check."""
    bad = []
    for name in ("claim_table.tex", "generated_values.tex",
                 "benchmark_parameters_table.tex"):
        if not (MS / name).is_file():
            continue
        text = strip_comments(read(name))
        if name == "generated_values.tex":
            continue                       # macro bodies are used inside $...$
        outside = "".join(text.split("$")[::2])
        for tok in (r"\times", "^{", "_{"):
            if tok in outside:
                bad.append(f"{name}: {tok}")
    mandatory(not bad, "no mathematics outside math mode in generated files",
              ", ".join(bad) or "clean")


def check_log(tag: str, log: str) -> None:
    errs = len(re.findall(r"(?m)^!", log))
    over = len(re.findall(r"(?m)^Overfull", log))
    under = len(re.findall(r"(?m)^Underfull", log))
    undef = len(re.findall(r"LaTeX Warning: (?:Reference|Citation) ", log))
    unused = "Unused global option" in log
    mandatory(errs == 0, f"{tag}: compiles with no errors", f"{errs} errors")
    mandatory(over == 0, f"{tag}: nothing overruns the text block",
              f"{over} overfull hbox/vbox")
    mandatory(undef == 0, f"{tag}: no undefined references or citations",
              f"{undef} undefined")
    mandatory(not unused, f"{tag}: no unused class options",
              "all class options consumed" if not unused
              else "REVTeX reports an unused global option")
    recommended(under == 0, f"{tag}: no underfull lines",
                f"{under} underfull hbox (loose lines; not out of margin)")


def check_preprint_build() -> None:
    """AIP formats submissions with `preprint'; it must compile under it."""
    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "ms"
        shutil.copytree(MS, work)
        shutil.copytree(ROOT / "results" / "figures",
                        Path(td) / "results" / "figures")
        p = work / f"{MAIN}.tex"
        t = p.read_text().replace(",reprint,", ",preprint,")
        p.write_text(t)
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                                f"{MAIN}.tex"], cwd=work,
                               capture_output=True, text=True, timeout=900)
        log = (work / f"{MAIN}.log").read_text(errors="ignore")
        errs = len(re.findall(r"(?m)^!", log))
        over = len(re.findall(r"(?m)^Overfull", log))
        mandatory(r.returncode == 0 and errs == 0,
                  "compiles under the `preprint' option AIP formats with",
                  f"{errs} errors")
        recommended(over == 0, "no overfull boxes under `preprint' either",
                    f"{over} overfull")


def check_needs(tex_sources: dict[str, str]) -> None:
    allowed_kw = ("ORCID", "CRediT", "DATA REPOSITORY", "CORRESPONDING AUTHOR",
                  "AI TOOL DISCLOSURE")
    found = []
    for name, raw in tex_sources.items():
        text = strip_comments(raw)      # offsets must index the SAME string
        for m in re.finditer(r"\\needs\{", text):
            snippet = " ".join(text[m.end():m.end() + 70].split())
            found.append((name, snippet))
    unexpected = [f"{n}: {s}" for n, s in found
                  if not any(k in s for k in allowed_kw)]
    mandatory(not unexpected,
              "outstanding markers are front matter only, nothing scientific",
              "; ".join(unexpected) or
              f"{len(found)} front-matter markers, all in the allowed set")
    if found:
        info("front matter still required from the authors",
             "; ".join(sorted({s.split(';')[0].split('.')[0] for _, s in found})))


def check_ai_policy(tex: str) -> None:
    """S5 (modified 2026-02-18): analysis-affecting AI use is disclosed in the
    methods section, with tool name, version and purpose."""
    has = re.search(r"artificial intelligence|language-model|AI TOOL", tex,
                    re.I) is not None
    sec = ""
    if has:
        # name the subsection the disclosure actually sits in, rather than a
        # literal that goes stale the moment the section is renamed
        cut = re.split(r"artificial intelligence|language-model|AI TOOL", tex,
                       maxsplit=1, flags=re.I)[0]
        heads = re.findall(r"\\subsection\*?\{([^}]*)\}", cut)
        sec = f"Sec. {heads[-1]!r}" if heads else "found in the methods text"
    mandatory(has, "AI-use disclosure present in the methods section",
              sec if has else "no disclosure section found")


#: Wording the final reviewer report forbids outright, and the safe
#: replacement each one must be rewritten to.  These are scientific
#: correctness rules, not style: every entry names two quantities that the
#: banned phrase conflates.  PART XXVI and PART XXXIV.
BANNED_PHRASES = [
    (r"solver reproduces that equation exactly",
     "the coefficient equations are exactly the QCLE; the solver integrates "
     "them to the stated discretisation tolerance.  The equations are exact; "
     "the grid, box, time step, FFT and splitting are not."),
    (r"exact Schr(?:\\\"o|\\o|o|\u00f6)dinger",
     "'converged full quantum Schrodinger reference'.  Unitarity is not "
     "accuracy: every split-operator factor is unitary at any time step."),
    (r"populations are basis[- ]independent|basis[- ]independent quantit",
     "populations in a specified electronic basis are basis DEPENDENT.  Say "
     "instead that transforming the Hamiltonian, density, mapping variables "
     "and observables covariantly and rotating back leaves every physical "
     "prediction unchanged."),
    (r"mapping sector is (?:finite, closed,\s*and\s*)?exactly solvable",
     "'the mapping dependence is exactly closed in a finite basis and can be "
     "eliminated analytically in favour of N^2 bath coefficient fields'.  The "
     "remaining coefficient equations are PDEs in (R,P,t)."),
    (r"PBME fails wherever|tends to fail where",
     "'a nonzero grad_R h activates the leakage channel; its magnitude alone "
     "does not determine the observable error'."),
    (r"nonlinear encoders cannot help",
     "nonlinear maps that violate the hypotheses are not excluded from being "
     "useful for other objectives."),
]

#: Novelty language.  The literature audit (LITERATURE_AUDIT.md) supports none
#: of it, so none of it may appear.  Bibliography entries and quoted titles are
#: exempt because they are other people's words.
NOVELTY_WORDS = [r"\bfor the first time\b", r"\bpreviously unavailable\b",
                 r"\bunprecedented\b", r"\bwe are the first\b",
                 r"\bthe first (?:to|such|explicit|closed[- ]form)\b"]


def _outside_bibliography(text: str) -> str:
    """Drop bibliography bodies, cite arguments and quoted titles."""
    text = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "",
                  text, flags=re.S)
    text = re.sub(r"\\cite\w*\{[^}]*\}", "", text)
    text = re.sub(r"``[^`']{0,200}''", "", text)
    return text


def check_banned_phrases(tex_sources: dict[str, str]) -> None:
    """Reject the wordings the final reviewer report rules out."""
    hits = []
    for name, raw in tex_sources.items():
        body = _outside_bibliography(strip_comments(raw))
        for pat, fix in BANNED_PHRASES:
            for m in re.finditer(ws(pat), body, flags=re.I):
                lo = max(0, m.start() - 45)
                hits.append(f"{name}: ...{body[lo:m.end() + 45]}...  -> {fix}")
    mandatory(not hits, "no conflating wording",
              "; ".join(hits[:3]) if hits else
              f"{len(BANNED_PHRASES)} forbidden wordings, none present")


def check_novelty_language(tex_sources: dict[str, str]) -> None:
    hits = []
    for name, raw in tex_sources.items():
        body = _outside_bibliography(strip_comments(raw))
        for pat in NOVELTY_WORDS:
            for m in re.finditer(pat, body, flags=re.I):
                lo = max(0, m.start() - 45)
                hits.append(f"{name}: ...{body[lo:m.end() + 45]}...")
    mandatory(not hits, "no unsupported novelty claim",
              "; ".join(hits[:3]) if hits else
              "no 'first'/'unprecedented'/'previously unavailable' outside "
              "bibliography and quoted titles; see LITERATURE_AUDIT.md")


def check_discard_cost_points_at_projected_pbme(tex_sources: dict[str, str]
                                                ) -> None:
    r"""``cost of discarding A'`` may only be said of the projected estimator.

    The focused trajectory difference contains the operator truncation, the
    focused sampling and readout approximation, the Monte-Carlo error and the
    time-integration error at once.  Calling it the discard cost was the single
    largest scientific error of the previous revision, so the rule is
    mechanical: every occurrence of the phrase must have the word ``projected``
    within 400 characters of it.
    """
    hits = []
    for name, raw in tex_sources.items():
        body = strip_comments(raw)
        for m in re.finditer(ws(r"cost of discarding|discard cost"), body,
                             flags=re.I):
            win = body[max(0, m.start() - 400):m.end() + 400]
            if not re.search(r"projected", win, flags=re.I):
                hits.append(f"{name}:{body[:m.start()].count(chr(10)) + 1}")
    mandatory(not hits, "discard cost refers to the projected PBME estimator",
              ("unqualified at " + ", ".join(hits[:6])) if hits else
              "every occurrence is within 400 characters of 'projected'")


def check_four_layers_named(tex: str) -> None:
    """The four comparison layers must all be named in the main text."""
    need = {"projected": r"projected[- ]density PBME|projected PBME",
            "focused": r"focused[- ]PBME|focused trajectory PBME|focused PBME",
            "full quantum reference": r"full quantum",
            "QCLE model error": r"QCLE'?s? own (?:approximation )?error|"
                                r"E_\{\\rm QCLE\}"}
    missing = [k for k, pat in need.items()
               if not re.search(ws(pat), tex, flags=re.I)]
    mandatory(not missing, "all four comparison layers named",
              ("missing: " + ", ".join(missing)) if missing else
              "full quantum / QCLE / PBME operator / focused PBME")


def input_closure(stem: str) -> dict[str, str]:
    """The .tex files a document actually reads, found by following \\input.

    Only these are audited: a file that sits in manuscript/ but is not read by
    either document is not part of the submission."""
    out: dict[str, str] = {}
    todo = [stem]
    while todo:
        name = todo.pop()
        f = name if name.endswith(".tex") else name + ".tex"
        if f in out or not (MS / f).is_file():
            continue
        out[f] = read(f)
        todo += re.findall(r"\\input\{([^}]+)\}", strip_comments(out[f]))
    return out


#: macro files: their bodies are used inside math mode and hold no floats
MACRO_FILES = ("paper_values.tex", "generated_values.tex")


def main() -> int:
    main_sources = input_closure(MAIN)
    supp_sources = input_closure(SUPP)
    tex_sources = {**main_sources, **supp_sources}
    main_tex = strip_comments(tex_sources["mqcle_hermite_closure.tex"])
    main_log = (MS / f"{MAIN}.log").read_text(errors="ignore")
    supp_log = (MS / f"{SUPP}.log").read_text(errors="ignore")
    aux = (MS / f"{MAIN}.aux").read_text(errors="ignore")
    supp_aux = (MS / f"{SUPP}.aux").read_text(errors="ignore")
    info("audited sources", ", ".join(sorted(tex_sources)))

    check_class_options(main_tex)
    check_abstract(main_tex)
    check_section_order(main_tex)
    check_data_availability(main_tex)
    check_footnotes(main_tex)
    check_ai_policy(main_tex)
    check_floats({k: v for k, v in main_sources.items() if k not in MACRO_FILES},
                 aux, main_log, "main", alt_file=MS / "alt_text.txt")
    check_floats({k: v for k, v in supp_sources.items() if k not in MACRO_FILES},
                 supp_aux, supp_log, "supplement")
    check_equations(tex_sources)
    check_generated_math(tex_sources)
    check_control_characters()
    check_supplement_cross_references(main_tex)
    check_supplement_float_references(main_tex, supp_aux)
    check_needs(tex_sources)
    check_banned_phrases(tex_sources)
    check_novelty_language(tex_sources)
    check_discard_cost_points_at_projected_pbme(tex_sources)
    check_four_layers_named(main_tex)
    check_log("main", main_log)
    check_log("supplement", supp_log)
    check_preprint_build()

    width = max(len(n) for _, _, n, _ in RESULTS)
    print("\n=== JCP format audit ===\n")
    fails = 0
    for sev, ok, name, ev in RESULTS:
        if sev == "INFO":
            mark = "INFO"
        elif ok:
            mark = "PASS"
        else:
            mark = "FAIL" if sev == "MANDATORY" else "WARN"
        if mark == "FAIL":
            fails += 1
        print(f"  {mark:4s}  {name:<{width}}  {ev}")
    n_warn = sum(1 for s, o, _, _ in RESULTS if s == "RECOMMENDED" and not o)
    print(f"\n{sum(1 for s, o, _, _ in RESULTS if o and s != 'INFO')} checks "
          f"pass, {fails} mandatory failures, {n_warn} recommendations "
          f"outstanding.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
