"""Provenance table of the archived results.

Every stage writes a ``metadata`` block into its JSON output: the script, the
command line, the platform, the Python version, the UTC time stamp and the
wall time.  This script collects those blocks from ``results/raw`` and
``results/failed`` and writes them as one table, ``results/PROVENANCE.md``,
together with the size of the HDF5 archive that accompanies each output.
Nothing is recomputed and no result file is modified.

    python scripts/21_provenance_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "PROVENANCE.md"

#: The two aggregates of the leakage study carry no metadata block of their
#: own; both are written by one call, the one run_final_pipeline.ps1 makes.
AGGREGATE_COMMAND = ("scripts\\08_leakage_diagnostic.py --aggregate "
                     "--out .\\results\\raw\\12_leakage_reanalysis.json")
NO_METADATA = {
    "12_leakage_reanalysis.json": AGGREGATE_COMMAND,
    "12_coupling_scan_projected_pbme.json": AGGREGATE_COMMAND,
}

#: GitHub refuses files above 100 MiB and warns above 50 MiB.
LARGE_BYTES = 50 * 2**20


def _platform(text: str) -> str:
    if text.startswith("Windows"):
        return "Windows"
    if text.startswith("Linux"):
        return "Linux"
    return text or "--"


def _python(text: str) -> str:
    return text.split()[0] if text else "--"


def _wall(seconds) -> str:
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "--"
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def _size(path: Path) -> str:
    if not path.is_file():
        return ""
    n = path.stat().st_size
    mark = " (†)" if n > LARGE_BYTES else ""
    return f"{n / 1e6:.1f} MB{mark}"


def _row(path: Path) -> str:
    doc = json.loads(path.read_text(encoding="utf-8"))
    md = doc.get("metadata") if isinstance(doc, dict) else None
    if isinstance(md, dict):
        command = md.get("command_line") or md.get("script") or "--"
        platform = _platform(str(md.get("platform", "")))
        python = _python(str(md.get("python", "")))
        stamp = str(md.get("timestamp_utc", "--"))
        wall = _wall(md.get("wall_time_s"))
    else:
        command = NO_METADATA.get(path.name, "--")
        platform = python = stamp = wall = "--"
    passed = doc.get("all_passed") if isinstance(doc, dict) else None
    checks = {True: "pass", False: "FAIL"}.get(passed, "--")
    if passed is False:
        failed = [str(c.get("name", "?")).split(".")[-1]
                  for c in doc.get("checks", [])
                  if isinstance(c, dict) and c.get("passed") is False]
        if failed:
            checks += ": " + ", ".join(failed)
    rel = path.relative_to(RESULTS).as_posix()
    h5 = _size(path.with_suffix(".h5"))
    return (f"| `{rel}` | `{command}` | {platform} | {python} | {stamp} "
            f"| {wall} | {checks} | {h5} |")


def main() -> None:
    files = (sorted((RESULTS / "raw").glob("*.json"))
             + sorted((RESULTS / "failed").glob("*/*.json")))
    lines = [
        "# Provenance of the archived results",
        "",
        "Written by `scripts/21_provenance_table.py` from the `metadata` block "
        "of each JSON output; nothing here is typed by hand.  *Command* is "
        "the command line recorded by the run, relative to the repository "
        "root.  *Checks* is the output's own `all_passed` flag, followed by "
        "the names of any checks that did not pass.  *HDF5* is "
        "the size of the archive written next to the JSON file, if any; "
        "(†) marks archives larger than 50 MiB, which are kept out of the "
        "Git repository and are available from the corresponding author.",
        "",
        "`results/failed/P10_2048x1024_20260829` holds the P0 = 10 "
        "subsystem-QCLE run on the coarser 2048 x 1024 grid.  It is the "
        "coarse member of the two-grid acceptance of "
        "`scripts/07c_p10_refinement_acceptance.py`, which binds it by "
        "SHA-256, and is kept for that reason.",
        "",
        "| Output | Command | Platform | Python | Finished (UTC) | Wall time "
        "| Checks | HDF5 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [_row(f) for f in files]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(files)} outputs)")


if __name__ == "__main__":
    main()
