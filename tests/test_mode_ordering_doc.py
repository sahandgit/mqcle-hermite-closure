r"""The documented mode table must equal the programmatic one.

Specification item 13.  The supplementary material prints the $N=2$, $K=2$,
even-parity multi-indices with explicit integer labels, and the reader is
entitled to use those labels to index :math:`\mathsf R`, :math:`\mathsf G` and
:math:`B`.  An earlier revision of that table duplicated one tuple and left
the last one unlabelled, so the printed labels disagreed with
``active_modes`` from index 6 onward.  Nothing in the code could catch that,
because the table lived only in LaTeX.  This test closes the gap: it parses
the table out of the manuscript source and compares it, label by label, with
the ordering the code actually uses.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from mqcle.hermite.operator import active_modes

SUPPLEMENT = (Path(__file__).resolve().parents[1]
              / "manuscript" / "supplementary_material.tex")

# The contract, written out.  Duplicated deliberately rather than derived from
# active_modes, so that a change to the enumeration cannot silently rewrite
# both sides of the comparison at once.
DOCUMENTED = [
    (0, 0, 0, 0),
    (0, 0, 0, 2),
    (0, 0, 1, 1),
    (0, 0, 2, 0),
    (0, 1, 0, 1),
    (0, 1, 1, 0),
    (0, 2, 0, 0),
    (1, 0, 0, 1),
    (1, 0, 1, 0),
    (1, 1, 0, 0),
    (2, 0, 0, 0),
]


def _parse_supplement_table() -> dict[int, tuple[int, ...]]:
    r"""Return ``{label: multi-index}`` as printed in the supplement.

    The table is typeset two logical columns to the row, so a row carries up
    to two ``label & $(a,b,c,d)$`` cells.  Cells whose label is blank are
    returned under ``-1`` so that an unlabelled tuple is visible to the test
    rather than silently dropped.
    """
    text = SUPPLEMENT.read_text(encoding="utf-8")
    head = re.search(r"\\subsection\{Mode ordering used by the code\}", text)
    assert head, "mode-ordering subsection not found in the supplement"
    body = text[head.end():]
    body = body[: body.index(r"\end{tabular}")]
    # Drop the preamble and the column specification, so that the first data
    # row's label is not glued to the prose that introduces the table.
    start = re.search(r"\\begin\{tabular\}\{[^\n]*\}", body)
    assert start, "tabular environment not found in the mode-ordering section"
    body = body[start.end():]

    out: dict[int, tuple[int, ...]] = {}
    unlabelled: list[tuple[int, ...]] = []
    for row in body.split(r"\\"):
        cells = [c.strip() for c in row.split("&")]
        for label, tup in zip(cells[0::2], cells[1::2]):
            m = re.fullmatch(r"\$\((\d+),(\d+),(\d+),(\d+)\)\$", tup.strip())
            if not m:
                continue
            index = tuple(int(g) for g in m.groups())
            if re.fullmatch(r"\d+", label):
                key = int(label)
                assert key not in out, f"label {key} printed twice"
                out[key] = index
            else:
                unlabelled.append(index)
    assert not unlabelled, f"tuples printed without a label: {unlabelled}"
    return out


def test_documented_table_matches_active_modes():
    """Label by label, the printed table is the code's enumeration."""
    code = [tuple(int(x) for x in n) for n in active_modes(2, 2, parity="even")]
    assert code == DOCUMENTED, (
        "active_modes no longer produces the ordering the manuscript "
        f"documents:\n  code = {code}\n  doc  = {DOCUMENTED}"
    )


def test_documented_table_has_no_duplicates():
    """Eleven distinct multi-indices, every one of even total order."""
    assert len(DOCUMENTED) == 11
    assert len(set(DOCUMENTED)) == 11, "duplicate multi-index in the table"
    assert all(sum(n) % 2 == 0 for n in DOCUMENTED)
    assert sorted(DOCUMENTED) == DOCUMENTED, "table is not in lexicographic order"


def test_supplement_source_matches_documented_table():
    """The LaTeX the reader sees agrees with the contract above."""
    if not SUPPLEMENT.exists():          # pragma: no cover - source-tree only
        pytest.skip("manuscript source not present in this checkout")
    printed = _parse_supplement_table()
    assert sorted(printed) == list(range(11)), (
        f"labels printed in the supplement are {sorted(printed)}, "
        "expected 0..10 with none missing"
    )
    for label, index in printed.items():
        assert index == DOCUMENTED[label], (
            f"supplement prints mode {label} as {index}, "
            f"code gives {DOCUMENTED[label]}"
        )
