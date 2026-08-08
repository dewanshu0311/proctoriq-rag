"""Build and validate ``submission.csv``.

This module is the **only** place the submission-format flags are applied. The
rest of the codebase works in canonical form — document IDs are bare file stems,
section titles are verbatim ``##`` header text — and the translation into
whatever the grader wants happens here, at serialization time. When a leaderboard
probe resolves either unknown, exactly one file changes: ``config/default.yaml``.

Validation refuses to write on any defect. That is deliberate: a malformed
submission is not recoverable after the fact, and a *quietly* malformed one is
worse than a crash.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from proctoriq_rag.config import SectionFormat, SubmissionConfig

SUBMISSION_COLUMNS: tuple[str, ...] = (
    "question_id",
    "answer_text",
    "cited_docs",
    "cited_sections",
)

#: Matches "Section 2: Common Installation Errors" -> ("2", "Common Installation Errors").
#: Headers that do not match this shape (e.g. "Overview") are emitted verbatim
#: under every format variant — there is no number to extract and no prefix to strip.
SECTION_PATTERN = re.compile(r"^Section\s+(\d+)\s*:\s*(.+)$")


class SubmissionValidationError(Exception):
    """Raised when a submission fails validation. No file is written."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        body = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(
            f"Submission validation failed with {len(self.problems)} problem(s):\n{body}"
        )


@dataclass(frozen=True)
class Prediction:
    """One question's pipeline output, in canonical (unformatted) form.

    ``cited_docs`` holds bare document IDs and ``cited_sections`` holds verbatim
    ``##`` header text. They are positionally aligned — the Nth section belongs
    to the Nth document — and that alignment survives serialization.
    """

    question_id: str
    answer_text: str
    cited_docs: list[str] = field(default_factory=list)
    cited_sections: list[str] = field(default_factory=list)

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.cited_docs, self.cited_sections))


def format_doc(doc_id: str, *, extension: bool) -> str:
    """Render a document ID for the ``cited_docs`` column.

    ``extension=False`` -> ``"01_windows_installation_login_guide"``
    ``extension=True``  -> ``"01_windows_installation_login_guide.md"``

    Idempotent: a doc_id that already ends in ``.md`` is not double-suffixed, and
    the extension is stripped when ``extension=False``.
    """
    stem = doc_id[:-3] if doc_id.endswith(".md") else doc_id
    return f"{stem}.md" if extension else stem


def format_section(section_title: str, fmt: SectionFormat | str) -> str:
    """Render a section header for the ``cited_sections`` column.

    Given ``"Section 2: Common Installation Errors"``:

    ``FULL_HEADER`` -> ``"Section 2: Common Installation Errors"``
    ``NUMBER_ONLY`` -> ``"Section 2"``
    ``TITLE_ONLY``  -> ``"Common Installation Errors"``

    A header that does not match ``Section N: ...`` — ``"Overview"``, for
    instance — is returned verbatim under every variant.
    """
    fmt = SectionFormat(fmt)
    title = section_title.strip()

    if fmt is SectionFormat.FULL_HEADER:
        return title

    match = SECTION_PATTERN.match(title)
    if match is None:
        return title

    number, label = match.group(1), match.group(2).strip()
    return f"Section {number}" if fmt is SectionFormat.NUMBER_ONLY else label


def read_expected_header(sample_submission: Path | str) -> tuple[str, ...]:
    """Read the authoritative column header off ``sample_submission.csv``.

    Preferred over the ``SUBMISSION_COLUMNS`` constant when the file is
    available — if the organisers change the header, we match it rather than
    our own assumption.
    """
    path = Path(sample_submission)
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    return tuple(h.strip() for h in header)


def _join(values: Iterable[str], separator: str) -> str:
    return separator.join(values)


def build_rows(
    predictions: Sequence[Prediction],
    question_ids: Sequence[str],
    config: SubmissionConfig,
) -> list[dict[str, str]]:
    """Serialize predictions into rows, ordered by ``question_ids``.

    Ordering follows ``test.csv``, not the order predictions arrive in. Any
    question with no prediction produces an empty row here and is caught by
    :func:`validate_rows` — it is not silently dropped.
    """
    by_id = {p.question_id: p for p in predictions}
    rows: list[dict[str, str]] = []

    for question_id in question_ids:
        prediction = by_id.get(question_id)
        if prediction is None:
            rows.append(
                {
                    "question_id": question_id,
                    "answer_text": "",
                    "cited_docs": "",
                    "cited_sections": "",
                }
            )
            continue

        docs = [
            format_doc(d, extension=config.doc_extension) for d in prediction.cited_docs
        ]
        sections = [
            format_section(s, config.section_format) for s in prediction.cited_sections
        ]

        rows.append(
            {
                "question_id": question_id,
                "answer_text": prediction.answer_text,
                # Duplicates are preserved. Q44 cites one document twice with two
                # different sections; collapsing it would break positional alignment.
                "cited_docs": _join(docs, config.separator),
                "cited_sections": _join(sections, config.separator),
            }
        )

    return rows


def validate_rows(
    rows: Sequence[dict[str, str]],
    question_ids: Sequence[str],
    expected_header: Sequence[str] = SUBMISSION_COLUMNS,
    separator: str = "|",
) -> None:
    """Raise :class:`SubmissionValidationError` if anything is wrong.

    Checks, in order: row count, question-ID order, header, then per-row
    completeness.

    The ``answer_text`` check is not incidental. The starter notebook's
    generation loop swallows per-question exceptions and writes
    ``answer, cited_docs, cited_sections = "", "", ""``. That produces a row that
    *looks* structurally valid while scoring zero across four of the five
    dimensions. Failing loudly here is the difference between noticing three
    blank answers and submitting them.
    """
    problems: list[str] = []

    if len(rows) != len(question_ids):
        problems.append(
            f"expected {len(question_ids)} rows, got {len(rows)}"
        )

    actual_ids = [r.get("question_id", "") for r in rows]
    if actual_ids != list(question_ids):
        mismatches = [
            f"row {i}: expected {exp!r}, got {act!r}"
            for i, (exp, act) in enumerate(zip(question_ids, actual_ids))
            if exp != act
        ]
        preview = "; ".join(mismatches[:5]) or "row count differs"
        problems.append(f"question_id order does not match test.csv ({preview})")

    for row in rows:
        header = tuple(row.keys())
        if header != tuple(expected_header):
            problems.append(
                f"row {row.get('question_id', '?')}: columns {header} "
                f"!= expected {tuple(expected_header)}"
            )
            break

    empty_answers: list[str] = []
    empty_docs: list[str] = []
    empty_sections: list[str] = []
    misaligned: list[str] = []
    null_fields: list[str] = []

    for row in rows:
        question_id = row.get("question_id", "?")

        for column in expected_header:
            if row.get(column) is None:
                null_fields.append(f"{question_id}.{column}")

        answer = (row.get("answer_text") or "")
        if not answer.strip():
            empty_answers.append(question_id)

        docs_raw = (row.get("cited_docs") or "")
        sections_raw = (row.get("cited_sections") or "")

        if not docs_raw.strip():
            empty_docs.append(question_id)
        if not sections_raw.strip():
            empty_sections.append(question_id)

        docs = docs_raw.split(separator) if docs_raw else []
        sections = sections_raw.split(separator) if sections_raw else []

        if len(docs) != len(sections):
            misaligned.append(
                f"{question_id} ({len(docs)} docs vs {len(sections)} sections)"
            )
        if any(not d.strip() for d in docs):
            empty_docs.append(question_id)
        if any(not s.strip() for s in sections):
            empty_sections.append(question_id)

    if null_fields:
        problems.append(f"null field(s): {', '.join(sorted(set(null_fields)))}")
    if empty_answers:
        problems.append(
            f"empty or whitespace-only answer_text for {len(empty_answers)} "
            f"question(s): {', '.join(empty_answers)}"
        )
    if empty_docs:
        problems.append(
            f"empty cited_docs for: {', '.join(sorted(set(empty_docs)))}"
        )
    if empty_sections:
        problems.append(
            f"empty cited_sections for: {', '.join(sorted(set(empty_sections)))}"
        )
    if misaligned:
        problems.append(
            f"cited_docs/cited_sections positional misalignment: {', '.join(misaligned)}"
        )

    if problems:
        raise SubmissionValidationError(problems)


def write_submission(
    predictions: Sequence[Prediction],
    test_csv: Path | str,
    out_path: Path | str,
    config: SubmissionConfig | None = None,
    sample_submission: Path | str | None = None,
) -> Path:
    """Build, validate and write ``submission.csv``. Returns the written path.

    Nothing is written unless validation passes — the file is only opened after
    :func:`validate_rows` returns. Answers containing commas, quotes or newlines
    are escaped by :mod:`csv` with ``QUOTE_MINIMAL`` and round-trip cleanly.
    """
    config = config or SubmissionConfig()
    question_ids = pd.read_csv(test_csv)["question_id"].astype(str).tolist()

    expected_header: tuple[str, ...] = SUBMISSION_COLUMNS
    if sample_submission is not None:
        expected_header = read_expected_header(sample_submission)

    rows = build_rows(predictions, question_ids, config)
    validate_rows(rows, question_ids, expected_header, separator=config.separator)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(expected_header))
        writer.writeheader()
        writer.writerows(rows)

    return out_path
