"""Alternative readings of contested answer-key entries.

Four questions came out of Phase 1 with the same signature: their expected
sections were the *only* ones naive retrieval failed to surface at all, and three
of the four were already flagged low-confidence when the key was hand-built.

    Q14  low     10_general_support :: Section 4 unretrieved at rank 13
    Q28  low     08_reattempt       :: Section 1 unretrieved at rank 13
    Q31  medium  07_policy          :: Section 4 unretrieved at rank 14
    Q35  low     09_status          :: Section 4 unretrieved at rank 12

That is either four retrieval failures or four reading errors, and a bi-encoder
cannot tell the difference — it only measures surface similarity. A cross-encoder
models question-document relevance directly, so it gets a genuine vote.

**This module produces evidence, never a decision.** ``answer_key.yaml`` is not
modified by anything here. If the cross-encoder ranks a primary citation near the
top, the key survives a real test and that is worth knowing. If it ranks an
alternate near the top while burying the primary, that is worth knowing too. Which
reading is correct remains a human call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from proctoriq_rag.corpus.loader import Corpus
from proctoriq_rag.evaluation.answer_key import AnswerKey, AnswerKeyValidationError


@dataclass(frozen=True)
class AlternateReading:
    """One candidate reading of a question, held to the primary key's standards."""

    question_id: str
    label: str
    docs: tuple[str, ...]
    sections: tuple[str, ...]
    rationale: str | None = None

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.docs, self.sections))

    @property
    def doc_set(self) -> frozenset[str]:
        return frozenset(self.docs)


def _as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


def load_alternates(
    path: Path | str,
    corpus: Corpus,
    answer_key: AnswerKey | None = None,
    strict: bool = True,
) -> dict[str, list[AlternateReading]]:
    """Load and validate ``answer_key_alternates.yaml``.

    Held to exactly the same standard as the primary key — every document real,
    every section string a genuine ``##`` header **in its paired document**, and
    ``len(docs) == len(sections)``. An alternate reading that cannot be validated
    is not evidence about anything; it is a typo.

    Additionally, an alternate may only be offered for a question that exists in
    the primary key, and it must actually differ from the primary reading —
    otherwise the comparison is vacuous.

    A missing file returns ``{}``. Alternates are optional by design.
    """
    path = Path(path)
    if not path.exists():
        return {}

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("alternates") or {}

    problems: list[str] = []
    result: dict[str, list[AlternateReading]] = {}

    for question_id, readings in entries.items():
        question_id = str(question_id)

        if answer_key is not None and question_id not in answer_key:
            problems.append(f"{question_id}: not present in the primary answer key")
            continue

        if not isinstance(readings, list):
            problems.append(f"{question_id}: expected a list of readings")
            continue

        seen_labels: set[str] = set()
        for position, reading in enumerate(readings):
            if not isinstance(reading, dict):
                problems.append(f"{question_id}[{position}]: expected a mapping")
                continue

            label = str(reading.get("label", f"alt-{position}")).strip()
            if label in seen_labels:
                problems.append(f"{question_id}: duplicate label {label!r}")
            seen_labels.add(label)

            docs = _as_tuple(reading.get("docs"))
            sections = _as_tuple(reading.get("sections"))

            if len(docs) != len(sections):
                problems.append(
                    f"{question_id}/{label}: docs/sections length mismatch "
                    f"({len(docs)} vs {len(sections)})"
                )

            for doc_id, section_title in zip(docs, sections):
                if not corpus.has_document(doc_id):
                    problems.append(f"{question_id}/{label}: unknown document {doc_id!r}")
                elif not corpus.has_section(doc_id, section_title):
                    problems.append(
                        f"{question_id}/{label}: {section_title!r} is not a ## header "
                        f"in {doc_id!r}"
                    )

            alternate = AlternateReading(
                question_id=question_id,
                label=label,
                docs=docs,
                sections=sections,
                rationale=(str(reading["rationale"]).strip()
                           if reading.get("rationale") else None),
            )

            if answer_key is not None and question_id in answer_key:
                if set(alternate.pairs) == set(answer_key[question_id].pairs):
                    problems.append(
                        f"{question_id}/{label}: identical to the primary reading — "
                        "an alternate must actually differ to be evidence"
                    )

            result.setdefault(question_id, []).append(alternate)

    if problems and strict:
        raise AnswerKeyValidationError(problems)

    return result
