"""Load and validate the hand-built answer key.

The key is a **holdout test set**. It is the only independent notion of
correctness available — the competition ships no training set and no ground
truth. Its value depends entirely on it never influencing what the pipeline
does, so this module lives under ``evaluation/`` and nothing outside
``evaluation/``, ``scripts/`` and ``tests/`` may import it.

Validation collects *every* problem before raising, so one run tells you
everything that is wrong with a key rather than the first thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import yaml

from proctoriq_rag.corpus.loader import Corpus

VALID_KINDS = frozenset(
    {"lookup", "multi_doc", "multi_section", "adversarial", "trap"}
)
VALID_CONFIDENCES = frozenset({"high", "medium", "low"})

#: The competition's 50 questions, Q01..Q50. Generated, never typed out — a
#: literal list here would be the first step towards question-specific logic.
EXPECTED_QUESTION_IDS: tuple[str, ...] = tuple(f"Q{i:02d}" for i in range(1, 51))


class AnswerKeyValidationError(Exception):
    """Raised when the key does not agree with the corpus.

    The message contains one line per problem found. A clean key raises nothing.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        body = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(
            f"Answer key validation failed with {len(self.problems)} problem(s):\n{body}"
        )


@dataclass(frozen=True)
class KeyEntry:
    """One question's expected citation.

    ``docs`` and ``sections`` are positionally aligned: the Nth section belongs
    to the Nth document. That alignment is the whole point of the format and is
    validated on load.
    """

    id: str
    docs: tuple[str, ...]
    sections: tuple[str, ...]
    kind: str
    confidence: str
    notes: str | None = None

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        """The positional ``(doc, section)`` pairs — the citation ground truth."""
        return tuple(zip(self.docs, self.sections))

    @property
    def doc_set(self) -> frozenset[str]:
        """Distinct documents. Q44 cites one document twice; this collapses it."""
        return frozenset(self.docs)

    @property
    def is_adversarial(self) -> bool:
        return self.kind == "adversarial"

    @property
    def is_multi_source(self) -> bool:
        return len(self.docs) > 1


class AnswerKey:
    """The validated key, indexed by question ID."""

    def __init__(self, entries: Sequence[KeyEntry], version: int | None = None) -> None:
        self._entries = {entry.id: entry for entry in entries}
        self.version = version

    def __getitem__(self, question_id: str) -> KeyEntry:
        return self._entries[question_id]

    def __contains__(self, question_id: object) -> bool:
        return question_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[KeyEntry]:
        """Iterate entries in question-ID order."""
        for question_id in self.ids():
            yield self._entries[question_id]

    def ids(self) -> list[str]:
        return sorted(self._entries)

    def get(self, question_id: str) -> KeyEntry | None:
        return self._entries.get(question_id)

    def by_kind(self, kind: str) -> list[KeyEntry]:
        return [e for e in self if e.kind == kind]

    def by_confidence(self, confidence: str) -> list[KeyEntry]:
        return [e for e in self if e.confidence == confidence]

    def adversarial_ids(self) -> list[str]:
        return [e.id for e in self if e.is_adversarial]

    def counts(self, attribute: str) -> dict[str, int]:
        """Frequency table over ``kind`` or ``confidence``, for reporting."""
        table: dict[str, int] = {}
        for entry in self:
            value = getattr(entry, attribute)
            table[value] = table.get(value, 0) + 1
        return dict(sorted(table.items(), key=lambda kv: (-kv[1], kv[0])))

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"AnswerKey({len(self._entries)} entries, version={self.version})"


def _as_str_tuple(value: object) -> tuple[str, ...]:
    """Coerce a YAML scalar-or-list into a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


def load_answer_key(
    path: Path | str,
    corpus: Corpus,
    expected_ids: Sequence[str] | None = EXPECTED_QUESTION_IDS,
    strict: bool = True,
) -> AnswerKey:
    """Load ``answer_key.yaml`` and validate it against ``corpus``.

    Fails loudly on:

    - ``len(docs) != len(sections)`` — positional alignment is broken
    - a document ID that is not in the corpus
    - a section string that is not a real ``##`` header **in its paired document**
      (right section title, wrong document, is still an error)
    - a duplicated question ID
    - a question ID set that is not exactly ``expected_ids``
    - an unknown ``kind`` or ``confidence``

    Pass ``strict=False`` to collect problems without raising — used by
    ``scripts/validate_key.py`` so it can print a full report and choose its own
    exit code.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Answer key not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    meta = raw.get("meta") or {}
    raw_questions = raw.get("questions")

    if not isinstance(raw_questions, list) or not raw_questions:
        raise AnswerKeyValidationError(
            [f"{path.name}: top-level 'questions' key is missing or empty"]
        )

    problems: list[str] = []
    entries: list[KeyEntry] = []
    seen: set[str] = set()

    for position, item in enumerate(raw_questions):
        if not isinstance(item, dict):
            problems.append(f"entry #{position}: expected a mapping, got {type(item).__name__}")
            continue

        question_id = str(item.get("id", "")).strip()
        if not question_id:
            problems.append(f"entry #{position}: missing 'id'")
            continue
        if question_id in seen:
            problems.append(f"{question_id}: duplicated question id")
            continue
        seen.add(question_id)

        docs = _as_str_tuple(item.get("docs"))
        sections = _as_str_tuple(item.get("sections"))
        kind = str(item.get("kind", "")).strip()
        confidence = str(item.get("confidence", "")).strip()
        notes = item.get("notes")

        if len(docs) != len(sections):
            problems.append(
                f"{question_id}: docs/sections length mismatch "
                f"({len(docs)} docs vs {len(sections)} sections) — "
                "positional alignment is broken"
            )

        for doc_id, section_title in zip(docs, sections):
            if not corpus.has_document(doc_id):
                problems.append(f"{question_id}: unknown document {doc_id!r}")
            elif not corpus.has_section(doc_id, section_title):
                available = ", ".join(repr(t) for t in corpus.section_titles(doc_id))
                problems.append(
                    f"{question_id}: {section_title!r} is not a ## header in "
                    f"{doc_id!r}. Available: {available}"
                )

        if kind not in VALID_KINDS:
            problems.append(
                f"{question_id}: unknown kind {kind!r} "
                f"(valid: {', '.join(sorted(VALID_KINDS))})"
            )
        if confidence not in VALID_CONFIDENCES:
            problems.append(
                f"{question_id}: unknown confidence {confidence!r} "
                f"(valid: {', '.join(sorted(VALID_CONFIDENCES))})"
            )

        entries.append(
            KeyEntry(
                id=question_id,
                docs=docs,
                sections=sections,
                kind=kind,
                confidence=confidence,
                notes=str(notes).strip() if notes else None,
            )
        )

    if expected_ids is not None:
        expected = set(expected_ids)
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        if missing:
            problems.append(f"missing question id(s): {', '.join(missing)}")
        if extra:
            problems.append(f"unexpected question id(s): {', '.join(extra)}")

    if problems and strict:
        raise AnswerKeyValidationError(problems)

    key = AnswerKey(entries, version=meta.get("version"))
    key.problems = problems  # type: ignore[attr-defined]
    return key
