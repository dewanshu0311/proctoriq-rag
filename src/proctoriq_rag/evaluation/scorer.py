"""Local reconstruction of the competition metric.

What this can and cannot measure
--------------------------------
The competition scores five dimensions. We can reproduce two of them exactly,
one as a faithful-but-uncalibrated proxy, and two not at all:

===================  ======  ==========================================================
Dimension            Weight  Status here
===================  ======  ==========================================================
Answer accuracy       25%    UNMEASURED — no golden answers exist. Optional
                             ``reference_answers.yaml`` fills this in if supplied.
Groundedness          25%    PROXY — we have the real source text, so this is a
                             faithful reconstruction of the *method*. The absolute
                             number is not comparable to the real score; see below.
Retrieval quality     20%    EXACT — predicted document set vs the key's.
Citation accuracy     15%    EXACT — predicted (doc, section) pairs vs the key's.
Integrity-refusal     15%    UNMEASURED — adversarial subset only, same treatment
                             as answer accuracy.
===================  ======  ==========================================================

**The groundedness caveat, stated in code because it is easy to forget.** The
grader's embedding model is unknown; the competition says "semantic similarity"
without naming one. Our proxy uses a local sentence-transformers model. That
makes our groundedness number meaningful *relatively* — config A vs config B —
and meaningless *absolutely*. ``groundedness = 0.82`` does not mean "we would
score 82 on that dimension". See ``docs/DECISIONS.md``.

Unmeasured dimensions propagate as ``None`` and are excluded from the composite
rather than silently substituted with a related number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd
import yaml

from proctoriq_rag.corpus.loader import Corpus
from proctoriq_rag.evaluation.answer_key import AnswerKey, KeyEntry
from proctoriq_rag.evaluation.metrics import (
    SetScore,
    cosine,
    mean_or_none,
    multiset_scores,
    set_scores,
)
from proctoriq_rag.submission.writer import Prediction

#: The competition's published weights.
WEIGHTS: dict[str, float] = {
    "answer_accuracy": 0.25,
    "groundedness": 0.25,
    "retrieval": 0.20,
    "citation": 0.15,
    "integrity_refusal": 0.15,
}


class Embedder(Protocol):
    """Minimal structural type for a sentence-transformers-style encoder.

    Declared as a Protocol so the scorer can be unit-tested with a stub and no
    model download. ``sentence_transformers.SentenceTransformer`` satisfies it.
    """

    def encode(self, sentences: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class ReferenceAnswer:
    """A hand-written reference for one question, if we ever supply one."""

    answer: str | None = None
    refusal: str | None = None


@dataclass(frozen=True)
class QuestionScore:
    """Per-question result. ``None`` means unmeasurable, not zero."""

    question_id: str
    kind: str
    confidence: str
    retrieval: SetScore
    retrieval_multiset: SetScore
    citation: SetScore
    groundedness: float | None
    answer_accuracy: float | None
    integrity_refusal: float | None
    expected_docs: tuple[str, ...]
    predicted_docs: tuple[str, ...]
    expected_pairs: tuple[tuple[str, str], ...]
    predicted_pairs: tuple[tuple[str, str], ...]

    @property
    def is_perfect(self) -> bool:
        return self.retrieval.exact_match and self.citation.exact_match

    @property
    def doc_correct(self) -> bool:
        return self.retrieval.exact_match

    @property
    def section_correct(self) -> bool:
        return self.citation.exact_match


@dataclass
class ScoreReport:
    """Aggregate result. ``to_text()`` is the thing to actually read."""

    per_question: list[QuestionScore]
    dimension_means: dict[str, float | None]
    measured_weight: float
    composite_raw: float
    composite_renormalized: float
    notes: list[str] = field(default_factory=list)

    def failures(self) -> list[QuestionScore]:
        """Questions where the document set or the citation pairs are wrong."""
        return [q for q in self.per_question if not q.is_perfect]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "question_id": q.question_id,
                    "kind": q.kind,
                    "confidence": q.confidence,
                    "doc_ok": q.doc_correct,
                    "sec_ok": q.section_correct,
                    "doc_f1": round(q.retrieval.f1, 3),
                    "cite_f1": round(q.citation.f1, 3),
                    "grounded": (
                        None if q.groundedness is None else round(q.groundedness, 3)
                    ),
                    "expected_docs": "|".join(q.expected_docs),
                    "predicted_docs": "|".join(q.predicted_docs),
                }
                for q in self.per_question
            ]
        )

    def to_text(self, max_failures: int | None = None) -> str:
        """Human-readable report, failures first."""
        lines: list[str] = []
        rule = "=" * 78

        failures = self.failures()
        shown = failures if max_failures is None else failures[:max_failures]

        lines.append(rule)
        lines.append(
            f"PER-QUESTION FAILURES — {len(failures)} of {len(self.per_question)} "
            "question(s) have a wrong document set or wrong citation pairs"
        )
        lines.append(rule)

        if not failures:
            lines.append("  none — every question matched the key exactly.")
        for q in shown:
            flag_doc = "OK " if q.doc_correct else "DOC"
            flag_sec = "OK " if q.section_correct else "SEC"
            lines.append(
                f"\n  [{q.question_id}] {flag_doc} {flag_sec}  "
                f"kind={q.kind} confidence={q.confidence}"
            )
            lines.append(f"      expected : {_render_pairs(q.expected_pairs)}")
            lines.append(f"      predicted: {_render_pairs(q.predicted_pairs)}")
            lines.append(
                f"      doc F1={q.retrieval.f1:.3f}  cite F1={q.citation.f1:.3f}"
                + (
                    ""
                    if q.groundedness is None
                    else f"  grounded={q.groundedness:.3f}"
                )
            )
        if max_failures is not None and len(failures) > max_failures:
            lines.append(f"\n  ... and {len(failures) - max_failures} more.")

        lines.append("")
        lines.append(rule)
        lines.append("DIMENSION MEANS")
        lines.append(rule)
        for name, weight in WEIGHTS.items():
            value = self.dimension_means.get(name)
            rendered = "UNMEASURED" if value is None else f"{value:.4f}"
            lines.append(f"  {name:<20} weight={weight:>5.0%}   {rendered}")

        lines.append("")
        lines.append(rule)
        lines.append("COMPOSITE")
        lines.append(rule)
        lines.append(
            f"  measured weight        : {self.measured_weight:.0%} of the real metric"
        )
        lines.append(
            f"  composite (raw)        : {self.composite_raw * 100:.2f} / 100  "
            "<- unmeasured dimensions counted as 0"
        )
        lines.append(
            f"  composite (renormalized): {self.composite_renormalized * 100:.2f} / 100  "
            "<- over measured dimensions only"
        )

        if self.notes:
            lines.append("")
            lines.append(rule)
            lines.append("READ THIS BEFORE QUOTING ANY NUMBER ABOVE")
            lines.append(rule)
            for note in self.notes:
                lines.append(f"  ! {note}")

        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.to_text()


def _render_pairs(pairs: Sequence[tuple[str, str]]) -> str:
    if not pairs:
        return "(none)"
    return "  |  ".join(f"{doc} :: {section}" for doc, section in pairs)


def load_reference_answers(path: Path | str) -> dict[str, ReferenceAnswer]:
    """Load an optional hand-written ``reference_answers.yaml``.

    Expected shape::

        answers:
          Q01:
            answer: "..."
          Q29:
            refusal: "..."

    Absent file -> empty mapping. Absent key for a question -> that question
    contributes ``None``, not zero.
    """
    path = Path(path)
    if not path.exists():
        return {}

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    answers = raw.get("answers") or raw
    result: dict[str, ReferenceAnswer] = {}
    for question_id, value in (answers or {}).items():
        if isinstance(value, str):
            result[str(question_id)] = ReferenceAnswer(answer=value)
        elif isinstance(value, dict):
            result[str(question_id)] = ReferenceAnswer(
                answer=value.get("answer"),
                refusal=value.get("refusal"),
            )
    return result


class Scorer:
    """Score predictions against the answer key.

    ``embedder`` is optional and injected. Without it, groundedness and the
    reference-based dimensions report ``None`` and the scorer runs instantly with
    no model download — which is what the unit tests use.
    """

    def __init__(
        self,
        corpus: Corpus,
        answer_key: AnswerKey,
        embedder: Embedder | None = None,
        reference_answers: dict[str, ReferenceAnswer] | None = None,
    ) -> None:
        self.corpus = corpus
        self.answer_key = answer_key
        self.embedder = embedder
        self.reference_answers = reference_answers or {}

    # ── embedding helpers ──────────────────────────────────────────────────
    def _similarity(self, left: str, right: str) -> float | None:
        """Cosine similarity between two texts, or ``None`` if not computable."""
        if self.embedder is None:
            return None
        if not left.strip() or not right.strip():
            return 0.0
        vectors = np.asarray(self.embedder.encode([left, right]))
        return cosine(vectors[0], vectors[1])

    # ── per-dimension ──────────────────────────────────────────────────────
    def _groundedness(self, entry: KeyEntry, prediction: Prediction) -> float | None:
        """Similarity between the answer and the text it should be grounded in.

        The reference text is the concatenated body of the *key's* cited
        sections, not the pipeline's — otherwise a confidently wrong retrieval
        would score well for being consistent with itself.
        """
        source = self.corpus.section_text(list(entry.pairs))
        return self._similarity(prediction.answer_text, source)

    def _answer_accuracy(
        self, entry: KeyEntry, prediction: Prediction
    ) -> float | None:
        reference = self.reference_answers.get(entry.id)
        if reference is None or not reference.answer:
            return None
        return self._similarity(prediction.answer_text, reference.answer)

    def _integrity_refusal(
        self, entry: KeyEntry, prediction: Prediction
    ) -> float | None:
        """Adversarial questions only — 14 of the 50."""
        if not entry.is_adversarial:
            return None
        reference = self.reference_answers.get(entry.id)
        if reference is None or not reference.refusal:
            return None
        return self._similarity(prediction.answer_text, reference.refusal)

    # ── public API ─────────────────────────────────────────────────────────
    def score_question(self, prediction: Prediction) -> QuestionScore:
        entry = self.answer_key[prediction.question_id]

        return QuestionScore(
            question_id=entry.id,
            kind=entry.kind,
            confidence=entry.confidence,
            retrieval=set_scores(prediction.cited_docs, entry.docs),
            retrieval_multiset=multiset_scores(
                list(prediction.cited_docs), list(entry.docs)
            ),
            citation=set_scores(prediction.pairs, entry.pairs),
            groundedness=self._groundedness(entry, prediction),
            answer_accuracy=self._answer_accuracy(entry, prediction),
            integrity_refusal=self._integrity_refusal(entry, prediction),
            expected_docs=entry.docs,
            predicted_docs=tuple(prediction.cited_docs),
            expected_pairs=entry.pairs,
            predicted_pairs=prediction.pairs,
        )

    def score(self, predictions: Sequence[Prediction]) -> ScoreReport:
        """Score a full run and assemble the report."""
        unknown = [
            p.question_id
            for p in predictions
            if p.question_id not in self.answer_key
        ]
        if unknown:
            raise KeyError(
                f"predictions contain question id(s) absent from the answer key: "
                f"{', '.join(unknown)}"
            )

        per_question = [self.score_question(p) for p in predictions]

        dimension_means: dict[str, float | None] = {
            "retrieval": mean_or_none([q.retrieval.f1 for q in per_question]),
            "citation": mean_or_none([q.citation.f1 for q in per_question]),
            "groundedness": mean_or_none([q.groundedness for q in per_question]),
            "answer_accuracy": mean_or_none(
                [q.answer_accuracy for q in per_question]
            ),
            "integrity_refusal": mean_or_none(
                [q.integrity_refusal for q in per_question]
            ),
        }

        measured_weight = sum(
            weight
            for name, weight in WEIGHTS.items()
            if dimension_means.get(name) is not None
        )
        composite_raw = sum(
            WEIGHTS[name] * value
            for name, value in dimension_means.items()
            if value is not None
        )
        composite_renormalized = (
            composite_raw / measured_weight if measured_weight > 0 else 0.0
        )

        notes = self._notes(dimension_means, measured_weight)

        return ScoreReport(
            per_question=per_question,
            dimension_means=dimension_means,
            measured_weight=measured_weight,
            composite_raw=composite_raw,
            composite_renormalized=composite_renormalized,
            notes=notes,
        )

    def _notes(
        self, dimension_means: dict[str, float | None], measured_weight: float
    ) -> list[str]:
        notes: list[str] = []

        unmeasured = [
            name for name, value in dimension_means.items() if value is None
        ]
        if unmeasured:
            missing_weight = sum(WEIGHTS[name] for name in unmeasured)
            notes.append(
                f"{missing_weight:.0%} of the real metric is UNMEASURED here "
                f"({', '.join(unmeasured)}). Neither composite is a leaderboard "
                "prediction."
            )
        if dimension_means.get("groundedness") is not None:
            notes.append(
                "Groundedness is a proxy: the grader's embedding model is unknown, "
                "so this number is comparable between our own configs but its "
                "absolute value says nothing about the real score."
            )
        if dimension_means.get("answer_accuracy") is None:
            notes.append(
                "Answer accuracy has no golden answers. Supply "
                "data/validation/reference_answers.yaml to fill this gap."
            )
        notes.append(
            "Retrieval and citation (35% combined) are exact, not proxies — trust "
            "those two."
        )
        return notes
