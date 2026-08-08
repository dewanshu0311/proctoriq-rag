"""Scorer tests.

All fixtures are synthetic and small enough that the expected numbers are
checkable by hand. The scorer runs with no embedder by default, so no model is
downloaded and groundedness correctly reports ``None``.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from proctoriq_rag.evaluation.answer_key import load_answer_key
from proctoriq_rag.evaluation.scorer import (
    WEIGHTS,
    ReferenceAnswer,
    Scorer,
    load_reference_answers,
)
from proctoriq_rag.submission.writer import Prediction

KEY_YAML = """
meta:
  version: 1
questions:
  - id: Q01
    docs: [01_alpha_guide]
    sections: ["Section 1: Installing Alpha"]
    kind: lookup
    confidence: high
  - id: Q02
    docs: [01_alpha_guide, 02_beta_guide]
    sections: ["Section 2: Common Alpha Errors", "Section 1: Installing Beta"]
    kind: multi_doc
    confidence: high
  - id: Q03
    docs: [01_alpha_guide, 01_alpha_guide]
    sections: ["Overview", "Section 1: Installing Alpha"]
    kind: multi_section
    confidence: high
  - id: Q04
    docs: [01_alpha_guide]
    sections: ["Section 2: Common Alpha Errors"]
    kind: adversarial
    confidence: high
"""

IDS = ["Q01", "Q02", "Q03", "Q04"]


class StubEmbedder:
    """Deterministic fake encoder — no model download, no network.

    Maps text to a bag-of-words vector over a fixed vocabulary, so similarity is
    predictable: identical strings score 1.0, disjoint vocabularies score 0.0.
    """

    VOCAB = ["alpha", "beta", "install", "error", "overview", "refuse", "banana"]

    def encode(self, sentences, **kwargs):
        vectors = []
        for sentence in sentences:
            lowered = str(sentence).lower()
            vectors.append([float(lowered.count(word)) for word in self.VOCAB])
        return np.asarray(vectors)


@pytest.fixture
def key(tiny_corpus, tmp_path):
    path = tmp_path / "key.yaml"
    path.write_text(textwrap.dedent(KEY_YAML), encoding="utf-8")
    return load_answer_key(path, tiny_corpus, expected_ids=IDS)


@pytest.fixture
def scorer(tiny_corpus, key):
    return Scorer(tiny_corpus, key)


def pred(question_id, docs, sections, answer="An answer."):
    return Prediction(question_id, answer, list(docs), list(sections))


PERFECT_PREDICTIONS = [
    pred("Q01", ["01_alpha_guide"], ["Section 1: Installing Alpha"]),
    pred(
        "Q02",
        ["01_alpha_guide", "02_beta_guide"],
        ["Section 2: Common Alpha Errors", "Section 1: Installing Beta"],
    ),
    pred(
        "Q03",
        ["01_alpha_guide", "01_alpha_guide"],
        ["Overview", "Section 1: Installing Alpha"],
    ),
    pred("Q04", ["01_alpha_guide"], ["Section 2: Common Alpha Errors"]),
]


# ── exact dimensions ───────────────────────────────────────────────────────
def test_perfect_prediction_scores_one(scorer):
    report = scorer.score(PERFECT_PREDICTIONS)
    assert report.dimension_means["retrieval"] == 1.0
    assert report.dimension_means["citation"] == 1.0
    assert report.failures() == []


def test_fully_wrong_prediction_scores_zero(scorer):
    report = scorer.score(
        [pred("Q01", ["02_beta_guide"], ["Section 1: Installing Beta"])]
    )
    assert report.dimension_means["retrieval"] == 0.0
    assert report.dimension_means["citation"] == 0.0
    assert len(report.failures()) == 1


def test_partial_multi_doc_match(scorer):
    """Q02 expects 2 docs; predict 1 -> P=1.0, R=0.5, F1=0.667."""
    result = scorer.score_question(
        pred("Q02", ["01_alpha_guide"], ["Section 2: Common Alpha Errors"])
    )
    assert result.retrieval.precision == 1.0
    assert result.retrieval.recall == 0.5
    assert result.retrieval.f1 == pytest.approx(2 / 3)
    assert result.citation.f1 == pytest.approx(2 / 3)
    assert not result.is_perfect


def test_right_document_wrong_section(scorer):
    """Retrieval is satisfied, citation is not. The two dimensions are separable."""
    result = scorer.score_question(
        pred("Q01", ["01_alpha_guide"], ["Section 2: Common Alpha Errors"])
    )
    assert result.retrieval.exact_match
    assert result.retrieval.f1 == 1.0
    assert result.citation.f1 == 0.0
    assert result.doc_correct and not result.section_correct


def test_q44_shaped_duplicate_document(scorer):
    """Q03 mirrors Q44: one document, two sections. Both citations count."""
    result = scorer.score_question(
        pred(
            "Q03",
            ["01_alpha_guide", "01_alpha_guide"],
            ["Overview", "Section 1: Installing Alpha"],
        )
    )
    assert result.citation.exact_match
    assert len(result.expected_pairs) == 2
    assert result.retrieval.exact_match  # set semantics: {alpha} == {alpha}
    assert result.retrieval_multiset.exact_match  # multiset agrees here too


def test_q44_half_answered_diverges_between_set_and_multiset(scorer):
    """Citing the doc once when it was expected twice: set says fine, multiset does not.

    This is precisely the ambiguity logged in DECISIONS.md — we report both
    because we do not know which the grader uses.
    """
    result = scorer.score_question(pred("Q03", ["01_alpha_guide"], ["Overview"]))
    assert result.retrieval.exact_match  # set: {alpha} == {alpha}
    assert not result.retrieval_multiset.exact_match  # multiset: 1 != 2
    assert result.citation.f1 == pytest.approx(2 / 3)


# ── unmeasured dimensions ──────────────────────────────────────────────────
def test_groundedness_is_none_without_an_embedder(scorer):
    report = scorer.score(PERFECT_PREDICTIONS)
    assert report.dimension_means["groundedness"] is None
    assert all(q.groundedness is None for q in report.per_question)


def test_answer_accuracy_is_none_without_references(scorer):
    report = scorer.score(PERFECT_PREDICTIONS)
    assert report.dimension_means["answer_accuracy"] is None
    assert report.dimension_means["integrity_refusal"] is None


def test_measured_weight_excludes_unmeasured_dimensions(scorer):
    """Retrieval (20%) + citation (15%) = 35% when nothing else is measurable."""
    report = scorer.score(PERFECT_PREDICTIONS)
    assert report.measured_weight == pytest.approx(0.35)
    assert report.composite_raw == pytest.approx(0.35)
    assert report.composite_renormalized == pytest.approx(1.0)


def test_unmeasured_dimensions_are_never_silently_zero(scorer):
    """A perfect run must renormalize to 1.0, not be dragged down by None."""
    report = scorer.score(PERFECT_PREDICTIONS)
    assert report.composite_renormalized == pytest.approx(1.0)
    assert report.composite_raw < report.composite_renormalized


# ── with an embedder ───────────────────────────────────────────────────────
def test_groundedness_is_measured_with_an_embedder(tiny_corpus, key):
    scorer = Scorer(tiny_corpus, key, embedder=StubEmbedder())
    report = scorer.score(PERFECT_PREDICTIONS)
    assert report.dimension_means["groundedness"] is not None
    assert report.measured_weight == pytest.approx(0.60)


def test_groundedness_compares_against_the_key_not_the_prediction(tiny_corpus, key):
    """A confidently wrong retrieval must not score well for self-consistency."""
    scorer = Scorer(tiny_corpus, key, embedder=StubEmbedder())
    grounded = scorer.score_question(
        pred("Q01", ["01_alpha_guide"], ["Section 1: Installing Alpha"],
             answer="Install steps for alpha.")
    )
    ungrounded = scorer.score_question(
        pred("Q01", ["02_beta_guide"], ["Section 1: Installing Beta"],
             answer="banana banana banana")
    )
    assert grounded.groundedness > ungrounded.groundedness


def test_empty_answer_scores_zero_groundedness(tiny_corpus, key):
    scorer = Scorer(tiny_corpus, key, embedder=StubEmbedder())
    result = scorer.score_question(
        pred("Q01", ["01_alpha_guide"], ["Section 1: Installing Alpha"], answer="   ")
    )
    assert result.groundedness == 0.0


def test_integrity_refusal_is_adversarial_only(tiny_corpus, key):
    references = {
        "Q01": ReferenceAnswer(answer="alpha install", refusal="refuse"),
        "Q04": ReferenceAnswer(answer="alpha error", refusal="refuse"),
    }
    scorer = Scorer(tiny_corpus, key, embedder=StubEmbedder(), reference_answers=references)
    report = scorer.score(PERFECT_PREDICTIONS)
    by_id = {q.question_id: q for q in report.per_question}
    assert by_id["Q04"].integrity_refusal is not None  # adversarial
    assert by_id["Q01"].integrity_refusal is None  # lookup, despite a refusal ref
    assert by_id["Q02"].answer_accuracy is None  # no reference supplied


def test_all_dimensions_measured_reaches_full_weight(tiny_corpus, key):
    references = {qid: ReferenceAnswer(answer="a", refusal="r") for qid in IDS}
    scorer = Scorer(tiny_corpus, key, embedder=StubEmbedder(), reference_answers=references)
    report = scorer.score(PERFECT_PREDICTIONS)
    assert report.measured_weight == pytest.approx(sum(WEIGHTS.values()))
    assert report.measured_weight == pytest.approx(1.0)


# ── report and misc ────────────────────────────────────────────────────────
def test_report_notes_flag_the_unmeasured_weight(scorer):
    report = scorer.score(PERFECT_PREDICTIONS)
    joined = " ".join(report.notes)
    assert "UNMEASURED" in joined
    assert "exact" in joined


def test_report_notes_flag_the_groundedness_caveat(tiny_corpus, key):
    scorer = Scorer(tiny_corpus, key, embedder=StubEmbedder())
    report = scorer.score(PERFECT_PREDICTIONS)
    assert any("embedding model is unknown" in n for n in report.notes)


def test_report_text_leads_with_failures(scorer):
    report = scorer.score(
        [pred("Q01", ["02_beta_guide"], ["Section 1: Installing Beta"])]
    )
    text = report.to_text()
    assert "PER-QUESTION FAILURES" in text
    assert "Q01" in text
    assert "expected" in text and "predicted" in text
    assert text.index("PER-QUESTION FAILURES") < text.index("DIMENSION MEANS")


def test_report_dataframe_has_one_row_per_question(scorer):
    frame = scorer.score(PERFECT_PREDICTIONS).to_dataframe()
    assert len(frame) == len(IDS)
    assert set(frame["question_id"]) == set(IDS)


def test_unknown_question_id_raises(scorer):
    with pytest.raises(KeyError, match="absent from the answer key"):
        scorer.score([pred("Q99", ["01_alpha_guide"], ["Overview"])])


def test_load_reference_answers_missing_file_is_empty(tmp_path):
    assert load_reference_answers(tmp_path / "absent.yaml") == {}


def test_load_reference_answers_accepts_both_shapes(tmp_path):
    path = tmp_path / "refs.yaml"
    path.write_text(
        textwrap.dedent(
            """
            answers:
              Q01: "a bare string answer"
              Q04:
                answer: "structured answer"
                refusal: "structured refusal"
            """
        ),
        encoding="utf-8",
    )
    refs = load_reference_answers(path)
    assert refs["Q01"].answer == "a bare string answer"
    assert refs["Q01"].refusal is None
    assert refs["Q04"].refusal == "structured refusal"
