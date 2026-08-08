"""Metric primitives. Every expected number here is computable on paper."""

from __future__ import annotations

import numpy as np
import pytest

from proctoriq_rag.evaluation.metrics import (
    cosine,
    mean_or_none,
    multiset_scores,
    set_scores,
)


def test_perfect_match():
    score = set_scores(["a", "b"], ["a", "b"])
    assert (score.precision, score.recall, score.f1) == (1.0, 1.0, 1.0)
    assert score.exact_match


def test_disjoint_sets_score_zero():
    score = set_scores(["a"], ["b"])
    assert (score.precision, score.recall, score.f1) == (0.0, 0.0, 0.0)
    assert not score.exact_match


def test_partial_multi_doc_recall():
    """Predict 1 of 2 correct docs: P=1.0, R=0.5, F1=2*1*.5/1.5=0.667."""
    score = set_scores(["a"], ["a", "b"])
    assert score.precision == 1.0
    assert score.recall == 0.5
    assert score.f1 == pytest.approx(2 / 3)
    assert not score.exact_match


def test_over_prediction_costs_precision():
    """Predict 2 where 1 was expected: P=0.5, R=1.0, F1=0.667."""
    score = set_scores(["a", "b"], ["a"])
    assert score.precision == 0.5
    assert score.recall == 1.0
    assert score.f1 == pytest.approx(2 / 3)


def test_order_does_not_matter():
    assert set_scores(["b", "a"], ["a", "b"]).exact_match


def test_duplicates_collapse_under_set_semantics():
    assert set_scores(["a", "a"], ["a"]).exact_match


def test_both_empty_scores_perfect():
    score = set_scores([], [])
    assert score.f1 == 1.0
    assert score.exact_match


def test_one_empty_scores_zero():
    assert set_scores([], ["a"]).f1 == 0.0
    assert set_scores(["a"], []).f1 == 0.0


def test_pairs_are_hashable_and_document_scoped():
    """A right section paired with the wrong doc is not a match."""
    expected = [("doc1", "Section 1"), ("doc2", "Section 2")]
    predicted = [("doc2", "Section 1"), ("doc1", "Section 2")]
    assert set_scores(predicted, expected).f1 == 0.0


# ── multiset variant (the Q44 question) ────────────────────────────────────
def test_multiset_honours_repeats():
    """Q44 shape: the same doc twice. Set semantics say 1, multiset says 2."""
    assert set_scores(["d", "d"], ["d", "d"]).exact_match
    assert multiset_scores(["d", "d"], ["d", "d"]).exact_match
    assert not multiset_scores(["d"], ["d", "d"]).exact_match
    assert multiset_scores(["d"], ["d", "d"]).recall == 0.5
    assert multiset_scores(["d"], ["d", "d"]).precision == 1.0


def test_multiset_empty_conventions_match_set():
    assert multiset_scores([], []).exact_match
    assert multiset_scores([], ["a"]).f1 == 0.0


# ── cosine ─────────────────────────────────────────────────────────────────
def test_cosine_identical_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_opposite_is_minus_one():
    assert cosine(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_cosine_zero_vector_returns_zero_not_nan():
    result = cosine(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    assert result == 0.0
    assert not np.isnan(result)


def test_cosine_is_scale_invariant():
    a, b = np.array([1.0, 2.0]), np.array([10.0, 20.0])
    assert cosine(a, b) == pytest.approx(1.0)


# ── mean_or_none ───────────────────────────────────────────────────────────
def test_mean_or_none_ignores_none():
    assert mean_or_none([1.0, None, 3.0]) == pytest.approx(2.0)


def test_mean_or_none_all_none_is_none():
    """An unmeasurable dimension must stay None, never become 0.0."""
    assert mean_or_none([None, None]) is None
    assert mean_or_none([]) is None
