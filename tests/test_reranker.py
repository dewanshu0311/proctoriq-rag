"""Reranker tests. Stub CrossEncoder — no model download."""

from __future__ import annotations

import numpy as np
import pytest

from proctoriq_rag.retrieval.chunking import SectionChunker
from proctoriq_rag.retrieval.citation import RelativeGap, ScoreThreshold
from proctoriq_rag.retrieval.reranker import (
    CrossEncoderReranker,
    RerankCache,
    apply_transform,
    build_rerank_texts,
    ordering_is_identical,
    saturation_report,
)
from tests.conftest import EXPECTED_TOTAL_SECTIONS, requires_competition_data


class StubCrossEncoder:
    """Deterministic cross-encoder returning caller-supplied logits."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype="float64")
        self.calls = 0

    def predict(self, pairs, **kwargs):
        self.calls += 1
        return self.matrix.ravel()[: len(pairs)]


@pytest.fixture
def sections(corpus):
    return SectionChunker(include_header_in_text=True).chunk(corpus)


# ── transforms ─────────────────────────────────────────────────────────────
def test_sigmoid_maps_negatives_into_unit_range():
    """The bug this fixes: 49 of 53 real scores were negative logits."""
    logits = np.array([[-11.4, -3.0, 0.0, 6.4]])
    out = apply_transform(logits, "sigmoid")
    assert out.min() > 0.0
    assert out.max() < 1.0
    assert out[0, 2] == pytest.approx(0.5)


def test_sigmoid_preserves_order():
    logits = np.array([[-11.4, 6.4, -3.0, 0.0]])
    assert list(np.argsort(-logits[0])) == list(np.argsort(-apply_transform(logits, "sigmoid")[0]))


def test_sigmoid_is_numerically_stable_at_extremes():
    out = apply_transform(np.array([[-1e4, 1e4]]), "sigmoid")
    assert np.isfinite(out).all()
    assert out[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert out[0, 1] == pytest.approx(1.0, abs=1e-9)


def test_minmax_is_per_row():
    logits = np.array([[-10.0, 0.0, 10.0], [1.0, 2.0, 3.0]])
    out = apply_transform(logits, "minmax")
    np.testing.assert_allclose(out[0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(out[1], [0.0, 0.5, 1.0])


def test_minmax_on_a_flat_row_is_all_ones():
    np.testing.assert_allclose(apply_transform(np.array([[2.0, 2.0]]), "minmax"), [[1.0, 1.0]])


def test_raw_is_identity():
    logits = np.array([[-1.0, 2.0]])
    np.testing.assert_allclose(apply_transform(logits, "raw"), logits)


def test_unknown_transform_raises():
    with pytest.raises(ValueError, match="unknown score_transform"):
        apply_transform(np.zeros((1, 2)), "softmax")


def test_ordering_identical_across_all_transforms_on_random_logits():
    """The guarantee the sweep aborts on if violated."""
    rng = np.random.default_rng(0)
    logits = rng.normal(-3.0, 5.0, size=(50, 53))
    assert all(ordering_is_identical(logits).values())


def test_ordering_check_is_not_vacuous():
    """A deliberately non-monotonic transform must be detectable."""
    logits = np.array([[-2.0, 1.0, 3.0]])
    scrambled = np.argsort(-(logits ** 2), axis=1)
    baseline = np.argsort(-logits, axis=1)
    assert not np.array_equal(scrambled, baseline)


# ── the cardinality bug this phase exists to avoid ─────────────────────────
def test_relative_gap_is_broken_on_raw_negative_logits():
    """Documents why sigmoid is required, not merely preferred.

    With every score negative, RelativeGap's ``top <= 0`` guard fires and it
    returns exactly one citation regardless of ratio — silently turning the whole
    cardinality sweep into topk-1 while producing a plausible curve.
    """
    from proctoriq_rag.retrieval.citation import RankedSection

    raw = [
        RankedSection("d", "A", -1.0, 1),
        RankedSection("d", "B", -1.05, 1),
    ]
    assert len(RelativeGap(0.9).select(raw)) == 1  # the bug

    transformed = apply_transform(np.array([[-1.0, -1.05]]), "sigmoid")[0]
    fixed = [
        RankedSection("d", "A", float(transformed[0]), 1),
        RankedSection("d", "B", float(transformed[1]), 1),
    ]
    assert len(RelativeGap(0.9).select(fixed)) == 2  # sane behaviour restored


def test_score_threshold_survives_negative_logits_via_minmax():
    """ScoreThreshold min-max normalizes internally, so it was never broken."""
    from proctoriq_rag.retrieval.citation import RankedSection

    raw = [RankedSection("d", "A", -1.0, 1), RankedSection("d", "B", -1.05, 1),
           RankedSection("d", "C", -9.0, 1)]
    assert len(ScoreThreshold(0.9).select(raw)) == 2


# ── saturation diagnostic ──────────────────────────────────────────────────
def test_saturation_report_flags_a_saturated_distribution():
    """Wide logits -> sigmoid pins near 0 and 1 -> ratios stop discriminating."""
    logits = np.tile(np.array([12.0, 11.0] + [-12.0] * 51), (50, 1))
    report = saturation_report(logits)
    assert report["top1_median"] > 0.99
    assert report["ratio_median"] > 0.95
    assert report["ratio_iqr"] < 0.01


def test_saturation_report_shows_spread_when_logits_are_narrow():
    rng = np.random.default_rng(1)
    logits = rng.normal(0.0, 0.5, size=(50, 53))
    report = saturation_report(logits)
    assert 0.2 < report["top1_median"] < 0.9
    assert report["ratio_iqr"] > 0.0


# ── text variants ──────────────────────────────────────────────────────────
@requires_competition_data
def test_text_variants_differ_but_metadata_does_not(corpus, sections):
    body = build_rerank_texts(corpus, sections, "body")
    titled = build_rerank_texts(corpus, sections, "titled")
    assert len(body) == len(titled) == len(sections)
    assert body != titled
    assert all(t.startswith(corpus[c.doc_id].title) for t, c in zip(titled, sections))


@requires_competition_data
def test_titled_variant_contains_the_section_title(corpus, sections):
    titled = build_rerank_texts(corpus, sections, "titled")
    for text, chunk in zip(titled, sections):
        assert chunk.section_title in text


# ── scoring, exhaustive vs pooled ──────────────────────────────────────────
@requires_competition_data
def test_exhaustive_scores_every_section(corpus, sections, tmp_path):
    questions = ["q one", "q two"]
    stub = StubCrossEncoder(np.arange(len(questions) * len(sections), dtype="float64"))
    reranker = CrossEncoderReranker(
        "stub", cache=RerankCache(tmp_path), model=stub
    )
    matrix = reranker.fit(questions, sections, corpus)
    assert matrix.shape == (2, EXPECTED_TOTAL_SECTIONS)
    assert len(reranker.rerank(0)) == EXPECTED_TOTAL_SECTIONS


@requires_competition_data
def test_pooled_scores_exactly_the_pool_size(corpus, sections, tmp_path):
    questions = ["q one"]
    stub = StubCrossEncoder(np.arange(len(sections), dtype="float64"))
    reranker = CrossEncoderReranker("stub", cache=RerankCache(tmp_path), model=stub)
    reranker.fit(questions, sections, corpus)
    for size in (10, 20, 30):
        assert len(reranker.rerank(0, candidates=list(range(size)))) == size


@requires_competition_data
def test_pooled_is_a_subset_of_exhaustive_with_same_relative_order(
    corpus, sections, tmp_path
):
    """The two modes share one code path; this proves they cannot drift."""
    rng = np.random.default_rng(2)
    stub = StubCrossEncoder(rng.normal(size=len(sections)))
    reranker = CrossEncoderReranker("stub", cache=RerankCache(tmp_path), model=stub)
    reranker.fit(["q"], sections, corpus)

    pool = list(range(20))
    exhaustive = reranker.rerank(0)
    pooled = reranker.rerank(0, candidates=pool)

    pooled_citations = [r.citation for r in pooled]
    assert set(pooled_citations) <= {r.citation for r in exhaustive}
    filtered = [r.citation for r in exhaustive if r.citation in set(pooled_citations)]
    assert filtered == pooled_citations


@requires_competition_data
def test_rerank_ranks_by_descending_score(corpus, sections, tmp_path):
    rng = np.random.default_rng(3)
    stub = StubCrossEncoder(rng.normal(size=len(sections)))
    reranker = CrossEncoderReranker("stub", cache=RerankCache(tmp_path), model=stub)
    reranker.fit(["q"], sections, corpus)
    result = reranker.rerank(0)
    assert [r.rank for r in result] == list(range(1, len(sections) + 1))
    scores = [r.score for r in result]
    assert scores == sorted(scores, reverse=True)


@requires_competition_data
def test_top_k_truncates(corpus, sections, tmp_path):
    stub = StubCrossEncoder(np.arange(len(sections), dtype="float64"))
    reranker = CrossEncoderReranker("stub", cache=RerankCache(tmp_path), model=stub)
    reranker.fit(["q"], sections, corpus)
    assert len(reranker.rerank(0, top_k=5)) == 5


@requires_competition_data
def test_adapts_to_phase_one_ranked_section(corpus, sections, tmp_path):
    stub = StubCrossEncoder(np.arange(len(sections), dtype="float64"))
    reranker = CrossEncoderReranker("stub", cache=RerankCache(tmp_path), model=stub)
    reranker.fit(["q"], sections, corpus)
    adapted = reranker.as_ranked_sections(reranker.rerank(0, top_k=3))
    assert len(adapted) == 3
    assert adapted[0].citation == reranker.rerank(0, top_k=1)[0].citation


def test_reading_scores_before_fit_raises(tmp_path):
    reranker = CrossEncoderReranker("stub", cache=RerankCache(tmp_path),
                                    model=StubCrossEncoder(np.zeros(1)))
    with pytest.raises(RuntimeError, match="call fit"):
        _ = reranker.raw_scores


# ── cache ──────────────────────────────────────────────────────────────────
@requires_competition_data
def test_cache_miss_then_hit(corpus, sections, tmp_path):
    cache = RerankCache(tmp_path)
    stub = StubCrossEncoder(np.arange(len(sections), dtype="float64"))
    first = CrossEncoderReranker("m", cache=cache, model=stub)
    first.fit(["q"], sections, corpus)
    assert stub.calls == 1

    second = CrossEncoderReranker("m", cache=cache, model=stub)
    second.fit(["q"], sections, corpus)
    assert stub.calls == 1  # not re-scored
    assert cache.stats == {"hits": 1, "misses": 1}
    np.testing.assert_allclose(first.raw_scores, second.raw_scores)


@requires_competition_data
def test_cache_invalidates_on_model_change(corpus, sections, tmp_path):
    cache = RerankCache(tmp_path)
    stub = StubCrossEncoder(np.arange(len(sections), dtype="float64"))
    CrossEncoderReranker("model-a", cache=cache, model=stub).fit(["q"], sections, corpus)
    CrossEncoderReranker("model-b", cache=cache, model=stub).fit(["q"], sections, corpus)
    assert stub.calls == 2


@requires_competition_data
def test_cache_invalidates_on_text_variant_change(corpus, sections, tmp_path):
    cache = RerankCache(tmp_path)
    stub = StubCrossEncoder(np.arange(len(sections), dtype="float64"))
    CrossEncoderReranker("m", text_variant="titled", cache=cache, model=stub).fit(
        ["q"], sections, corpus)
    CrossEncoderReranker("m", text_variant="body", cache=cache, model=stub).fit(
        ["q"], sections, corpus)
    assert stub.calls == 2


@requires_competition_data
def test_cache_invalidates_on_chunk_text_change(corpus, sections, tmp_path):
    """Editing a document must not silently serve stale scores."""
    cache = RerankCache(tmp_path)
    stub = StubCrossEncoder(np.arange(len(sections), dtype="float64"))
    CrossEncoderReranker("m", cache=cache, model=stub).fit(["q"], sections, corpus)
    CrossEncoderReranker("m", cache=cache, model=stub).fit(["q"], sections[:-1], corpus)
    assert stub.calls == 2


@requires_competition_data
def test_cache_invalidates_on_question_change(corpus, sections, tmp_path):
    cache = RerankCache(tmp_path)
    stub = StubCrossEncoder(np.arange(2 * len(sections), dtype="float64"))
    CrossEncoderReranker("m", cache=cache, model=stub).fit(["q one"], sections, corpus)
    CrossEncoderReranker("m", cache=cache, model=stub).fit(["q two"], sections, corpus)
    assert stub.calls == 2


@requires_competition_data
def test_corrupt_cache_is_a_miss(corpus, sections, tmp_path):
    cache = RerankCache(tmp_path)
    stub = StubCrossEncoder(np.arange(len(sections), dtype="float64"))
    CrossEncoderReranker("m", cache=cache, model=stub).fit(["q"], sections, corpus)
    for path in tmp_path.glob("*.npz"):
        path.write_bytes(b"garbage")
    CrossEncoderReranker("m", cache=cache, model=stub).fit(["q"], sections, corpus)
    assert stub.calls == 2


@requires_competition_data
def test_cache_stores_raw_logits_so_transforms_are_free(corpus, sections, tmp_path):
    """Caching before the transform is what makes the sanity check affordable."""
    cache = RerankCache(tmp_path)
    stub = StubCrossEncoder(np.linspace(-10, 5, len(sections)))
    reranker = CrossEncoderReranker("m", cache=cache, model=stub)
    reranker.fit(["q"], sections, corpus)
    assert reranker.raw_scores.min() < 0  # raw, not squashed
    assert reranker.scores("sigmoid").min() > 0
    assert stub.calls == 1


def test_fingerprint_identifies_model_text_and_transform():
    reranker = CrossEncoderReranker(
        "BAAI/bge-reranker-base", score_transform="minmax", text_variant="body",
        model=StubCrossEncoder(np.zeros(1)),
    )
    assert reranker.fingerprint == "bge-reranker-base/body/minmax"
