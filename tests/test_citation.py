"""Citation strategy tests.

Every case is constructed so the expected answer is checkable by hand. This is
the module with the point consequences, so the tests are arithmetic, not smoke.
"""

from __future__ import annotations

import pytest

from proctoriq_rag.retrieval.chunking import Chunk
from proctoriq_rag.retrieval.citation import (
    CitationPolicy,
    RankedSection,
    RelativeGap,
    ScoreThreshold,
    SectionAggregator,
    TopK,
    build_strategies,
    section_ranks,
)
from proctoriq_rag.retrieval.retriever import ScoredChunk


def sec(name: str, score: float, doc: str = "doc_a", count: int = 1) -> RankedSection:
    return RankedSection(doc_id=doc, section_title=name, score=score, chunk_count=count)


def chunk(doc: str, section: str, text: str = "t", index: int = 0) -> Chunk:
    return Chunk(f"{doc}::{section}::{index}", doc, section, text, index)


def scored(doc: str, section: str, score: float, index: int = 0) -> ScoredChunk:
    return ScoredChunk(chunk(doc, section, index=index), score)


# ── TopK ───────────────────────────────────────────────────────────────────
def test_topk_1_takes_the_leader():
    sections = [sec("A", 0.9), sec("B", 0.8), sec("C", 0.1)]
    assert [s.section_title for s in TopK(1).select(sections)] == ["A"]


def test_topk_2_takes_two_regardless_of_gap():
    """Fixed cardinality ignores the gap — that is exactly its weakness."""
    sections = [sec("A", 0.9), sec("B", 0.05)]
    assert [s.section_title for s in TopK(2).select(sections)] == ["A", "B"]


def test_topk_handles_fewer_sections_than_k():
    assert len(TopK(2).select([sec("A", 0.9)])) == 1


def test_topk_rejects_zero():
    with pytest.raises(ValueError):
        TopK(0)


def test_topk_on_empty_input():
    assert TopK(1).select([]) == []


# ── ScoreThreshold — min-max normalized within the query ───────────────────
def test_threshold_keeps_only_the_top_when_the_tail_is_far():
    """Scores 1.0 / 0.5 / 0.0 normalize to 1.0 / 0.5 / 0.0. t=0.9 keeps one."""
    sections = [sec("A", 1.0), sec("B", 0.5), sec("C", 0.0)]
    assert [s.section_title for s in ScoreThreshold(0.9).select(sections)] == ["A"]


def test_threshold_keeps_two_when_the_runner_up_is_close():
    """1.0 / 0.9 / 0.0 -> normalized 1.0 / 0.9 / 0.0. t=0.85 keeps two."""
    sections = [sec("A", 1.0), sec("B", 0.9), sec("C", 0.0)]
    assert [s.section_title for s in ScoreThreshold(0.85).select(sections)] == ["A", "B"]


def test_threshold_always_returns_at_least_one():
    """Min-max puts the leader at 1.0, so a citation is always produced."""
    sections = [sec("A", 0.11), sec("B", 0.10)]
    assert len(ScoreThreshold(1.0).select(sections)) >= 1


def test_threshold_on_an_exact_tie_keeps_both():
    """A flat score vector normalizes to all-ones — no basis to discriminate."""
    result = ScoreThreshold(0.9).select([sec("A", 0.7), sec("B", 0.7)])
    assert len(result) == 2


def test_threshold_respects_max_citations():
    sections = [sec(n, 1.0) for n in "ABCDE"]
    assert len(ScoreThreshold(0.5, max_citations=3).select(sections)) == 3


def test_threshold_single_section_passes_through():
    assert len(ScoreThreshold(0.99).select([sec("A", 0.4)])) == 1


# ── RelativeGap — ratio to the leader, scale free ──────────────────────────
def test_gap_keeps_second_when_within_ratio():
    """0.90 / 1.00 = 0.90, so r=0.90 keeps it (>= is inclusive)."""
    result = RelativeGap(0.90).select([sec("A", 1.0), sec("B", 0.90)])
    assert [s.section_title for s in result] == ["A", "B"]


def test_gap_drops_second_when_outside_ratio():
    """0.89 / 1.00 = 0.89 < 0.90."""
    result = RelativeGap(0.90).select([sec("A", 1.0), sec("B", 0.89)])
    assert [s.section_title for s in result] == ["A"]


def test_gap_is_scale_free():
    """Multiplying every score by 100 must not change the decision."""
    small = RelativeGap(0.9).select([sec("A", 0.010), sec("B", 0.0095)])
    large = RelativeGap(0.9).select([sec("A", 1.000), sec("B", 0.9500)])
    assert len(small) == len(large) == 2


def test_gap_on_exact_tie_keeps_both():
    assert len(RelativeGap(0.95).select([sec("A", 0.6), sec("B", 0.6)])) == 2


def test_gap_ratio_1_requires_an_exact_tie():
    assert len(RelativeGap(1.0).select([sec("A", 0.6), sec("B", 0.599)])) == 1
    assert len(RelativeGap(1.0).select([sec("A", 0.6), sec("B", 0.600)])) == 2


def test_gap_respects_max_citations():
    sections = [sec(n, 1.0) for n in "ABCDE"]
    assert len(RelativeGap(0.5, max_citations=2).select(sections)) == 2


def test_gap_with_nonpositive_top_returns_one():
    assert len(RelativeGap(0.9).select([sec("A", 0.0), sec("B", 0.0)])) == 1


def test_gap_rejects_out_of_range_ratio():
    with pytest.raises(ValueError):
        RelativeGap(0.0)
    with pytest.raises(ValueError):
        RelativeGap(1.5)


def test_gap_on_empty_input():
    assert RelativeGap(0.9).select([]) == []


# ── SectionAggregator: max vs sum ──────────────────────────────────────────
def test_aggregator_max_takes_the_best_chunk():
    chunks = [scored("d", "S1", 0.9), scored("d", "S1", 0.3, 1), scored("d", "S2", 0.8)]
    result = SectionAggregator(top_n=10, mode="max").aggregate(chunks)
    assert result[0].section_title == "S1"
    assert result[0].score == pytest.approx(0.9)
    assert result[0].chunk_count == 2


def test_aggregator_sum_adds_chunk_scores():
    chunks = [scored("d", "S1", 0.5), scored("d", "S1", 0.4, 1), scored("d", "S2", 0.8)]
    result = SectionAggregator(top_n=10, mode="sum").aggregate(chunks)
    assert result[0].section_title == "S1"
    assert result[0].score == pytest.approx(0.9)


def test_max_and_sum_disagree_on_a_constructed_case():
    """One strong chunk vs three mediocre ones — the modes must rank differently.

    S2 has the single best chunk (0.70). S1 has three chunks summing to 0.90.
    'max' should prefer S2; 'sum' should prefer S1. This is exactly the split
    SubsectionChunker creates in doc 01 Section 2.
    """
    chunks = [
        scored("d", "S2", 0.70, 0),
        scored("d", "S1", 0.35, 1),
        scored("d", "S1", 0.30, 2),
        scored("d", "S1", 0.25, 3),
    ]
    by_max = SectionAggregator(top_n=10, mode="max").aggregate(chunks)
    by_sum = SectionAggregator(top_n=10, mode="sum").aggregate(chunks)
    assert by_max[0].section_title == "S2"
    assert by_sum[0].section_title == "S1"


def test_aggregator_is_identical_for_one_chunk_per_section():
    """With SectionChunker the two modes must agree — the clean control."""
    chunks = [scored("d", "S1", 0.9), scored("d", "S2", 0.8), scored("d", "S3", 0.7)]
    by_max = SectionAggregator(top_n=10, mode="max").aggregate(chunks)
    by_sum = SectionAggregator(top_n=10, mode="sum").aggregate(chunks)
    assert [s.citation for s in by_max] == [s.citation for s in by_sum]
    assert [s.score for s in by_max] == [s.score for s in by_sum]


def test_aggregator_top_n_bounds_the_pool():
    chunks = [scored("d", f"S{i}", 1.0 - i / 100, i) for i in range(20)]
    assert len(SectionAggregator(top_n=5).aggregate(chunks)) == 5


def test_aggregator_distinguishes_same_section_name_in_different_docs():
    """'Section 3: Login Issues' exists in both install guides — never merge them."""
    chunks = [scored("doc_a", "Section 3: Login Issues", 0.9),
              scored("doc_b", "Section 3: Login Issues", 0.8)]
    result = SectionAggregator(top_n=10).aggregate(chunks)
    assert len(result) == 2
    assert {s.doc_id for s in result} == {"doc_a", "doc_b"}


def test_aggregator_rejects_bad_arguments():
    with pytest.raises(ValueError):
        SectionAggregator(mode="median")
    with pytest.raises(ValueError):
        SectionAggregator(top_n=0)


def test_aggregator_on_empty_input():
    assert SectionAggregator().aggregate([]) == []


# ── CitationPolicy end to end ──────────────────────────────────────────────
def test_policy_returns_aligned_docs_and_sections():
    chunks = [scored("doc_a", "S1", 1.0), scored("doc_b", "S2", 0.99)]
    policy = CitationPolicy(SectionAggregator(top_n=10), RelativeGap(0.9))
    docs, sections = policy.cite(chunks)
    assert len(docs) == len(sections) == 2
    assert (docs[0], sections[0]) == ("doc_a", "S1")


def test_policy_can_cite_one_document_twice():
    """The Q44 shape: same doc, two sections. Both must survive."""
    chunks = [scored("doc_a", "S1", 1.0), scored("doc_a", "S2", 1.0)]
    docs, sections = CitationPolicy(
        SectionAggregator(top_n=10), RelativeGap(0.95)
    ).cite(chunks)
    assert docs == ["doc_a", "doc_a"]
    assert sorted(sections) == ["S1", "S2"]


def test_policy_fingerprint_identifies_both_stages():
    policy = CitationPolicy(SectionAggregator(top_n=15, mode="sum"), TopK(2))
    assert policy.fingerprint == "sum@15|topk-2"


# ── section_ranks ──────────────────────────────────────────────────────────
def test_section_ranks_reports_1_based_positions():
    sections = [sec("A", 0.9), sec("B", 0.8), sec("C", 0.7)]
    assert section_ranks(sections, [("doc_a", "A")]) == [1]
    assert section_ranks(sections, [("doc_a", "C")]) == [3]


def test_section_ranks_marks_absent_sections_past_the_end():
    sections = [sec("A", 0.9), sec("B", 0.8)]
    assert section_ranks(sections, [("doc_a", "Z")]) == [3]


def test_section_ranks_handles_multi_source():
    sections = [sec("A", 0.9), sec("B", 0.8), sec("C", 0.7)]
    assert section_ranks(sections, [("doc_a", "A"), ("doc_a", "C")]) == [1, 3]


# ── the sweep grid ─────────────────────────────────────────────────────────
def test_build_strategies_covers_the_full_curve():
    strategies = build_strategies()
    fingerprints = [s.fingerprint for s in strategies]
    assert "topk-1" in fingerprints
    assert "topk-2" in fingerprints
    assert sum(f.startswith("thresh-") for f in fingerprints) == 6
    assert sum(f.startswith("gap-") for f in fingerprints) == 6
    assert len(set(fingerprints)) == len(fingerprints)


def test_default_max_citations_allows_three():
    """The key's maximum is 2, so a third citation is measurable, not assumed."""
    sections = [sec(n, 1.0) for n in "ABCD"]
    for strategy in build_strategies():
        assert len(strategy.select(sections)) <= 3
