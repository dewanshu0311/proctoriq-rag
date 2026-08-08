"""Chunks in, citations out. The module with the point consequences.

Retrieval returns ranked chunks; a submission needs a set of ``(doc, section)``
citations. The mapping between them is a decision, it is fully measurable, and it
is worth more points than it looks.

**The tradeoff.** In the answer key, 36 of 50 questions have exactly one citation
and 14 have two. So:

- always cite 1 -> a guaranteed recall miss on 14 questions
- always cite 2 -> a guaranteed precision hit on 36

Neither fixed rule can serve both tails, which is why this phase reports the whole
curve and picks nothing. The correct answer is provably per-question, and that
belongs to the router in a later phase — locking a threshold now on aggregate F1
would bake in the average and lose both ends of the distribution.

**Two composable stages.**

*Stage A* — ``SectionAggregator`` turns ranked chunks into ranked sections.
*Stage B* — a ``CitationStrategy`` turns ranked sections into a cited set.

Splitting them matters because they answer different questions. Stage A decides
whether a section that placed three mediocre chunks beats one that placed a single
excellent chunk. Stage B decides how many sections to commit to. For
``SectionChunker`` the two aggregation modes are identical (one chunk per
section), which makes it a clean control; for ``SubsectionChunker`` they should
diverge sharply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Protocol, Sequence, runtime_checkable

import numpy as np

from proctoriq_rag.retrieval.retriever import ScoredChunk

AggregationMode = Literal["max", "sum"]


@dataclass(frozen=True)
class RankedSection:
    """One candidate citation with its aggregated score."""

    doc_id: str
    section_title: str
    score: float
    chunk_count: int

    @property
    def citation(self) -> tuple[str, str]:
        return (self.doc_id, self.section_title)


class SectionAggregator:
    """Stage A: group ranked chunks into ranked sections.

    ``mode="max"`` scores a section by its single best chunk — it rewards one
    precise match and is indifferent to how much of the section is irrelevant.
    ``mode="sum"`` adds the scores of every retrieved chunk belonging to the
    section — it rewards breadth, and systematically favours sections that were
    split into more pieces.

    That bias is not a bug but it is a real confound: under ``SubsectionChunker``,
    ``sum`` gives doc 01 Section 2 three chances to accumulate score while a
    single-chunk section gets one. Both modes are swept so the effect is visible
    rather than assumed.

    ``top_n`` bounds how deep into the chunk ranking to look. Chunks below it are
    ignored entirely, which keeps a long tail of weak matches from accumulating
    under ``sum``.
    """

    def __init__(self, top_n: int = 10, mode: AggregationMode = "max") -> None:
        if mode not in ("max", "sum"):
            raise ValueError(f"mode must be 'max' or 'sum', got {mode!r}")
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")
        self.top_n = top_n
        self.mode = mode

    @property
    def fingerprint(self) -> str:
        return f"{self.mode}@{self.top_n}"

    def aggregate(self, scored_chunks: Sequence[ScoredChunk]) -> list[RankedSection]:
        """Rank sections, descending by score. Ties broken by first appearance."""
        totals: dict[tuple[str, str], float] = {}
        counts: dict[tuple[str, str], int] = {}
        order: list[tuple[str, str]] = []

        for scored in list(scored_chunks)[: self.top_n]:
            citation = scored.citation
            if citation not in totals:
                totals[citation] = scored.score
                counts[citation] = 1
                order.append(citation)
                continue
            counts[citation] += 1
            totals[citation] = (
                max(totals[citation], scored.score)
                if self.mode == "max"
                else totals[citation] + scored.score
            )

        sections = [
            RankedSection(
                doc_id=citation[0],
                section_title=citation[1],
                score=totals[citation],
                chunk_count=counts[citation],
            )
            for citation in order
        ]
        # Stable sort preserves retrieval order among exact ties.
        sections.sort(key=lambda s: -s.score)
        return sections


@runtime_checkable
class CitationStrategy(Protocol):
    """Stage B: ranked sections -> the sections we actually cite."""

    @property
    def fingerprint(self) -> str: ...

    def select(self, sections: Sequence[RankedSection]) -> list[RankedSection]: ...


def _cap(sections: Iterable[RankedSection], limit: int) -> list[RankedSection]:
    return list(sections)[:limit]


class TopK:
    """Fixed cardinality. The baseline the adaptive strategies must beat.

    ``k=1`` and ``k=2`` bracket the problem: one is right for 36 questions, two is
    right for 14. Both are reported so the cost of each fixed choice is explicit.
    """

    def __init__(self, k: int = 1) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k

    @property
    def fingerprint(self) -> str:
        return f"topk-{self.k}"

    def select(self, sections: Sequence[RankedSection]) -> list[RankedSection]:
        return _cap(sections, self.k)


class ScoreThreshold:
    """Cite every section at or above ``threshold`` of the per-query score range.

    Section scores are min-max normalized within the query before comparison, so
    one threshold means the same thing across dense, sparse and hybrid — whose raw
    scales differ by orders of magnitude. A consequence of min-max: the top-ranked
    section always normalizes to 1.0, so at least one citation is always produced.

    This is genuinely different from :class:`RelativeGap`. Min-max asks "where does
    this section sit between the best and worst candidate?", which is sensitive to
    how bad the tail is. Ratio-to-top asks only about the leader.
    """

    def __init__(self, threshold: float, max_citations: int = 3) -> None:
        self.threshold = threshold
        self.max_citations = max_citations

    @property
    def fingerprint(self) -> str:
        return f"thresh-{self.threshold:g}"

    def select(self, sections: Sequence[RankedSection]) -> list[RankedSection]:
        if not sections:
            return []
        if len(sections) == 1:
            return list(sections)

        scores = np.array([s.score for s in sections], dtype="float64")
        low, high = scores.min(), scores.max()
        normalized = (
            np.ones_like(scores) if high - low < 1e-12 else (scores - low) / (high - low)
        )
        kept = [
            section
            for section, value in zip(sections, normalized)
            if value >= self.threshold
        ]
        return _cap(kept or [sections[0]], self.max_citations)


class RelativeGap:
    """Cite section *i* only if ``score_i >= ratio * score_1``.

    Scale-free by construction: it compares candidates to the leader, never to an
    absolute value, so it transfers across retriever modes without retuning. The
    first section is always cited.

    This is the strategy most likely to behave sensibly on the multi-source
    questions, because a genuine two-source question should produce two sections
    with close scores while a single-source question should produce one clear
    leader — which is exactly the signal a ratio measures.
    """

    def __init__(self, ratio: float, max_citations: int = 3) -> None:
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        self.ratio = ratio
        self.max_citations = max_citations

    @property
    def fingerprint(self) -> str:
        return f"gap-{self.ratio:g}"

    def select(self, sections: Sequence[RankedSection]) -> list[RankedSection]:
        if not sections:
            return []
        top = sections[0].score
        if top <= 0:
            return [sections[0]]
        kept = [s for s in sections if s.score >= self.ratio * top]
        return _cap(kept or [sections[0]], self.max_citations)


class CitationPolicy:
    """Stage A + Stage B, the full chunks-to-citations mapping."""

    def __init__(
        self, aggregator: SectionAggregator, strategy: CitationStrategy
    ) -> None:
        self.aggregator = aggregator
        self.strategy = strategy

    @property
    def fingerprint(self) -> str:
        return f"{self.aggregator.fingerprint}|{self.strategy.fingerprint}"

    def cite(
        self, scored_chunks: Sequence[ScoredChunk]
    ) -> tuple[list[str], list[str]]:
        """Return positionally aligned ``(cited_docs, cited_sections)``."""
        selected = self.strategy.select(self.aggregator.aggregate(scored_chunks))
        return (
            [s.doc_id for s in selected],
            [s.section_title for s in selected],
        )


def build_strategies(
    thresholds: Sequence[float] = (0.55, 0.65, 0.75, 0.80, 0.85, 0.90),
    ratios: Sequence[float] = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97),
    max_citations: int = 3,
) -> list[CitationStrategy]:
    """The full cardinality sweep.

    ``max_citations`` defaults to 3 while the key's maximum is 2, so "does a third
    citation ever pay?" becomes a measured answer rather than an assumption.
    """
    strategies: list[CitationStrategy] = [TopK(1), TopK(2)]
    strategies += [ScoreThreshold(t, max_citations) for t in thresholds]
    strategies += [RelativeGap(r, max_citations) for r in ratios]
    return strategies


def section_ranks(
    sections: Sequence[RankedSection], expected: Iterable[tuple[str, str]]
) -> list[int]:
    """1-based rank of each expected citation, or ``len+1`` when absent.

    This is the number that separates a Phase 1 problem from a Phase 2 problem: a
    section that never appears was never retrieved, while a section at rank 4 was
    retrieved and then dropped by the cardinality rule — which a reranker can fix.
    """
    positions = {s.citation: index for index, s in enumerate(sections, start=1)}
    return [positions.get(citation, len(sections) + 1) for citation in expected]
