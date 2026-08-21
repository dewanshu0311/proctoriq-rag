"""RAG Fusion — multi-query retrieval merged by reciprocal rank fusion.

The idea: one phrasing of a question retrieves one neighbourhood. Ask an LLM to
rephrase it several ways, retrieve for each, and merge the ranked lists. A
section that ranks well across several phrasings is more likely to be the right
one than a section that wins under exactly one.

Reciprocal rank fusion is the standard merge:

    score(section) = sum over queries of  1 / (k + rank)

with ``k=60`` by convention. It uses only *ranks*, never raw scores, which is
what makes it safe to merge lists from retrievers on different scales.

The prediction on record, before measuring
------------------------------------------
Same as HyDE: on this corpus recall@10 is already 93.75%, so there is almost no
headroom for a technique whose entire job is to find sections a single query
missed. Query rephrasing helps most when the corpus is large and the vocabulary
mismatch is severe; here there are 53 sections and a cross-encoder already scores
every one of them exhaustively.

Reported honestly either way, including the ``policy_boundary`` subset as its own
row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from proctoriq_rag.retrieval.chunking import Chunk
from proctoriq_rag.retrieval.retriever import ScoredChunk

RRF_K = 60

FUSION_PROMPT = """\
Rewrite this student support question in {n} different ways. Each rewrite must ask for
the same information but use different words — a student describing the same problem
differently, or the phrasing a support document would use for it.

Question: {question}

Output exactly {n} lines, one rewrite per line, nothing else. No numbering, no bullets,
no commentary."""


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]], k: int = RRF_K, n_items: int | None = None
) -> np.ndarray:
    """Merge ranked index lists into a fused score per item.

    Rank-based by construction, so lists from retrievers with incomparable score
    scales merge safely — which is the whole reason RRF is the standard choice.
    """
    size = n_items if n_items is not None else (
        max((max(l) for l in ranked_lists if len(l)), default=-1) + 1
    )
    fused = np.zeros(size, dtype="float64")
    for ranked in ranked_lists:
        for rank, index in enumerate(ranked, start=1):
            if 0 <= index < size:
                fused[index] += 1.0 / (k + rank)
    return fused


@dataclass
class RAGFusion:
    """Generate query variants, retrieve for each, merge by RRF.

    ``score_fn`` maps a query string to a per-chunk score array, so this works
    over dense, sparse or hybrid retrieval without knowing which.
    """

    client: object
    score_fn: Callable[[str], np.ndarray]
    chunks: Sequence[Chunk]
    n_queries: int = 3
    k: int = RRF_K
    variants: dict[str, list[str]] = field(default_factory=dict)
    fallbacks: int = 0

    def generate_queries(self, question: str) -> list[str]:
        """The original question plus ``n_queries`` rewrites. Cached."""
        if question in self.variants:
            return self.variants[question]
        try:
            raw = self.client.complete(
                FUSION_PROMPT.format(question=question, n=self.n_queries)
            )
        except Exception:  # noqa: BLE001 - degrade to the original question alone
            raw = ""

        lines = [
            line.strip().lstrip("0123456789.-) ").strip()
            for line in (raw or "").splitlines()
            if line.strip()
        ]
        lines = [l for l in lines if len(l) > 10][: self.n_queries]
        if not lines:
            self.fallbacks += 1

        # The original always participates: a rewrite can drift, and fusion
        # should never be able to score worse than the query it started from.
        queries = [question] + lines
        self.variants[question] = queries
        return queries

    def scores(self, question: str) -> np.ndarray:
        ranked_lists = [
            list(np.argsort(-self.score_fn(query)))
            for query in self.generate_queries(question)
        ]
        return reciprocal_rank_fusion(ranked_lists, self.k, n_items=len(self.chunks))

    def retrieve(self, question: str, k: int = 10) -> list[ScoredChunk]:
        fused = self.scores(question)
        order = np.argsort(-fused)[:k]
        return [ScoredChunk(self.chunks[i], float(fused[i])) for i in order]
