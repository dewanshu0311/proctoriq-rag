"""Dense, sparse and hybrid retrieval over chunks.

Dense is FAISS ``IndexFlatIP`` over L2-normalized vectors, so inner product is
cosine similarity. Exact search — with at most a few hundred chunks there is
nothing to approximate.

Sparse is BM25. It earns its place on specific questions: several turn on exact
strings the corpus uses verbatim — *Session Start Error*, *Element not found*,
*Under Verification*, *Resolved — Flag Upheld*. Dense retrieval is weakest exactly
where the match should be trivial, because an embedding of a short error name
carries little more signal than an embedding of any other short error name.

Hybrid blends the two. Because BM25 scores are unbounded and cosine is not, the
two score lists are min-max normalized per query before blending — otherwise the
weight would mean something different for every query.

Every retriever exposes both ``scores`` (the full score vector, which the sweep
uses) and ``retrieve`` (ranked chunks, which a pipeline uses).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from proctoriq_rag.retrieval.chunking import Chunk

TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float

    @property
    def citation(self) -> tuple[str, str]:
        return self.chunk.citation


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization, shared by BM25 indexing and querying."""
    return TOKEN_RE.findall(text.lower())


def minmax(scores: np.ndarray) -> np.ndarray:
    """Scale a score vector to [0, 1]. A flat vector maps to all zeros.

    Applied per query. Without it, a single weight cannot mean the same thing
    across queries, because BM25's scale varies with query length and term rarity.
    """
    scores = np.asarray(scores, dtype="float32")
    low, high = float(scores.min()), float(scores.max())
    if high - low < 1e-12:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


class BaseRetriever:
    """Shared ranking behaviour. Subclasses implement ``scores``."""

    chunks: Sequence[Chunk]

    def scores(self, query_index: int) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def rank(self, query_index: int) -> np.ndarray:
        """Chunk indices ordered by descending score."""
        return np.argsort(-self.scores(query_index), kind="stable")

    def retrieve(self, query_index: int, k: int = 10) -> list[ScoredChunk]:
        scores = self.scores(query_index)
        order = np.argsort(-scores, kind="stable")[:k]
        return [ScoredChunk(self.chunks[i], float(scores[i])) for i in order]


class DenseRetriever(BaseRetriever):
    """FAISS ``IndexFlatIP`` over normalized chunk vectors.

    Query vectors are supplied precomputed: the sweep encodes all 50 questions
    once per model, and re-encoding per configuration would dominate runtime.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        chunk_vectors: np.ndarray,
        query_vectors: np.ndarray,
    ) -> None:
        import faiss

        self.chunks = list(chunks)
        self.chunk_vectors = np.asarray(chunk_vectors, dtype="float32")
        self.query_vectors = np.asarray(query_vectors, dtype="float32")

        self.index = faiss.IndexFlatIP(self.chunk_vectors.shape[1])
        self.index.add(self.chunk_vectors)

        # One search for the whole query set: k = every chunk, so the sweep can
        # compute recall at any depth without re-searching.
        similarity, indices = self.index.search(
            self.query_vectors, len(self.chunks)
        )
        self._scores = np.zeros(
            (len(self.query_vectors), len(self.chunks)), dtype="float32"
        )
        for row in range(len(self.query_vectors)):
            self._scores[row, indices[row]] = similarity[row]

    def scores(self, query_index: int) -> np.ndarray:
        return self._scores[query_index]


class BM25Retriever(BaseRetriever):
    """Lexical retrieval. Independent of the embedding model by construction."""

    def __init__(self, chunks: Sequence[Chunk], queries: Sequence[str]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        self._scores = np.vstack(
            [
                np.asarray(self.bm25.get_scores(tokenize(q)), dtype="float32")
                for q in queries
            ]
        )

    def scores(self, query_index: int) -> np.ndarray:
        return self._scores[query_index]


class HybridRetriever(BaseRetriever):
    """``alpha * dense + (1 - alpha) * sparse`` over per-query normalized scores.

    ``alpha=1.0`` reduces exactly to dense ranking and ``alpha=0.0`` exactly to
    sparse — asserted in the tests, because a hybrid that quietly fails to
    degenerate would make every blended result uninterpretable.
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        alpha: float = 0.5,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if len(dense.chunks) != len(sparse.chunks):
            raise ValueError("dense and sparse retrievers must share a chunk set")
        self.chunks = dense.chunks
        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha

    def scores(self, query_index: int) -> np.ndarray:
        dense = minmax(self.dense.scores(query_index))
        sparse = minmax(self.sparse.scores(query_index))
        return self.alpha * dense + (1.0 - self.alpha) * sparse
