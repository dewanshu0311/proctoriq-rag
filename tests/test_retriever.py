"""Retriever tests. Stub vectors for dense — no model download."""

from __future__ import annotations

import numpy as np
import pytest

from proctoriq_rag.retrieval.chunking import Chunk
from proctoriq_rag.retrieval.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    minmax,
    tokenize,
)

CHUNKS = [
    Chunk("c0", "01_windows", "Section 2: Common Installation Errors",
          "Session Start Error occurs when PSB cannot reach the assessment server", 0),
    Chunk("c1", "01_windows", "Section 2: Common Installation Errors",
          "Element not found means a kiosk mode driver failed to register", 1),
    Chunk("c2", "09_status", "Section 2: Possible Statuses",
          "Under Verification means the support team is cross referencing logs", 2),
    Chunk("c3", "07_policy", "Section 1: Prohibited Actions",
          "Swiping with three or four fingers on a touchpad is not permitted", 3),
]

QUERIES = [
    "PSB shows Session Start Error on my laptop",   # lexical: c0
    "my status says Under Verification",            # lexical: c2
]


def orthogonal_vectors(n: int, dim: int = 4) -> np.ndarray:
    """One-hot rows, so cosine similarity is exactly 1 on a match and 0 elsewhere."""
    matrix = np.zeros((n, dim), dtype="float32")
    for i in range(n):
        matrix[i, i % dim] = 1.0
    return matrix


@pytest.fixture
def dense() -> DenseRetriever:
    chunk_vectors = orthogonal_vectors(len(CHUNKS))
    # Query 0 points at chunk 2, query 1 points at chunk 3.
    query_vectors = np.array(
        [[0, 0, 1, 0], [0, 0, 0, 1]], dtype="float32"
    )
    return DenseRetriever(CHUNKS, chunk_vectors, query_vectors)


@pytest.fixture
def sparse() -> BM25Retriever:
    return BM25Retriever(CHUNKS, QUERIES)


# ── tokenize / minmax ──────────────────────────────────────────────────────
def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Session Start Error!") == ["session", "start", "error"]


def test_tokenize_keeps_digits():
    assert tokenize("Section 2: v1.5") == ["section", "2", "v1", "5"]


def test_minmax_maps_to_unit_range():
    result = minmax(np.array([2.0, 4.0, 6.0]))
    np.testing.assert_allclose(result, [0.0, 0.5, 1.0])


def test_minmax_on_a_flat_vector_is_all_zeros():
    """No basis to discriminate — must not divide by zero."""
    np.testing.assert_allclose(minmax(np.array([3.0, 3.0, 3.0])), [0.0, 0.0, 0.0])


def test_minmax_handles_negative_scores():
    np.testing.assert_allclose(minmax(np.array([-1.0, 0.0, 1.0])), [0.0, 0.5, 1.0])


# ── dense ──────────────────────────────────────────────────────────────────
def test_dense_ranks_the_matching_chunk_first(dense):
    assert dense.retrieve(0, k=1)[0].chunk.chunk_id == "c2"
    assert dense.retrieve(1, k=1)[0].chunk.chunk_id == "c3"


def test_dense_scores_are_cosine(dense):
    top = dense.retrieve(0, k=1)[0]
    assert top.score == pytest.approx(1.0, abs=1e-5)


def test_dense_returns_every_chunk_when_k_is_large(dense):
    assert len(dense.retrieve(0, k=99)) == len(CHUNKS)


def test_dense_scores_vector_covers_all_chunks(dense):
    assert dense.scores(0).shape == (len(CHUNKS),)


def test_dense_rank_is_descending(dense):
    scores = dense.scores(0)
    ordered = scores[dense.rank(0)]
    assert list(ordered) == sorted(ordered, reverse=True)


def test_retrieved_chunks_carry_their_citation(dense):
    assert dense.retrieve(0, k=1)[0].citation == (
        "09_status", "Section 2: Possible Statuses"
    )


# ── sparse ─────────────────────────────────────────────────────────────────
def test_bm25_finds_the_exact_error_string(sparse):
    """The case BM25 was added for: 'Session Start Error' appears verbatim."""
    assert sparse.retrieve(0, k=1)[0].chunk.chunk_id == "c0"


def test_bm25_finds_the_exact_status_string(sparse):
    assert sparse.retrieve(1, k=1)[0].chunk.chunk_id == "c2"


def test_bm25_is_independent_of_any_embedding_model(sparse):
    """No vectors were supplied — that is the point of reporting it once."""
    other = BM25Retriever(CHUNKS, QUERIES)
    np.testing.assert_allclose(sparse.scores(0), other.scores(0))


# ── hybrid ─────────────────────────────────────────────────────────────────
def test_hybrid_alpha_1_reduces_to_dense(dense, sparse):
    hybrid = HybridRetriever(dense, sparse, alpha=1.0)
    np.testing.assert_allclose(hybrid.scores(0), minmax(dense.scores(0)))
    assert [c.chunk.chunk_id for c in hybrid.retrieve(0, k=4)] == [
        c.chunk.chunk_id for c in dense.retrieve(0, k=4)
    ]


def test_hybrid_alpha_0_reduces_to_sparse(dense, sparse):
    hybrid = HybridRetriever(dense, sparse, alpha=0.0)
    np.testing.assert_allclose(hybrid.scores(0), minmax(sparse.scores(0)))
    assert hybrid.retrieve(0, k=1)[0].chunk.chunk_id == "c0"


def test_hybrid_blends_both_signals(dense, sparse):
    """Dense says c2, sparse says c0; a 50/50 blend must reflect both."""
    hybrid = HybridRetriever(dense, sparse, alpha=0.5)
    scores = hybrid.scores(0)
    ids = [c.chunk_id for c in CHUNKS]
    assert scores[ids.index("c0")] > 0
    assert scores[ids.index("c2")] > 0
    expected = 0.5 * minmax(dense.scores(0)) + 0.5 * minmax(sparse.scores(0))
    np.testing.assert_allclose(scores, expected, rtol=1e-5)


def test_hybrid_scores_stay_in_unit_range(dense, sparse):
    for alpha in (0.0, 0.3, 0.5, 0.7, 1.0):
        scores = HybridRetriever(dense, sparse, alpha).scores(0)
        assert scores.min() >= -1e-6
        assert scores.max() <= 1.0 + 1e-6


def test_hybrid_rejects_alpha_out_of_range(dense, sparse):
    with pytest.raises(ValueError, match="alpha must be in"):
        HybridRetriever(dense, sparse, alpha=1.5)


def test_hybrid_rejects_mismatched_chunk_sets(dense):
    other = BM25Retriever(CHUNKS[:2], QUERIES)
    with pytest.raises(ValueError, match="share a chunk set"):
        HybridRetriever(dense, other, alpha=0.5)
