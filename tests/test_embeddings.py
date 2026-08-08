"""Embedding backend and cache tests. Stub backend — no model download."""

from __future__ import annotations

import numpy as np
import pytest

from proctoriq_rag.retrieval.embeddings import (
    QUERY_PREFIXES,
    SWEEP_MODELS,
    CacheKey,
    EmbeddingCache,
    SentenceTransformerBackend,
    content_hash,
    short_model_name,
)


class CountingBackend:
    """Deterministic stub that records how many times it actually encoded."""

    def __init__(self, name: str = "stub-model", dim: int = 8) -> None:
        self._name = name
        self.dim = dim
        self.document_calls = 0
        self.query_calls = 0

    @property
    def name(self) -> str:
        return self._name

    def _vectors(self, texts):
        rows = []
        for text in texts:
            vector = np.zeros(self.dim, dtype="float32")
            for i, char in enumerate(text[: self.dim]):
                vector[i] = (ord(char) % 13) + 1
            rows.append(vector)
        matrix = np.vstack(rows)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def encode_documents(self, texts):
        self.document_calls += 1
        return self._vectors(texts)

    def encode_queries(self, texts):
        self.query_calls += 1
        return self._vectors(texts)


@pytest.fixture
def cache(tmp_path):
    return EmbeddingCache(tmp_path / "embeddings")


TEXTS = ["alpha section body", "beta section body", "gamma section body"]


# ── miss then hit ──────────────────────────────────────────────────────────
def test_first_call_is_a_miss_and_encodes(cache):
    backend = CountingBackend()
    vectors = cache.encode(backend, TEXTS, fingerprint="section-hdr")
    assert vectors.shape == (3, 8)
    assert backend.document_calls == 1
    assert cache.stats == {"hits": 1 - 1, "misses": 1}


def test_second_call_is_a_hit_and_does_not_encode(cache):
    backend = CountingBackend()
    first = cache.encode(backend, TEXTS, fingerprint="section-hdr")
    second = cache.encode(backend, TEXTS, fingerprint="section-hdr")
    assert backend.document_calls == 1  # not re-encoded
    assert cache.stats == {"hits": 1, "misses": 1}
    np.testing.assert_allclose(first, second)


def test_hit_survives_a_new_cache_object(tmp_path):
    """The cache is on disk, so a fresh process still hits."""
    backend = CountingBackend()
    EmbeddingCache(tmp_path / "e").encode(backend, TEXTS, fingerprint="fp")
    EmbeddingCache(tmp_path / "e").encode(backend, TEXTS, fingerprint="fp")
    assert backend.document_calls == 1


# ── invalidation ───────────────────────────────────────────────────────────
def test_changing_the_model_invalidates(cache):
    a, b = CountingBackend("model-a"), CountingBackend("model-b")
    cache.encode(a, TEXTS, fingerprint="fp")
    cache.encode(b, TEXTS, fingerprint="fp")
    assert a.document_calls == 1 and b.document_calls == 1
    assert cache.stats["misses"] == 2


def test_changing_the_chunker_fingerprint_invalidates(cache):
    backend = CountingBackend()
    cache.encode(backend, TEXTS, fingerprint="section-hdr")
    cache.encode(backend, TEXTS, fingerprint="recursive-256-32-hdr")
    assert backend.document_calls == 2


def test_changing_the_content_invalidates(cache):
    """Editing a document must not silently serve stale vectors."""
    backend = CountingBackend()
    cache.encode(backend, TEXTS, fingerprint="fp")
    cache.encode(backend, TEXTS[:2] + ["edited body"], fingerprint="fp")
    assert backend.document_calls == 2


def test_documents_and_queries_do_not_collide(cache):
    """Same texts, different kind — bge prefixes queries, so they must differ."""
    backend = CountingBackend()
    cache.encode(backend, TEXTS, fingerprint="fp", kind="documents")
    cache.encode(backend, TEXTS, fingerprint="fp", kind="queries")
    assert backend.document_calls == 1
    assert backend.query_calls == 1


def test_corrupt_cache_file_is_a_miss_not_a_crash(cache):
    backend = CountingBackend()
    cache.encode(backend, TEXTS, fingerprint="fp")
    for path in cache.cache_dir.glob("*.npz"):
        path.write_bytes(b"not an npz")
    cache.encode(backend, TEXTS, fingerprint="fp")
    assert backend.document_calls == 2


def test_length_mismatch_forces_a_re_encode(cache):
    key = CacheKey("stub-model", "fp", content_hash(TEXTS))
    cache.put(key, np.zeros((99, 8), dtype="float32"))
    backend = CountingBackend()
    vectors = cache.encode(backend, TEXTS, fingerprint="fp")
    assert vectors.shape[0] == 3


# ── keys and hashing ───────────────────────────────────────────────────────
def test_content_hash_is_order_sensitive():
    assert content_hash(["a", "b"]) != content_hash(["b", "a"])


def test_content_hash_resists_concatenation_collisions():
    """['ab'] and ['a','b'] must not hash alike — the null separator prevents it."""
    assert content_hash(["ab"]) != content_hash(["a", "b"])


def test_cache_key_digest_is_stable_and_distinct():
    a = CacheKey("m", "fp", "hash")
    b = CacheKey("m", "fp", "hash")
    c = CacheKey("m", "fp", "other")
    assert a.digest() == b.digest()
    assert a.digest() != c.digest()


def test_put_writes_a_readable_sidecar(cache):
    import json

    cache.encode(CountingBackend(), TEXTS, fingerprint="section-hdr")
    sidecar = next(cache.cache_dir.glob("*.json"))
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["chunker"] == "section-hdr"
    assert meta["shape"] == [3, 8]


# ── backend wiring ─────────────────────────────────────────────────────────
def test_bge_gets_its_documented_query_prefix():
    """Omitting it measurably handicaps bge, which would make the sweep unfair."""
    backend = SentenceTransformerBackend("BAAI/bge-small-en-v1.5")
    assert backend.query_prefix == QUERY_PREFIXES["BAAI/bge-small-en-v1.5"]


def test_other_models_get_no_prefix():
    assert SentenceTransformerBackend(
        "sentence-transformers/all-MiniLM-L6-v2"
    ).query_prefix == ""


def test_query_prefix_can_be_overridden():
    backend = SentenceTransformerBackend("BAAI/bge-small-en-v1.5", query_prefix="")
    assert backend.query_prefix == ""


def test_model_is_not_loaded_on_construction():
    """Constructing a backend must not download or load anything."""
    backend = SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2")
    assert backend._model is None


def test_sweep_covers_at_least_two_models():
    assert len(SWEEP_MODELS) >= 2
    assert len(set(SWEEP_MODELS)) == len(SWEEP_MODELS)


def test_short_model_name_strips_the_org():
    assert short_model_name("BAAI/bge-small-en-v1.5") == "bge-small-en-v1.5"
    assert short_model_name("bare-name") == "bare-name"
