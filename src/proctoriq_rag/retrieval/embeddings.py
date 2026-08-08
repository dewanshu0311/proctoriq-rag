"""Pluggable embedding backends with a disk cache.

Local sentence-transformers only this phase — no API calls, no keys. The cache
exists because the sweep re-embeds the same (model, chunker) pairs across
retriever modes and citation policies, and re-encoding on every run would make
iteration slow enough to discourage running the sweep at all.

Cache invalidation is content-addressed: the key includes a hash of the chunk
texts themselves, so editing a document or changing a chunker parameter produces
a different key automatically. There is no way to silently serve stale vectors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from proctoriq_rag.config import REPO_ROOT

DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "embeddings"

#: Models whose card specifies an asymmetric query prefix. Omitting it measurably
#: handicaps the model, so leaving it out would make the comparison unfair rather
#: than neutral — but it is a real confound, so it is declared here rather than
#: buried in the backend. See DECISIONS D-012.
QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Anything that turns text into L2-normalized vectors."""

    @property
    def name(self) -> str: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product equals cosine similarity."""
    matrix = np.asarray(matrix, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class SentenceTransformerBackend:
    """A local sentence-transformers model. Loaded lazily on first use."""

    def __init__(self, model_name: str, query_prefix: str | None = None) -> None:
        self.model_name = model_name
        self.query_prefix = (
            QUERY_PREFIXES.get(model_name, "") if query_prefix is None else query_prefix
        )
        self._model = None

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return _normalize(np.asarray(vectors))

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if self.query_prefix:
            texts = [f"{self.query_prefix}{t}" for t in texts]
        return self._encode(texts)


@dataclass(frozen=True)
class CacheKey:
    model_name: str
    fingerprint: str
    content_hash: str
    kind: str = "documents"

    def digest(self) -> str:
        raw = f"{self.kind}|{self.model_name}|{self.fingerprint}|{self.content_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def content_hash(texts: Sequence[str]) -> str:
    """Stable hash over the exact texts being embedded."""
    hasher = hashlib.sha256()
    for text in texts:
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:16]


class EmbeddingCache:
    """Disk cache of embedding matrices, keyed by model + config + content."""

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.hits = 0
        self.misses = 0

    def _path(self, key: CacheKey) -> Path:
        return self.cache_dir / f"{key.digest()}.npz"

    def get(self, key: CacheKey) -> np.ndarray | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                return data["vectors"]
        except (OSError, KeyError, ValueError):
            # A truncated or corrupt cache file is a miss, not a crash.
            return None

    def put(self, key: CacheKey, vectors: np.ndarray) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        np.savez_compressed(path, vectors=np.asarray(vectors, dtype="float32"))
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "model": key.model_name,
                    "chunker": key.fingerprint,
                    "kind": key.kind,
                    "content_hash": key.content_hash,
                    "shape": list(np.shape(vectors)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def encode(
        self,
        backend: EmbeddingBackend,
        texts: Sequence[str],
        fingerprint: str,
        kind: str = "documents",
    ) -> np.ndarray:
        """Return cached vectors, encoding and storing them on a miss."""
        key = CacheKey(
            model_name=backend.name,
            fingerprint=fingerprint,
            content_hash=content_hash(texts),
            kind=kind,
        )
        cached = self.get(key)
        if cached is not None and cached.shape[0] == len(texts):
            self.hits += 1
            return cached

        self.misses += 1
        vectors = (
            backend.encode_queries(texts)
            if kind == "queries"
            else backend.encode_documents(texts)
        )
        self.put(key, vectors)
        return vectors

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


#: The sweep grid. Any embedding model is permitted by the competition rules.
SWEEP_MODELS: tuple[str, ...] = (
    "sentence-transformers/all-MiniLM-L6-v2",  # Phase 0 diagnostic baseline
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
)


def short_model_name(model_name: str) -> str:
    """Trailing path component, for readable leaderboard rows."""
    return model_name.rsplit("/", 1)[-1]
