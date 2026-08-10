"""Cross-encoder reranking.

Phase 1 established the mandate: 35 of 39 citation failures had every expected
section already inside the top 10. Retrieval finds the right sections and orders
them badly — recall@10 is 0.9375 while recall@1 is 0.42–0.52. A cross-encoder
scores the question and the passage jointly rather than embedding them
independently, which is what closes that gap.

The corpus is 53 sections, so scoring every question against every section is
2,650 pairs — cheap enough to do exhaustively and cache. When that is affordable
there is no candidate pool to fall out of and the recall@10 ceiling stops
existing.

On reading cross-encoder scores
-------------------------------
These models emit raw logits, not probabilities. Measured on this corpus with
``ms-marco-MiniLM-L-6-v2``, one question's 53 scores ranged from **-11.4 to
+6.4, with 49 of 53 negative**.

That matters because :class:`~proctoriq_rag.retrieval.citation.RelativeGap`
compares ``score_i / score_1``. Ratios of negative numbers are meaningless — and
worse, ``RelativeGap`` short-circuits to a single citation when the top score is
non-positive, so on raw logits the entire cardinality sweep would have quietly
degenerated into ``topk-1`` while producing a plausible-looking curve.

Both model families here are trained as binary relevance classifiers with BCE
loss, so a sigmoid is the principled way to read their output, not a patch. It is
also **strictly monotonic**, so it cannot change the ranking — only the spacing.
``score_transform`` makes the choice explicit and testable, and
:func:`saturation_report` exists because a sigmoid over wide logits saturates,
which can flatten ratio-based rules for a completely different reason.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np

from proctoriq_rag.config import REPO_ROOT
from proctoriq_rag.corpus.loader import Corpus
from proctoriq_rag.retrieval.chunking import Chunk
from proctoriq_rag.retrieval.citation import RankedSection

DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "rerank"

ScoreTransform = Literal["auto", "sigmoid", "minmax", "raw"]
TextVariant = Literal["body", "titled"]

#: Cross-encoders worth testing on CPU, with throughput measured on this machine.
#: Note the differing output conventions — see :func:`apply_transform`.
RERANKER_MODELS: tuple[str, ...] = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2",   # ~19.1 pairs/s, logits -11.4..+6.4
    "BAAI/bge-reranker-base",                 # ~3.9 pairs/s, logits
    "mixedbread-ai/mxbai-rerank-base-v1",     # ~2.9 pairs/s, ALREADY [0,1]
)


#: Directories searched for a pre-downloaded copy of the reranker, before falling
#: back to the HuggingFace hub. Kaggle competitions sometimes run notebooks with
#: internet disabled, in which case the model has to arrive as an attached
#: Dataset instead — this is what makes that possible without a code change.
#: Override with PROCTORIQ_MODEL_DIR.
LOCAL_MODEL_ROOTS: tuple[str, ...] = (
    "/kaggle/input",
    "/kaggle/working/models",
)


def resolve_model_source(model_name: str) -> str:
    """Return a local directory holding ``model_name`` if one exists, else the name.

    Looks for a directory whose name matches the model's final path component,
    e.g. ``ms-marco-MiniLM-L-6-v2``. Falls through to the hub name unchanged when
    nothing local is found, so behaviour is identical when internet is available.
    """
    override = os.environ.get("PROCTORIQ_MODEL_DIR", "").strip()
    short = model_name.rsplit("/", 1)[-1]

    candidate_roots = [Path(override)] if override else [Path(r) for r in LOCAL_MODEL_ROOTS]
    for root in candidate_roots:
        if not root.is_dir():
            continue
        direct = root / short
        if (direct / "config.json").exists():
            return str(direct)
        for child in root.iterdir():
            nested = child / short
            if child.is_dir() and (nested / "config.json").exists():
                return str(nested)
    return model_name


@runtime_checkable
class CrossEncoderLike(Protocol):
    """Structural type for a cross-encoder. Lets tests inject a stub."""

    def predict(self, pairs: Sequence[tuple[str, str]], **kwargs) -> np.ndarray: ...


@dataclass(frozen=True)
class RerankedSection:
    """One section with its cross-encoder score and 1-based rank."""

    doc_id: str
    section_title: str
    score: float
    rank: int

    @property
    def citation(self) -> tuple[str, str]:
        return (self.doc_id, self.section_title)

    def as_ranked_section(self) -> RankedSection:
        """Adapt to the Phase 1 type so citation strategies apply unchanged."""
        return RankedSection(
            doc_id=self.doc_id,
            section_title=self.section_title,
            score=self.score,
            chunk_count=1,
        )


def build_rerank_texts(
    corpus: Corpus, chunks: Sequence[Chunk], variant: TextVariant = "titled"
) -> list[str]:
    """The passage side of each ``(question, passage)`` pair.

    ``body``   — the section body alone.
    ``titled`` — ``"{doc_title} — {section_title}\\n{body}"``.

    Phase 1 found the equivalent flag made no measurable difference for
    bi-encoders. That conclusion is deliberately **not** carried across: a
    bi-encoder embeds the passage in isolation, while a cross-encoder attends over
    the question and passage jointly, so a title that names the document's topic
    can participate in the match in a way it cannot for a bi-encoder. Measured
    again rather than assumed.
    """
    texts: list[str] = []
    for chunk in chunks:
        body = chunk.text
        if variant == "body":
            texts.append(body)
            continue
        doc_title = corpus[chunk.doc_id].title if chunk.doc_id in corpus else chunk.doc_id
        texts.append(f"{doc_title} — {chunk.section_title}\n{body}")
    return texts


def emits_probabilities(scores: np.ndarray) -> bool:
    """True when a model's output already lives on a [0, 1] probability scale.

    Not every cross-encoder emits logits. Measured on this corpus:
    ``ms-marco-MiniLM-L-6-v2`` returned -11.4..+6.4, while
    ``mxbai-rerank-base-v1`` returned 0.01..0.94 with no negatives at all.
    """
    scores = np.asarray(scores, dtype="float64")
    return bool(scores.min() >= 0.0 and scores.max() <= 1.0)


def apply_transform(scores: np.ndarray, transform: ScoreTransform) -> np.ndarray:
    """Map model output onto a scale the citation strategies can use.

    Every transform here is order-preserving. ``sigmoid`` and ``raw`` are globally
    monotonic; ``minmax`` is affine with positive scale within each query row,
    which preserves order per row — all that ranking needs.

    ``auto`` is the right default across a mixed model set. The transform's job is
    to put scores on a probability scale so ``RelativeGap`` ratios mean something;
    if a model already emits probabilities, squashing them again through a sigmoid
    compresses the range toward 0.5–0.73 and distorts exactly the ratio geometry
    the cardinality sweep is trying to measure. So ``auto`` applies a sigmoid only
    when the output is not already in [0, 1].

    ``minmax`` is deliberately **not** the default despite being the most
    scale-agnostic: it forces the top score to exactly 1.0, which collapses
    ``RelativeGap(r)`` and ``ScoreThreshold(r)`` into the same function and would
    silently erase the distinction between two strategies we are trying to compare.
    """
    scores = np.asarray(scores, dtype="float64")
    if transform == "auto":
        transform = "raw" if emits_probabilities(scores) else "sigmoid"
    if transform == "raw":
        return scores
    if transform == "sigmoid":
        # Numerically stable logistic: avoids overflow at large |x|.
        return np.where(
            scores >= 0,
            1.0 / (1.0 + np.exp(-np.clip(scores, -700, 700))),
            np.exp(np.clip(scores, -700, 700)) / (1.0 + np.exp(np.clip(scores, -700, 700))),
        )
    if transform == "minmax":
        low = scores.min(axis=-1, keepdims=True)
        high = scores.max(axis=-1, keepdims=True)
        span = high - low
        return np.where(span < 1e-12, np.ones_like(scores), (scores - low) / np.maximum(span, 1e-12))
    raise ValueError(f"unknown score_transform {transform!r}")


def content_digest(*parts: object) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:32]


class RerankCache:
    """Disk cache of **raw** cross-encoder logits.

    Caching before the transform is deliberate: it makes comparing sigmoid
    against minmax against raw free, so the transform can be audited without
    re-running an hour of CPU scoring.
    """

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.hits = 0
        self.misses = 0

    def key(self, model_name: str, variant: str, texts: Sequence[str],
            questions: Sequence[str]) -> str:
        return content_digest(model_name, variant, "|".join(texts), "|".join(questions))

    def _path(self, digest: str) -> Path:
        return self.cache_dir / f"{digest}.npz"

    def get(self, digest: str, shape: tuple[int, int]) -> np.ndarray | None:
        path = self._path(digest)
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                matrix = data["scores"]
        except (OSError, KeyError, ValueError):
            return None
        return matrix if matrix.shape == shape else None

    def put(self, digest: str, matrix: np.ndarray) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._path(digest), scores=np.asarray(matrix, dtype="float32"))

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


class CrossEncoderReranker:
    """Scores ``(question, section)`` pairs jointly and ranks sections.

    Exhaustive and pooled reranking share one code path — ``rerank`` takes an
    optional candidate index list — so the two modes cannot drift apart. A pool is
    literally a subset of the exhaustive score matrix, which also means all pool
    sizes come free from a single scoring pass.
    """

    def __init__(
        self,
        model_name: str,
        score_transform: ScoreTransform = "sigmoid",
        text_variant: TextVariant = "titled",
        cache: RerankCache | None = None,
        model: CrossEncoderLike | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.score_transform = score_transform
        self.text_variant = text_variant
        self.cache = cache if cache is not None else RerankCache()
        self.batch_size = batch_size
        self._model = model
        self._chunks: list[Chunk] = []
        self._raw: np.ndarray | None = None

    @property
    def model(self) -> CrossEncoderLike:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(resolve_model_source(self.model_name))
        return self._model

    @property
    def fingerprint(self) -> str:
        short = self.model_name.rsplit("/", 1)[-1]
        return f"{short}/{self.text_variant}/{self.score_transform}"

    # ── scoring ────────────────────────────────────────────────────────────
    def fit(
        self,
        questions: Sequence[str],
        chunks: Sequence[Chunk],
        corpus: Corpus,
    ) -> np.ndarray:
        """Score every (question, chunk) pair. Returns the raw logit matrix."""
        self._chunks = list(chunks)
        texts = build_rerank_texts(corpus, self._chunks, self.text_variant)
        shape = (len(questions), len(texts))

        digest = self.cache.key(self.model_name, self.text_variant, texts, questions)
        cached = self.cache.get(digest, shape)
        if cached is not None:
            self.cache.hits += 1
            self._raw = cached.astype("float64")
            return self._raw

        self.cache.misses += 1
        pairs = [(q, t) for q in questions for t in texts]
        flat = np.asarray(
            self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False),
            dtype="float64",
        )
        self._raw = flat.reshape(shape)
        self.cache.put(digest, self._raw)
        return self._raw

    @property
    def raw_scores(self) -> np.ndarray:
        if self._raw is None:
            raise RuntimeError("call fit() before reading scores")
        return self._raw

    def scores(self, transform: ScoreTransform | None = None) -> np.ndarray:
        """The transformed score matrix. Transform is applied on read."""
        return apply_transform(self.raw_scores, transform or self.score_transform)

    # ── ranking ────────────────────────────────────────────────────────────
    def rerank(
        self,
        query_index: int,
        candidates: Sequence[int] | None = None,
        top_k: int | None = None,
        transform: ScoreTransform | None = None,
    ) -> list[RerankedSection]:
        """Rank sections for one question.

        ``candidates=None`` scores every chunk (exhaustive). Passing first-stage
        indices restricts to that pool.
        """
        row = self.scores(transform)[query_index]
        indices = (
            np.arange(len(self._chunks)) if candidates is None else np.asarray(candidates, dtype=int)
        )
        order = indices[np.argsort(-row[indices], kind="stable")]
        if top_k is not None:
            order = order[:top_k]

        return [
            RerankedSection(
                doc_id=self._chunks[i].doc_id,
                section_title=self._chunks[i].section_title,
                score=float(row[i]),
                rank=rank,
            )
            for rank, i in enumerate(order, start=1)
        ]

    def as_ranked_sections(self, reranked: Sequence[RerankedSection]) -> list[RankedSection]:
        """Adapt to Phase 1's type so citation strategies apply unchanged."""
        return [r.as_ranked_section() for r in reranked]


# ── diagnostics ────────────────────────────────────────────────────────────
def ordering_is_identical(raw: np.ndarray) -> dict[str, bool]:
    """Verify every transform yields the same per-query ordering.

    Sigmoid is strictly monotonic and min-max is affine with positive scale, so
    ordering *must* be preserved. Any difference here is a bug in the transform,
    not a modelling choice — and since the whole cardinality result is read
    through the transform, this is checked on the real score matrix for every
    model, not only on stubs.
    """
    baseline = np.argsort(-raw, axis=1, kind="stable")
    result = {}
    for transform in ("auto", "sigmoid", "minmax", "raw"):
        other = np.argsort(-apply_transform(raw, transform), axis=1, kind="stable")
        result[transform] = bool(np.array_equal(baseline, other))
    return result


def saturation_report(raw: np.ndarray) -> dict[str, float]:
    """Measure how much the sigmoid compresses the top of the distribution.

    A sigmoid over wide logits saturates: if most transformed values pin near 0
    or 1, then ``RelativeGap`` ratios cluster near 1.0 and the gap rule loses
    discrimination — a different failure from the negative-logit one, with the
    same symptom of a flat cardinality curve. This quantifies it before any
    conclusion is drawn from that curve.
    """
    sig = apply_transform(raw, "auto")
    order = np.argsort(-sig, axis=1)
    top1 = np.take_along_axis(sig, order[:, :1], axis=1).ravel()
    top2 = np.take_along_axis(sig, order[:, 1:2], axis=1).ravel()
    ratio = np.divide(top2, top1, out=np.zeros_like(top2), where=top1 > 0)

    mm = apply_transform(raw, "minmax")
    mm_order = np.argsort(-mm, axis=1)
    mm_top2 = np.take_along_axis(mm, mm_order[:, 1:2], axis=1).ravel()

    return {
        "top1_median": float(np.median(top1)),
        "top1_p10": float(np.percentile(top1, 10)),
        "top1_p90": float(np.percentile(top1, 90)),
        "top1_frac_above_0.99": float((top1 > 0.99).mean()),
        "top2_median": float(np.median(top2)),
        "top2_frac_below_0.01": float((top2 < 0.01).mean()),
        "ratio_median": float(np.median(ratio)),
        "ratio_p10": float(np.percentile(ratio, 10)),
        "ratio_p90": float(np.percentile(ratio, 90)),
        "ratio_iqr": float(np.percentile(ratio, 75) - np.percentile(ratio, 25)),
        "ratio_frac_above_0.95": float((ratio > 0.95).mean()),
        "minmax_top2_median": float(np.median(mm_top2)),
        "minmax_top2_iqr": float(
            np.percentile(mm_top2, 75) - np.percentile(mm_top2, 25)
        ),
    }
