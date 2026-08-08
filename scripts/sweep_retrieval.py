#!/usr/bin/env python
"""Retrieval configuration sweep.

    python scripts/sweep_retrieval.py

Cross-product over chunker x embedding model x retriever mode x aggregator x
citation strategy, scored against the answer key with the Phase 0 ``Scorer``.

How selection works here, and why it is not an argmax
----------------------------------------------------
Roughly 4,000 configurations scored against 50 questions is an overfitting
machine. The best row is substantially luck, and picking it would be optimising
toward a visible metric until it reads perfect while the private leaderboard
disagrees. So:

1. **Selection is on section-level recall@10, not citation F1.** Phase 2 adds a
   cross-encoder that reorders the candidate pool. A config chosen for rank-1
   precision is optimised for a pipeline we are not building. What Phase 2 needs
   handed to it is the right section *present in the pool*; ordering is its job.
   Citation F1 is reported on every row but is diagnostic this phase.

2. **Plateaus, not peaks.** Every config carries the mean score of its immediate
   neighbourhood in config space — the same config with one dimension moved one
   step. A config sitting on a broad flat region is trustworthy. A config that
   beats its neighbours by several points on 50 questions is noise wearing a
   crown, and ``delta_vs_nbhd`` is the column that exposes it.

3. **A noise band, computed not asserted.** Recall is a proportion over a finite
   number of expected sections, so its standard error is known. Two configs
   within two standard errors of each other are not distinguishable on this
   sample, and the shortlist says so out loud.

4. **Two leaderboards.** One over all 50 questions, one over the 36 marked
   ``confidence: high``. 14 entries are medium or low confidence and Phase 0
   showed naive retrieval disagreeing outright with the key on Q35. If the top
   configs differ materially between the two, that is a finding about the key,
   not about retrieval.

5. **No cardinality policy is chosen.** The full curve is reported; the decision
   belongs to the router in a later phase, because 36 questions want one citation
   and 14 want two and no fixed rule serves both.

No LLM, no API keys, no reranker. Local models only.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from proctoriq_rag.config import load_config  # noqa: E402
from proctoriq_rag.corpus.loader import load_corpus  # noqa: E402
from proctoriq_rag.evaluation.answer_key import load_answer_key  # noqa: E402
from proctoriq_rag.evaluation.scorer import Scorer  # noqa: E402
from proctoriq_rag.retrieval.chunking import build_chunkers, validate_chunks  # noqa: E402
from proctoriq_rag.retrieval.citation import (  # noqa: E402
    CitationPolicy,
    SectionAggregator,
    build_strategies,
    section_ranks,
)
from proctoriq_rag.retrieval.embeddings import (  # noqa: E402
    SWEEP_MODELS,
    EmbeddingCache,
    SentenceTransformerBackend,
    short_model_name,
)
from proctoriq_rag.retrieval.retriever import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    ScoredChunk,
)
from proctoriq_rag.submission.writer import Prediction  # noqa: E402

DEFAULT_KEY_PATH = REPO_ROOT / "data" / "validation" / "answer_key.yaml"
RECALL_DEPTHS = (1, 3, 5, 10)
HYBRID_ALPHAS = (0.3, 0.5, 0.7)
AGGREGATOR_TOP_N = 15

#: Ordered dimensions. Neighbours differ by exactly one step along one of these,
#: or by any value along a categorical dimension (model, header, mode, aggregator).
CHUNK_SIZE_AXIS = [256, 400, 512, 768, 10_000]  # 10_000 stands in for "no split"
ALPHA_AXIS = [0.0, 0.3, 0.5, 0.7, 1.0]


@dataclass(frozen=True)
class RetrievalConfig:
    """One (chunker, model, mode, alpha, aggregator) point in config space."""

    chunker_fp: str
    chunker_family: str
    chunk_size: int
    header: bool
    model: str
    mode: str
    alpha: float
    agg_mode: str

    @property
    def label(self) -> str:
        """Uniquely identifies the config.

        The chunk size must appear: without it every ``recursive`` row renders
        identically, which silently collides the ranked-pool cache used for the
        winner's failure report.
        """
        model = "—" if self.mode == "sparse" else short_model_name(self.model)
        alpha = f"@{self.alpha:g}" if self.mode == "hybrid" else ""
        header = "hdr" if self.header else "nohdr"
        family = (
            f"rec{self.chunk_size}"
            if self.chunker_family == "recursive"
            else self.chunker_family
        )
        return (
            f"{family}/{header} · {model} · {self.mode}{alpha} · agg={self.agg_mode}"
        )

    @property
    def axes(self) -> dict[str, object]:
        return {
            "chunker_family": self.chunker_family,
            "chunk_size": self.chunk_size,
            "header": self.header,
            "model": self.model,
            "mode": self.mode,
            "alpha": self.alpha,
            "agg_mode": self.agg_mode,
        }


def is_neighbour(a: RetrievalConfig, b: RetrievalConfig) -> bool:
    """True when ``b`` differs from ``a`` in exactly one dimension, by one step."""
    left, right = a.axes, b.axes
    differing = [key for key in left if left[key] != right[key]]
    if len(differing) != 1:
        return False
    key = differing[0]

    if key == "chunk_size":
        if a.chunker_family != b.chunker_family:
            return False
        try:
            gap = abs(CHUNK_SIZE_AXIS.index(a.chunk_size) - CHUNK_SIZE_AXIS.index(b.chunk_size))
        except ValueError:
            return False
        return gap == 1
    if key == "alpha":
        try:
            gap = abs(ALPHA_AXIS.index(a.alpha) - ALPHA_AXIS.index(b.alpha))
        except ValueError:
            return False
        return gap == 1
    if key == "chunker_family":
        # Families are categorical, but only compare like-for-like chunk sizes.
        return a.chunk_size == b.chunk_size
    return True  # header, model, mode, agg_mode are categorical


def recall_at(ranks_per_question: list[list[int]], depth: int) -> float:
    """Fraction of expected sections retrieved within the top ``depth``."""
    total = sum(len(r) for r in ranks_per_question)
    if not total:
        return 0.0
    hit = sum(sum(1 for rank in r if rank <= depth) for r in ranks_per_question)
    return hit / total


def noise_band(proportion: float, n: int) -> float:
    """Two standard errors of a binomial proportion — the resolution limit.

    At p=1.0 the textbook standard error is exactly zero, which would claim we can
    resolve arbitrarily small differences from a perfect score. That is an artefact
    of the formula, not a fact about the data. At the boundaries we fall back to the
    rule of three: with no observed failures in ``n`` trials, the true rate could
    still be as high as ~3/n.
    """
    if n <= 0:
        return 0.0
    if proportion <= 0.0 or proportion >= 1.0:
        return 3.0 / n
    return 2.0 * float(np.sqrt(proportion * (1 - proportion) / n))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--models", nargs="*", default=list(SWEEP_MODELS))
    parser.add_argument("--top-n", type=int, default=AGGREGATOR_TOP_N)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--quick", action="store_true", help="MiniLM only, header on")
    args = parser.parse_args()

    started = time.time()
    config = load_config()
    corpus = load_corpus(config.paths.kb_dir)
    key = load_answer_key(args.key, corpus)
    questions = pd.read_csv(config.paths.test_csv)
    qids = questions["question_id"].astype(str).tolist()
    qtexts = questions["question"].astype(str).tolist()

    header_variants = (True,) if args.quick else (True, False)
    models = args.models[:1] if args.quick else args.models

    chunkers = build_chunkers(header_variants=header_variants)
    strategies = build_strategies()
    aggregators = [
        SectionAggregator(top_n=args.top_n, mode="max"),
        SectionAggregator(top_n=args.top_n, mode="sum"),
    ]

    print(f"corpus: {len(corpus)} documents, {len(corpus.sections())} sections")
    print(f"key   : {len(key)} entries")
    print(f"grid  : {len(chunkers)} chunkers x {len(models)} models "
          f"x {len(aggregators)} aggregators x {len(strategies)} strategies")

    cache = EmbeddingCache()
    scorer = Scorer(corpus, key)
    expected = {qid: list(key[qid].pairs) for qid in qids}
    high_conf = {e.id for e in key.by_confidence("high")}

    # ── build every chunk set once ─────────────────────────────────────────
    chunk_sets = {}
    for chunker in chunkers:
        chunks = chunker.chunk(corpus)
        validate_chunks(chunks, corpus)
        chunk_sets[chunker.fingerprint] = (chunker, chunks)
        print(f"  {chunker.fingerprint:<28} {len(chunks):>4} chunks")

    # ── BM25 once per chunk set: independent of the embedding model ────────
    print("\nbuilding BM25 indexes (model-independent)...")
    sparse_by_fp = {
        fp: BM25Retriever(chunks, qtexts) for fp, (_, chunks) in chunk_sets.items()
    }

    rows: list[dict] = []
    best_cache: dict[str, object] = {}

    def evaluate(cfg: RetrievalConfig, retriever, chunks) -> None:
        """Score one retrieval config across all aggregators and strategies."""
        pools = [
            retriever.retrieve(i, k=min(args.top_n, len(chunks)))
            for i in range(len(qtexts))
        ]
        for aggregator in aggregators:
            if aggregator.mode != cfg.agg_mode:
                continue
            ranked = [aggregator.aggregate(pool) for pool in pools]
            ranks = [
                section_ranks(ranked[i], expected[qid]) for i, qid in enumerate(qids)
            ]
            ranks_high = [
                r for r, qid in zip(ranks, qids) if qid in high_conf
            ]
            recalls = {d: recall_at(ranks, d) for d in RECALL_DEPTHS}
            recalls_high = {d: recall_at(ranks_high, d) for d in RECALL_DEPTHS}

            for strategy in strategies:
                preds, preds_high = [], []
                for i, qid in enumerate(qids):
                    selected = strategy.select(ranked[i])
                    prediction = Prediction(
                        qid,
                        "",
                        [s.doc_id for s in selected],
                        [s.section_title for s in selected],
                    )
                    preds.append(prediction)
                    if qid in high_conf:
                        preds_high.append(prediction)

                report = scorer.score(preds)
                report_high = scorer.score(preds_high)

                row = {
                    "config": cfg.label,
                    "chunker": cfg.chunker_fp,
                    "chunker_family": cfg.chunker_family,
                    "chunk_size": cfg.chunk_size,
                    "header": cfg.header,
                    "model": "—" if cfg.mode == "sparse" else short_model_name(cfg.model),
                    "mode": cfg.mode,
                    "alpha": cfg.alpha,
                    "agg_mode": cfg.agg_mode,
                    "strategy": strategy.fingerprint,
                    "n_chunks": len(chunks),
                    "doc_f1": report.dimension_means["retrieval"],
                    "doc_exact": np.mean(
                        [q.retrieval.exact_match for q in report.per_question]
                    ),
                    "cite_f1": report.dimension_means["citation"],
                    "cite_exact": np.mean(
                        [q.citation.exact_match for q in report.per_question]
                    ),
                    "cite_f1_high": report_high.dimension_means["citation"],
                    "mean_citations": np.mean(
                        [len(p.cited_docs) for p in preds]
                    ),
                }
                for depth in RECALL_DEPTHS:
                    row[f"recall@{depth}"] = recalls[depth]
                    row[f"recall@{depth}_high"] = recalls_high[depth]
                rows.append(row)

            # Keep the ranked pools for the failure report of the eventual winner.
            best_cache[f"{cfg.label}"] = ranked

    # ── sparse: one pass per chunk set ─────────────────────────────────────
    print("scoring sparse configurations...")
    for fp, (chunker, chunks) in chunk_sets.items():
        family, size = _family_and_size(chunker)
        for agg in ("max", "sum"):
            cfg = RetrievalConfig(
                fp, family, size, _header_of(chunker), "", "sparse", 0.0, agg
            )
            evaluate(cfg, sparse_by_fp[fp], chunks)

    # ── dense and hybrid: one model load per model ─────────────────────────
    for model_name in models:
        print(f"\nloading {model_name} ...")
        backend = SentenceTransformerBackend(model_name)
        qvecs = cache.encode(backend, qtexts, fingerprint="test-questions", kind="queries")

        for fp, (chunker, chunks) in chunk_sets.items():
            cvecs = cache.encode(
                backend, [c.text for c in chunks], fingerprint=fp, kind="documents"
            )
            dense = DenseRetriever(chunks, cvecs, qvecs)
            family, size = _family_and_size(chunker)
            header = _header_of(chunker)

            for agg in ("max", "sum"):
                evaluate(
                    RetrievalConfig(fp, family, size, header, model_name, "dense", 1.0, agg),
                    dense, chunks,
                )
                for alpha in HYBRID_ALPHAS:
                    evaluate(
                        RetrievalConfig(
                            fp, family, size, header, model_name, "hybrid", alpha, agg
                        ),
                        HybridRetriever(dense, sparse_by_fp[fp], alpha), chunks,
                    )
        print(f"  cache: {cache.stats}")

    frame = pd.DataFrame(rows)

    # ── neighbourhood means over the retrieval-config grid ─────────────────
    frame = _attach_neighbourhood(frame)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = args.out_dir / f"sweep_retrieval_{stamp}.csv"
    frame.to_csv(path, index=False)
    frame.to_csv(args.out_dir / "sweep_retrieval_latest.csv", index=False)

    _report(frame, key, qids, expected, high_conf, best_cache, args.out_dir)

    print(f"\n{len(frame)} rows written to {path}")
    print(f"elapsed: {time.time() - started:.1f}s")
    return 0


def _family_and_size(chunker) -> tuple[str, int]:
    fp = chunker.fingerprint
    if fp.startswith("section"):
        return "section", 10_000
    if fp.startswith("subsection"):
        return "subsection", 10_000
    return "recursive", int(fp.split("-")[1])


def _header_of(chunker) -> bool:
    return chunker.fingerprint.endswith("-hdr")


def _attach_neighbourhood(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``nbhd_mean`` / ``nbhd_n`` / ``delta_vs_nbhd`` on recall@10.

    Neighbourhood is computed over the *retrieval* grid — the citation strategy is
    held fixed, because recall@10 does not depend on it.
    """
    grid = (
        frame.drop_duplicates(
            subset=["chunker_family", "chunk_size", "header", "model", "mode",
                    "alpha", "agg_mode"]
        )
        .reset_index(drop=True)
    )
    configs = [
        RetrievalConfig(
            row.chunker, row.chunker_family, int(row.chunk_size), bool(row.header),
            row.model, row.mode, float(row.alpha), row.agg_mode,
        )
        for row in grid.itertuples()
    ]
    values = grid["recall@10"].to_numpy()

    means, counts = [], []
    for i, cfg in enumerate(configs):
        neighbours = [
            values[j] for j, other in enumerate(configs)
            if j != i and is_neighbour(cfg, other)
        ]
        means.append(float(np.mean(neighbours)) if neighbours else float(values[i]))
        counts.append(len(neighbours))

    grid = grid.assign(nbhd_mean=means, nbhd_n=counts)
    keys = ["chunker_family", "chunk_size", "header", "model", "mode", "alpha", "agg_mode"]
    merged = frame.merge(grid[keys + ["nbhd_mean", "nbhd_n"]], on=keys, how="left")
    merged["delta_vs_nbhd"] = merged["recall@10"] - merged["nbhd_mean"]
    return merged


def _leaderboard(frame: pd.DataFrame, recall_col: str, cite_col: str, title: str,
                 n_expected: int, top: int = 10) -> pd.DataFrame:
    """Rank retrieval configs by recall@10, then recall@5. Cardinality-independent."""
    grid = (
        frame.groupby(
            ["config", "chunker_family", "chunk_size", "header", "model", "mode",
             "alpha", "agg_mode", "n_chunks", "nbhd_mean", "nbhd_n"],
            dropna=False,
        )
        .agg(**{
            "recall@10": (recall_col, "max"),
            "recall@5": (recall_col.replace("10", "5"), "max"),
            "recall@3": (recall_col.replace("10", "3"), "max"),
            "recall@1": (recall_col.replace("10", "1"), "max"),
            "best_cite_f1": (cite_col, "max"),
        })
        .reset_index()
        .sort_values(["recall@10", "recall@5"], ascending=False)
    )
    grid["delta_vs_nbhd"] = grid["recall@10"] - grid["nbhd_mean"]

    band = noise_band(float(grid["recall@10"].iloc[0]), n_expected)
    print("\n" + "=" * 112)
    print(f"{title}  —  sorted by recall@10, then recall@5")
    print(f"n expected sections = {n_expected}   noise band (2 s.e.) = ±{band:.4f} "
          f"({band * 100:.1f} pp)")
    print("=" * 112)
    print(f"{'#':>2}  {'config':<64} {'r@10':>6} {'r@5':>6} {'r@1':>6} "
          f"{'nbhd':>6} {'d-nbhd':>7} {'citeF1':>7}")
    print("-" * 112)
    for i, (_, row) in enumerate(grid.head(top).iterrows(), start=1):
        print(
            f"{i:>2}  {str(row['config'])[:64]:<64} "
            f"{row['recall@10']:>6.4f} {row['recall@5']:>6.4f} "
            f"{row['recall@1']:>6.4f} {row['nbhd_mean']:>6.4f} "
            f"{row['delta_vs_nbhd']:>+7.4f} {row['best_cite_f1']:>7.4f}"
        )
    return grid


def _report(frame, key, qids, expected, high_conf, best_cache, out_dir) -> None:
    n_all = sum(len(v) for v in expected.values())
    n_high = sum(len(v) for qid, v in expected.items() if qid in high_conf)

    board_all = _leaderboard(
        frame, "recall@10", "cite_f1", "LEADERBOARD A — all 50 questions", n_all
    )
    board_high = _leaderboard(
        frame, "recall@10_high", "cite_f1_high",
        f"LEADERBOARD B — {len(high_conf)} high-confidence questions only", n_high
    )

    _agreement(board_all, board_high)
    shortlist = _shortlist(board_all, n_all)
    _cardinality_curve(frame)

    winner, strategy, ranked = _winner_context(frame, board_all, best_cache)
    if ranked is not None:
        _by_kind(winner, strategy, ranked, qids, key, expected)
        _failures(winner, strategy, ranked, qids, key, expected)
    _write_shortlist(shortlist, out_dir)


def _winner_context(frame, board, best_cache):
    """The top config, its best-scoring citation strategy, and its ranked pools."""
    from proctoriq_rag.retrieval.citation import build_strategies as _bs

    winner = board["config"].iloc[0]
    subset = frame[frame["config"] == winner]
    best_strategy_fp = subset.loc[subset["cite_f1"].idxmax(), "strategy"]
    strategy = next(s for s in _bs() if s.fingerprint == best_strategy_fp)
    return winner, strategy, best_cache.get(winner)


def _predictions_for(strategy, ranked, qids):
    preds = []
    for i, qid in enumerate(qids):
        selected = strategy.select(ranked[i])
        preds.append(
            Prediction(qid, "", [s.doc_id for s in selected],
                       [s.section_title for s in selected])
        )
    return preds


def _agreement(board_all: pd.DataFrame, board_high: pd.DataFrame) -> None:
    """Compare the two leaderboards by tied-at-max SETS, not by row order.

    Comparing top-10 row identity is meaningless here: most configs tie exactly at
    the top, so the sort order among them is arbitrary and a naive identity
    comparison reports a dramatic disagreement that is pure tie-breaking noise.
    The honest question is whether the two subsets pick out overlapping *sets* of
    best configurations.
    """
    best_a = float(board_all["recall@10"].max())
    best_b = float(board_high["recall@10"].max())
    set_a = set(board_all[board_all["recall@10"] == best_a]["config"])
    set_b = set(board_high[board_high["recall@10"] == best_b]["config"])
    overlap = set_a & set_b
    jaccard = len(overlap) / len(set_a | set_b) if (set_a | set_b) else 0.0

    print("\n" + "=" * 112)
    print("DO THE TWO LEADERBOARDS AGREE?")
    print("=" * 112)
    print(f"  all-50    : {len(set_a)} configs tied at recall@10 = {best_a:.4f}")
    print(f"  high-only : {len(set_b)} configs tied at recall@10 = {best_b:.4f}")
    print(f"  configs top-tier on BOTH: {len(overlap)}   (Jaccard {jaccard:.3f})")

    if not overlap:
        print("  -> !! MATERIAL DISAGREEMENT. No configuration is best on both subsets.")
        print("     The medium/low-confidence entries are driving selection. Treat this")
        print("     as a finding about the key, not about retrieval.")
    else:
        print(f"  -> Substantial agreement: {len(overlap)} configurations are top-tier")
        print("     on both subsets, so config choice is not an artefact of the")
        print("     uncertain key entries. Ranking WITHIN the tied set is not evidence.")

    models_a = sorted({c.split("·")[1].strip() for c in set_a})
    models_b = sorted({c.split("·")[1].strip() for c in set_b})
    print(f"\n  models in the all-50 tied set   : {', '.join(models_a)}")
    print(f"  models in the high-only tied set: {', '.join(models_b)}")


def _shortlist(board: pd.DataFrame, n_expected: int, size: int = 5) -> pd.DataFrame:
    best = float(board["recall@10"].iloc[0])
    band = noise_band(best, n_expected)
    within = board[board["recall@10"] >= best - band]

    print("\n" + "=" * 108)
    print(f"SHORTLIST — {len(within)} configs are within one noise band (±{band * 100:.1f} pp) "
          f"of the top score")
    print("=" * 108)
    print("  These are statistically indistinguishable on n=50. Ranking among them is not")
    print("  evidence. Preferring the one on the broadest plateau (highest nbhd, smallest")
    print("  Δnbhd) is the defensible tie-break.\n")

    robust = within.sort_values(["nbhd_mean", "recall@10"], ascending=False).head(size)
    for i, (_, row) in enumerate(robust.iterrows(), start=1):
        print(f"  {i}. {row['config']}")
        print(
            f"       recall@10={row['recall@10']:.4f}  recall@5={row['recall@5']:.4f}  "
            f"neighbourhood={row['nbhd_mean']:.4f} (n={int(row['nbhd_n'])})  "
            f"d={row['delta_vs_nbhd']:+.4f}  chunks={int(row['n_chunks'])}"
        )
    return robust


def _cardinality_curve(frame: pd.DataFrame) -> None:
    print("\n" + "=" * 108)
    print("CITATION CARDINALITY CURVE — reported, NOT selected on")
    print("=" * 108)
    print("  36 of 50 questions want one citation, 14 want two. No fixed rule serves both;")
    print("  this decision belongs to the router in a later phase.\n")
    curve = (
        frame.groupby("strategy")
        .agg(
            mean_citations=("mean_citations", "mean"),
            doc_f1=("doc_f1", "mean"),
            cite_f1=("cite_f1", "mean"),
            cite_exact=("cite_exact", "mean"),
            best_cite_f1=("cite_f1", "max"),
        )
        .sort_values("mean_citations")
    )
    print(f"  {'strategy':<14} {'mean cites':>11} {'doc F1':>8} {'cite F1':>9} "
          f"{'cite exact':>11} {'best cite F1':>13}")
    print("  " + "-" * 72)
    for name, row in curve.iterrows():
        print(f"  {name:<14} {row.mean_citations:>11.3f} {row.doc_f1:>8.4f} "
              f"{row.cite_f1:>9.4f} {row.cite_exact:>11.4f} {row.best_cite_f1:>13.4f}")


def _by_kind(winner, strategy, ranked, qids, key, expected) -> None:
    """Per-kind and per-confidence tables for the winning config.

    The headline number hides the trade that matters: a cardinality rule tuned on
    the 36 single-citation questions can win overall while destroying the
    multi-source ones.
    """
    from proctoriq_rag.config import load_config as _lc
    from proctoriq_rag.corpus.loader import load_corpus as _load

    corpus = _load(_lc().paths.kb_dir)
    report = Scorer(corpus, key).score(_predictions_for(strategy, ranked, qids))

    for attribute in ("kind", "confidence"):
        print("\n" + "=" * 112)
        print(f"BY-{attribute.upper()} BREAKDOWN — {winner}")
        print(f"citation strategy: {strategy.fingerprint} (best for this config — "
              f"diagnostic only, not a Phase 1 choice)")
        print("=" * 112)
        print(f"  {attribute:<16} {'n':>4} {'doc F1':>9} {'doc exact':>11} "
              f"{'cite F1':>9} {'cite exact':>11}")
        print("  " + "-" * 66)
        for name, stats in report.breakdown(attribute).items():
            print(f"  {name:<16} {int(stats['n']):>4} {stats['doc_f1']:>9.4f} "
                  f"{stats['doc_exact']:>11.4f} {stats['cite_f1']:>9.4f} "
                  f"{stats['cite_exact']:>11.4f}")


def _failures(winner, strategy, ranked, qids, key, expected) -> None:
    """Per-question failure list for the winning config, with expected-section ranks."""
    print("\n" + "=" * 112)
    print(f"PER-QUESTION FAILURES — {winner}")
    print(f"citation strategy: {strategy.fingerprint}")
    print("=" * 112)
    print("  'rank' is where each expected section actually landed in the aggregated ranking.")
    print("  rank<=10 means retrieval FOUND it and the cardinality rule dropped it — a Phase 2")
    print("  reranker can recover that. A missing rank means retrieval never surfaced it.\n")

    misses = 0
    for i, qid in enumerate(qids):
        selected = strategy.select(ranked[i])
        predicted = [(s.doc_id, s.section_title) for s in selected]
        if set(predicted) == set(expected[qid]):
            continue
        misses += 1
        entry = key[qid]
        ranks = section_ranks(ranked[i], expected[qid])
        print(f"  [{qid}] kind={entry.kind} confidence={entry.confidence}")
        for (doc, sec), rank in zip(expected[qid], ranks):
            marker = "FOUND" if rank <= 10 else "MISSED"
            print(f"      expected  : {doc} :: {sec}   [rank {rank}, {marker}]")
        for doc, sec in predicted:
            flag = "" if (doc, sec) in expected[qid] else "  <-- wrong"
            print(f"      predicted : {doc} :: {sec}{flag}")
        print()

    recoverable = 0
    for i, qid in enumerate(qids):
        selected = strategy.select(ranked[i])
        if set((s.doc_id, s.section_title) for s in selected) == set(expected[qid]):
            continue
        if all(r <= 10 for r in section_ranks(ranked[i], expected[qid])):
            recoverable += 1
    print(f"  {misses} questions with inexact citations.")
    print(f"  {recoverable} of them have EVERY expected section already inside the top 10")
    print("  -> recoverable by reranking alone in Phase 2, with no retrieval change.")
    print(f"  {misses - recoverable} need better retrieval, not better ordering.")


def _write_shortlist(shortlist: pd.DataFrame, out_dir: Path) -> None:
    path = out_dir / "sweep_shortlist.csv"
    shortlist.to_csv(path, index=False)
    print(f"\nshortlist written to {path}")


if __name__ == "__main__":
    raise SystemExit(main())
