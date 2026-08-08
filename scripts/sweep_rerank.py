#!/usr/bin/env python
"""Cross-encoder reranking sweep.

    python scripts/sweep_rerank.py

Grid: 3 reranker models x 2 scored-text variants x 4 pool sizes x 14 citation
strategies = 336 rows, from 15,900 cross-encoder pairs scored once and cached.

Reporting discipline, carried over from Phase 1 and tightened
---------------------------------------------------------------
Citation-exact is a proportion over 50 questions, so its resolution is
``2*sqrt(p(1-p)/50)`` — roughly **14 percentage points**. Most of this grid will
land inside that. So the shortlist is printed FIRST, framed as "these N configs
are statistically indistinguishable, here is the simplest," and the ranked
leaderboard comes after. Rank ordering is persuasive out of proportion to its
evidential weight, and showing it first invites over-reading.

The one comparison expected to survive the band is exhaustive vs pooled, because
that is structural rather than a hyperparameter. It gets its own table.

Citation-exact and citation-F1 are always shown together. Phase 1 found them
disagreeing violently — topk-1 scored 0.359 exact vs topk-2 at 0.048 while F1
moved 0.463 to 0.464 — which means the grader's matching method is worth more
than any configuration choice here. Collapsing to one headline would hide that.

No LLM, no API keys, no generation. Local cross-encoders only.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from proctoriq_rag.config import load_config  # noqa: E402
from proctoriq_rag.corpus.loader import load_corpus  # noqa: E402
from proctoriq_rag.evaluation.alternates import load_alternates  # noqa: E402
from proctoriq_rag.evaluation.answer_key import load_answer_key  # noqa: E402
from proctoriq_rag.evaluation.scorer import Scorer  # noqa: E402
from proctoriq_rag.retrieval.chunking import SectionChunker  # noqa: E402
from proctoriq_rag.retrieval.citation import build_strategies, section_ranks  # noqa: E402
from proctoriq_rag.retrieval.embeddings import (  # noqa: E402
    EmbeddingCache,
    SentenceTransformerBackend,
)
from proctoriq_rag.retrieval.reranker import (  # noqa: E402
    CrossEncoderReranker,
    RerankCache,
    ordering_is_identical,
    saturation_report,
)
from proctoriq_rag.retrieval.retriever import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
)
from proctoriq_rag.submission.writer import Prediction  # noqa: E402

KEY_PATH = REPO_ROOT / "data" / "validation" / "answer_key.yaml"
ALT_PATH = REPO_ROOT / "data" / "validation" / "answer_key_alternates.yaml"

MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-base",
    "mixedbread-ai/mxbai-rerank-base-v1",
]
TEXT_VARIANTS = ["titled", "body"]
POOL_SIZES = [10, 20, 30, None]  # None = exhaustive
RECALL_DEPTHS = (1, 3, 5, 10)

# Phase 1 provisional default, fixed so pool size is the only thing varying.
FIRST_STAGE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FIRST_STAGE_ALPHA = 0.7


def noise_band(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    if p <= 0.0 or p >= 1.0:
        return 3.0 / n
    return 2.0 * float(np.sqrt(p * (1 - p) / n))


def pool_label(size: int | None) -> str:
    return "exhaustive" if size is None else f"pool{size}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    started = time.time()
    config = load_config()
    corpus = load_corpus(config.paths.kb_dir)
    key = load_answer_key(KEY_PATH, corpus)
    alternates = load_alternates(ALT_PATH, corpus, key)

    questions = pd.read_csv(config.paths.test_csv)
    qids = questions["question_id"].astype(str).tolist()
    qtexts = questions["question"].astype(str).tolist()

    chunks = SectionChunker(include_header_in_text=True).chunk(corpus)
    expected = {qid: list(key[qid].pairs) for qid in qids}
    high_conf = {e.id for e in key.by_confidence("high")}
    scorer = Scorer(corpus, key)
    strategies = build_strategies()

    models = args.models[:1] if args.quick else args.models
    variants = TEXT_VARIANTS[:1] if args.quick else TEXT_VARIANTS

    print(f"corpus: {len(corpus)} docs, {len(chunks)} sections")
    print(f"grid  : {len(models)} models x {len(variants)} text variants x "
          f"{len(POOL_SIZES)} pools x {len(strategies)} strategies")
    print(f"pairs : {len(models) * len(variants) * len(qtexts) * len(chunks):,}\n")

    first_stage_pool = _build_first_stage(corpus, chunks, qtexts, config)
    rerank_cache = RerankCache()

    rows: list[dict] = []
    pools_by_config: dict[str, list] = {}
    diagnostics: list[dict] = []

    for model_name in models:
        for variant in variants:
            print(f"scoring {model_name}  text={variant} ...", flush=True)
            t0 = time.time()
            reranker = CrossEncoderReranker(
                model_name, score_transform="auto", text_variant=variant,
                cache=rerank_cache,
            )
            raw = reranker.fit(qtexts, chunks, corpus)
            print(f"  {raw.size:,} pairs in {time.time() - t0:.1f}s  "
                  f"cache={rerank_cache.stats}")

            # ── AMENDMENT 2: verify the transform cannot have changed ranking ──
            ordering = ordering_is_identical(raw)
            from proctoriq_rag.retrieval.reranker import emits_probabilities
            diagnostics.append(
                {"model": model_name, "variant": variant,
                 "output_scale": "probabilities" if emits_probabilities(raw) else "logits",
                 "raw_min": float(raw.min()), "raw_max": float(raw.max()),
                 **{f"order_{k}": v for k, v in ordering.items()},
                 **saturation_report(raw)}
            )
            if not all(ordering.values()):
                print(f"  !! ORDERING DIFFERS ACROSS TRANSFORMS: {ordering}")
                print("  !! Stopping. Sigmoid is monotonic, so this is a transform bug.")
                return 1
            print(f"  ordering identical under raw/sigmoid/minmax: OK")

            for pool_size in POOL_SIZES:
                label = f"{model_name.rsplit('/', 1)[-1]} · {variant} · {pool_label(pool_size)}"
                ranked_per_q = []
                for i in range(len(qtexts)):
                    candidates = None if pool_size is None else first_stage_pool[i][:pool_size]
                    ranked_per_q.append(
                        reranker.as_ranked_sections(reranker.rerank(i, candidates=candidates))
                    )
                pools_by_config[label] = ranked_per_q

                ranks = [section_ranks(ranked_per_q[i], expected[q])
                         for i, q in enumerate(qids)]
                recalls = {d: _recall_at(ranks, d) for d in RECALL_DEPTHS}
                ranks_high = [r for r, q in zip(ranks, qids) if q in high_conf]
                recalls_high = {d: _recall_at(ranks_high, d) for d in RECALL_DEPTHS}

                for strategy in strategies:
                    preds, preds_high = [], []
                    for i, qid in enumerate(qids):
                        chosen = strategy.select(ranked_per_q[i])
                        p = Prediction(qid, "", [s.doc_id for s in chosen],
                                       [s.section_title for s in chosen])
                        preds.append(p)
                        if qid in high_conf:
                            preds_high.append(p)

                    report = scorer.score(preds)
                    report_high = scorer.score(preds_high)
                    kinds = report.breakdown("kind")

                    rows.append({
                        "config": label,
                        "model": model_name.rsplit("/", 1)[-1],
                        "text_variant": variant,
                        "pool": pool_label(pool_size),
                        "pool_size": -1 if pool_size is None else pool_size,
                        "exhaustive": pool_size is None,
                        "strategy": strategy.fingerprint,
                        "cite_exact": float(np.mean(
                            [q.citation.exact_match for q in report.per_question])),
                        "cite_f1": report.dimension_means["citation"],
                        "doc_exact": float(np.mean(
                            [q.retrieval.exact_match for q in report.per_question])),
                        "doc_f1": report.dimension_means["retrieval"],
                        "cite_exact_high": float(np.mean(
                            [q.citation.exact_match for q in report_high.per_question])),
                        "cite_f1_high": report_high.dimension_means["citation"],
                        "mean_citations": float(np.mean([len(p.cited_docs) for p in preds])),
                        "lookup_doc_f1": kinds.get("lookup", {}).get("doc_f1", float("nan")),
                        "adversarial_doc_f1": kinds.get("adversarial", {}).get(
                            "doc_f1", float("nan")),
                        "adversarial_gap": (
                            kinds.get("lookup", {}).get("doc_f1", float("nan"))
                            - kinds.get("adversarial", {}).get("doc_f1", float("nan"))
                        ),
                        "adversarial_cite_f1": kinds.get("adversarial", {}).get(
                            "cite_f1", float("nan")),
                        **{f"recall@{d}": recalls[d] for d in RECALL_DEPTHS},
                        **{f"recall@{d}_high": recalls_high[d] for d in RECALL_DEPTHS},
                    })

    frame = pd.DataFrame(rows)
    frame = _attach_neighbourhood(frame)
    diag = pd.DataFrame(diagnostics)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    frame.to_csv(args.out_dir / f"sweep_rerank_{stamp}.csv", index=False)
    frame.to_csv(args.out_dir / "sweep_rerank_latest.csv", index=False)
    diag.to_csv(args.out_dir / "rerank_diagnostics.csv", index=False)

    _report_transform_diagnostics(diag)
    shortlist = _report_shortlist(frame)
    _report_exhaustive_vs_pooled(frame)
    _report_leaderboards(frame)
    _report_metric_divergence(frame)
    _report_cardinality(frame)
    _report_by_kind(frame, shortlist)
    _report_alternates(pools_by_config, frame, key, alternates, qids)
    _report_failures(pools_by_config, frame, key, qids, expected, strategies)

    print(f"\n{len(frame)} rows -> {args.out_dir / 'sweep_rerank_latest.csv'}")
    print(f"elapsed: {time.time() - started:.1f}s")
    return 0


def _recall_at(ranks_per_q: list[list[int]], depth: int) -> float:
    total = sum(len(r) for r in ranks_per_q)
    if not total:
        return 0.0
    return sum(sum(1 for r in rr if r <= depth) for rr in ranks_per_q) / total


def _build_first_stage(corpus, chunks, qtexts, config) -> list[np.ndarray]:
    """Phase 1 provisional default, used only to form pools for pooled modes."""
    cache = EmbeddingCache()
    backend = SentenceTransformerBackend(FIRST_STAGE_MODEL)
    qvecs = cache.encode(backend, qtexts, fingerprint="test-questions", kind="queries")
    cvecs = cache.encode(backend, [c.text for c in chunks],
                         fingerprint="section-hdr", kind="documents")
    dense = DenseRetriever(chunks, cvecs, qvecs)
    sparse = BM25Retriever(chunks, qtexts)
    hybrid = HybridRetriever(dense, sparse, alpha=FIRST_STAGE_ALPHA)
    return [hybrid.rank(i) for i in range(len(qtexts))]


def _attach_neighbourhood(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean cite_exact of configs differing in exactly one dimension."""
    grid = frame.groupby(["config", "model", "text_variant", "pool"], as_index=False)[
        "cite_exact"
    ].max()
    means, counts = [], []
    for _, row in grid.iterrows():
        mask = (
            ((grid["model"] != row["model"])
             & (grid["text_variant"] == row["text_variant"])
             & (grid["pool"] == row["pool"]))
            | ((grid["text_variant"] != row["text_variant"])
               & (grid["model"] == row["model"]) & (grid["pool"] == row["pool"]))
            | ((grid["pool"] != row["pool"])
               & (grid["model"] == row["model"])
               & (grid["text_variant"] == row["text_variant"]))
        )
        neighbours = grid[mask]["cite_exact"]
        means.append(float(neighbours.mean()) if len(neighbours) else float(row["cite_exact"]))
        counts.append(int(len(neighbours)))
    grid = grid.assign(nbhd_mean=means, nbhd_n=counts).drop(columns=["cite_exact"])
    merged = frame.merge(grid, on=["config", "model", "text_variant", "pool"], how="left")
    merged["delta_vs_nbhd"] = merged["cite_exact"] - merged["nbhd_mean"]
    return merged


def _best_per_config(frame: pd.DataFrame) -> pd.DataFrame:
    idx = frame.groupby("config")["cite_exact"].idxmax()
    return frame.loc[idx].sort_values(["cite_exact", "cite_f1"], ascending=False)


def _report_transform_diagnostics(diag: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("TRANSFORM SANITY CHECK — does the sigmoid change anything it shouldn't?")
    print("=" * 100)
    print("  Sigmoid is strictly monotonic and min-max is affine with positive scale, so")
    print("  per-query ordering MUST be identical under all three. Verified on the real")
    print("  50x53 matrix for every model, not just on stubs.\n")
    for _, row in diag.iterrows():
        ok = all(row[f"order_{t}"] for t in ("auto", "sigmoid", "minmax", "raw"))
        print(f"  {row['model'].rsplit('/', 1)[-1]:<32} text={row['variant']:<7} "
              f"raw {row['raw_min']:>7.2f}..{row['raw_max']:<6.2f} "
              f"({row['output_scale']:<13}) ordering identical: {'YES' if ok else 'NO !!'}")

    print("\n  SIGMOID SATURATION — does the gap rule still have room to discriminate?")
    print(f"  {'model / text':<40} {'top1 med':>9} {'top2 med':>9} {'ratio med':>10} "
          f"{'ratio IQR':>10} {'r>0.95':>8}")
    print("  " + "-" * 92)
    for _, row in diag.iterrows():
        name = f"{row['model'].rsplit('/', 1)[-1]}/{row['variant']}"
        print(f"  {name[:40]:<40} {row['top1_median']:>9.4f} {row['top2_median']:>9.4f} "
              f"{row['ratio_median']:>10.4f} {row['ratio_iqr']:>10.4f} "
              f"{row['ratio_frac_above_0.95']:>8.2f}")
    print("\n  Read: if ratio IQR is tiny and 'r>0.95' is near 1.0, sigmoid has saturated")
    print("  and RelativeGap cannot discriminate — a different failure from the negative-")
    print("  logit one, with the same flat-curve symptom. minmax_top2 columns in")
    print("  outputs/rerank_diagnostics.csv show the alternative scale.")


def _report_shortlist(frame: pd.DataFrame) -> pd.DataFrame:
    """Printed FIRST, deliberately: the band constrains the conclusion."""
    best_rows = _best_per_config(frame)
    best = float(best_rows["cite_exact"].iloc[0])
    band = noise_band(best, 50)
    within = best_rows[best_rows["cite_exact"] >= best - band]

    print("\n" + "=" * 100)
    print("HEADLINE — WHAT IS ACTUALLY DISTINGUISHABLE")
    print("=" * 100)
    print(f"  Best citation-exact: {best:.4f}. Noise band on n=50 is +/-{band:.4f} "
          f"({band * 100:.1f} pp).")
    print(f"  {len(within)} of {len(best_rows)} configurations fall inside it.")
    print()
    if len(within) > 1:
        print(f"  ==> These {len(within)} configurations are STATISTICALLY")
        print("      INDISTINGUISHABLE on this sample. Their ranking below is not")
        print("      evidence. Prefer the simplest: fewest stages, smallest model.")
    else:
        print("  ==> One configuration separates from the field by more than the noise band.")

    simplest = within.copy()
    simplest["simplicity"] = (
        simplest["exhaustive"].astype(int) * 2
        + simplest["model"].str.contains("L-6").astype(int)
        + (simplest["text_variant"] == "body").astype(int)
    )
    simplest = simplest.sort_values(["simplicity", "cite_exact"], ascending=False)

    print("\n  SHORTLIST (simplest first among the indistinguishable):")
    print(f"  {'#':>2}  {'config':<52} {'cite_ex':>8} {'cite_F1':>8} {'doc_F1':>8} "
          f"{'strategy':>12}")
    print("  " + "-" * 94)
    for i, (_, row) in enumerate(simplest.head(5).iterrows(), start=1):
        print(f"  {i:>2}  {row['config'][:52]:<52} {row['cite_exact']:>8.4f} "
              f"{row['cite_f1']:>8.4f} {row['doc_f1']:>8.4f} {row['strategy']:>12}")
    return simplest


def _report_exhaustive_vs_pooled(frame: pd.DataFrame) -> None:
    """Structural comparison, expected to survive the band. Own table."""
    print("\n" + "=" * 100)
    print("EXHAUSTIVE vs POOLED — the structural comparison")
    print("=" * 100)
    print("  Scoring all 53 sections removes the candidate pool entirely, so the recall@10")
    print("  ceiling stops existing. This is a structural difference, not a hyperparameter.\n")
    table = (
        frame.groupby("pool")
        .agg(recall_at_1=("recall@1", "max"), recall_at_10=("recall@10", "max"),
             cite_exact=("cite_exact", "max"), cite_f1=("cite_f1", "max"),
             doc_f1=("doc_f1", "max"), adv_gap=("adversarial_gap", "min"))
        .reindex(["pool10", "pool20", "pool30", "exhaustive"])
    )
    print(f"  {'pool':<12} {'recall@1':>9} {'recall@10':>10} {'cite_ex':>9} {'cite_F1':>9} "
          f"{'doc_F1':>8} {'adv gap':>9}")
    print("  " + "-" * 74)
    for name, row in table.iterrows():
        print(f"  {name:<12} {row['recall_at_1']:>9.4f} {row['recall_at_10']:>10.4f} "
              f"{row['cite_exact']:>9.4f} {row['cite_f1']:>9.4f} {row['doc_f1']:>8.4f} "
              f"{row['adv_gap']:>9.4f}")

    ex = float(table.loc["exhaustive", "cite_exact"])
    p10 = float(table.loc["pool10", "cite_exact"])
    band = noise_band(max(ex, p10), 50)
    delta = ex - p10
    print(f"\n  exhaustive - pool10 on citation-exact: {delta:+.4f}  "
          f"(band +/-{band:.4f})")
    print(f"  -> {'SURVIVES the noise band' if abs(delta) > band else 'INSIDE the noise band'}")


def _report_leaderboards(frame: pd.DataFrame) -> None:
    for subset, exact_col, f1_col, n in (
        ("ALL 50 QUESTIONS", "cite_exact", "cite_f1", 50),
        ("36 HIGH-CONFIDENCE ONLY", "cite_exact_high", "cite_f1_high", 36),
    ):
        idx = frame.groupby("config")[exact_col].idxmax()
        board = frame.loc[idx].sort_values([exact_col, f1_col], ascending=False)
        band = noise_band(float(board[exact_col].iloc[0]), n)
        tied = int((board[exact_col] >= board[exact_col].iloc[0] - band).sum())

        print("\n" + "=" * 100)
        print(f"LEADERBOARD — {subset}   (ordering is NOT evidence: {tied}/{len(board)} "
              f"within +/-{band * 100:.1f} pp)")
        print("=" * 100)
        print(f"  {'#':>2}  {'config':<50} {'cite_ex':>8} {'cite_F1':>8} {'nbhd':>7} "
              f"{'d-nbhd':>8} {'strategy':>12}")
        print("  " + "-" * 96)
        for i, (_, row) in enumerate(board.head(10).iterrows(), start=1):
            print(f"  {i:>2}  {row['config'][:50]:<50} {row[exact_col]:>8.4f} "
                  f"{row[f1_col]:>8.4f} {row['nbhd_mean']:>7.4f} "
                  f"{row['delta_vs_nbhd']:>+8.4f} {row['strategy']:>12}")


def _report_metric_divergence(frame: pd.DataFrame) -> None:
    by_exact = set(_best_per_config(frame).head(5)["config"])
    idx = frame.groupby("config")["cite_f1"].idxmax()
    by_f1 = set(frame.loc[idx].sort_values("cite_f1", ascending=False).head(5)["config"])
    print("\n" + "=" * 100)
    print("DOES THE CHOICE OF METRIC CHANGE THE ANSWER?")
    print("=" * 100)
    print(f"  top-5 by citation-exact vs by citation-F1: {len(by_exact & by_f1)}/5 overlap")
    if len(by_exact & by_f1) < 4:
        print("  -> !! The two metrics prefer DIFFERENT configurations. Since we do not know")
        print("     which the grader uses, this is an unresolved risk, not a tuning choice.")
    else:
        print("  -> The metrics broadly agree on configuration, so the grader-method unknown")
        print("     does not change which config to ship (it still changes the score).")


def _report_cardinality(frame: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("CARDINALITY UNDER RERANKER SCORES")
    print("=" * 100)
    print("  The question: can a score-gap rule separate the 36 single-citation questions")
    print("  from the 14 two-citation ones now that scores are discriminative?\n")
    curve = (
        frame.groupby("strategy")
        .agg(mean_cites=("mean_citations", "mean"), cite_exact=("cite_exact", "mean"),
             best_cite_exact=("cite_exact", "max"), cite_f1=("cite_f1", "mean"),
             doc_f1=("doc_f1", "mean"))
        .sort_values("mean_cites")
    )
    print(f"  {'strategy':<14} {'mean cites':>11} {'cite_exact':>11} {'best ex':>9} "
          f"{'cite_F1':>9} {'doc_F1':>8}")
    print("  " + "-" * 68)
    for name, row in curve.iterrows():
        print(f"  {name:<14} {row['mean_cites']:>11.3f} {row['cite_exact']:>11.4f} "
              f"{row['best_cite_exact']:>9.4f} {row['cite_f1']:>9.4f} {row['doc_f1']:>8.4f}")

    topk1 = curve.loc["topk-1", "best_cite_exact"]
    best_adaptive = curve.drop(index=["topk-1", "topk-2"])["best_cite_exact"].max()
    band = noise_band(max(topk1, best_adaptive), 50)
    print(f"\n  best adaptive - topk-1 on citation-exact: {best_adaptive - topk1:+.4f} "
          f"(band +/-{band:.4f})")
    if best_adaptive - topk1 > band:
        print("  -> A score-gap rule BEATS fixed topk-1 beyond noise. Cardinality is")
        print("     partly a score decision, which makes the Phase 3 router's job easier.")
    else:
        print("  -> No score-gap rule beats fixed topk-1 beyond noise. Cardinality is an")
        print("     INTENT decision, not a score decision — Phase 3 must classify, not threshold.")


def _report_by_kind(frame: pd.DataFrame, shortlist: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("BY-KIND — does reranking close the adversarial gap?")
    print("=" * 100)
    print("  Phase 1 baseline: lookup doc F1 0.839, adversarial 0.450, gap 0.389\n")
    print(f"  {'config':<46} {'strategy':>12} {'lookup':>8} {'advers':>8} {'gap':>8} "
          f"{'adv citeF1':>11}")
    print("  " + "-" * 96)
    for _, row in shortlist.head(5).iterrows():
        print(f"  {row['config'][:46]:<46} {row['strategy']:>12} "
              f"{row['lookup_doc_f1']:>8.4f} {row['adversarial_doc_f1']:>8.4f} "
              f"{row['adversarial_gap']:>8.4f} {row['adversarial_cite_f1']:>11.4f}")

    best_gap = float(frame["adversarial_gap"].min())
    print(f"\n  smallest adversarial gap anywhere in the sweep: {best_gap:.4f} "
          f"(Phase 1: 0.3891)")
    if best_gap < 0.20:
        print("  -> Reranking substantially CLOSES the gap.")
    else:
        print("  -> The gap PERSISTS. Adversarial questions are not a ranking problem;")
        print("     they need intent classification BEFORE retrieval, in Phase 3.")


def _report_alternates(pools, frame, key, alternates, qids) -> None:
    best_config = _best_per_config(frame)["config"].iloc[0]
    ranked = pools.get(best_config)
    if ranked is None:
        return

    print("\n" + "=" * 100)
    print(f"ALTERNATE-KEY EVIDENCE — {best_config}")
    print("=" * 100)
    print("  The cross-encoder models question-document relevance directly, so it gets a")
    print("  real vote on the four contested entries. EVIDENCE ONLY — answer_key.yaml is")
    print("  untouched and the reading remains a human decision.\n")

    index = {q: i for i, q in enumerate(qids)}
    for qid, readings in alternates.items():
        i = index[qid]
        sections = ranked[i]
        entry = key[qid]
        print(f"  [{qid}] kind={entry.kind} confidence={entry.confidence}")

        def describe(pairs):
            ranks = section_ranks(sections, pairs)
            parts = []
            for (doc, sec), rank in zip(pairs, ranks):
                score = next((s.score for s in sections if s.citation == (doc, sec)), 0.0)
                parts.append(f"{doc.split('_')[0]} §{sec.split(':')[0]:<10} rank {rank:>2} "
                             f"({score:.3f})")
            return parts, max(ranks) if ranks else 999

        parts, worst = describe(list(entry.pairs))
        print(f"      PRIMARY                       worst-rank {worst:>2}")
        for part in parts:
            print(f"          {part}")
        for reading in readings:
            parts, alt_worst = describe(list(reading.pairs))
            verdict = "BETTER" if alt_worst < worst else ("same" if alt_worst == worst else "worse")
            print(f"      alt: {reading.label:<24} worst-rank {alt_worst:>2}   [{verdict}]")
            for part in parts:
                print(f"          {part}")
        print()


def _report_failures(pools, frame, key, qids, expected, strategies) -> None:
    best = _best_per_config(frame).iloc[0]
    ranked = pools.get(best["config"])
    if ranked is None:
        return
    strategy = next(s for s in strategies if s.fingerprint == best["strategy"])

    print("=" * 100)
    print(f"PER-QUESTION FAILURES — {best['config']} · {best['strategy']}")
    print("=" * 100)
    misses = 0
    for i, qid in enumerate(qids):
        chosen = strategy.select(ranked[i])
        predicted = [(s.doc_id, s.section_title) for s in chosen]
        if set(predicted) == set(expected[qid]):
            continue
        misses += 1
        entry = key[qid]
        ranks = section_ranks(ranked[i], expected[qid])
        print(f"  [{qid}] kind={entry.kind} confidence={entry.confidence}")
        for (doc, sec), rank in zip(expected[qid], ranks):
            print(f"      expected  : {doc} :: {sec}   [rank {rank}]")
        for doc, sec in predicted:
            flag = "" if (doc, sec) in expected[qid] else "  <-- wrong"
            print(f"      predicted : {doc} :: {sec}{flag}")
        print()
    print(f"  {misses} of {len(qids)} questions have inexact citations.")


if __name__ == "__main__":
    raise SystemExit(main())
