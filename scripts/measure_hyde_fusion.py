#!/usr/bin/env python
"""Do HyDE and RAG Fusion help on this corpus?

    python scripts/measure_hyde_fusion.py

Both are mandated components. Both are implemented properly and measured
honestly. The prediction on record, made before running: on a 4,000-word corpus
where recall@10 is already 93.75%, neither should move retrieval much.

Reported per arm: recall@1/3/5/10, document F1, citation F1, and the
``policy_boundary`` subset as **its own row** — an effect on 14 questions can
disappear inside a mean over 50, and policy-boundary questions are where the
question-vocabulary / answer-vocabulary gap is widest and HyDE should help most.

If the answer is "no effect", that is the finding. A documented negative with
numbers is a stronger result than a contrived use case.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from proctoriq_rag.config import load_config  # noqa: E402
from proctoriq_rag.corpus.loader import load_corpus  # noqa: E402
from proctoriq_rag.evaluation.answer_key import load_answer_key  # noqa: E402
from proctoriq_rag.evaluation.scorer import Scorer  # noqa: E402
from proctoriq_rag.generation.groq_client import GroqChatClient, load_dotenv  # noqa: E402
from proctoriq_rag.retrieval.chunking import SectionChunker  # noqa: E402
from proctoriq_rag.retrieval.citation import TopK, section_ranks  # noqa: E402
from proctoriq_rag.retrieval.embeddings import (  # noqa: E402
    EmbeddingCache,
    SentenceTransformerBackend,
)
from proctoriq_rag.retrieval.fusion import RAGFusion  # noqa: E402
from proctoriq_rag.retrieval.hyde import HyDERetriever  # noqa: E402
from proctoriq_rag.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from proctoriq_rag.retrieval.retriever import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    minmax,
)
from proctoriq_rag.routing.router import QueryRouter  # noqa: E402
from proctoriq_rag.submission.writer import Prediction  # noqa: E402

KEY_PATH = REPO_ROOT / "data" / "validation" / "answer_key.yaml"
CACHE = REPO_ROOT / ".cache" / "router_decisions.json"
DEPTHS = (1, 3, 5, 10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    config = load_config()
    corpus = load_corpus(config.paths.kb_dir)
    key = load_answer_key(KEY_PATH, corpus)
    scorer = Scorer(corpus, key)

    questions = pd.read_csv(config.paths.test_csv)
    qids = questions["question_id"].astype(str).tolist()
    qtexts = questions["question"].astype(str).tolist()
    expected = {q: list(key[q].pairs) for q in qids}

    chunks = SectionChunker(include_header_in_text=True).chunk(corpus)
    cache = EmbeddingCache()
    backend = SentenceTransformerBackend(config.embedding.diagnostic_model)
    qvecs = cache.encode(backend, qtexts, fingerprint="test-questions", kind="queries")
    cvecs = cache.encode(backend, [c.text for c in chunks],
                         fingerprint="section-hdr", kind="documents")

    dense = DenseRetriever(chunks, cvecs, qvecs)
    sparse = BM25Retriever(chunks, qtexts)
    hybrid = HybridRetriever(dense, sparse, alpha=0.7)

    print("reranking (the shipped baseline) ...")
    reranker = CrossEncoderReranker(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", score_transform="auto", text_variant="body"
    )
    reranker.fit(qtexts, chunks, corpus)

    client = GroqChatClient()
    router = QueryRouter(client=client, cache_path=CACHE)
    decisions = {d.question_id: d for d in router.classify_all(qids, qtexts)}
    boundary = {q for q in qids if decisions[q].is_policy_boundary}
    print(f"policy_boundary subset: {len(boundary)} questions")

    print("HyDE: generating hypothetical answers ...", flush=True)
    hyde = HyDERetriever(client=client, embedder=backend, chunks=chunks,
                         chunk_vectors=cvecs)
    hyde_scores = []
    for i, q in enumerate(qtexts):
        hyde_scores.append(hyde.scores(q))
        if (i + 1) % 15 == 0:
            print(f"  {i + 1}/{len(qtexts)}", flush=True)
    print(f"  fallbacks: {hyde.fallbacks}")

    print("RAG Fusion: generating query variants ...", flush=True)
    q_index = {q: i for i, q in enumerate(qtexts)}

    def hybrid_score(query: str) -> np.ndarray:
        """Score a query string. Known questions reuse the cached hybrid row."""
        if query in q_index:
            return hybrid.scores(q_index[query])
        vec = np.asarray(backend.encode_queries([query]))[0]
        d = minmax(np.asarray(cvecs) @ vec)
        s = minmax(np.asarray(sparse.bm25.get_scores(query.lower().split()),
                              dtype="float32"))
        return 0.7 * d + 0.3 * s

    fusion = RAGFusion(client=client, score_fn=hybrid_score, chunks=chunks, n_queries=3)
    fusion_scores = []
    for i, q in enumerate(qtexts):
        fusion_scores.append(fusion.scores(q))
        if (i + 1) % 15 == 0:
            print(f"  {i + 1}/{len(qtexts)}", flush=True)
    print(f"  fallbacks: {fusion.fallbacks}")

    arms = {
        "hybrid (Phase 1 first stage)": [hybrid.scores(i) for i in range(len(qtexts))],
        "HyDE": hyde_scores,
        "RAG Fusion": fusion_scores,
        "cross-encoder (SHIPPED)": [reranker.scores()[i] for i in range(len(qtexts))],
    }

    rows = []
    for name, score_lists in arms.items():
        rows.append(_evaluate(name, score_lists, chunks, qids, expected, boundary, scorer))

    frame = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "hyde_fusion.csv", index=False)
    _report(frame)
    return 0


def _evaluate(name, score_lists, chunks, qids, expected, boundary, scorer) -> dict:
    from proctoriq_rag.retrieval.citation import RankedSection

    ranked_per_q, preds = [], []
    for i, qid in enumerate(qids):
        order = np.argsort(-np.asarray(score_lists[i]))
        seen, sections = set(), []
        for idx in order:
            citation = chunks[idx].citation
            if citation in seen:
                continue
            seen.add(citation)
            sections.append(RankedSection(citation[0], citation[1],
                                          float(score_lists[i][idx]), 1))
        ranked_per_q.append(sections)
        chosen = TopK(1).select(sections)
        preds.append(Prediction(qid, "x", [s.doc_id for s in chosen],
                                [s.section_title for s in chosen]))

    ranks = [section_ranks(ranked_per_q[i], expected[q]) for i, q in enumerate(qids)]
    b_ranks = [r for r, q in zip(ranks, qids) if q in boundary]
    report = scorer.score(preds)
    b_report = scorer.score([p for p in preds if p.question_id in boundary])

    row = {"arm": name,
           "doc_f1": report.dimension_means["retrieval"],
           "cite_f1": report.dimension_means["citation"],
           "boundary_doc_f1": b_report.dimension_means["retrieval"],
           "boundary_cite_f1": b_report.dimension_means["citation"]}
    for d in DEPTHS:
        row[f"recall@{d}"] = _recall(ranks, d)
        row[f"boundary_recall@{d}"] = _recall(b_ranks, d)
    return row


def _recall(ranks, depth) -> float:
    total = sum(len(r) for r in ranks)
    if not total:
        return 0.0
    return sum(sum(1 for x in r if x <= depth) for r in ranks) / total


def _report(frame) -> None:
    rule = "=" * 100
    print("\n" + rule)
    print("HyDE / RAG FUSION vs THE SHIPPED PIPELINE — all 50 questions")
    print(rule)
    print(f"  {'arm':<30} {'r@1':>7} {'r@3':>7} {'r@5':>7} {'r@10':>7} "
          f"{'doc_F1':>8} {'cite_F1':>8}")
    print("  " + "-" * 82)
    for _, r in frame.iterrows():
        print(f"  {r['arm']:<30} {r['recall@1']:>7.4f} {r['recall@3']:>7.4f} "
              f"{r['recall@5']:>7.4f} {r['recall@10']:>7.4f} "
              f"{r['doc_f1']:>8.4f} {r['cite_f1']:>8.4f}")

    print("\n" + rule)
    print("POLICY_BOUNDARY SUBSET — its own row, where HyDE should help most")
    print(rule)
    print(f"  {'arm':<30} {'r@1':>7} {'r@3':>7} {'r@5':>7} {'r@10':>7} "
          f"{'doc_F1':>8} {'cite_F1':>8}")
    print("  " + "-" * 82)
    for _, r in frame.iterrows():
        print(f"  {r['arm']:<30} {r['boundary_recall@1']:>7.4f} "
              f"{r['boundary_recall@3']:>7.4f} {r['boundary_recall@5']:>7.4f} "
              f"{r['boundary_recall@10']:>7.4f} {r['boundary_doc_f1']:>8.4f} "
              f"{r['boundary_cite_f1']:>8.4f}")

    base = frame[frame["arm"].str.startswith("hybrid")].iloc[0]
    print("\n  vs the hybrid first stage they replace:")
    for _, r in frame.iterrows():
        if r["arm"] == base["arm"]:
            continue
        print(f"    {r['arm']:<30} recall@10 {r['recall@10'] - base['recall@10']:+.4f}   "
              f"boundary recall@10 {r['boundary_recall@10'] - base['boundary_recall@10']:+.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
