#!/usr/bin/env python
"""Router A/B: does routing improve the citation half, or regress it?

    python scripts/sweep_router.py

The router is the first component in this project that can make retrieval
*worse*. Lookup document-F1 is already 1.000 locally, so on that class there is
nothing to gain and everything to lose. This script therefore reports router-off
versus router-on **broken down by kind**, and never nets the two halves against
each other — if routing helps adversarial while hurting lookup, the right answer
may be to route only the classes where it helps.

Also reports per-axis classification accuracy against the answer key, and the
alternate-key comparison as a standing column.
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
from proctoriq_rag.evaluation.alternates import load_alternates  # noqa: E402
from proctoriq_rag.evaluation.answer_key import load_answer_key  # noqa: E402
from proctoriq_rag.evaluation.scorer import Scorer  # noqa: E402
from proctoriq_rag.generation.groq_client import GroqChatClient, load_dotenv  # noqa: E402
from proctoriq_rag.retrieval.chunking import SectionChunker  # noqa: E402
from proctoriq_rag.retrieval.citation import TopK  # noqa: E402
from proctoriq_rag.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from proctoriq_rag.routing.policy import RoutingWeights, apply_routing, cardinality_strategy  # noqa: E402
from proctoriq_rag.routing.router import PHASE_DOCS, QueryRouter  # noqa: E402
from proctoriq_rag.submission.writer import Prediction  # noqa: E402

KEY_PATH = REPO_ROOT / "data" / "validation" / "answer_key.yaml"
ALT_PATH = REPO_ROOT / "data" / "validation" / "answer_key_alternates.yaml"
CACHE = REPO_ROOT / ".cache" / "router_decisions.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-llm", action="store_true", help="fallback classifier only")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    config = load_config()
    corpus = load_corpus(config.paths.kb_dir)
    key = load_answer_key(KEY_PATH, corpus)
    alternates = load_alternates(ALT_PATH, corpus, key)

    questions = pd.read_csv(config.paths.test_csv)
    qids = questions["question_id"].astype(str).tolist()
    qtexts = questions["question"].astype(str).tolist()

    chunks = SectionChunker(include_header_in_text=True).chunk(corpus)
    reranker = CrossEncoderReranker(
        config_get(config, "model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        score_transform="auto", text_variant="body",
    )
    print("reranking ...")
    reranker.fit(qtexts, chunks, corpus)
    base_ranked = [
        reranker.as_ranked_sections(reranker.rerank(i)) for i in range(len(qtexts))
    ]

    client = None if args.no_llm else GroqChatClient()
    router = QueryRouter(client=client, cache_path=CACHE, use_llm=not args.no_llm)
    print("classifying ...")
    decisions = router.classify_all(qids, qtexts)
    print(f"  {router.stats}")

    _report_accuracy(decisions, key, qids)
    frame = _report_ab(base_ranked, decisions, qids, corpus, key)
    _report_alternates(base_ranked, decisions, key, alternates, qids)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([d.as_dict() for d in decisions]).to_csv(
        args.out_dir / "router_decisions.csv", index=False
    )
    frame.to_csv(args.out_dir / "router_ab.csv", index=False)
    print(f"\nwritten to {args.out_dir / 'router_ab.csv'}")
    return 0


def config_get(config, name, default):
    return default


def _predictions(ranked_lists, decisions, qids, strategy=None):
    out = []
    for i, qid in enumerate(qids):
        sel = (strategy or cardinality_strategy(decisions[i])).select(ranked_lists[i])
        out.append(Prediction(qid, "x", [s.doc_id for s in sel],
                              [s.section_title for s in sel]))
    return out


def _report_accuracy(decisions, key, qids) -> None:
    """Per-axis accuracy against what the key implies."""
    rule = "=" * 92
    print("\n" + rule)
    print("ROUTER CLASSIFICATION ACCURACY")
    print(rule)

    by_id = {d.question_id: d for d in decisions}

    # intent: the key's `kind` maps adversarial -> policy_boundary/compound
    intent_ok = sum(
        1 for e in key
        if (e.kind == "adversarial") == by_id[e.id].is_policy_boundary
    )
    print(f"  intent (adversarial <-> policy_boundary): {intent_ok}/{len(key)} "
          f"= {intent_ok / len(key):.1%}")

    # platform: only the 9 questions whose citations are platform-specific
    plat_total = plat_ok = 0
    for e in key:
        docs = e.doc_set
        expected = None
        if docs == {"01_windows_installation_login_guide"}:
            expected = "windows"
        elif docs == {"02_mac_installation_login_guide"}:
            expected = "mac"
        if expected:
            plat_total += 1
            plat_ok += by_id[e.id].platform == expected
    print(f"  platform (on the {plat_total} platform-specific questions): "
          f"{plat_ok}/{plat_total} = {plat_ok / max(plat_total, 1):.1%}")

    # phase: by cited document group
    phase_total = phase_ok = 0
    for e in key:
        prefixes = {d[:2] for d in e.doc_set}
        labels = {p for p, ds in PHASE_DOCS.items() if prefixes & set(ds)}
        if len(labels) == 1:
            phase_total += 1
            phase_ok += by_id[e.id].phase in labels
    print(f"  phase (on the {phase_total} single-phase questions): "
          f"{phase_ok}/{phase_total} = {phase_ok / max(phase_total, 1):.1%}")

    card_ok = sum(1 for e in key if by_id[e.id].cardinality == min(len(e.docs), 2))
    print(f"  cardinality: {card_ok}/{len(key)} = {card_ok / len(key):.1%}")
    two = [e.id for e in key if len(e.docs) == 2]
    caught = [q for q in two if by_id[q].cardinality == 2]
    print(f"    of the {len(two)} genuinely two-source questions, "
          f"{len(caught)} identified: {', '.join(caught) or 'none'}")
    false_pos = [e.id for e in key if len(e.docs) == 1 and by_id[e.id].cardinality == 2]
    print(f"    false positives (single-source called two): {len(false_pos)} "
          f"{', '.join(false_pos[:8])}")


def _report_ab(base_ranked, decisions, qids, corpus, key) -> pd.DataFrame:
    scorer = Scorer(corpus, key)
    rule = "=" * 92
    rows = []

    arms = {
        "router OFF (topk-1)": (
            [apply_routing(r, d, RoutingWeights(mode="off"))
             for r, d in zip(base_ranked, decisions)], TopK(1)),
        "router ON  (bias, routed cardinality)": (
            [apply_routing(r, d, RoutingWeights(mode="bias"))
             for r, d in zip(base_ranked, decisions)], None),
        "router ON  (bias, topk-1)": (
            [apply_routing(r, d, RoutingWeights(mode="bias"))
             for r, d in zip(base_ranked, decisions)], TopK(1)),
        "router ON  (filter, topk-1)": (
            [apply_routing(r, d, RoutingWeights(mode="filter"))
             for r, d in zip(base_ranked, decisions)], TopK(1)),
        # Selective: take the cardinality decision, drop the score biasing.
        # The by-kind split showed biasing gains nothing on adversarial and
        # costs a lookup question, while cardinality is the only real win.
        "router ON  (cardinality only, NO bias)": (
            [apply_routing(r, d, RoutingWeights(mode="off"))
             for r, d in zip(base_ranked, decisions)], None),
    }

    reports = {}
    for name, (ranked, strategy) in arms.items():
        preds = _predictions(ranked, decisions, qids, strategy)
        report = scorer.score(preds)
        reports[name] = report
        rows.append({
            "arm": name,
            "doc_f1": report.dimension_means["retrieval"],
            "cite_f1": report.dimension_means["citation"],
            "cite_exact": float(np.mean(
                [q.citation.exact_match for q in report.per_question])),
            "weighted_citation_half": (
                20 * report.dimension_means["retrieval"]
                + 15 * report.dimension_means["citation"]),
            "mean_citations": float(np.mean([len(p.cited_docs) for p in preds])),
        })

    print("\n" + rule)
    print("ROUTER A/B — the citation half, on the real 20/15 weighting")
    print(rule)
    print(f"  {'arm':<40} {'doc_F1':>8} {'cite_F1':>8} {'cite_ex':>8} "
          f"{'/35':>7} {'cites':>6}")
    print("  " + "-" * 82)
    baseline = rows[0]["weighted_citation_half"]
    for row in rows:
        delta = row["weighted_citation_half"] - baseline
        print(f"  {row['arm']:<40} {row['doc_f1']:>8.4f} {row['cite_f1']:>8.4f} "
              f"{row['cite_exact']:>8.4f} {row['weighted_citation_half']:>7.2f} "
              f"{row['mean_citations']:>6.2f}"
              + (f"   {delta:+.2f}" if delta else ""))

    print("\n" + rule)
    print("BY KIND — reported separately, never netted")
    print(rule)
    kinds = sorted(reports["router OFF (topk-1)"].breakdown("kind"))
    for name in arms:
        table = reports[name].breakdown("kind")
        print(f"\n  {name}")
        print(f"    {'kind':<16} {'n':>3} {'doc_F1':>8} {'cite_F1':>8}")
        for kind in kinds:
            stats = table[kind]
            print(f"    {kind:<16} {int(stats['n']):>3} {stats['doc_f1']:>8.4f} "
                  f"{stats['cite_f1']:>8.4f}")

    off = reports["router OFF (topk-1)"].breakdown("kind")
    on = reports["router ON  (bias, routed cardinality)"].breakdown("kind")
    print("\n  DELTA (routed - off), by kind:")
    for kind in kinds:
        d_doc = on[kind]["doc_f1"] - off[kind]["doc_f1"]
        d_cite = on[kind]["cite_f1"] - off[kind]["cite_f1"]
        flag = "  <-- REGRESSION" if (d_doc < -1e-9 or d_cite < -1e-9) else ""
        print(f"    {kind:<16} doc {d_doc:+.4f}   cite {d_cite:+.4f}{flag}")

    return pd.DataFrame(rows)


def _report_alternates(base_ranked, decisions, key, alternates, qids) -> None:
    """Standing column from now on: does the router prefer an alternate reading?"""
    from proctoriq_rag.retrieval.citation import section_ranks

    rule = "=" * 92
    print("\n" + rule)
    print("ALTERNATE-KEY COMPARISON (standing column)")
    print(rule)
    print("  The router is the first component that reasons about INTENT rather than")
    print("  similarity. If it independently prefers an alternate reading, that is a")
    print("  third signal. Still evidence, still the author's call.\n")

    index = {q: i for i, q in enumerate(qids)}
    for qid, readings in alternates.items():
        i = index[qid]
        routed = apply_routing(base_ranked[i], decisions[i], RoutingWeights(mode="bias"))
        entry = key[qid]
        d = decisions[i]
        primary = max(section_ranks(routed, list(entry.pairs)))
        print(f"  [{qid}] router: intent={d.intent} phase={d.phase} card={d.cardinality}")
        print(f"      primary worst-rank {primary}")
        for reading in readings:
            alt = max(section_ranks(routed, list(reading.pairs)))
            verdict = "BETTER" if alt < primary else ("same" if alt == primary else "worse")
            print(f"      alt {reading.label:<28} worst-rank {alt:>2}  [{verdict}]")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
