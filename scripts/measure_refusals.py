#!/usr/bin/env python
"""Does a reasoned refusal beat pasted policy text on the adversarial subset?

    python scripts/measure_refusals.py

The Phase 4 router A/B falsified the retrieval half of the Phase 2 adversarial
hypothesis: routing moved adversarial document-F1 by exactly zero. This script
tests the other half — that we already retrieve the right policy section and then
paste it verbatim instead of refusing.

Citations are held fixed at the locked config throughout, so only ``answer_text``
varies. That is the same discipline as probes 2-5 and it is what makes probe 6a
interpretable.

Reported per arm: groundedness proxy (with the D-007 caveat), whether the text
actually declines, whether it addresses the student, and the Q33 compound case
verbatim.
"""

from __future__ import annotations

import argparse
import json
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
from proctoriq_rag.generation.answerer import ExtractiveAnswerer  # noqa: E402
from proctoriq_rag.generation.groq_client import GroqChatClient, load_dotenv  # noqa: E402
from proctoriq_rag.generation.prompts import count_padding  # noqa: E402
from proctoriq_rag.generation.refusal import (  # noqa: E402
    RefusalAnswerer,
    addresses_student,
    looks_like_refusal,
)
from proctoriq_rag.pipeline import Pipeline, PipelineConfig  # noqa: E402
from proctoriq_rag.routing.router import QueryRouter  # noqa: E402
from proctoriq_rag.submission.writer import Prediction  # noqa: E402

KEY_PATH = REPO_ROOT / "data" / "validation" / "answer_key.yaml"
CACHE = REPO_ROOT / ".cache" / "router_decisions.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    config = load_config()
    corpus = load_corpus(config.paths.kb_dir)
    key = load_answer_key(KEY_PATH, corpus)

    questions = pd.read_csv(config.paths.test_csv)
    qids = questions["question_id"].astype(str).tolist()
    qtexts = questions["question"].astype(str).tolist()

    print("reranking ...")
    pipeline = Pipeline(corpus=corpus, config=PipelineConfig())
    pipeline.fit(qtexts)
    citations = [pipeline.citations_for(i) for i in range(len(qtexts))]

    client = GroqChatClient()
    router = QueryRouter(client=client, cache_path=CACHE)
    decisions = {d.question_id: d for d in router.classify_all(qids, qtexts)}

    extractive = ExtractiveAnswerer(corpus).fit_focuser(qtexts, pipeline.reranker)
    refuser = RefusalAnswerer(corpus=corpus, base=extractive, client=client)

    print("Loading embedder for the groundedness proxy ...")
    from sentence_transformers import SentenceTransformer

    scorer = Scorer(
        corpus, key, embedder=SentenceTransformer(config.embedding.diagnostic_model)
    )

    adversarial = {e.id for e in key if e.kind == "adversarial"}
    rows, answers = [], {}

    for arm in ("extractive", "refusal"):
        print(f"\n=== {arm} ===", flush=True)
        texts = []
        for i, qid in enumerate(qids):
            decision = decisions[qid]
            fire = decision.is_policy_boundary or qid in adversarial
            if arm == "refusal" and fire:
                texts.append(refuser.refuse(
                    qtexts[i], citations[i], compound=decision.intent == "compound"
                ))
            else:
                texts.append(extractive.answer(qtexts[i], citations[i]))
            if (i + 1) % 15 == 0:
                print(f"  {i + 1}/{len(qids)}", flush=True)
        answers[arm] = texts

        preds = [
            Prediction(qid, t, [d for d, _ in citations[i]], [s for _, s in citations[i]])
            for i, (qid, t) in enumerate(zip(qids, texts))
        ]
        report = scorer.score(preds)
        adv = [q for q in report.per_question if q.question_id in adversarial]
        adv_texts = [t for qid, t in zip(qids, texts) if qid in adversarial]

        rows.append({
            "arm": arm,
            "grounded_all": report.dimension_means["groundedness"],
            "grounded_adversarial": float(np.mean([q.groundedness for q in adv])),
            "declines": sum(looks_like_refusal(t) for t in adv_texts),
            "addresses_student": sum(addresses_student(t) for t in adv_texts),
            "mean_chars_adv": float(np.mean([len(t) for t in adv_texts])),
            "padding": sum(count_padding(t) for t in adv_texts),
            "n_adversarial": len(adv_texts),
        })

    frame = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "refusal_comparison.csv", index=False)
    (args.out_dir / "refusal_answers.json").write_text(
        json.dumps({"question_ids": qids, "answers": answers}, indent=2), encoding="utf-8"
    )

    _report(frame, qids, qtexts, answers, adversarial)
    return 0


def _report(frame, qids, qtexts, answers, adversarial) -> None:
    rule = "=" * 96
    print("\n" + rule)
    print("REFUSAL vs PASTED POLICY TEXT — adversarial subset")
    print(rule)
    print(f"  {'arm':<14} {'grounded(all)':>14} {'grounded(adv)':>14} {'declines':>10} "
          f"{'addresses':>11} {'chars':>7} {'padding':>8}")
    print("  " + "-" * 84)
    for _, row in frame.iterrows():
        n = int(row["n_adversarial"])
        print(f"  {row['arm']:<14} {row['grounded_all']:>14.4f} "
              f"{row['grounded_adversarial']:>14.4f} "
              f"{int(row['declines']):>7}/{n} {int(row['addresses_student']):>8}/{n} "
              f"{row['mean_chars_adv']:>7.0f} {int(row['padding']):>8}")

    print("\n  !! D-007 CAVEAT: groundedness uses a LOCAL embedding model; the grader's is")
    print("     unknown. Valid for ranking arms against each other, not as a score estimate.")
    print("     'declines' and 'addresses' are structural checks, not similarity — an answer")
    print("     that recites policy without declining scores 0 there regardless of similarity.")

    index = {q: i for i, q in enumerate(qids)}
    for qid in ("Q33", "Q29", "Q50"):
        if qid not in index:
            continue
        i = index[qid]
        print("\n" + rule)
        print(f"HAND CHECK — {qid}")
        print(rule)
        print(f"  Q: {qtexts[i]}\n")
        for arm, texts in answers.items():
            print(f"  [{arm}]")
            print(f"    {texts[i][:600]}{'...' if len(texts[i]) > 600 else ''}\n")


if __name__ == "__main__":
    raise SystemExit(main())
