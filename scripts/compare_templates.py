#!/usr/bin/env python
"""Compare candidate prompt templates on all 50 questions.

    python scripts/compare_templates.py

Runs each template through Groq, scores the answers with the Phase 0 scorer's
groundedness proxy, and reports mean answer length and padding-phrase counts
alongside.

READ THE CAVEAT BEFORE READING THE NUMBERS (D-007)
---------------------------------------------------
Groundedness here is cosine similarity between the generated answer and the
concatenated body of the answer key's cited sections, using a LOCAL embedding
model. The competition grader's embedding model is unknown. So these numbers are
valid for comparing template A against template B, and tell us nothing about what
any of them would score on the leaderboard. A groundedness of 0.82 does not mean
"we would score 82 on that dimension".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from proctoriq_rag.config import load_config  # noqa: E402
from proctoriq_rag.corpus.loader import load_corpus  # noqa: E402
from proctoriq_rag.evaluation.answer_key import load_answer_key  # noqa: E402
from proctoriq_rag.evaluation.scorer import Scorer  # noqa: E402
from proctoriq_rag.generation.answerer import ExtractiveAnswerer, GroqAnswerer  # noqa: E402
from proctoriq_rag.generation.groq_client import GroqChatClient, load_dotenv  # noqa: E402
from proctoriq_rag.generation.prompts import TEMPLATES, count_padding  # noqa: E402
from proctoriq_rag.pipeline import Pipeline, PipelineConfig  # noqa: E402

KEY_PATH = REPO_ROOT / "data" / "validation" / "answer_key.yaml"
TRAP_QUESTION_HINT = "mock test right before"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    config = load_config()
    corpus = load_corpus(config.paths.kb_dir)
    key = load_answer_key(KEY_PATH, corpus)

    questions = pd.read_csv(config.paths.test_csv)
    if args.limit:
        questions = questions.head(args.limit)
    qids = questions["question_id"].astype(str).tolist()
    qtexts = questions["question"].astype(str).tolist()

    print("Loading embedder for the groundedness proxy ...")
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(config.embedding.diagnostic_model)
    scorer = Scorer(corpus, key, embedder=embedder)

    # Citations are identical across templates — only the answer text varies, so
    # any groundedness difference is attributable to the prompt alone.
    print("Reranking once (citations held constant across templates) ...")
    pipeline = Pipeline(corpus=corpus, config=PipelineConfig(generation_mode="extractive"))
    pipeline.fit(qtexts)
    citations = [pipeline.citations_for(i) for i in range(len(qtexts))]

    client = GroqChatClient()
    rows: list[dict] = []
    answers_by_template: dict[str, list[str]] = {}

    variants = [("extractive", None)] + [(t.name, t) for t in TEMPLATES]

    for name, template in variants:
        print(f"\n=== {name} ===", flush=True)
        started = time.time()
        if template is None:
            answerer = ExtractiveAnswerer(corpus)
        else:
            answerer = GroqAnswerer(corpus, client=client, template_name=template.name)

        answers: list[str] = []
        for index, (qid, question) in enumerate(zip(qids, qtexts)):
            answers.append(answerer.answer(question, citations[index]))
            if (index + 1) % 10 == 0:
                print(f"  {index + 1}/{len(qtexts)}", flush=True)
        answers_by_template[name] = answers

        predictions = [
            _prediction(qid, answer, citations[i])
            for i, (qid, answer) in enumerate(zip(qids, answers))
        ]
        report = scorer.score(predictions)

        rows.append({
            "template": name,
            "groundedness": report.dimension_means["groundedness"],
            "mean_chars": float(np.mean([len(a) for a in answers])),
            "mean_words": float(np.mean([len(a.split()) for a in answers])),
            "padding_hits": int(sum(count_padding(a) for a in answers)),
            "empty_answers": int(sum(1 for a in answers if not a.strip())),
            "seconds": round(time.time() - started, 1),
        })
        print(f"  groundedness={rows[-1]['groundedness']:.4f}  "
              f"chars={rows[-1]['mean_chars']:.0f}  padding={rows[-1]['padding_hits']}")

    frame = pd.DataFrame(rows).sort_values("groundedness", ascending=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "template_comparison.csv", index=False)
    (args.out_dir / "template_answers.json").write_text(
        json.dumps({"question_ids": qids, "answers": answers_by_template}, indent=2),
        encoding="utf-8",
    )

    _report(frame, qids, qtexts, answers_by_template)
    return 0


def _prediction(qid, answer, citation_pairs):
    from proctoriq_rag.submission.writer import Prediction

    return Prediction(
        question_id=qid,
        answer_text=answer,
        cited_docs=[d for d, _ in citation_pairs],
        cited_sections=[s for _, s in citation_pairs],
    )


def _report(frame, qids, qtexts, answers_by_template) -> None:
    rule = "=" * 92
    print("\n" + rule)
    print("TEMPLATE COMPARISON")
    print(rule)
    print(f"  {'template':<26} {'grounded':>9} {'chars':>7} {'words':>7} "
          f"{'padding':>8} {'empty':>6} {'secs':>7}")
    print("  " + "-" * 76)
    for _, row in frame.iterrows():
        print(f"  {row['template']:<26} {row['groundedness']:>9.4f} "
              f"{row['mean_chars']:>7.0f} {row['mean_words']:>7.0f} "
              f"{row['padding_hits']:>8} {row['empty_answers']:>6} {row['seconds']:>7.1f}")

    print("\n  !! D-007 CAVEAT: groundedness uses a LOCAL embedding model; the grader's is")
    print("     unknown. These numbers rank templates against each other. Their absolute")
    print("     values say nothing about leaderboard score.")

    # ── the Q15 nuance trap, checked by hand ───────────────────────────────
    trap_index = next(
        (i for i, q in enumerate(qtexts) if TRAP_QUESTION_HINT in q.lower()), None
    )
    if trap_index is None:
        return
    print("\n" + rule)
    print(f"NUANCE TRAP — {qids[trap_index]}")
    print(rule)
    print(f"  Q: {qtexts[trap_index]}")
    print("  Source (doc 03 §2): the 24-hour window is \"not a hard technical requirement\"")
    print("  but is strongly advised. An answer that says \"no, you must\" contradicts the")
    print("  passage it cites and scores badly against it.\n")
    for name, answers in answers_by_template.items():
        text = answers[trap_index]
        flat = any(p in text.lower() for p in (
            "you must", "not allowed", "cannot do", "is required", "mandatory", "not permitted"
        ))
        nuanced = any(p in text.lower() for p in (
            "not a hard", "strongly", "recommend", "advis", "should", "best"
        ))
        verdict = "FLAT PROHIBITION" if (flat and not nuanced) else (
            "nuance preserved" if nuanced else "unclear")
        print(f"  [{name}] -> {verdict}")
        print(f"      {text[:400]}{'...' if len(text) > 400 else ''}\n")


if __name__ == "__main__":
    raise SystemExit(main())
