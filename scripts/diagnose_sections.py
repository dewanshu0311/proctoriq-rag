#!/usr/bin/env python
"""Where does naive semantic retrieval actually land?

    python scripts/diagnose_sections.py

Embeds each of the 50 test questions and all 53 section bodies with a local
sentence-transformers model, then for every question prints the top-5 sections by
cosine similarity, marking the section(s) the answer key expects and the rank
each landed at.

Why this exists
---------------
Three key entries are low-confidence (Q14, Q28, Q35) and eleven are medium. This
script replaces one person's reading of the corpus with evidence about the
corpus: if naive retrieval puts the key's expected section at rank 1 for a
low-confidence question, that is weak corroboration; if it puts a different
section at rank 1 with a large margin, that is a probe worth running.

It also establishes the baseline retrieval ceiling before any reranking, HyDE or
fusion exists — every later phase is measured against these numbers.

No API calls, no keys. The model is local and cached after first download.
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

DEFAULT_KEY_PATH = REPO_ROOT / "data" / "validation" / "answer_key.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--kb-dir", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--json", type=Path, default=None, help="also dump raw ranks to this path"
    )
    args = parser.parse_args()

    config = load_config()
    kb_dir = args.kb_dir or config.paths.kb_dir
    test_csv = args.test_csv or config.paths.test_csv
    model_name = args.model or config.embedding.diagnostic_model

    corpus = load_corpus(kb_dir)
    key = load_answer_key(args.key, corpus)
    questions = pd.read_csv(test_csv)

    sections = corpus.sections()
    labels = [f"{s.doc_id} :: {s.section_title}" for s in sections]
    # Embed header + body: the header carries real signal ("Common Installation
    # Errors"), and a body alone often reads as a bare list of steps.
    section_texts = [f"{s.section_title}\n{s.body}" for s in sections]

    print(f"Loading embedding model: {model_name}")
    print("(first run downloads ~90 MB, then it is cached; no API key involved)\n")
    from sentence_transformers import SentenceTransformer  # local import: heavy

    model = SentenceTransformer(model_name)

    section_vectors = model.encode(
        section_texts, normalize_embeddings=True, show_progress_bar=False
    )
    question_vectors = model.encode(
        questions["question"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Normalized vectors -> inner product is cosine similarity.
    similarity = np.asarray(question_vectors) @ np.asarray(section_vectors).T

    rule = "=" * 84
    print(rule)
    print(f"NAIVE SEMANTIC RETRIEVAL — {len(questions)} questions vs {len(sections)} sections")
    print(f"model: {model_name}")
    print(rule)

    expected_ranks: dict[str, list[int]] = {}
    dump: list[dict] = []

    for row_index, row in questions.iterrows():
        question_id = str(row["question_id"])
        question_text = str(row["question"])
        entry = key.get(question_id)
        expected_pairs = list(entry.pairs) if entry else []
        expected_labels = {f"{d} :: {s}" for d, s in expected_pairs}

        order = np.argsort(-similarity[row_index])
        rank_of = {labels[idx]: rank for rank, idx in enumerate(order, start=1)}

        ranks = sorted(rank_of[label] for label in expected_labels if label in rank_of)
        expected_ranks[question_id] = ranks

        marker = "" if not entry else f"  [{entry.kind}/{entry.confidence}]"
        print(f"\n[{question_id}]{marker} {question_text[:88]}")

        for rank, idx in enumerate(order[: args.top_k], start=1):
            label = labels[idx]
            hit = "<<< EXPECTED" if label in expected_labels else ""
            print(f"    {rank}. {similarity[row_index][idx]:.4f}  {label} {hit}")

        if not expected_labels:
            print("    (no answer-key entry for this question)")
        else:
            missed = [
                (label, rank_of[label])
                for label in sorted(expected_labels)
                if rank_of.get(label, 10**9) > args.top_k
            ]
            for label, rank in missed:
                print(f"    !! NOT IN TOP {args.top_k} — expected {label} at rank {rank}")

        dump.append(
            {
                "question_id": question_id,
                "kind": entry.kind if entry else None,
                "confidence": entry.confidence if entry else None,
                "expected": sorted(expected_labels),
                "expected_ranks": ranks,
                "top_k": [
                    {"rank": r, "score": float(similarity[row_index][i]), "section": labels[i]}
                    for r, i in enumerate(order[: args.top_k], start=1)
                ],
            }
        )

    _print_summary(key, expected_ranks, args.top_k)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dump, indent=2), encoding="utf-8")
        print(f"\nraw ranks written to {args.json}")

    return 0


def _print_summary(key, expected_ranks: dict[str, list[int]], top_k: int) -> None:
    rule = "=" * 84
    print()
    print(rule)
    print("SUMMARY — baseline retrieval ceiling before any reranking")
    print(rule)

    def recall_at(n: int) -> float:
        """Fraction of expected sections retrieved within the top n."""
        total = sum(len(r) for r in expected_ranks.values())
        hit = sum(sum(1 for rank in r if rank <= n) for r in expected_ranks.values())
        return hit / total if total else 0.0

    def all_at(n: int) -> float:
        """Fraction of questions where EVERY expected section is within the top n."""
        eligible = [r for r in expected_ranks.values() if r]
        hit = sum(1 for r in eligible if max(r) <= n)
        return hit / len(eligible) if eligible else 0.0

    for n in (1, 3, 5, 10):
        print(
            f"  section recall@{n:<3} {recall_at(n):>6.1%}      "
            f"all-expected-within-{n:<3} {all_at(n):>6.1%}"
        )

    print()
    print("  Low- and medium-confidence entries (the ones worth probing):")
    for confidence in ("low", "medium"):
        for entry in key.by_confidence(confidence):
            ranks = expected_ranks.get(entry.id, [])
            rendered = ", ".join(str(r) for r in ranks) or "n/a"
            verdict = "OK" if ranks and max(ranks) <= top_k else "MISS"
            print(
                f"    [{entry.id}] {confidence:<6} {verdict:<4} "
                f"expected section rank(s): {rendered}"
            )

    print()
    print(
        "  Read this as evidence about the corpus, not as a verdict on the key —\n"
        "  naive cosine has no notion of the phase disambiguation or the adversarial\n"
        "  framing that several of these questions turn on."
    )


if __name__ == "__main__":
    raise SystemExit(main())
