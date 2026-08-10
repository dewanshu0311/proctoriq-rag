#!/usr/bin/env python
"""Generate submission files locally, including the full probe set.

    python scripts/run_pipeline.py --probes      # all probe variants
    python scripts/run_pipeline.py               # baseline only

Writes to ``outputs/probes/`` so submissions can be diffed before anything is
uploaded, and **asserts that ``answer_text`` is byte-identical across every
probe** — the property the entire probe design rests on. If that assertion ever
fails, the probes are measuring two things at once and their deltas are
uninterpretable.

Nothing here uploads anything.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from proctoriq_rag.config import SectionFormat, SubmissionConfig, load_config  # noqa: E402
from proctoriq_rag.corpus.loader import load_corpus  # noqa: E402
from proctoriq_rag.pipeline import Pipeline, PipelineConfig  # noqa: E402
from proctoriq_rag.submission.writer import Prediction, write_submission  # noqa: E402

#: The probe sequence. Each entry names what it changes FROM THE CURRENT BASELINE
#: — the baseline is rebuilt after each format question is resolved, because
#: citation is scored on (doc, section) pairs and a wrong document format makes
#: every section probe read as "no effect". See docs/PROBES.md.
PROBES: list[tuple[str, dict]] = [
    ("p1-baseline", {}),
    ("p2-doc-extension", {"doc_extension": True}),
    ("p3-section-number", {"section_format": SectionFormat.NUMBER_ONLY}),
    ("p4-section-title", {"section_format": SectionFormat.TITLE_ONLY}),
    ("p5-topk2", {"citation_strategy": "topk-2"}),
    ("p6-topk3", {"citation_strategy": "topk-3"}),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", action="store_true", help="generate all probe variants")
    parser.add_argument("--mode", default="extractive", choices=["extractive", "generative"])
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "probes")
    args = parser.parse_args()

    config = load_config()
    corpus = load_corpus(config.paths.kb_dir)
    questions = pd.read_csv(config.paths.test_csv)
    qids = questions["question_id"].astype(str).tolist()
    qtexts = questions["question"].astype(str).tolist()

    # Rerank once. Every probe shares the same ranking; only serialization and
    # cardinality differ, which is exactly what makes the deltas attributable.
    print("reranking (shared across all probes) ...")
    base = Pipeline(corpus=corpus, config=PipelineConfig(generation_mode=args.mode))
    base.fit(qtexts)

    variants = PROBES if args.probes else PROBES[:1]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fingerprints: dict[str, str] = {}
    rows: list[dict] = []

    for name, changes in variants:
        pipeline_config = PipelineConfig(
            generation_mode=args.mode,
            citation_strategy=changes.get("citation_strategy", "topk-1"),
        )
        pipeline = Pipeline(
            corpus=corpus, config=pipeline_config,
            reranker=base.reranker, answerer=base.answerer, chunks=base.chunks,
        )
        predictions = []
        for index, (qid, question) in enumerate(zip(qids, qtexts)):
            citations = pipeline.citations_for(index)
            predictions.append(
                Prediction(
                    question_id=qid,
                    answer_text=pipeline.answerer.answer(question, citations),
                    cited_docs=[d for d, _ in citations],
                    cited_sections=[sec for _, sec in citations],
                )
            )

        submission_config = SubmissionConfig(
            doc_extension=changes.get("doc_extension", False),
            section_format=changes.get("section_format", SectionFormat.FULL_HEADER),
        )
        path = args.out_dir / f"submission_{name}.csv"
        write_submission(
            predictions, config.paths.test_csv, path, submission_config,
            sample_submission=config.paths.sample_submission,
        )

        answers = [p.answer_text for p in predictions]
        digest = hashlib.sha256("\x00".join(answers).encode("utf-8")).hexdigest()[:16]
        fingerprints[name] = digest
        rows.append({
            "probe": name,
            "doc_extension": submission_config.doc_extension,
            "section_format": submission_config.section_format.value,
            "citation_strategy": pipeline_config.citation_strategy,
            "mean_citations": sum(len(p.cited_docs) for p in predictions) / len(predictions),
            "answer_sha": digest,
            "file": path.name,
        })
        print(f"  {name:<20} -> {path.name}   answer_text sha={digest}")

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "probe_manifest.csv", index=False)

    print("\n" + "=" * 78)
    print("ANSWER-TEXT IDENTITY CHECK")
    print("=" * 78)
    unique = set(fingerprints.values())
    for name, digest in fingerprints.items():
        print(f"  {name:<20} {digest}")
    if len(unique) == 1:
        print("\n  PASS — answer_text is byte-identical across every probe.")
        print("  Any leaderboard delta is therefore attributable to the citation half alone.")
    else:
        print(f"\n  FAIL — {len(unique)} distinct answer sets. The probes are measuring")
        print("  two variables at once and their deltas are NOT interpretable.")
        return 1

    print(f"\n{len(frame)} submissions in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
