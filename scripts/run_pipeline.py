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

#: Probes 1-5 are RESOLVED and kept only as a historical record — rerunning them
#: is how the locked config was reached, not something to repeat. The format
#: baseline now comes from config/default.yaml, which is locked.
#:
#: Probes 6a and 6b vary ONLY answer_text, with citations frozen at the locked
#: config, mirroring how 2-5 varied only citations. See docs/PROBES.md.
HISTORICAL_PROBES: list[tuple[str, dict]] = [
    ("h2-doc-extension", {"doc_extension": True}),
    ("h4-section-title", {"section_format": SectionFormat.TITLE_ONLY}),
    ("h5-topk2", {"citation_strategy": "topk-2"}),
]

PROBES: list[tuple[str, dict]] = [
    ("p1-locked", {}),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", action="store_true", help="generate all probe variants")
    parser.add_argument("--historical", action="store_true",
                        help="also regenerate the resolved probes 2-5 (record only)")
    parser.add_argument("--probe-6a", action="store_true",
                        help="answer-half probe: refusals on router-classified "
                             "boundary questions, citations frozen")
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

    if args.probe_6a:
        return _probe_6a(config, corpus, qids, qtexts, base, args.out_dir)

    variants = list(PROBES) if args.probes else PROBES[:1]
    if args.historical:
        variants += HISTORICAL_PROBES
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

        # Defaults come from the LOCKED config, not from literals — probes 2-5
        # resolved these and a hardcoded default here would silently re-break them.
        submission_config = SubmissionConfig(
            doc_extension=changes.get("doc_extension", config.submission.doc_extension),
            section_format=changes.get("section_format", config.submission.section_format),
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


def _probe_6a(config, corpus, qids, qtexts, base, out_dir: Path) -> int:
    """Answer-half probe: only answer_text varies, citations frozen.

    Asserts the citation columns are byte-identical to the locked baseline before
    writing. If they are not, the probe measures two things at once and must not
    be submitted — the same guarantee the answer-hash check gave probes 2-5,
    pointed the other way.
    """
    from proctoriq_rag.generation.groq_client import GroqChatClient, load_dotenv
    from proctoriq_rag.generation.refusal import RefusalAnswerer
    from proctoriq_rag.routing.router import QueryRouter

    load_dotenv(REPO_ROOT / ".env")
    client = GroqChatClient()
    router = QueryRouter(
        client=client, cache_path=REPO_ROOT / ".cache" / "router_decisions.json"
    )
    decisions = {d.question_id: d for d in router.classify_all(qids, qtexts)}

    # NOTE: fit_focuser() is deliberately NOT called here.
    #
    # The cross-encoder subsection fix changes answer_text on exactly one
    # question (Q03, which was receiving the Session Start Error passage instead
    # of Unspecified Error). Applying it here would make probe 6a vary TWO things
    # against the 79.27 submission — refusals on 19 questions AND a subsection
    # correction on a 20th — and the delta would no longer be attributable.
    #
    # The fix is real and wanted; it ships in a later submission, isolated.
    refuser = RefusalAnswerer(corpus=corpus, base=base.answerer, client=client)

    citations = [base.citations_for(i) for i in range(len(qtexts))]
    fired, predictions = [], []
    for i, qid in enumerate(qids):
        decision = decisions[qid]
        if decision.is_policy_boundary:
            variant = "compound" if decision.intent == "compound" else "pure_boundary"
            text = refuser.answer_with_variant(qtexts[i], citations[i], variant)
            fired.append(qid)
        else:
            text = base.answerer.answer(qtexts[i], citations[i])
        predictions.append(Prediction(
            question_id=qid, answer_text=text,
            cited_docs=[d for d, _ in citations[i]],
            cited_sections=[sec for _, sec in citations[i]],
        ))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "submission_p6a-refusals.csv"
    write_submission(predictions, config.paths.test_csv, path, config.submission,
                     sample_submission=config.paths.sample_submission)

    print("\n  refusal fired on %d questions: %s" % (len(fired), ", ".join(fired)))

    baseline = out_dir / "submission_p1-locked.csv"
    if baseline.exists():
        old, new = pd.read_csv(baseline), pd.read_csv(path)
        same = (old["cited_docs"].equals(new["cited_docs"])
                and old["cited_sections"].equals(new["cited_sections"]))
        changed = int((old["answer_text"] != new["answer_text"]).sum())
        print("\n" + "=" * 78)
        print("CITATION-IDENTITY CHECK")
        print("=" * 78)
        print("  citation columns identical to locked baseline: %s" % same)
        print("  answer_text rows changed: %d" % changed)
        if not same:
            print("\n  FAIL - citations moved. This probe would measure two things.")
            return 1
        print("\n  PASS - only answer_text varies. Any delta is the answer half.")
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
