#!/usr/bin/env python
"""Run the RAG Triad over all 50 questions, for both selected arms.

    python scripts/run_triad.py

Mandated component. Also the only evaluation in this project that does NOT consult
the answer key — every other measurement scores against a hand-built holdout and
inherits its opinions, including on the four contested entries. The triad judges
the pipeline on its own terms.

Reported by kind, because the interesting comparison is adversarial: the whole
argument for shipping refusals is that pasted policy text scores badly on ANSWER
RELEVANCY (it does not respond to the student) while remaining perfectly faithful.
If that shows up here, the triad independently corroborates a decision that the
leaderboard could not resolve.
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
from proctoriq_rag.evaluation.triad import Eval  # noqa: E402
from proctoriq_rag.generation.answerer import ExtractiveAnswerer  # noqa: E402
from proctoriq_rag.generation.groq_client import GroqChatClient, load_dotenv  # noqa: E402
from proctoriq_rag.generation.refusal import RefusalAnswerer  # noqa: E402
from proctoriq_rag.pipeline import Pipeline, PipelineConfig  # noqa: E402
from proctoriq_rag.routing.router import QueryRouter  # noqa: E402

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

    contexts = [
        "\n\n".join(
            f"[{d} - {s}]\n{corpus.get_section(d, s).body}"
            for d, s in cits if corpus.get_section(d, s)
        )
        for cits in citations
    ]

    client = GroqChatClient()
    router = QueryRouter(client=client, cache_path=CACHE)
    decisions = {d.question_id: d for d in router.classify_all(qids, qtexts)}

    # Slot 2 arm: plain extractive. Slot 1 arm: refusals + subsection fix.
    slot2 = ExtractiveAnswerer(corpus)
    slot1_base = ExtractiveAnswerer(corpus).fit_focuser(qtexts, pipeline.reranker)
    refuser = RefusalAnswerer(corpus=corpus, base=slot1_base, client=client)

    arms: dict[str, list[str]] = {}
    arms["slot2 extractive (79.27)"] = [
        slot2.answer(qtexts[i], citations[i]) for i in range(len(qtexts))
    ]

    print("generating slot 1 answers ...", flush=True)
    slot1 = []
    for i, qid in enumerate(qids):
        decision = decisions[qid]
        if decision.is_policy_boundary:
            variant = "compound" if decision.intent == "compound" else "pure_boundary"
            slot1.append(refuser.answer_with_variant(
                qtexts[i], citations[i], variant, question_id=qid))
        else:
            slot1.append(slot1_base.answer(qtexts[i], citations[i]))
        if (i + 1) % 15 == 0:
            print(f"  {i + 1}/{len(qids)}", flush=True)
    arms["slot1 refusals+subsection (80.15)"] = slot1

    kinds = {e.id: e.kind for e in key}
    rows = []
    for name, answers in arms.items():
        print(f"\njudging: {name}", flush=True)
        evaluator = Eval(client=client)
        scores = evaluator.evaluate_all(qids, qtexts, contexts, answers)
        for s in scores:
            rows.append({
                "arm": name, "question_id": s.question_id, "kind": kinds[s.question_id],
                "context_relevancy": s.context_relevancy,
                "faithfulness": s.faithfulness,
                "answer_relevancy": s.answer_relevancy,
                "mean": s.mean,
            })
        print(f"  unparseable judge replies: {len(evaluator.unparseable)}")

    frame = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "rag_triad.csv", index=False)
    _report(frame)
    return 0


def _report(frame: pd.DataFrame) -> None:
    rule = "=" * 96
    cols = ["context_relevancy", "faithfulness", "answer_relevancy", "mean"]

    print("\n" + rule)
    print("RAG TRIAD — all 50 questions")
    print(rule)
    print(f"  {'arm':<36} {'context':>9} {'faithful':>10} {'answer rel':>12} {'mean':>7}")
    print("  " + "-" * 78)
    for arm, group in frame.groupby("arm", sort=False):
        m = group[cols].mean()
        print(f"  {arm:<36} {m['context_relevancy']:>9.3f} {m['faithfulness']:>10.3f} "
              f"{m['answer_relevancy']:>12.3f} {m['mean']:>7.3f}")

    print("\n" + rule)
    print("BY KIND — adversarial is the comparison that matters")
    print(rule)
    for arm, group in frame.groupby("arm", sort=False):
        print(f"\n  {arm}")
        print(f"    {'kind':<16} {'n':>3} {'context':>9} {'faithful':>10} {'answer rel':>12}")
        for kind, sub in group.groupby("kind"):
            m = sub[cols].mean()
            print(f"    {kind:<16} {len(sub):>3} {m['context_relevancy']:>9.3f} "
                  f"{m['faithfulness']:>10.3f} {m['answer_relevancy']:>12.3f}")

    adv = frame[frame["kind"] == "adversarial"]
    if len(adv["arm"].unique()) == 2:
        a, b = list(adv["arm"].unique())
        da = adv[adv["arm"] == a][cols].mean()
        db = adv[adv["arm"] == b][cols].mean()
        print(f"\n  adversarial delta ({b} minus {a}):")
        for c in cols:
            print(f"    {c:<20} {db[c] - da[c]:+.3f}")

    print("\n  !! This is an LLM judge. Scores are ordinal, not calibrated — valid for")
    print("     comparing arms and finding the worst questions, not as absolute quality.")
    print("     Same caution as D-007 on the groundedness proxy.")


if __name__ == "__main__":
    raise SystemExit(main())
