#!/usr/bin/env python
"""Sweep answer configurations, judged by the RAG Triad.

    python scripts/sweep_answers.py

**Why the Triad and not the groundedness proxy.** The proxy measures cosine
similarity to the source excerpt, so extraction wins it *by construction* — an
extractive answer IS the excerpt. It also covers only the 25% groundedness
dimension. The Triad's **answer relevancy** measures whether the text answers the
question asked, which is what the 25% answer-accuracy dimension actually rewards,
and it never consults the answer key.

**Citations are frozen at the locked config for every arm.** Only ``answer_text``
varies, so any arm here is directly submittable as a probe.

What this sweep is chasing
--------------------------
The Triad on the shipped extractive arm, by kind:

    lookup          n=29   answer relevancy 0.776
    adversarial     n=14   answer relevancy 0.493   <- refusals fix this (-> 0.871)
    multi_doc       n=5    answer relevancy 0.360   <- UNADDRESSED
    multi_section   n=1    answer relevancy 0.200   <- UNADDRESSED

The multi-part questions are the worst per-question failure in the system, and the
cause is structural: ``answer_from="top1"`` answers one half of a two-part
question. That was frozen in Phase 3 so probes could vary citations cleanly.
Widening the answer's *context* to the top-2 reranked sections fixes it while
leaving the citation columns byte-identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from proctoriq_rag.config import load_config  # noqa: E402
from proctoriq_rag.corpus.loader import load_corpus  # noqa: E402
from proctoriq_rag.evaluation.answer_key import load_answer_key  # noqa: E402
from proctoriq_rag.evaluation.triad import Eval  # noqa: E402
from proctoriq_rag.generation.answerer import ExtractiveAnswerer, GroqAnswerer  # noqa: E402
from proctoriq_rag.generation.groq_client import GroqChatClient, load_dotenv  # noqa: E402
from proctoriq_rag.generation.prompts import count_padding  # noqa: E402
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
    kinds = {e.id: e.kind for e in key}

    questions = pd.read_csv(config.paths.test_csv)
    qids = questions["question_id"].astype(str).tolist()
    qtexts = questions["question"].astype(str).tolist()

    print("reranking ...")
    pipeline = Pipeline(corpus=corpus, config=PipelineConfig())
    pipeline.fit(qtexts)
    citations = [pipeline.citations_for(i) for i in range(len(qtexts))]

    # Top-3 reranked sections per question — the widened ANSWER context. Citations
    # remain whatever the citation strategy chose (topk-1), untouched.
    ranked = [
        [s.citation for s in pipeline.reranker.as_ranked_sections(
            pipeline.reranker.rerank(i, top_k=3))]
        for i in range(len(qtexts))
    ]

    client = GroqChatClient()
    router = QueryRouter(client=client, cache_path=CACHE)
    decisions = {d.question_id: d for d in router.classify_all(qids, qtexts)}

    base = ExtractiveAnswerer(corpus).fit_focuser(qtexts, pipeline.reranker)

    def extractive(answer_from: str, max_chars: int = 700):
        a = ExtractiveAnswerer(corpus, max_chars=max_chars, answer_from=answer_from)
        a.fit_focuser(qtexts, pipeline.reranker)
        return lambda i: a.answer(qtexts[i], citations[i], ranked[i])

    def generative(template: str, answer_from: str, refusals: bool):
        g = GroqAnswerer(corpus, client=client, template_name=template,
                         answer_from=answer_from)
        r = RefusalAnswerer(corpus=corpus, base=base, client=client) if refusals else None

        def run(i: int) -> str:
            qid = qids[i]
            if r is not None and decisions[qid].is_policy_boundary:
                variant = "compound" if decisions[qid].intent == "compound" else "pure_boundary"
                return r.answer_with_variant(qtexts[i], citations[i], variant, question_id=qid)
            return g.answer(qtexts[i], citations[i], ranked[i])
        return run

    arms = {
        "A extractive top1 (SHIPPED)":       extractive("top1"),
        "B extractive top2":                 extractive("top2"),
        "C generative structured top2":      generative("structured-steps", "top2", False),
        "D generative answer-first top2":    generative("answer-first-explained", "top2", False),
        "E generative terse top2":           generative("terse-extractive", "top2", False),
        "F generative structured top2 + refusals":
                                             generative("structured-steps", "top2", True),
    }

    contexts = [
        "\n\n".join(
            f"[{d} - {s}]\n{corpus.get_section(d, s).body}"
            for d, s in ranked[i][:2] if corpus.get_section(d, s)
        )
        for i in range(len(qtexts))
    ]

    rows, answers = [], {}
    for name, fn in arms.items():
        print(f"\n=== {name} ===", flush=True)
        texts = []
        for i in range(len(qtexts)):
            texts.append(fn(i))
            if (i + 1) % 15 == 0:
                print(f"  {i + 1}/{len(qtexts)}", flush=True)
        answers[name] = texts

        evaluator = Eval(client=client)
        print("  judging ...", flush=True)
        for i, qid in enumerate(qids):
            rows.append({
                "arm": name, "question_id": qid, "kind": kinds[qid],
                "answer_relevancy": evaluator.get_answer_relevancy(qtexts[i], texts[i], qid),
                "faithfulness": evaluator.get_faithfulness_score(contexts[i], texts[i], qid),
                "chars": len(texts[i]),
                "padding": count_padding(texts[i]),
            })
        print(f"  unparseable: {len(evaluator.unparseable)}")

    frame = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "answer_sweep.csv", index=False)
    (args.out_dir / "answer_sweep_answers.json").write_text(
        json.dumps({"question_ids": qids, "answers": answers}, indent=2), encoding="utf-8"
    )
    _report(frame)
    return 0


def _report(frame: pd.DataFrame) -> None:
    rule = "=" * 100
    print("\n" + rule)
    print("ANSWER CONFIGURATION SWEEP — judged by RAG Triad, citations frozen")
    print(rule)
    print(f"  {'arm':<44} {'answer rel':>11} {'faithful':>10} {'chars':>7} {'padding':>8}")
    print("  " + "-" * 84)
    order = (frame.groupby("arm", sort=False)["answer_relevancy"].mean()
             .sort_values(ascending=False))
    for arm in order.index:
        g = frame[frame.arm == arm]
        print(f"  {arm:<44} {g.answer_relevancy.mean():>11.3f} "
              f"{g.faithfulness.mean():>10.3f} {g.chars.mean():>7.0f} "
              f"{int(g.padding.sum()):>8}")

    print("\n" + rule)
    print("ANSWER RELEVANCY BY KIND — multi_doc and multi_section are the target")
    print(rule)
    pivot = frame.pivot_table(index="arm", columns="kind",
                              values="answer_relevancy", aggfunc="mean")
    cols = [c for c in ("lookup", "adversarial", "multi_doc", "multi_section", "trap")
            if c in pivot.columns]
    header = "  " + f"{'arm':<44}" + "".join(f"{c:>15}" for c in cols)
    print(header)
    print("  " + "-" * (44 + 15 * len(cols)))
    for arm in order.index:
        line = f"  {arm:<44}"
        for c in cols:
            line += f"{pivot.loc[arm, c]:>15.3f}"
        print(line)

    base_arm = [a for a in frame.arm.unique() if a.startswith("A ")][0]
    base = frame[frame.arm == base_arm]
    print(f"\n  delta vs {base_arm}:")
    for arm in order.index:
        if arm == base_arm:
            continue
        g = frame[frame.arm == arm]
        print(f"    {arm:<44} answer_rel {g.answer_relevancy.mean() - base.answer_relevancy.mean():+.3f}   "
              f"faithful {g.faithfulness.mean() - base.faithfulness.mean():+.3f}")

    print("\n  !! LLM judge — ordinal, not calibrated. Valid for ranking arms.")


if __name__ == "__main__":
    raise SystemExit(main())
