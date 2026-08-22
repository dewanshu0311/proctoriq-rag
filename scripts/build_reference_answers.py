#!/usr/bin/env python
"""Generate synthetic reference answers, so answer accuracy has ANY local instrument.

    python scripts/build_reference_answers.py

The gap this fills
------------------
Answer accuracy is 25% of the score and has been measured locally by **nothing**.
``Scorer._answer_accuracy`` has existed since Phase 0 and always returned ``None``,
because ``data/validation/reference_answers.yaml`` was never written. Every answer
decision so far has therefore rested on two instruments that are both blind in the
same place:

    groundedness proxy   cosine to the source. Rewards echoing the source, and
                         has no opinion about whether the question was answered.
    RAG Triad            an LLM judge. Rewards answering the question, and does
                         not penalise saying it at three times the necessary
                         length.

**Neither instrument penalises verbosity.** That matters right now because the
hybrid arms under test are 800-1000 characters against a shipped arm of 425, and
the real grader compares against a golden answer that is almost certainly short.
Length is the one lever where both existing instruments are biased in the *same*
direction, so no combination of them can resolve it.

What this is, and what it is not
--------------------------------
For each question, an LLM writes the answer it believes the documentation
supports, from the **answer key's** cited sections. That is a measurement use of
the key — the key decides what we compare against, never what the pipeline does —
and it is why this lives in ``scripts/`` and writes to ``data/validation/``, which
no module under ``src/`` may read.

The number it produces is **not** an estimate of the leaderboard's answer
accuracy. Our golden answer is not theirs. Its value is comparative, and it comes
from having a bias that runs *opposite* to the groundedness proxy:

    groundedness proxy   biased toward extraction (an extract IS the source)
    this proxy           biased toward generation (an LLM wrote the reference)
    RAG Triad            biased toward neither, but blind to length

An arm that wins on all three is winning for a reason no single bias explains.
An arm that wins on one is telling you about the instrument.

Reruns are cached: an existing file is loaded and only missing questions are
generated, so this costs nothing after the first pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from proctoriq_rag.config import load_config  # noqa: E402
from proctoriq_rag.corpus.loader import load_corpus  # noqa: E402
from proctoriq_rag.evaluation.answer_key import load_answer_key  # noqa: E402
from proctoriq_rag.generation.groq_client import GroqChatClient, load_dotenv  # noqa: E402

KEY_PATH = REPO_ROOT / "data" / "validation" / "answer_key.yaml"
OUT_PATH = REPO_ROOT / "data" / "validation" / "reference_answers.yaml"

ANSWER_PROMPT = """\
You are writing the OFFICIAL reference answer for a support-documentation quality
benchmark. This is the answer other systems will be scored against, so it must be
what the documentation actually says — not longer, not hedged, not padded.

Documentation passages:
{context}

Student's question: {question}

Write the reference answer. Rules:
- Answer the question directly and completely, using the passages only.
- Two to four sentences. Prose. No greeting, no sign-off, no bullet list.
- If the question has two parts, answer both.
- If the passages say the request is not permitted or not possible, say so plainly,
  give the documentation's reason, and state what the student can do instead.
- Include nothing the passages do not state.

Reference answer:"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--force", action="store_true",
                        help="regenerate every question, ignoring the cache")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    config = load_config()
    corpus = load_corpus(config.paths.kb_dir)
    key = load_answer_key(KEY_PATH, corpus)
    by_id = {entry.id: entry for entry in key}

    questions = pd.read_csv(config.paths.test_csv)
    qids = questions["question_id"].astype(str).tolist()
    qtexts = questions["question"].astype(str).tolist()

    existing: dict[str, dict[str, str]] = {}
    if args.out.exists() and not args.force:
        raw = yaml.safe_load(args.out.read_text(encoding="utf-8")) or {}
        existing = raw.get("answers") or {}
        print(f"loaded {len(existing)} cached reference answers from {args.out.name}")

    todo = [i for i, qid in enumerate(qids) if qid not in existing]
    if not todo:
        print("nothing to generate — all questions cached")
        return 0
    print(f"generating {len(todo)} reference answers ...")

    client = GroqChatClient()
    answers = dict(existing)
    for count, index in enumerate(todo, start=1):
        qid, question = qids[index], qtexts[index]
        entry = by_id.get(qid)
        if entry is None:
            continue
        context = "\n\n".join(
            f"[{doc_id} - {section_title}]\n{section.body}"
            for doc_id, section_title in entry.pairs
            if (section := corpus.get_section(doc_id, section_title)) is not None
        )
        text = client.complete(ANSWER_PROMPT.format(context=context, question=question))
        text = " ".join(text.split()).strip()
        if not text:
            print(f"  {qid}: EMPTY — skipped, will retry on next run")
            continue

        record = {"answer": text}
        # The 15% integrity dimension is scored against a refusal, not an answer.
        # For adversarial questions the reference answer IS the refusal, so it is
        # recorded under both keys rather than prompting twice.
        if entry.is_adversarial:
            record["refusal"] = text
        answers[qid] = record
        if count % 10 == 0 or count == len(todo):
            print(f"  {count}/{len(todo)}", flush=True)

    args.out.write_text(
        "# GENERATED by scripts/build_reference_answers.py — NOT hand-written.\n"
        "#\n"
        "# Synthetic references for measuring answer accuracy and integrity refusal\n"
        "# locally. These are one model's reading of the answer key's cited sections,\n"
        "# not the competition's golden answers. Absolute values are meaningless;\n"
        "# only the ranking of arms against each other is informative, and even that\n"
        "# carries a bias toward generated phrasing (see the script's docstring).\n"
        "#\n"
        "# Regenerate:  python scripts/build_reference_answers.py --force\n\n"
        + yaml.safe_dump({"answers": answers}, sort_keys=True, allow_unicode=True,
                         default_flow_style=False, width=88),
        encoding="utf-8",
    )
    lengths = [len(v["answer"]) for v in answers.values()]
    print(f"\nwrote {args.out}")
    print(f"  {len(answers)} references, mean {sum(lengths) / len(lengths):.0f} chars, "
          f"range {min(lengths)}-{max(lengths)}")
    print(f"  {sum(1 for v in answers.values() if 'refusal' in v)} also recorded as refusals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
