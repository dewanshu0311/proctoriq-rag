# ProctorIQ RAG

A retrieval-augmented generation pipeline for proctored-assessment support, built
for the NIAT "Building & Optimizing RAG" Kaggle challenge — and, more
interestingly, a measurement harness for a problem that ships **no ground truth**.

---

## The problem

ProctorIQ runs proctored online exams. Students hit issues across five phases —
installing the secure browser, mock-test readiness, network drops mid-exam,
camera and face-verification warnings, and post-exam re-attempt status — and a
support assistant has to answer from a ten-document knowledge base while doing
two things at once:

1. **Cite its sources.** Every answer names the document(s) and section(s) it came
   from, pipe-separated and aligned by position: the Nth section belongs to the
   Nth document.
2. **Refuse cleanly when it should.** 14 of the 50 test questions are adversarial.
   They ask how to turn off face verification, whether a friend can watch the
   screen, whether a medical condition exempts a student from an integrity flag.
   The correct answer is a grounded refusal — and several of them wrap that
   request around a *legitimate* question that must still be answered helpfully.

Given `test.csv` (50 questions), produce `submission.csv`:

| Column | Meaning |
|---|---|
| `question_id` | from `test.csv`, unchanged |
| `answer_text` | the pipeline's final answer |
| `cited_docs` | source document(s), `|`-separated |
| `cited_sections` | section(s), `|`-separated, positionally aligned with `cited_docs` |

## Scoring

Automated — semantic similarity plus rule-based matching, no LLM judge. Out of 100.

| Dimension | Weight | Method |
|---|---|---|
| Answer accuracy | 25% | Semantic similarity to a hidden golden answer |
| Groundedness | 25% | Semantic similarity to the actual source excerpt |
| Retrieval quality | 20% | Match against the correct source document(s) |
| Citation accuracy | 15% | Match against the correct section(s) |
| Integrity-refusal | 15% | Semantic similarity to the correct refusal (adversarial only) |

The public leaderboard covers 40% of the questions; the private leaderboard covers
the other 60% and decides the final rank.

## The interesting constraint

**There is no training set and no ground truth in the competition data.** You
cannot measure whether a change helped. Every tuning decision is otherwise made
by submitting and squinting at a leaderboard that moves for reasons you cannot
see, on 40% of a 50-question sample.

So the first thing built here is not retrieval. It is a way to know whether
retrieval works.

`data/validation/answer_key.yaml` is a hand-built holdout: all 50 questions read
against all 10 documents, each tagged with its expected document(s), section(s), a
`kind` (`lookup` / `multi_doc` / `multi_section` / `adversarial` / `trap`) and an
honest `confidence` (`high` / `medium` / `low`) — including three entries marked
low because they are genuinely contested, with notes on what the alternative
reading would be.

That key is worth something only as long as it never touches the pipeline. Two
reasons: the competition rules forbid hardcoded answers, and a key that has
shaped the pipeline can no longer tell you whether the pipeline works — only that
it remembers. So the boundary is enforced structurally, by an AST walk over every
module in `src/`, rather than by good intentions:

> Every module under `src/proctoriq_rag/` **except** `evaluation/` may not import
> the evaluation package, name `answer_key`, or read `data/validation/`. And no
> module anywhere under `src/` may contain a `Q01`–`Q50` literal in executable
> code.

See [`tests/test_no_key_leakage.py`](tests/test_no_key_leakage.py).

## What can actually be measured locally

| Dimension | Weight | Status |
|---|---|---|
| Retrieval quality | 20% | **Exact** — predicted document set vs the key's |
| Citation accuracy | 15% | **Exact** — predicted `(doc, section)` pairs vs the key's |
| Groundedness | 25% | Proxy — faithful method, uncalibrated magnitude |
| Answer accuracy | 25% | Unmeasured — no golden answers exist |
| Integrity-refusal | 15% | Unmeasured — adversarial subset only |

35% is measured exactly. The scorer reports the composite twice — raw, and
renormalized over measured weight — and prints how much of the real metric is
invisible, every time. Unmeasured dimensions report `None` and are excluded; they
are never quietly substituted with a correlated number, because a visible gap can
be closed and a papered-over one cannot.

**One caveat is repeated in three places on purpose:** the grader's embedding
model is unknown. Our groundedness figure compares our own configs against each
other. It is not an estimate of the real score, and `groundedness 0.82` does not
mean "we would score 82". See [`docs/DECISIONS.md`](docs/DECISIONS.md) D-007.

## Architecture

```
src/proctoriq_rag/
├── config.py              # the two unresolved submission-format flags live here
├── corpus/loader.py       # markdown -> 53 sections. Splits on ## only, never ###
├── submission/writer.py   # the only place format flags are applied
├── evaluation/            # measurement only. The pipeline may never import this
│   ├── answer_key.py
│   ├── metrics.py
│   └── scorer.py
└── retrieval/ routing/ generation/   # Phase 1+
```

Full detail in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### Two design choices worth naming

**`###` is content, not a section boundary.** Document 01's "Section 2: Common
Installation Errors" contains three distinct errors as `###` subheadings, and Q01,
Q02 and Q03 each ask about one of them — but all three cite the same section,
because that is the section they live in. Splitting on `###` would invent sections
the grader has never seen and break citations on roughly a third of the test set.
The 53-section total is asserted in the test suite so a parsing regression fails
loudly.

**Two format questions are unresolved, so they are flags rather than guesses.**
Does `cited_docs` carry the `.md` extension? `sample_submission.csv` says no, the
competition description says "filename". What shape is `cited_sections` — the full
header, `Section 2`, or `Common Installation Errors`? Every variant is implemented
and tested; both will be settled by leaderboard probe. Guessing wrong costs up to
35% of the score, and supporting all of them costs one serialization branch.

## Running it

Requires Python ≥ 3.10.

```bash
pip install -e ".[dev,embed]"
```

Run the test suite — this is the gate for every phase:

```bash
python -m pytest
```

Validate the answer key against the corpus (expects 53 sections, 50 entries, zero problems):

```bash
python scripts/validate_key.py
```

See where naive semantic retrieval lands before any reranking exists:

```bash
python scripts/diagnose_sections.py
```

The diagnostic embeds all 50 questions against all 53 sections with a local
sentence-transformers model and prints the top-5 per question, marking the key's
expected section and its rank. It needs no API key and makes no network calls
beyond a one-time model download. Its purpose is to replace one person's reading
of the corpus with evidence about the corpus — particularly for the three
low-confidence key entries.

### Competition data

`data/raw/` is gitignored: the competition rules prohibit redistributing the
provided data, so the knowledge base, `test.csv` and `sample_submission.csv` are
not committed. `data/validation/answer_key.yaml` **is** committed — it is our own
work, not theirs.

To run against the real data, place the 10 `.md` files in `data/raw/kb/` and
`test.csv` / `sample_submission.csv` in `data/raw/`. Tests that need them skip
cleanly when they are absent.

## Decision log

Every non-obvious choice is recorded in [`docs/DECISIONS.md`](docs/DECISIONS.md)
with its alternatives, reasoning and evidence — including the ones still open and
the one place the stated spec had to be corrected to be enforceable.

## Status

**Phase 0 complete** — repository scaffold, corpus loader, answer-key validator,
local scorer, submission writer, leakage guard, retrieval diagnostic. 165 tests
passing. No retrieval, embedding, LLM or pipeline logic yet; that is Phase 1.
