# ProctorIQ RAG

A retrieval-augmented generation pipeline for proctored-assessment support, built
for the NIAT "Building & Optimizing RAG" Kaggle challenge.

**Final: 80.15 public.** Leader 87.00.

More interestingly, this is a case study in measuring a system that ships **no
ground truth** — and in the several ways that measurement went wrong before it went
right.

---

## The problem

ProctorIQ runs proctored online exams. Students hit issues across five phases —
installing the secure browser, mock-test readiness, network drops mid-exam, camera
and face-verification warnings, and post-exam re-attempt status. A support
assistant must answer from a ten-document knowledge base while doing two things at
once:

1. **Cite its sources.** Every answer names the document(s) and section(s) it drew
   on, pipe-separated and aligned by position.
2. **Refuse cleanly when it should.** 14 of the 50 test questions are adversarial —
   how to disable face verification, whether a friend may watch the screen, whether
   a medical condition exempts an integrity flag. Several wrap that request around a
   *legitimate* question that must still be answered helpfully.

Given `test.csv` (50 questions), produce `submission.csv` with `question_id`,
`answer_text`, `cited_docs`, `cited_sections`.

## Scoring

Automated — semantic similarity plus rule-based matching, no LLM judge.

| Dimension | Weight |
|---|---|
| Answer accuracy | 25% |
| Groundedness | 25% |
| Retrieval quality | 20% |
| Citation accuracy | 15% |
| Integrity-refusal | 15% |

Public leaderboard: 40% of questions. Private: the other 60%, and it decides rank.

---

## The central constraint

**There is no training set and no ground truth in the competition data.** You cannot
tell whether a change helped. Every decision is otherwise made by submitting and
squinting at a leaderboard that moves for invisible reasons on a ~20-question
sample.

So the first thing built was not retrieval. It was a way to know whether retrieval
works.

`data/validation/answer_key.yaml` is a hand-built holdout: all 50 questions read
against all 10 documents, each tagged with expected document(s), section(s), a
`kind`, and an honest `confidence` — including three entries marked low because
they are genuinely contested.

That key is worth something only while it never touches the pipeline. Two
independent guards enforce it structurally:

- **`tests/test_no_key_leakage.py`** — AST walk over `src/`. No pipeline module may
  import the evaluation package, name `answer_key`, or read `data/validation/`. No
  module anywhere may contain a `Q01`–`Q50` literal in executable code.
- **`tests/test_no_key_leakage_scripts.py`** — taint analysis over `scripts/`.
  Measurement harnesses legitimately read the key, so the rule is directional:
  **the key may determine what we MEASURE, never what the PIPELINE DOES.**

The second guard exists because the first one missed a real leak: a harness fired
refusals on `qid in adversarial`, letting the key decide pipeline behaviour.

---

## Results

### What the leaderboard resolved

Five probes, one variable each, `answer_text` held byte-identical:

| probe | change | public | Δ | resolved |
|---|---|---|---|---|
| 1 | baseline (`full_header`) | 68.02 | — | reference |
| 2 | `.md` extension added | 52.36 | −15.66 | **no extension** |
| 3 | `number_only` sections | **79.27** | +11.25 | **`number_only`** |
| 4 | `title_only` sections | 68.02 | **0.00** | grader matches **exactly** |
| 5 | `topk-2` | 76.90 | −2.37 | **F1 partial credit** |
| 6b | full generative | 79.25 | −0.02\* | **generation buys nothing broadly** |

\* against the extractive baseline of 79.27. The headline −0.90 against probe 7
is a three-variable comparison: 6b also switched refusals off (+0.56) and left
`SUBSECTION_FIX` inert (+0.32).

Probe 4 is the decisive one: `title_only` tying `full_header` **to the cent** is
only possible if both are simply wrong. That fixes the grader model as *exact string
match per element, F1 across the set*.

Probe 2 also measures the answer half directly — it zeroes both citation dimensions,
so its 52.36 **is** the answer half. Full decomposition:

| | achieved | available | headroom |
|---|---|---|---|
| retrieval | 15.66 | 20 | 4.34 |
| citation | 11.25 | 15 | 3.75 |
| answer half | 52.36 | 65 | 12.64 |
| **total** | **79.27** | 100 | |

### What measured negative

Kept in the codebase with their numbers, because a documented negative is worth
more than the code is as a feature:

| component | result |
|---|---|
| **HyDE** | recall@10 −0.078, citation F1 −0.200. Failed even on `policy_boundary`, its predicted best case |
| **RAG Fusion** | beats the hybrid first stage (doc F1 0.767 vs 0.733) but stays well below the cross-encoder, at 4 LLM calls per question |
| **Router score biasing** | −1.40 out of 35; gained **exactly 0.0000** on adversarial, the class it was built for; regressed lookup 1.000 → 0.966 |
| **Router cardinality** | 26.88 vs 26.93 off — neutral. Mechanism found later: the rank1→rank2 gap is *larger* on multi-section questions (0.132 vs 0.028), so the signal is **inverted** |
| **Full generative** | −0.02 on the leaderboard against a local estimate of **+5.1** |
| **Answer length** | swept 231→1050 chars; flat. Groundedness and answer accuracy trade at par |
| **Chunk size** | all six chunkers tied at recall@10 0.9375 |

### What measured positive

| change | evidence |
|---|---|
| **Cross-encoder reranking** | +17 points at the *median* config (0.36 → 0.53 citation-exact). Lookup document-F1 reaches 1.000 |
| **Refusal template** | 8/14 → 13/14 declining; **RAG Triad answer-relevancy on adversarial 0.493 → 0.871** |
| **Subsection fix** | Q03 was receiving the wrong error's passage |

The refusal decision rests on the Triad — the only evaluation here that never
consults the answer key — not on the leaderboard, which returned an inconclusive
+0.56.

---

## Architecture

```
src/proctoriq_rag/
├── config.py              probe-resolved submission format
├── corpus/loader.py       markdown -> 53 sections. Splits on ## only, never ###
├── retrieval/
│   ├── chunking.py        section / recursive / subsection
│   ├── embeddings.py      content-addressed disk cache
│   ├── retriever.py       dense (FAISS) / BM25 / hybrid
│   ├── reranker.py        cross-encoder, exhaustive over all 53 sections
│   ├── citation.py        aggregation + cardinality strategies
│   ├── hyde.py            mandated; measured negative
│   └── fusion.py          mandated; measured negative
├── routing/               LLM intent classifier; ships for refusal firing only
├── generation/            extractive + Groq, refusal templates
├── evaluation/            answer key, scorer, alternates, RAG Triad
└── submission/writer.py   the only place format flags are applied
```

Full detail in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Four ways a wrong submission almost shipped

Each produced a **valid-looking 50-row `submission.csv` that was not the arm it
claimed to be**. Each is now a hard guard that raises before writing:

1. **No Groq key** → router silently fell back to keywords, refusals no-op'd.
2. **Model deprecated mid-competition** → `llama-3.1-8b-instant` returned 404 after
   it had already produced a scored submission. Every router call failed; the
   `stats["llm"] == 0` guard caught it.
3. **Reasoning-model truncation** → `finish_reason="length"` with empty content fell
   back to extractive text. The `compound` prompt ran the longest traces and sat 35
   tokens from the cap.
4. **Flags not flipped** → the run summary correctly printed the arm, and it was
   submitted as a different one anyway. A description is not a check, so
   `EXPECTED_ARM` now asserts.

The lesson generalises: **a wrong artefact that looks correct is worse than a
crash**, and every one of these was caught by a guard rather than by review.

## Three measurement errors, same shape

All three were *comparisons where more than one thing changed*:

- **Phase 1** — a max over 312 configs compared against a max over 24 looked like a
  regression. Corrected by subsampling: the cross-encoder is +17 at the median.
- **Phase 2** — leaderboard agreement compared by row identity across 36 *tied*
  configs reported a false "MATERIAL DISAGREEMENT". It was measuring the sort's
  tie-breaking.
- **Phase 4** — a prompt edit between two router runs was reported as sampling
  nondeterminism. That claim reached the config file before being retracted (D-036).

## Prediction calibration

Every probe stated a predicted delta before submitting. Four consecutive landed low,
and the ratio **falls as the effect shrinks**:

| probe | predicted | observed | ratio |
|---|---|---|---|
| 2 | −16.9 | −15.66 | 0.93 |
| 5 | −3.4 | −2.37 | 0.70 |
| 6a | +1.5 | +0.56 | 0.37 |
| 7 | +1.5 | +0.32 | 0.21 |
| 6b | +3.6 | −0.02 | **0.00** |

Mechanism reasoning establishes direction well and magnitude badly:
**×0.90 above 10 points, ×0.70 for 3–10, ×0.30 below 3.**

Probe 6b broke that rule, and the split is instructive. Probes 2 and 5 were
predicted from **grader mechanism** — what the scorer could string-match — and
shrinkage worked. Probe 6b was predicted from **local proxies**, and no shrinkage
factor produces −0.02 from +5.1. Proxy-derived predictions get direction only.

### Three instruments agreed, and all three were wrong

The strongest result in the project is a negative one about its own method.
Generation was recommended on +4.4 to +5.3 from three deliberately
differently-biased instruments:

| instrument | documented bias | verdict on generation |
|---|---|---|
| groundedness (cosine to source) | toward extraction | −0.064 |
| RAG Triad relevancy | blind to length | +0.284 |
| synthetic reference answers | **toward generation** | +0.131 |

The third instrument's bias was written in its own docstring before it ever ran,
and it was still allowed to carry a quarter of the weighted estimate undiscounted.
The only proxy that predicted correctly was the one measured against text the
competition actually contains. **Naming a bias is not controlling for it.**

---

## Running it

Requires Python ≥ 3.10.

```bash
pip install -e ".[dev,embed]"
```

```bash
python -m pytest
```

```bash
python scripts/validate_key.py
```

```bash
python scripts/export_notebook.py
```

Other entry points: `sweep_retrieval.py`, `sweep_rerank.py`, `sweep_router.py`,
`measure_refusals.py`, `measure_hyde_fusion.py`, `run_triad.py`,
`run_pipeline.py --probe-6a`.

Generative paths need `GROQ_API_KEY` (or `GROQ_API_KEY_1..9`) in `.env`. The
extractive path needs no key at all.

### Competition data

`data/raw/` is gitignored — the rules prohibit redistributing it. Place the 10
`.md` files in `data/raw/kb/` and `test.csv` / `sample_submission.csv` in
`data/raw/`. Tests needing them skip cleanly when absent.
`data/validation/answer_key.yaml` **is** committed: it is our own work.

---

## Documentation

- [`docs/DECISIONS.md`](docs/DECISIONS.md) — 46 decisions with alternatives,
  reasoning and evidence, including every retraction
- [`docs/SUBMISSION_LOG.md`](docs/SUBMISSION_LOG.md) — every probe, predicted before
  observed
- [`docs/PROBES.md`](docs/PROBES.md) — probe design and pre-registered decision rules
- [`docs/FINAL_SUBMISSION.md`](docs/FINAL_SUBMISSION.md) — the two selected arms
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Status

**Complete.** 508 tests passing. Two submissions selected:
`probe-7-subsection-fix-retry` (80.15) and `probe-3-section-number` (79.27, tagged
`v1.0-locked-79.27`, deterministic, no runtime LLM dependency).
