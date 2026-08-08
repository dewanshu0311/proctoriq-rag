# Decision Log

Every non-obvious choice, why it was made, and what evidence supports it. Later
phases append; nothing is deleted. When a decision is overturned, the original
entry stays and a new one supersedes it — the reasoning that turned out wrong is
as informative as the reasoning that held.

Entries are written as explanations rather than shorthand, because this file is
also the script for the required demo video.

---

## D-001 — `###` subheadings are content, not section boundaries

**Date:** 2026-08-08 · **Phase:** 0 · **Status:** settled

**Decision.** The corpus loader splits documents on `##` headers only. `###`
subheadings stay inside the body of their parent section.

**Alternatives considered.** Split on both `##` and `###`, producing finer-grained
retrieval units. Finer chunks generally help retrieval, so this was not an
obviously bad idea — it is wrong for a specific reason.

**Reasoning.** The competition scores citations against a section name, and the
section a piece of content *belongs to* is defined by the `##` header above it.
Splitting on `###` would invent sections that do not exist in the ground truth,
and every question answered from one of them would cite a section string the
grader has never seen.

**Evidence.** Document 01, "Section 2: Common Installation Errors", contains three
distinct errors as `###` subheadings — "Element not found", "Session Start Error"
and "Unspecified Error". Questions Q01, Q02 and Q03 ask about one each, and the
answer key cites all three to that one section. Documents 01 and 02 both do this.
Splitting on `###` breaks citations on roughly a third of the test set.
Locked in by `tests/test_corpus_loader.py::test_h3_subheadings_are_content_not_boundaries`
and by the total-section-count assertion (53).

---

## D-002 — The answer key is a holdout, never a pipeline input

**Date:** 2026-08-08 · **Phase:** 0 · **Status:** settled

**Decision.** `data/validation/answer_key.yaml` may be read only by `tests/`,
`scripts/`, and the `src/proctoriq_rag/evaluation/` package. No pipeline code may
import the evaluation package, name `answer_key`, or read `data/validation/`.
Enforced by an AST walk in `tests/test_no_key_leakage.py`, not by convention.

**Alternatives considered.** Enforce by code review and discipline. Rejected —
the pressure to special-case a stubborn question grows as the leaderboard
deadline approaches, and a rule that depends on willpower fails exactly when it
matters most.

**Reasoning.** Two independent reasons. First, the competition rules explicitly
forbid hardcoded answers, so a pipeline that consults the key is disqualifiable.
Second and more practically: the key is the *only* independent notion of
correctness we have — there is no training set and no ground truth in the
competition data. A key that has influenced the pipeline can no longer tell us
whether the pipeline works; it only tells us the pipeline remembers.

**Note on the literal wording.** The instruction was "fail if any module under
`src/` references the answer key". Taken literally that fails
`evaluation/scorer.py`, whose entire purpose is to compare against the key. The
enforceable formulation of the same guarantee is a one-way dependency: evaluation
may import the pipeline, the pipeline may never import evaluation. The pipeline
therefore cannot reach the key even transitively.

**Evidence.** `tests/test_no_key_leakage.py` walks every module under `src/` and
asserts the boundary per-module, and includes a test proving each check actually
fires on a violating module — a guard that cannot fail is not a guard.

---

## D-003 — Hardcoded question IDs are banned in executable code

**Date:** 2026-08-08 · **Phase:** 0 · **Status:** settled

**Decision.** No `Q01`–`Q50` literal may appear in executable code anywhere under
`src/` — string literals and identifiers included. Docstrings and comments are
exempt.

**Alternatives considered.** Rely on D-002 alone. Rejected: D-002 stops the key
leaking, but a developer could still hand-write `if qid == "Q28"` from memory
without touching the key at all.

**Reasoning.** Question-specific branching is the first step towards the
hardcoded answers the rules forbid, and it is far easier to catch mechanically
than in review. Prose is exempt because explaining *why* the Q44 duplicate-document
case is handled a certain way makes the code better, and a docstring cannot
branch. `EXPECTED_QUESTION_IDS` is built with a comprehension rather than typed
out, for exactly this reason.

**Evidence.** The check found five violations on its first run — all of them my
own docstrings — which is what prompted the docstring exemption and confirmed the
check is not vacuous.

---

## D-004 — Document name format is a config flag, pending a leaderboard probe

**Date:** 2026-08-08 · **Phase:** 0 · **Status:** OPEN — resolve by probe

**Decision.** `submission.doc_extension` controls whether `cited_docs` carries
`.md`. Both variants are implemented and tested. Default: `false`.

**Alternatives considered.** Commit to one and hope.

**Reasoning.** The evidence genuinely conflicts, so this is not resolvable by
reading. `sample_submission.csv` omits the extension; the competition description
says "filename", which implies it. Guessing costs up to 35% of the score
(retrieval + citation) if wrong, and the cost of supporting both is one
serialization branch. The default follows `sample_submission.csv` because a
machine-generated sample file is weaker evidence than a human-written description
in general, but is *direct* evidence of the parser's expected shape here.

**Evidence.** `data/raw/sample_submission.csv` row 2:
`Q01,Sample answer text goes here.,01_windows_installation_login_guide,Section 1`
— no `.md`. To be settled by submitting two runs differing only in this flag.

---

## D-005 — Section string format is a config flag, pending a leaderboard probe

**Date:** 2026-08-08 · **Phase:** 0 · **Status:** OPEN — resolve by probe

**Decision.** `submission.section_format` selects between `full_header`,
`number_only` and `title_only`. All three are implemented and tested. Default:
`full_header`.

**Reasoning.** The header is `## Section 2: Common Installation Errors`, and three
different substrings of it are plausible submissions. `sample_submission.csv`
shows `Section 1`, which points at `number_only` — but that file is dummy data
with the same value repeated on all 50 rows, so it demonstrates the *column*, not
the expected value. `full_header` is the default because it carries the most
information: if the grader does substring or semantic matching, the full header
contains both other variants; if it does exact matching, we probe.

**Open sub-case.** Headers with no `Section N:` prefix — `## Overview` — are
emitted verbatim under all three variants. The answer key never cites an Overview
section, so this path is currently unreachable in practice, but the formatter is
total rather than partial.

**Evidence.** All three variants covered in
`tests/test_submission_writer.py::test_all_three_section_variants`. To be settled
by probe.

---

## D-006 — Document-set scoring deduplicates; citation scoring does not

**Date:** 2026-08-08 · **Phase:** 0 · **Status:** OPEN — unknown, reported both ways

**Decision.** The local scorer computes retrieval quality over the *set* of
predicted documents, and citation accuracy over the set of `(doc, section)`
*pairs*. It additionally reports a multiset variant of the document score.

**Reasoning.** Q44 is the only entry citing one document twice — doc 09, two
different sections. As a set its documents are `{09}`, size 1; as a list they are
`[09, 09]`, length 2. We do not know whether the grader deduplicates before
comparing, and the two conventions disagree on exactly one realistic failure
mode: citing the document once when two citations were expected. Set semantics
call that a perfect document score, multiset calls it 0.5 recall.

**Why this is left open rather than guessed.** It affects one question out of
fifty, so the cost of being wrong is small and bounded, while the cost of
resolving it is a leaderboard submission we would rather spend on D-004 and
D-005. Reporting both keeps the ambiguity visible instead of buried in a
convention.

**Evidence.** `tests/test_scorer.py::test_q44_half_answered_diverges_between_set_and_multiset`
pins the exact case where the two disagree.

---

## D-007 — The grader's embedding model is unknown, so groundedness is relative only

**Date:** 2026-08-08 · **Phase:** 0 · **Status:** settled (a permanent caveat, not a task)

**Decision.** The groundedness proxy uses `sentence-transformers/all-MiniLM-L6-v2`
locally. Its output is treated as a **relative** signal — valid for comparing our
config A against our config B — and never as an estimate of the real score.

**Reasoning.** The competition describes the metric as "semantic similarity"
without naming a model. Cosine similarity under two different embedding models is
not on a shared scale: the same pair of texts can score 0.62 under one model and
0.84 under another, and the *ordering* of candidate answers can differ too, though
it is far more stable than the absolute value. Our reconstruction is faithful in
*method* — we compare the generated answer against the real source text, which we
have — but uncalibrated in magnitude.

**The specific error this exists to prevent.** In three weeks, someone reading
`groundedness 0.82` will be tempted to conclude "we would score 82 on that
dimension." That inference is invalid, and it is invalid in an unhelpful
direction: it would make a mediocre pipeline look finished. The number supports
one kind of claim only — "config B grounds better than config A" — and even that
holds only when both were measured under the same model.

**Evidence.** The caveat is printed in every scorer report (`ScoreReport.notes`),
asserted by `tests/test_scorer.py::test_report_notes_flag_the_groundedness_caveat`,
and repeated in the `evaluation/scorer.py` module docstring and
`config/default.yaml`. Three copies is deliberate — this is the caveat most likely
to be forgotten.

---

## D-008 — Unmeasured dimensions report `None` and are excluded, never substituted

**Date:** 2026-08-08 · **Phase:** 0 · **Status:** settled

**Decision.** Answer accuracy (25%) and integrity-refusal (15%) have no ground
truth, so they report `None`. The composite is published twice: raw (unmeasured
dimensions counted as zero) and renormalized (over measured weight only), always
alongside an explicit statement of how much of the real metric is unmeasured.

**Alternatives considered.** Substitute groundedness for answer accuracy, since
the two correlate. Rejected — it would produce a single confident-looking number
that silently double-counts one measurement and hides that 40% of the metric is
invisible to us.

**Reasoning.** A gap that is visible can be closed. A gap that has been papered
over with a plausible substitute cannot, because nobody remembers it is there.

**Evidence.** With no references supplied, `measured_weight` is 0.35 — retrieval
20% plus citation 15%, the two exact dimensions. Adding a local embedder raises it
to 0.60. Only a hand-written `reference_answers.yaml` reaches 1.00. Asserted in
`tests/test_scorer.py::test_measured_weight_excludes_unmeasured_dimensions`.

---

## D-009 — The submission writer rejects empty `answer_text`

**Date:** 2026-08-08 · **Phase:** 0 · **Status:** settled

**Decision.** `validate_rows` raises on any empty or whitespace-only
`answer_text`, naming the offending question IDs, and no file is written.

**Reasoning.** The starter notebook's generation loop wraps each question in
`try/except` and, on failure, writes `answer, cited_docs, cited_sections = "", "", ""`.
That produces a row which is structurally valid — right ID, right column count,
parses cleanly — while scoring zero on four of the five dimensions. It is the
failure mode most likely to survive a visual check of the output file, because
nothing about it looks wrong until you read all fifty rows.

Failing loudly converts a silent 6% score loss into a crash that names the three
questions to re-run.

**Evidence.** `tests/test_submission_writer.py::test_empty_answer_text_raises_and_names_the_questions`
and `::test_nothing_is_written_when_validation_fails`.

---

## D-010 — Target Python 3.10, not 3.11

**Date:** 2026-08-08 · **Phase:** 0 · **Status:** settled

**Decision.** `requires-python = ">=3.10"`. `SectionFormat` derives from
`(str, Enum)` rather than `enum.StrEnum`.

**Reasoning.** 3.10.11 is the only interpreter on the development machine, and
`enum.StrEnum` is 3.11+ — it would fail at import, not at use. Nothing in this
phase needs a 3.11-only feature, and `(str, Enum)` is behaviourally equivalent for
everything we rely on (members compare equal to their string values). Kaggle's
runtime is 3.11 and remains compatible with a 3.10 floor.

**Evidence.** `tests/test_config.py::test_section_format_members_compare_as_strings`.

---

# Phase 1 — Retrieval baseline

Sweep: 312 retrieval configurations (12 chunkers × 3 embedding models × dense /
sparse / hybrid×3α × 2 aggregators) × 14 citation strategies = 4,368 scored rows.
Full data in `outputs/sweep_retrieval_latest.csv`.

## D-011 — Selection is on recall@10, not citation F1

**Date:** 2026-08-08 · **Phase:** 1 · **Status:** settled

**Decision.** Configurations are ranked by section-level recall@10, with recall@5
as the tie-break. Citation F1 and exact-match are reported on every row but are
diagnostic this phase, not the objective.

**Reasoning.** Phase 2 adds a cross-encoder that reorders the candidate pool. A
config chosen for rank-1 precision is optimised for a pipeline we are not
building — the reranker can fix ordering, but it cannot retrieve something that
was never in the pool. What Phase 1 owes Phase 2 is the right section *present*,
not the right section *first*.

**Evidence.** At the chosen config, 39 of 50 questions have inexact citations —
but **35 of those 39 have every expected section already inside the top 10**. Only
4 need better retrieval rather than better ordering. Optimising citation F1 now
would have traded away pool quality to chase the 35 that reranking gets for free.

---

## D-012 — Almost nothing in the sweep is distinguishable on n=50

**Date:** 2026-08-08 · **Phase:** 1 · **Status:** settled — the central finding

**Decision.** No single "winning" configuration is declared. A shortlist of robust
configurations is reported instead, and `config/default.yaml` is marked
**provisional**.

**The numbers.** Of 312 retrieval configurations:

| | count |
|---|---|
| tied at the top recall@10 (0.9375) | **36** |
| within one noise band (±6.1 pp) of it | **217** |

The noise band is two standard errors of a binomial proportion over the 64
expected sections: `2 × sqrt(p(1-p)/64) ≈ 0.0605`. Differences smaller than that
are not measurable on this sample.

**Per-dimension best recall@10, all 50 questions:**

| Dimension | Values | Best recall@10 |
|---|---|---|
| Chunker | section / rec256 / rec400 / rec512 / rec768 / subsection | **0.9375 for all six** |
| Model | MiniLM-L6 / bge-small / mpnet-base | 0.9375 / 0.9219 / 0.9375 |
| Mode | dense / hybrid / sparse | 0.9375 / 0.9375 / 0.8125 |
| Header in text | on / off | 0.9375 / 0.9375 |
| Aggregator | max / sum | 0.9375 / 0.9375 |

Every chunker ties. Both header settings tie. Both aggregators tie. Two of three
models tie. **The only dimension that separates at all is retriever mode**, and
only because sparse-alone is clearly worse.

**Why this is written down rather than quietly resolved by picking the top row.**
4,368 rows scored against 50 questions is an overfitting machine. The top row beats
its neighbours by roughly 2 pp on a metric whose resolution is about 6 pp. Picking
it would be selecting noise and calling it a decision. The `delta_vs_nbhd` column
exists to make that visible: the best configs sit on plateaus with Δ between +0.008
and +0.023, entirely inside the noise band.

**Consequence for the demo.** The competition asks for at least 2 chunk sizes and 2
embedding models to be tested with a justified choice. We tested five chunking
strategies and three models. The justified finding is that **on this corpus, at this
sample size, chunk size does not matter** — a stronger result than a fabricated
one-point winner, because it is reproducible.

---

## D-013 — Chunk size is a weak lever on this corpus, and we knew why in advance

**Date:** 2026-08-08 · **Phase:** 1 · **Status:** settled

**Decision.** `SectionChunker` (no split) is the provisional default.

**Reasoning, established before running anything.** Section bodies are min 134 /
median 347 / mean 432 / max 1317 characters. Only **5 of 53** sections exceed 768,
so a 768-character window leaves 48 sections untouched.

| chunk_size | chunks | vs. 53 |
|---|---|---|
| 256 | 129 | 2.4× |
| 400 | 84 | 1.6× |
| 512 | 72 | 1.4× |
| 768 | 59 | 1.1× |
| no-split (section) | 53 | — |
| subsection | 61 | 1.2× |

All six achieve identical best recall@10 (0.9375). `SectionChunker` is preferred
because retrieval unit and citation unit coincide exactly — there is no aggregation
step that can go wrong — and it has the highest mean neighbourhood score (0.9297).

**The disappointment worth recording.** `SubsectionChunker` was the strong prior
going in: all 13 `###` subsections sit in the 5 largest sections, which serve the
densest lookup region in the test set (Q01–Q03, Q05, Q07–Q11, Q13). Splitting
should have sharpened retrieval on the three Windows install errors. It did not
move recall@10 at all. The hypothesis was reasonable and it was wrong — the corpus
is small enough that a 1317-character passage is not actually hard to match.

---

## D-014 — Hybrid ties dense; sparse alone is clearly worse

**Date:** 2026-08-08 · **Phase:** 1 · **Status:** settled

**Decision.** Provisional default is hybrid with `alpha=0.7` (70% dense).

**Evidence.**

| Mode | best recall@10 | best recall@5 | best recall@1 | mean recall@10 |
|---|---|---|---|---|
| dense | 0.9375 | 0.8281 | **0.6250** | 0.8991 |
| hybrid | 0.9375 | **0.8438** | 0.5938 | 0.8992 |
| sparse (BM25) | 0.8125 | 0.7344 | 0.4531 | 0.7760 |

**Did lexical retrieval earn its place?** Partially, and less than expected. BM25
was added because several questions turn on strings the corpus uses verbatim —
*Session Start Error*, *Element not found*, *Under Verification*, *Resolved — Flag
Upheld*. Alone it is 12 points worse at recall@10. Blended it buys about 1.6 pp at
recall@5 over pure dense while costing about 3 pp at recall@1 — both inside the
noise band. **The honest reading: hybrid is not measurably better than dense here.**
It is kept as the default because the α sweep is smooth and 0.7 sits mid-plateau,
but this is a coin-flip dressed as a decision and should be revisited when the
reranker changes what the pool is for.

---

## D-015 — The header-in-text variable, resolved as "no effect"

**Date:** 2026-08-08 · **Phase:** 1 · **Status:** settled

**Decision.** Keep `include_header_in_text: true`.

**Reasoning.** Phase 0's diagnostic embedded `section_title + body` without ever
testing whether the header helped, leaving an unexplained variable in the baseline.
Sweeping it: best recall@10 is 0.9375 either way; mean recall@10 is 0.8912 with
headers vs 0.8881 without, and mean recall@5 is 0.7919 vs 0.7766. A consistent but
tiny lean toward headers, far inside the noise band.

Kept on because it is marginally ahead on the mean and because it makes chunk text
self-describing when a human reads it. **Not** because it was shown to matter.

---

## D-016 — The two leaderboards agree, and the tie-aware check is why we know

**Date:** 2026-08-08 · **Phase:** 1 · **Status:** settled

**Decision.** Configuration selection is not an artefact of the uncertain key
entries. 30 configurations are top-tier on both the all-50 and
high-confidence-only leaderboards.

**A reporting bug worth recording, because it nearly produced a false conclusion.**
The first version compared the *top-10 rows by identity* between the two
leaderboards and reported **0/10 overlap — "MATERIAL DISAGREEMENT"**. That was
wrong. With 36 configs tied at exactly the same recall@10, the sort order among
them is arbitrary, so comparing row identity measures the sort's tie-breaking
rather than agreement. The corrected check compares the *sets* of configs tied at
the maximum:

| | all-50 | high-confidence only |
|---|---|---|
| configs tied at max | 36 | 76 |
| max recall@10 | 0.9375 | **1.0000** |
| top-tier on both | 30 (Jaccard 0.366) | |

**The finding that matters more than the agreement.** On the 36 high-confidence
questions, 76 configurations achieve **recall@10 = 1.000** — every expected section
retrieved, no misses at all. All four genuine retrieval misses on the full set are
**Q14, Q28, Q31, Q35**: the three low-confidence entries plus one medium.

| Question | confidence | expected section not retrieved | rank |
|---|---|---|---|
| Q14 | low | 10_general_support · Section 4: A Note on What Support Will Never Do | 13 |
| Q28 | low | 08_reattempt · Section 1: Reporting an Interruption | 13 |
| Q31 | medium | 07_policy · Section 4: This Policy Cannot Be Configured or Bypassed | 14 |
| Q35 | low | 09_status · Section 4: What This Document Does Not Do | 12 |

Retrieval is not the bottleneck on the questions the key is sure about. The
residual misses sit exactly where the key itself is uncertain — corroborating the
Phase 0 diagnostic, which already showed naive retrieval disagreeing with the key
on those same entries. That is evidence about the key, and those four entries
should be re-read before any effort is spent making retrieval find them.

---

## D-017 — No citation cardinality policy is chosen in Phase 1

**Date:** 2026-08-08 · **Phase:** 1 · **Status:** OPEN — belongs to the router

**Decision.** The full cardinality curve is reported; `citation_strategy` is left
`null` in config.

**The curve** (averaged across all retrieval configs; `best cite F1` is the best
single config at that strategy):

| strategy | mean citations | doc F1 | cite F1 | cite exact | best cite F1 |
|---|---|---|---|---|---|
| topk-1 | 1.000 | 0.6323 | 0.4633 | **0.3589** | **0.7333** |
| gap-0.97 | 1.167 | 0.6418 | 0.4734 | 0.3227 | 0.7047 |
| thresh-0.9 | 1.256 | 0.6475 | 0.4800 | 0.2963 | 0.7013 |
| gap-0.95 | 1.291 | 0.6474 | 0.4789 | 0.2963 | 0.7013 |
| thresh-0.85 | 1.416 | 0.6544 | 0.4847 | 0.2579 | 0.6813 |
| thresh-0.8 | 1.568 | **0.6555** | **0.4848** | 0.2221 | 0.6767 |
| gap-0.9 | 1.607 | 0.6557 | 0.4798 | 0.2232 | 0.6560 |
| gap-0.85 | 1.883 | 0.6495 | 0.4690 | 0.1634 | 0.6533 |
| topk-2 | 2.000 | **0.6695** | 0.4642 | **0.0479** | 0.5900 |
| thresh-0.55 | 2.297 | 0.6418 | 0.4574 | 0.0984 | 0.5893 |

**The shape, which is the point.** Document F1 rises monotonically with citation
count (0.632 → 0.670) because extra citations can only add recall on the doc set.
Citation exact-match *collapses* in the opposite direction (0.359 → 0.048) because
every surplus citation breaks exactness on the 36 single-citation questions. Cite
F1 is nearly flat across the whole range — it peaks at 0.485 around 1.4–1.6 mean
citations and never varies by more than 0.03.

**Why no fixed rule is adopted.** 36 of 50 questions want one citation and 14 want
two. `topk-1` guarantees a recall miss on 14; `topk-2` guarantees a precision hit on
36 and drops exact-match by a factor of seven. The adaptive strategies sit between
the two without escaping the trade — the flatness of cite F1 across a 2.3× range of
mean citations *is* the evidence that no fixed cardinality is right.

Locking a threshold now on aggregate F1 would bake in the average and lose both
tails. This is the one dimension where the correct answer is provably per-question,
so it goes to the router.

**A third citation never pays.** `max_citations` was set to 3 while the key's
maximum is 2, to make this measurable rather than assumed. Every strategy allowing a
third scores worse on exact-match than its two-citation equivalent.

---

# Phase 2 — Cross-encoder reranking

Sweep: 3 reranker models × 2 scored-text variants × 4 pool sizes × 14 citation
strategies = 336 rows, from 15,900 cross-encoder pairs scored once (56 min CPU)
and cached. Data in `outputs/sweep_rerank_latest.csv`, transform diagnostics in
`outputs/rerank_diagnostics.csv`.

## D-018 — Reranking helps, but the honest comparison is the median, not the max

**Date:** 2026-08-08 · **Phase:** 2 · **Status:** settled — read this before D-019..D-023

**The naive comparison says reranking failed.** Phase 1's best citation-exact was
**0.6000**; Phase 2's best is **0.5600**. Taken at face value, the cross-encoder
made things worse.

**That comparison is invalid, and the reason matters.** Phase 1 selected a maximum
over **312** configurations; Phase 2 over **24**. A larger grid has more chances to
draw a lucky configuration, so its maximum is biased upward. Subsampling Phase 1's
grid down to 24 configurations, 3,000 times:

| | citation-exact |
|---|---|
| Phase 1 max over all 312 configs | 0.6000 |
| Phase 1 max over 24 random configs (mean of 3,000 draws) | **0.5324** |
| Phase 2 max over its 24 configs | **0.5600** |

At equal grid size, Phase 2 is ahead by +0.028. That is inside the noise band and
proves nothing on its own.

**The comparison that does hold is the median.** Holding the citation strategy
fixed at `topk-1`:

| | n configs | median | p90 | max |
|---|---|---|---|---|
| Phase 1 bi-encoder | 312 | **0.3600** | 0.4800 | 0.6000 |
| Phase 2 cross-encoder | 24 | **0.5300** | 0.5600 | 0.5600 |

**+17 points at the median.** The cross-encoder is not better at its luckiest; it is
dramatically better *typically*. Its worst configuration beats the bi-encoder's
median. That is robustness rather than a draw from the tail, and robustness is what
survives contact with a private leaderboard.

**Like-for-like on the Phase 1 provisional default** (same 53 sections, same
`topk-1`, bi-encoder hybrid@0.7 vs cross-encoder exhaustive):

| metric | Phase 1 | Phase 2 | Δ |
|---|---|---|---|
| citation-exact | 0.4800 | 0.5600 | **+0.0800** |
| citation-F1 | 0.5867 | 0.6667 | **+0.0800** |
| document F1 | 0.7333 | 0.8467 | **+0.1133** |
| document exact | 0.6000 | 0.7000 | **+0.1000** |
| recall@1 | 0.5000 | 0.5625 | +0.0625 |
| inexact-citation questions | 39 / 50 | **22 / 50** | −17 |

**Chosen model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.** All 24 configurations are
within the ±14.0 pp noise band, so this is a tie-break, not a win — and it is
tie-broken on grounds that are not the score: it is the **smallest** model (~90 MB)
and the **fastest by 5×** (19.1 pairs/s against 3.9 for bge-reranker-base and 2.9
for mxbai-rerank-base-v1), while sitting at the top of both leaderboards.

---

## D-019 — Exhaustive scoring ties pooled exactly; prefer it for having fewer parts

**Date:** 2026-08-08 · **Phase:** 2 · **Status:** settled

**Decision.** Default to exhaustive cross-encoder scoring of all 53 sections. Keep
the Phase 1 retriever — this decides the *default path*, not what exists.

| pool | recall@1 | recall@10 | cite-exact | cite-F1 | doc F1 |
|---|---|---|---|---|---|
| pool10 | 0.5781 | **0.9375** | 0.5600 | 0.6733 | 0.8467 |
| pool20 | 0.5781 | 0.9062 | 0.5600 | 0.6733 | 0.8467 |
| pool30 | 0.5781 | 0.8906 | 0.5600 | 0.6733 | 0.8467 |
| exhaustive | 0.5781 | 0.8906 | 0.5600 | 0.6733 | 0.8467 |

**Citation-exact is identical to four decimal places across every pool size.**
`exhaustive − pool10 = +0.0000`, against a ±0.1404 band. The structural comparison
that was expected to survive the noise band instead produced a perfect tie.

Exhaustive is preferred on the pre-committed tie-break — fewer moving parts, no
first-stage retriever in the default path, and no recall ceiling to reason about —
not because it scored better. It scored identically.

**An unexpected result worth recording.** Exhaustive recall@10 (0.8906) is *lower*
than pool10's (0.9375). That is not a bug: pool10's recall@10 is by construction the
bi-encoder's own recall@10, since the reranker can only reorder ten candidates. The
cross-encoder, scoring all 53, pushes some expected sections below rank 10 that the
bi-encoder had inside it. **The cross-encoder sharpens the top of the ranking and
loses some of the tail.** Since we cite one or two sections, the top is what pays —
but this is the direct reason recall@10 was retired as the selection criterion this
phase, and it would matter again if a later phase wanted a deep pool.

---

## D-020 — Scored text: no effect, same as for bi-encoders

**Date:** 2026-08-08 · **Phase:** 2 · **Status:** settled

Best citation-exact by model and text variant:

| model | body | titled |
|---|---|---|
| ms-marco-MiniLM-L-6-v2 | **0.5600** | **0.5600** |
| bge-reranker-base | 0.5000 | 0.5400 |
| mxbai-rerank-base-v1 | 0.5000 | 0.5200 |

Phase 1's conclusion was deliberately **not** carried across — a bi-encoder embeds
the passage alone while a cross-encoder attends over question and passage jointly,
so titles could plausibly have participated in the match differently. They did not.
Every difference here is inside the ±14 pp band, and for the chosen model the two
variants are exactly tied.

`body` is the default: identical score, shorter sequences, marginally faster, and
the winner's adversarial document F1 is better under `body` (0.6429) than `titled`
(0.4286) — itself inside the band, but pointing the same way.

---

## D-021 — Score transform: `auto`, after a bug and a near-miss

**Date:** 2026-08-08 · **Phase:** 2 · **Status:** settled

**The bug, found before the sweep rather than after.** `ms-marco-MiniLM-L-6-v2`
emits raw logits — measured range **−11.5 to +9.6, with 49 of 53 scores negative on
a sample question**. `RelativeGap` compares `score_i / score_1`, which is
meaningless for negatives, and its `top <= 0` short-circuit returns exactly one
citation regardless of ratio. The entire cardinality re-sweep would have degenerated
into `topk-1` while drawing a plausible-looking curve. Pinned by
`tests/test_reranker.py::test_relative_gap_is_broken_on_raw_negative_logits`.

**The near-miss.** The obvious fix — always apply a sigmoid — would itself have been
wrong. Only one of the three models emits logits:

| model | raw output range | scale |
|---|---|---|
| ms-marco-MiniLM-L-6-v2 | −11.54 … 9.58 | logits |
| bge-reranker-base | 0.00 … 1.00 | probabilities |
| mxbai-rerank-base-v1 | 0.00 … 0.99 | probabilities |

Squashing an already-normalised score through a sigmoid compresses it toward
0.5–0.73 and distorts exactly the ratio geometry the cardinality sweep measures. So
the default is `auto`: sigmoid **only** when output is not already in [0, 1].

**`minmax` was rejected as a universal transform** despite being the most
scale-agnostic option. It forces the top score to exactly 1.0, which makes
`RelativeGap(r)` and `ScoreThreshold(r)` compute the same function — silently
erasing the distinction between two strategies under comparison.

**Ordering sanity check, run on the real 50×53 matrix for all six model/text
combinations.** Sigmoid is strictly monotonic and min-max is affine with positive
scale, so per-query ordering must be identical under every transform. It was, in all
six. The sweep aborts if it ever is not, because a difference there would be a
transform bug rather than a modelling choice.

**Saturation check — the concern was real but the answer is negative.**

| model / text | top-1 median | top-2 median | ratio median | ratio IQR | ratio > 0.95 |
|---|---|---|---|---|---|
| ms-marco/titled | 0.966 | 0.440 | 0.799 | 0.552 | 0.36 |
| ms-marco/body | 0.974 | 0.586 | 0.791 | 0.563 | 0.40 |
| bge/titled | 0.980 | 0.409 | 0.574 | 0.628 | 0.24 |
| mxbai/body | 0.669 | 0.428 | 0.820 | 0.270 | 0.24 |

If sigmoid had saturated, ratios would cluster near 1.0 with a tiny IQR. Instead the
IQR is 0.27–0.63 and only 18–40% of ratios exceed 0.95. **`RelativeGap` had ample
room to discriminate.** Its failure in D-022 is therefore a real finding about the
problem, not an artefact of the scale.

---

## D-022 — Cardinality is an intent decision, not a score decision

**Date:** 2026-08-08 · **Phase:** 2 · **Status:** settled — this is the Phase 3 mandate

**Decision.** `topk-1` remains the default. No adaptive strategy is adopted.

| strategy | mean citations | cite-exact (mean) | best cite-exact | cite-F1 | doc F1 |
|---|---|---|---|---|---|
| **topk-1** | 1.000 | **0.5250** | **0.5600** | **0.6394** | 0.7964 |
| gap-0.97 | 1.253 | 0.4083 | 0.4800 | 0.6086 | 0.7697 |
| gap-0.95 | 1.356 | 0.3617 | 0.4200 | 0.5981 | 0.7604 |
| thresh-0.9 | 1.475 | 0.3250 | 0.4000 | 0.5840 | 0.7429 |
| gap-0.9 | 1.484 | 0.3217 | 0.3800 | 0.5819 | 0.7419 |
| thresh-0.8 | 1.697 | 0.2617 | 0.3200 | 0.5695 | 0.7428 |
| topk-2 | 2.000 | 0.0817 | 0.1200 | 0.5403 | 0.7535 |

Best adaptive minus `topk-1` on citation-exact: **−0.0800**. Not merely inside the
noise band — *below* the fixed baseline, and monotonically so: every strategy that
cites more, scores worse.

**This is the answer the phase was built to get, and it is the negative one.** Phase
1 left cardinality open because bi-encoder scores were uncalibrated and the F1 curve
was flat. Cross-encoder scores are genuinely discriminative (D-021 proves the ratio
scale had room), and a score-gap rule still cannot separate the 36 single-citation
questions from the 14 two-citation ones.

**Consequence for Phase 3.** Whether a question needs one citation or two is not
recoverable from retrieval scores. It is a property of what the question is *asking*
— "what's different about X versus Y" wants two sources because it is a comparison,
not because two sections score similarly. So the router must **classify intent**,
not threshold scores. That is now an evidence-backed requirement rather than a
design preference.

---

## D-023 — Reranking lifts adversarial questions but does not close the gap

**Date:** 2026-08-08 · **Phase:** 2 · **Status:** settled — second Phase 3 mandate

Document F1 by kind, Phase 1 winner vs Phase 2 winner:

| kind | n | Phase 1 | Phase 2 | Δ |
|---|---|---|---|---|
| lookup | 29 | 0.8391 | **1.0000** | +0.1609 |
| adversarial | 14 | 0.4500 | **0.6429** | +0.1929 |
| **gap (lookup − adversarial)** | | **0.3891** | **0.3571** | −0.0320 |

**Lookup retrieval is now perfect** — all 29 questions, correct document set. That is
the clearest single win of the phase.

**Adversarial improved by nearly as much in absolute terms (+0.193) but the gap
barely moved**, because lookup improved too. Reading the gap alone would have been
misleading in both directions: adversarial questions genuinely got much better, and
they are still by far the weakest class. Adversarial citation-F1 is 0.4524 against
lookup's near-perfect retrieval.

The smallest adversarial gap anywhere in the sweep is 0.1728, but that belongs to a
configuration with weaker lookup performance — it narrows the gap by getting worse
at the easy questions, which is not an improvement.

**Why this is structural.** Adversarial questions ask about policy boundaries whose
sections share vocabulary across four documents — "cannot be configured or
bypassed", "what support will never do", "what this document does not do". A
reranker scores relevance to the question *as asked*; when a student asks how to
disable face verification, the genuinely most relevant passage is the one about face
verification, not the one stating the policy cannot be bypassed. **Better ranking
cannot fix this because the ranking is not wrong — the task is.** Confirmed by Q31:
its expected section sits at rank 39 of 53 under the cross-encoder.

Adversarial questions need intent classification **before** retrieval, in Phase 3.

---

## D-024 — Alternate-key evidence: two independent model families bury the same four sections

**Date:** 2026-08-08 · **Phase:** 2 · **Status:** EVIDENCE — the key is unchanged, the call is the author's

`answer_key.yaml` was not modified. Ranks are from the winning config, exhaustive
over 53 sections.

| Q | reading | sections and ranks | verdict |
|---|---|---|---|
| **Q14** | primary | 01 §3 rank **1** · 10 §4 rank **45** | |
| | *single-source-login* | 01 §3 rank **1** | **BETTER** |
| **Q28** | primary | 07 §3 rank **2** · 08 §1 rank **15** | |
| | *policy-only* | 07 §3 rank **2** | **BETTER** |
| | *policy-plus-aftermath* | 07 §3 rank 2 · 07 §5 rank **18** | worse |
| **Q31** | primary | 07 §4 rank **39** | |
| | *prohibition-plus-boundary* | 07 §1 rank **8** · 07 §4 rank **39** | same worst-rank |
| **Q35** | primary | 06 §3 rank **12** · 09 §4 rank **18** | |
| | *handoff-sections* | 06 §4 rank **8** · 09 §4 rank 18 | same worst-rank |
| | *warning-vs-termination-only* | 06 §3 rank **12** | **BETTER** |

**The strongest signal is Q14.** Its first citation ranks **1st of 53**; its second
ranks **45th of 53** — in the bottom sixth of the corpus. A cross-encoder asked
whether "can my roommate receive my OTP and read it out to me" is answered by "A Note
on What Support Will Never Do" places 44 sections above it.

**Q28** is nearly as clear: 07 §3 at rank 2, 08 §1 at rank 15. The alternate
`policy-plus-aftermath` (07 §5) fares *worse* at rank 18, which is itself
informative — it suggests the re-attempt half of that question may have no good home
in the corpus at all, consistent with doc 08's Overview scoping itself to technical
interruptions.

**Q31 is the one where the cross-encoder does not rescue any reading.** Both the
primary and the alternate put 07 §4 at rank 39. The alternate's other section (07 §1,
"Prohibited Actions") lands at rank 8, so if Q31 has a retrievable answer it is that
one — but neither reading is well-supported by the model.

**Why this is worth more than the Phase 1 signal.** Phase 1 flagged these four using
a bi-encoder, which measures surface similarity — a weak witness. A cross-encoder
models question-document relevance directly and is a genuinely different kind of
evidence. **Both bury the same four sections, and the cross-encoder buries them
deeper** (Q14: rank 13 → 45; Q31: rank 14 → 39). Two independent method families
agreeing is much stronger than one method twice.

This does not prove the readings wrong. Three of the four are adversarial or
multi-doc questions where the second citation is a *policy boundary* — exactly the
class D-023 shows rerankers systematically under-rank. The finding is genuinely
confounded with that weakness, and both explanations remain live.

---

## D-025 — OPEN RISK: the grader's matching method is the largest unknown in the project

**Date:** 2026-08-08 · **Phase:** 2 · **Status:** OPEN — highest-value leaderboard probe

**The measurement.** Across the cardinality sweep, moving from `topk-1` to `topk-2`:

| | citation-exact | citation-F1 |
|---|---|---|
| Phase 1 (bi-encoder) | 0.3589 → 0.0479 (**−31 pp**) | 0.4633 → 0.4642 (+0.1 pp) |
| Phase 2 (cross-encoder) | 0.5250 → 0.0817 (**−44 pp**) | 0.6394 → 0.5403 (−9.9 pp) |

**If the grader scores exact set match, citation cardinality is worth up to 44
points. If it scores F1, it is worth ~10.** No configuration choice anywhere in this
project comes close to that swing, and we do not know which applies.

**It also changes which configuration to ship.** The top-5 configurations by
citation-exact and by citation-F1 overlap on only **1 of 5**. This is not a tuning
choice we can defer — the two metrics prefer different systems.

**Probe design.** Two submissions differing only in `topk-1` vs `topk-2`. Under
exact-set-match grading the scores should differ by roughly 30–44 points on the
citation dimension; under F1 grading by under 10. This is a cleaner and more
valuable probe than D-004 (doc extension) or D-005 (section format), and should be
spent first.

**Until it is resolved, `topk-1` is the default** — it wins on citation-exact by a
wide margin and loses on citation-F1 only slightly, so it is the choice that is least
bad under the unfavourable branch. That is a decision made under uncertainty, not a
measured optimum.
