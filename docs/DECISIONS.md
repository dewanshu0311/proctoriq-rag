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
