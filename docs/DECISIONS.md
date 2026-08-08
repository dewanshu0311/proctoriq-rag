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
