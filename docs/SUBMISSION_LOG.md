# Submission Log

One row per submission. **Fill in the hypothesis and predicted delta before
looking at the score** — a prediction written afterwards is not a prediction, and
the whole value of the probe sequence is that a small movement becomes readable
only when you already know what magnitude was expected.

**Record public scores at full displayed precision.** With roughly 20 public
questions, one question moves one dimension by about 5%, so the signal that
distinguishes hypotheses lives in the decimals Kaggle shows. Rounding to whole
points would discard exactly the information the probes exist to collect.

> A good public score is a measurement, not proof. The private leaderboard is 60%
> of the questions and decides the rank. Treating a public number as confirmation
> is how the last competition was lost.

---

## Log

| # | Date | Config changed | Hypothesis | Predicted Δ | Observed public | Actual Δ | Conclusion |
|---|---|---|---|---|---|---|---|
| 1 | | baseline: no `.md`, `full_header`, `topk-1`, extractive | reference point | — | | — | |
| 2 | | `DOC_EXTENSION = True` | D-004: does `cited_docs` take `.md`? | −27 if baseline right / +27 if wrong / ~0 if fuzzy | | | |
| 3 | | `SECTION_FORMAT = "number_only"` | D-005a | −10 if `full_header` right | | | |
| 4 | | `SECTION_FORMAT = "title_only"` | D-005b | −10 if `full_header` right | | | |
| 5 | | `CITATION_STRATEGY = "topk-2"` | D-025: exact-match vs F1 grading | **−12.8** exact / **−3.4** F1 | | | |
| 6 | | `CITATION_STRATEGY = "topk-3"` | D-025 amplified — **only if 5 inconclusive** | **−19.2** exact / **−6.9** F1 | | | |

---

## Per-dimension breakdown

If the leaderboard reports dimensions separately, record them — it collapses
several probes into one observation, because a change confined to the citation
dimension immediately rules out the fuzzy-matching hypothesis.

| # | Answer 25% | Grounded 25% | Retrieval 20% | Citation 15% | Refusal 15% | Composite |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |

---

## Budget

Cap is 5 submissions per day; roughly 20 days remain. The sequence needs 5–6,
so budget is not the constraint — **sequencing is**. Probes 3 and 4 are
uninterpretable until probe 2 resolves the document format, because citation is
scored on `(doc, section)` pairs and a broken document half zeroes the citation
dimension regardless of the section string.

Do not batch probes to save days. One variable per submission, in order.

---

## Notes column

Anything that would change how a number should be read: a Kaggle outage, a
notebook that timed out, internet disabled, a config flag set wrong. A score with
an unrecorded caveat is worse than no score, because it will be trusted later.

| # | Notes |
|---|---|
| 1 | |
| 2 | |
| 3 | |
| 4 | |
| 5 | |
| 6 | |
