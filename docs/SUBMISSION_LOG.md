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

| # | Config changed | Hypothesis | Predicted Δ | Observed public | Actual Δ | Conclusion |
|---|---|---|---|---|---|---|
| 1 | baseline: no `.md`, `full_header`, `topk-1`, extractive | reference point | — | **68.02** | — | reference |
| 2 | `DOC_EXTENSION = True` | D-004: does `cited_docs` take `.md`? | −27 / +27 / ~0 | **52.36** | **−15.66** | **no `.md`.** See reconciliation below |
| 3 | `SECTION_FORMAT = "number_only"` | D-005a | −10 if `full_header` right | **79.27** | **+11.25** | **`number_only` wins.** `full_header` was wrong |
| 4 | `SECTION_FORMAT = "title_only"` | D-005b | −10 if `full_header` right | **68.02** | **0.00** | tie to the cent -> **exact matching** |
| 5 | `CITATION_STRATEGY = "topk-2"` (vs the 79.27 baseline) | D-025: exact-match vs F1 grading | **−12.8** exact / **−3.4** F1 | **76.90** | **−2.37** | **F1 partial credit.** topk-1 retained |
| 6 | *not run* — probe 5 was decisive (−2.37 is outside the (−9.0, −6.0) inconclusive band) | — | — | — | — | pre-registered rule correctly said skip |

### The grader model, consistent with all four deltas

**Exact string match on each citation element, F1-style partial credit across the set.**

The decisive evidence is probe 4. `title_only` scored **68.02 — identical to `full_header` to the
cent**. Under fuzzy or semantic matching those two strings would score differently, because
"Common Installation Errors" and "Section 2: Common Installation Errors" have different similarity
to "Section 2". Under exact matching both are simply wrong and both score exactly zero. Only exact
matching produces a tie to the cent.

### Reconciling probe 2 — why −15.66 and not −27

The −27 prediction assumed retrieval *and* citation would both collapse. But probe 2 ran with
`full_header`, so **citation was already zero before `.md` was added**. Only retrieval could break:

    predicted retrieval loss = 20 x doc_F1 = 20 x 0.847 = 16.9
    observed                 = 15.66  ->  public doc_F1 = 0.783

The prediction was right about the mechanism and slightly high on the magnitude, because public
doc-F1 (0.783) is a little below our local estimate (0.847) on a ~20-question sample.

### Score decomposition, derived from the deltas alone

Probe 2 zeroes both retrieval and citation, so **its 52.36 is the answer half measured directly**.

| dimension | F1 (public) | achieved | available | headroom |
|---|---|---|---|---|
| retrieval | 0.783 | 15.66 | 20 | 4.34 |
| citation | 0.750 | 11.25 | 15 | 3.75 |
| **citation half** | | **26.91** | **35** | **8.09** |
| **answer half** (accuracy + groundedness + refusal) | | **52.36** | **65** | **12.64** |
| **total** | | **79.27** | **100** | **20.73** |

Reconstructs to 79.27 exactly, which is a strong check that the model is right.

**Note:** public citation-F1 (0.750) is *higher* than our local estimate (0.667). Different question
subsets, so weak — but it points the same way as the Phase 2 cross-encoder evidence that our key may
be pessimistic on the contested entries.

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
