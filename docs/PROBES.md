# Leaderboard Probe Plan

Three questions cannot be answered locally. Each is worth more than every
configuration decision made in Phases 0–2 combined, and each needs a submission.

| Unknown | Decision | Local evidence |
|---|---|---|
| Does `cited_docs` carry `.md`? | D-004 | conflicting: `sample_submission.csv` omits it, the description says "filename" |
| What shape is `cited_sections`? | D-005 | `sample_submission.csv` shows `Section 1`, but it is dummy data repeated 50× |
| Does the grader match citations exactly or by F1? | D-025 | **none** — locally invisible, and worth up to 44 points |

**Nothing is uploaded by any script in this repo.** Probe files are generated into
`outputs/probes/` for inspection; submitting is a manual act.

---

## The one property that makes this work

Every probe submission holds **`answer_text` byte-identical**. Only the citation
columns change. So the entire observed score delta is attributable to the 35%
citation half (retrieval 20% + citation 15%), with answer accuracy, groundedness
and integrity-refusal held constant.

Two design choices enforce this:

1. **Extractive answers.** Deterministic, no model. An LLM cannot guarantee
   byte-identical output across separate Kaggle runs even at temperature 0.
2. **`ANSWER_FROM = "top1"`.** The answer is built from the top-ranked section
   regardless of how many sections are cited. Without this, probe 5 (`topk-1` →
   `topk-2`) would change the cited set, change the answer, and stop being a
   citation-only experiment.

Verified mechanically — `scripts/run_pipeline.py --probes` hashes the answers of
every variant and refuses to proceed unless all hashes match:

```
p1-baseline          b925179f4df75b6d
p2-doc-extension     b925179f4df75b6d
p3-section-number    b925179f4df75b6d
p4-section-title     b925179f4df75b6d
p5-topk2             b925179f4df75b6d
p6-topk3             b925179f4df75b6d
PASS — answer_text is byte-identical across every probe.
```

---

## Pre-flight checks — do these before probe 1

- [ ] **Confirm the competition's internet policy on the competition page.** The
      notebook downloads a cross-encoder from HuggingFace and `pip install`s three
      packages. The starter notebook's own `!pip install` strongly implies internet
      is permitted, but that is an inference, not a fact. **If internet must be
      off, the pipeline dies on submission rather than locally** — the most
      expensive possible place to discover it.
      *Fallback already built:* `resolve_model_source()` searches `/kaggle/input`
      for a local copy of the model before falling back to the hub, so attaching
      `ms-marco-MiniLM-L-6-v2` as a Kaggle Dataset requires no code change.
- [ ] Confirm the public/private split (assumed 40% public ≈ 20 questions).
- [ ] Confirm the daily submission cap (assumed 5).
- [ ] Run `Save & Run All (Commit)` once and confirm `submission.csv` has 50 rows.

---

## The sequence is a dependency chain, not a menu

**Citation accuracy is scored on `(doc, section)` pairs.** If the document format
is wrong, citation scores zero *regardless of which section string is used*. So
running the section probes against a broken document baseline would show
near-zero delta for **both** variants and read as "section format doesn't matter"
— a false negative that would cost us the 15% citation dimension permanently.

Each probe therefore adopts the winner of the previous one as its new baseline.

```
  ┌─────────────────────────────────────────────────────────────────┐
  │ PROBE 1 — baseline                                              │
  │   no .md · full_header · topk-1 · extractive                    │
  │   establishes the reference score                               │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
  ┌───────────────────────────▼─────────────────────────────────────┐
  │ PROBE 2 — cited_docs WITH .md            resolves D-004         │
  │   predicted: −27 (baseline right) │ +27 (baseline wrong) │ ~0 (fuzzy) │
  └───────────────────────────┬─────────────────────────────────────┘
                              │  ADOPT the winning document format
                              │  ↓ rebuild baseline before continuing
  ┌───────────────────────────▼─────────────────────────────────────┐
  │ PROBE 3 — sections as "Section N"        resolves D-005 (a)     │
  │ PROBE 4 — sections as title only         resolves D-005 (b)     │
  │   predicted: −10 each if full_header is right                   │
  └───────────────────────────┬─────────────────────────────────────┘
                              │  ADOPT the winning section format
                              │  ↓ baseline is now fully format-corrected
  ┌───────────────────────────▼─────────────────────────────────────┐
  │ PROBE 5 — topk-2                          resolves D-025        │
  │   predicted: −12.8 exact-match │ −3.4 F1                        │
  └───────────────────────────┬─────────────────────────────────────┘
                              │  ONLY IF probe 5 is inconclusive
  ┌───────────────────────────▼─────────────────────────────────────┐
  │ PROBE 6 — topk-3            amplified re-test of D-025          │
  │   predicted: −19.2 exact-match │ −6.9 F1                        │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Predicted deltas, computed in advance

All from the Phase 2 winning configuration measured against the answer key
(`ms-marco-MiniLM-L-6-v2 · body · exhaustive`). Composite weights: retrieval 20,
citation 15.

| cardinality | cite-exact | cite-F1 | doc-exact | doc-F1 | mean citations |
|---|---|---|---|---|---|
| `topk-1` | 0.5600 | 0.6667 | 0.7000 | 0.8467 | 1.00 |
| `topk-2` | 0.0800 | 0.5500 | 0.4200 | 0.7633 | 2.00 |
| `topk-3` | **0.0000** | 0.4580 | 0.1600 | 0.6560 | 3.00 |

### Probe 2 — document extension

If the format is wrong, **both** retrieval and citation collapse to zero, because
a citation is a `(doc, section)` pair and the doc half is unmatchable.

- baseline correct → **≈ −27** under F1 grading (`20×0.847 + 15×0.667`), **≈ −22**
  under exact grading
- baseline wrong → **≈ +22 to +27**
- **≈ 0** → the grader normalises document names, and D-004 is moot

This is the largest and least ambiguous probe. A movement under 5 points here
means fuzzy matching, which is itself a valuable finding.

### Probes 3 and 4 — section format

Citation only (15%); retrieval is untouched because the document half is
unchanged.

- `full_header` correct → **≈ −10** under F1, **≈ −8** under exact
- the probed variant correct → **≈ +8 to +10**
- **≈ 0 for both** → substring or semantic matching on sections

Run both. If they move in opposite directions, the larger positive wins. If both
land near zero *after* the document format is confirmed correct, the grader is
fuzzy-matching sections and `full_header` should be kept for carrying the most
information.

### Probe 5 — cardinality and the grader's matching method

| grading | Δ retrieval | Δ citation | **predicted composite** |
|---|---|---|---|
| exact set match | `20 × (0.42 − 0.70)` = −5.6 | `15 × (0.08 − 0.56)` = −7.2 | **−12.8** |
| F1 partial credit | `20 × (0.763 − 0.847)` = −1.7 | `15 × (0.550 − 0.667)` = −1.8 | **−3.4** |

Separation: **9.4 points**.

### Probe 6 — amplified re-test, conditional

`topk-3` drives citation-exact to **exactly 0.0000**, so under exact-set matching
the citation dimension is fully extinguished while F1 grading degrades gently.

| grading | **predicted composite** |
|---|---|
| exact set match | **−19.2** |
| F1 partial credit | **−6.9** |

Separation: **12.3 points** — 30% wider than probe 5.

---

## Decision rule for probe 6, pre-registered

Written before any number is seen, so the call cannot be made after the fact.

Let `D5` be the observed composite delta from probe 5 against its baseline.

- **`D5 ≤ −9.0`** → exact-set matching. D-025 resolved. **Do not run probe 6.**
  Cardinality is worth ~31 points and `topk-1` is mandatory.
- **`D5 ≥ −6.0`** → F1 partial credit. D-025 resolved. **Do not run probe 6.**
  Cardinality is nearly free and Phase 4's router should optimise recall over
  precision.
- **`−9.0 < D5 < −6.0`** → **inconclusive. Run probe 6.** Then apply the same
  rule with thresholds `−13.0` and `−10.0`.
- If probe 6 is also inconclusive, record D-025 as **unresolved** and keep
  `topk-1`, which is the least-bad choice under the unfavourable branch.

**Why the inconclusive band exists.** The public leaderboard covers roughly 20
questions. Each question moves each dimension by about 5%, so sampling noise is
genuinely comparable to a 3-point composite difference. Predicting −12.8 vs −3.4
and observing −7.5 would tell us nothing, and the honest response is to say so
rather than to pick the nearer hypothesis.

---

## What a null result means

A near-zero delta is not a failed probe. For probes 2–4 it is positive evidence
that the grader normalises or fuzzy-matches that field — which would mean the
format questions were never worth the anxiety, and that the 35% citation half is
more forgiving than assumed. Record it as a finding, not a non-event.

---

## Running a probe

1. `python scripts/run_pipeline.py --probes` — regenerate locally and confirm the
   answer-hash check passes.
2. `python scripts/export_notebook.py` — rebuild the notebook.
3. In the notebook's **CONFIGURATION** cell, change the one flag for this probe:

   | probe | flag |
   |---|---|
   | 2 | `DOC_EXTENSION = True` |
   | 3 | `SECTION_FORMAT = "number_only"` |
   | 4 | `SECTION_FORMAT = "title_only"` |
   | 5 | `CITATION_STRATEGY = "topk-2"` |
   | 6 | `CITATION_STRATEGY = "topk-3"` |

4. **Save & Run All (Commit)**, submit from the committed Output.
5. Record in `docs/SUBMISSION_LOG.md` **before** looking at the score: the
   hypothesis and the predicted delta. Then add the observed score at full
   displayed precision.
