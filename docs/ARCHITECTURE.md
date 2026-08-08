# Architecture

## The one diagram that matters

```
                    ┌───────────────────────────────────────┐
                    │  data/raw/kb/*.md   (10 documents)    │
                    └──────────────────┬────────────────────┘
                                       │
                              corpus/loader.py
                          split on ## only, never ###
                                       │
                                       v
                        ┌──────────────────────────┐
                        │  Corpus: 53 sections     │
                        └────────┬─────────────────┘
                                 │
       ┌─────────────────────────┼──────────────────────────┐
       │                         │                          │
       v                         v                          v
  retrieval/               generation/              submission/writer.py
  routing/                 (Phase 1+)          applies the format flags,
  (Phase 1+)                                   validates, writes 50 rows
       │                                                    │
       └────────────────► Prediction ───────────────────────┘
                               │
                               │  (measurement only — one way)
                               v
                     ┌──────────────────────┐
                     │  evaluation/         │◄──── data/validation/
                     │  scorer.py           │      answer_key.yaml
                     └──────────────────────┘      (HOLDOUT)

        The arrow into evaluation/ never reverses.
        Enforced by tests/test_no_key_leakage.py.
```

## Package responsibilities

| Package | Responsibility | Phase |
|---|---|---|
| `config` | Configuration, including the two unresolved submission-format flags | 0 |
| `corpus` | Parse markdown into documents and `##` sections | 0 |
| `submission` | Serialize predictions to `submission.csv`; the only place format flags apply | 0 |
| `evaluation` | Answer-key loading and local scoring. Measurement only | 0 |
| `retrieval` | Vector store, hybrid search, reranking, HyDE, RAG-Fusion | 1+ |
| `routing` | Query classification and retrieval-strategy selection | 1+ |
| `generation` | Prompt construction, LLM invocation, citation extraction | 1+ |

## Three invariants

**1. The evaluation boundary is one-way.** `evaluation/` may import the pipeline;
the pipeline may never import `evaluation/`. This is what keeps
`answer_key.yaml` a valid measurement rather than a training signal, and it is
checked per-module by an AST walk rather than by convention. See
[DECISIONS.md](DECISIONS.md) D-002.

**2. Format decisions live at the edge.** Everything internal works in canonical
form — document IDs are bare file stems, section titles are verbatim `##` header
text. The translation into whatever shape the grader expects happens once, in
`submission/writer.py`, at serialization time. Resolving either open format
question is a one-line change to `config/default.yaml`, not a code change.
See D-004 and D-005.

**3. Unmeasurable is not zero.** Two of the five scored dimensions have no ground
truth available to us. They report `None` and are excluded from the composite,
which is published alongside an explicit statement of how much of the real metric
it covers. See D-008.

## Data flow through a single question

1. `test.csv` supplies `question_id` and `question`.
2. *(Phase 1+)* routing classifies, retrieval fetches sections, generation writes
   an answer and names its sources.
3. The result becomes a `Prediction`: an answer plus positionally-aligned
   `cited_docs` and `cited_sections`, both in canonical form.
4. `submission/writer.py` applies the format flags, validates the whole batch, and
   writes `submission.csv` — or refuses to write and raises.
5. Separately and offline, `evaluation/scorer.py` compares predictions against the
   holdout key and produces a report whose first section is the list of
   mismatches.

Step 5 never feeds back into steps 2–4 automatically. A human reads the report and
changes the pipeline; the key does not.

## Scoring: what is real and what is a proxy

| Dimension | Weight | Local status |
|---|---|---|
| Answer accuracy | 25% | Unmeasured. Optional `reference_answers.yaml` fills it |
| Groundedness | 25% | Proxy. Faithful method, uncalibrated magnitude — D-007 |
| Retrieval quality | 20% | **Exact** |
| Citation accuracy | 15% | **Exact** |
| Integrity-refusal | 15% | Unmeasured, adversarial subset only (14 of 50) |

35% of the metric is measured exactly. That is the number to trust; everything
else on the report is context.

## Corpus shape

53 `##` sections across 10 documents:

| Document | Sections |
|---|---|
| `01_windows_installation_login_guide` | 5 |
| `02_mac_installation_login_guide` | 6 |
| `03_mock_test_device_readiness_guide` | 5 |
| `04_network_connectivity_issue_playbook` | 6 |
| `05_device_system_issue_playbook` | 5 |
| `06_camera_microphone_face_verification_guide` | 5 |
| `07_permitted_prohibited_actions_policy` | 6 |
| `08_reattempt_request_policy` | 5 |
| `09_status_verification_outcome_guide` | 5 |
| `10_general_support_escalation_guide` | 5 |

The count is asserted in `tests/test_corpus_loader.py` so a parsing regression
fails loudly instead of quietly shifting every citation.

## Answer key composition

50 entries, Q01–Q50, hand-built by reading all ten documents end to end.

| Kind | Count | | Confidence | Count |
|---|---|---|---|---|
| `lookup` | 29 | | `high` | 36 |
| `adversarial` | 14 | | `medium` | 11 |
| `multi_doc` | 5 | | `low` | 3 |
| `multi_section` | 1 | | | |
| `trap` | 1 | | | |

The three low-confidence entries — Q14, Q28, Q35 — are the ones
`scripts/diagnose_sections.py` exists to gather evidence on.
