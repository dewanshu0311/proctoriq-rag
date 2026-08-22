# Final Submission — Selected

**Final standing: 80.15 public, 2nd place. Leader 84.98.**

Two submissions selected manually. Kaggle's auto-selection was **not** allowed to
choose: it picks by best public score, and the public set is ~20 questions against
a private 30 that decides the rank.

| slot | submission | public | what it is |
|---|---|---|---|
| 1 | `probe-7-subsection-fix-retry` | **80.15** | refusals + subsection fix |
| 2 | `probe-3-section-number` | **79.27** | locked extractive, deterministic, no LLM |

**Note on slot 2:** the deterministic arm is **probe 3**, not probe 1. Probe 1 ran
with `full_header` and scored 68.02 — it predates the format resolution. Probe 3 is
the first submission at the fully corrected format (no `.md`, `number_only`,
`topk-1`) and is the artefact the `v1.0-locked-79.27` tag reproduces.

---

## Private scoring: resolved

**Output-scored, not re-executed.** The notebook runs once at commit time and its
committed `submission.csv` is scored against both splits.

Confirmed three ways: the project owner's direct experience in a prior competition
(rank 1 public, 5th private, with no re-run — the same artefact scored against the
held-out 60%); the starter notebook's own instruction to submit from the committed
Output rather than an uploaded CSV; and `test.csv` containing all 50 questions with
no hidden split to re-execute against.

**Consequence: candidate B's Groq dependency is retrospective only.** The artefact
is frozen the moment it is committed. A future model deprecation — which has
already happened once mid-competition (D-041) — cannot break a selected submission
after the fact. That removes the single largest objection to shipping the LLM arm,
and it is why both slots are used rather than pairing two deterministic variants.

The pairing is still a narrow hedge, and worth being honest about: both arms share
`pip install`, a HuggingFace cross-encoder download, and Internet: On at *commit*
time. What differs is only the Groq dependency. Post-commit, neither can break.

---

## Why probe 7 was chosen — mechanism, not the 0.88 gap

**The public gap is not the argument.** 80.15 versus 79.27 is 0.88 points on a
~20-question split, roughly one question's worth of movement. Both contributing
probes were individually inconclusive: probe 6a landed inside its pre-registered
(−1, +3) band, and probe 7 landed *below* even its shrunk band. Four consecutive
probes have come in at 0.93 → 0.70 → 0.37 → 0.21 of predicted effect. Treating
+0.88 as evidence of superiority would be exactly the error this project has spent
five phases guarding against.

**The argument is that probe 7 fixes two identified classes of error**, each found
against the hand-built answer key *before* any leaderboard submission:

1. **Policy dumps instead of refusals.** Extractive mode pasted policy prose on
   adversarial questions — the right source text, but not a refusal. It does not
   decline, does not address the student, does not explain. Integrity-refusal is
   15% and is scored against a *correct refusal*; reciting a policy cannot score
   well there however similar the text looks. Measured: **8/14 declining extractive
   versus 13/14 with the refusal template**, with groundedness essentially at parity
   (−0.003).

2. **The wrong subsection passage.** Q03 asked about "Unspecified Error" and
   received the "Session Start Error" text. The citation was correct either way, so
   this cost nothing on the 35% citation half and everything on the other 65%. Fixed
   by scoring `###` headers with the cross-encoder rather than lexical overlap.

Both are corrections to *behaviour that was demonstrably wrong*, validated against
an independent holdout before the grader ever saw them. That is a better basis for
selection than a sub-noise public delta, and it is the reason probe 7 takes slot 1
rather than being dismissed as unmeasurable.

**The private split favours it slightly.** 30 questions holds ~8–9 adversarial
rather than the ~5.6 in the public 20, so a genuine per-question refusal effect has
more room to appear there than it did here.

---

## Slot 1 — `probe-7-subsection-fix-retry`

**Config:** `ROUTER_ENABLED = True`, `REFUSAL_ENABLED = True`,
`SUBSECTION_FIX = True`, `EXPECTED_ARM = "refusals-plus-subsection"`,
`GROQ_MODEL = "openai/gpt-oss-120b"`. Format flags at the locked values.

**Runtime dependencies (commit time only):** pip install, HuggingFace cross-encoder
(~90 MB), Internet: On, Groq API with a key in Secrets, ~68 LLM calls.

**Failure modes — all fail loudly and write nothing:** model deprecated (preflight
raises), key unresolvable (guard raises), router falling back to keywords (guard
raises on `stats["llm"] == 0`), refusal truncated (guard raises), arm mismatch
(`EXPECTED_ARM` raises). The failure mode is *no submission*, never a degraded one.
That was hard-won across four distinct wrong-arm mechanisms in four days.

**Reproducibility: not bit-identical.** Kaggle fired refusals on 18 questions;
the local run fired on 17, differing on Q28. Irrelevant post-commit, since the
artefact is frozen — but it means the arm names a family of runs, not one output.

---

## Slot 2 — `probe-3-section-number` (`v1.0-locked-79.27`)

**Config:** all router/refusal/subsection flags off,
`EXPECTED_ARM = "extractive-locked"`. Reproduce with
`git checkout v1.0-locked-79.27 && python scripts/export_notebook.py`.

**Runtime dependencies:** pip install, HuggingFace cross-encoder, Internet: On.
**No LLM, no API key, no Groq.**

**Reproducibility: bit-identical.** Verified — the notebook's locked arm matched
the tagged submission with **0 of 50 rows differing**.

**Public decomposition**, derived from the probe deltas: retrieval 15.66/20,
citation 11.25/15, answer half 52.36/65 → 79.27 exactly.

**Expected on the private 30:** essentially unchanged. Nothing in this arm is tuned
to the public 20 — the format decisions were resolved by probe and apply uniformly,
and the retrieval configuration sits on a plateau where 217 of 312 candidates were
within one noise band. This is the arm least likely to surprise in either
direction.

---

## What was deliberately not done

The remaining measured headroom is ~8 points on citations and ~12 on answers, and
it was not pursued. Everything cheap has been measured; what remains is either
inconclusive (refusals, subsection fix) or negative (HyDE, RAG Fusion, router score
biasing, cardinality routing). With four probes landing at a falling fraction of
their predicted effect, further tuning would be optimising against noise on a
20-question public sample — which is precisely how a rank-1 public position became
5th private last time.

Defending 2nd with one reproducible floor and one mechanism-justified upside is the
better expected outcome.
