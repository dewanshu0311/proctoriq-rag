# Final Submission Recommendation

**Standing: 80.15, 2nd place. Leader 84.98.** We are defending a strong position,
not chasing from behind — which weights variance reduction over upside.

Kaggle selects two submissions. **Select both manually.** Auto-selection picks by
best public score, which is exactly how the previous competition was lost: the
public set is ~20 questions and the private 30 decides the rank.

---

## The uncomfortable part first: the two arms are not a hedge

The brief asked whether pairing `v1.0-locked-79.27` with
`refusals-plus-subsection` is a hedge. **It is, but not in the way "two
submissions" suggests** — and the reason matters more than the conclusion.

| | locked extractive | refusals-plus-subsection |
|---|---|---|
| cross-encoder from HuggingFace | **yes** | **yes** |
| `pip install` at runtime | **yes** | **yes** |
| Internet: On | **yes** | **yes** |
| Groq API | no | **yes** |
| a specific Groq model existing | no | **yes** |
| bit-reproducible output | **yes** | **no** |

They share three of six runtime dependencies. A Kaggle image change, a HuggingFace
outage, or internet being disabled at scoring time **takes out both simultaneously**.
The pairing hedges exactly one axis: the Groq dependency, which is the one that has
*already failed once* (D-041).

So: **pair them, but understand the hedge is narrow.** It protects against the
failure we have actually observed, and against nothing else.

---

## Does Kaggle re-execute at private scoring?

**This is the question that determines whether a post-deadline Groq deprecation can
break a selected submission, and I could not resolve it from the competition page.**

Kaggle notebook competitions come in two shapes:

- **Output-scored** (the common case for this format): you commit the notebook, it
  runs *once* at commit time, and `submission.csv` from that committed Output is
  what gets scored — against both public and private splits. **A later deprecation
  cannot break it.** The artefact is frozen.
- **Re-execution** (used in code competitions with hidden test sets): Kaggle re-runs
  the notebook against data you never see. **A later deprecation breaks it outright.**

**The evidence points to output-scored.** The starter notebook's own instructions
say *"Submitted from this notebook's committed Output, not an uploaded CSV"*, and
`test.csv` is fully visible with all 50 questions — there is no hidden split to
re-execute against. Under that reading the private 30 are simply the other 60% of
the same 50 questions we already answer.

**But this is inference, not a fact I verified, and it is load-bearing.** Please
confirm on the competition page before the deadline:

- If **output-scored**: the Groq risk is retrospective only. Both arms are safe once
  committed, and the pairing argument weakens considerably — the LLM arm's main
  liability disappears.
- If **re-executed**: `refusals-plus-subsection` is a **live hazard**. A model
  deprecation after the deadline turns it into a `RuntimeError` and a zero. Under
  that reading I would **not** select it, and would pair `v1.0-locked-79.27` with
  `extractive-subsection` instead — capturing the subsection fix with no LLM
  dependency at all.

---

## Candidate A — `v1.0-locked-79.27` (low variance)

**Config:** all flags off. `EXPECTED_ARM = "extractive-locked"`.
**Tagged:** `git checkout v1.0-locked-79.27`.

**Runtime dependencies:** `pip install` of sentence-transformers, one HuggingFace
cross-encoder download (~90 MB), Internet: On. **No LLM, no API key, no Groq.**

**Failure modes on Kaggle:**
- Internet disabled → model download fails. *Mitigated:* `resolve_model_source`
  searches `/kaggle/input` first, so attaching the model as a Dataset needs no code
  change.
- HuggingFace outage at commit time → same mitigation.
- Nothing else. There is no third-party API in this path.

**Reproducibility:** bit-identical across runs. Verified — the notebook's locked arm
matched the tagged submission with **0 of 50 rows differing**.

**Public: 79.27.** Decomposed from the probe deltas: retrieval 15.66/20, citation
11.25/15, answer half 52.36/65.

**Expected on the private 30:** essentially the same. Nothing in this arm is tuned
to the public 20 — the format decisions were resolved by probe and apply uniformly,
and the retrieval config sits on a plateau where 217 of 312 configurations were
within one noise band. **This is the arm least likely to surprise.**

---

## Candidate B — `refusals-plus-subsection` (higher expected value)

**Config:** `ROUTER_ENABLED = True`, `REFUSAL_ENABLED = True`,
`SUBSECTION_FIX = True`, `EXPECTED_ARM = "refusals-plus-subsection"`.

**Runtime dependencies:** everything in A, **plus** the Groq API, a valid key in
Secrets, and `openai/gpt-oss-120b` continuing to exist. ~68 LLM calls per run.

**Failure modes on Kaggle:**
- Groq model deprecated → **preflight raises**, no submission written. *This has
  already happened once, mid-competition, during active use.*
- Key not resolvable → **guard raises**, no submission written.
- Router falls back to keywords → **guard raises** (`stats["llm"] == 0`).
- Refusal truncates → **guard raises**.
- Rate limiting mid-run → retry with backoff, then raise.

Every one of these fails **loudly and writes nothing** — which is the correct
behaviour and was hard-won, but it means the failure mode is *no submission*, not a
degraded one.

**Reproducibility: NOT bit-identical.** Kaggle fired refusals on 18 questions; local
fired on 17. Same config, same temperature 0, different classification on Q28.

**Public: 80.15**, i.e. **+0.88 over A**, accumulated as +0.56 (refusals) and +0.32
(subsection fix). **Both individually inconclusive**: probe 6a landed inside its
pre-registered (−1, +3) band, and probe 7 below even its shrunk band.

**Expected on the private 30:** modestly better than A, with wider error bars. The
refusal mechanism argument is stronger than the measurement — 13/14 declining vs
8/14 — and the private split holds ~8–9 adversarial questions rather than ~5.6, so
a real per-question effect has more room to show. But the same instability that
gave 17-vs-18 locally applies there too.

---

## Recommendation

**Select A and B.** A is the floor; B is the upside, and its one unshared
dependency is the one that has already failed.

**Two conditions on B:**

1. **Verify the run before accepting it.** The Logs tab must show
   `DERIVED ARM: refusals-plus-subsection`, `router: {'llm': 50, ...}`, and
   `refusal fires on N questions` with N in the 16–19 range. If the summary shows
   anything else, the arm did not run — and four wrong-arm submissions in four days
   is the reason that check exists.
2. **If Kaggle re-executes at private scoring, drop B** and select
   `extractive-subsection` instead (`SUBSECTION_FIX = True`, everything else off,
   `EXPECTED_ARM = "extractive-subsection"`). That keeps the +0.32 measured
   improvement with zero LLM dependency. It has never been submitted, so it is
   unmeasured as a standalone arm — but it is strictly A plus one changed answer,
   and the risk of that regressing is very low.

**What I would not do:** chase the 4.83-point gap to the leader. The remaining
measured headroom is ~8 points on citations and ~12 on answers, but four
consecutive probes have landed at 0.93 → 0.70 → 0.37 → 0.21 of their predicted
effect. Everything cheap has been measured; what is left is either inconclusive or
negative. Defending 2nd with a reproducible floor is the better expected outcome
than spending the remaining days on effects we can no longer distinguish from
noise.
