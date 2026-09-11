#!/usr/bin/env python
"""Export a self-contained Kaggle notebook.

    python scripts/export_notebook.py

Kaggle cannot import our ``src`` package, so the notebook has to carry its own
copy of the pipeline. The exporter **reads the real source files** and inlines
them with intra-package imports stripped, rather than duplicating logic by hand —
one source of truth, no drift between what is tested and what is submitted.

What is inlined verbatim vs. reimplemented
------------------------------------------
Pure logic inlines from source: the corpus loader, chunkers, citation strategies,
cleaning, prompts, answerers, the reranker, and the submission writer's
formatting and validation. Only environment-bound values are rewritten — repo
paths become Kaggle paths.

The guarantee is enforced by test, not by inspection:
``tests/test_notebook_export.py`` asserts the notebook's ``format_doc`` and
``format_section`` produce output identical to the library's across all 53
section headers x 3 section formats x 2 extension settings, and executes the
whole notebook against a mocked input tree.

The export boundary is also a leakage boundary
----------------------------------------------
The build FAILS if any answer-key content, alternates content, evaluation import,
or ``Q01``-``Q50`` literal reaches the generated notebook. Same guarantee as
``tests/test_no_key_leakage.py``, enforced where the artefact leaves the repo.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "proctoriq_rag"
sys.path.insert(0, str(REPO_ROOT / "src"))

import nbformat as nbf  # noqa: E402

QUESTION_ID_LITERAL = re.compile(r"\bQ(?:0[1-9]|[1-4][0-9]|50)\b")
FORBIDDEN_TOKENS = ("answer_key", "data/validation", "answer_key_alternates")

#: Modules inlined in dependency order, with environment-bound lines rewritten.
INLINE_MODULES: list[tuple[str, list[tuple[str, str]]]] = [
    ("corpus/loader.py", []),
    ("retrieval/chunking.py", []),
    # retriever.py is inlined for ScoredChunk, which citation.py annotates against.
    # Its faiss / rank-bm25 imports are lazy and never fire in the notebook's
    # exhaustive path, so this costs nothing at runtime and removes a dependency
    # on postponed-annotation behaviour inside IPython cells.
    ("retrieval/retriever.py", []),
    ("retrieval/citation.py", []),
    ("retrieval/reranker.py", [
        (r'^DEFAULT_CACHE_DIR = .*$', 'DEFAULT_CACHE_DIR = Path(WORKING_ROOT) / "rerank_cache"'),
    ]),
    ("generation/cleaning.py", []),
    ("generation/prompts.py", []),
    ("generation/answerer.py", []),
    ("routing/router.py", []),
    ("routing/policy.py", []),
    ("generation/refusal.py", []),
]

IMPORT_LINE = re.compile(r"^\s*(from|import)\s+proctoriq_rag[\w.]*\s+import\s+.*$|^\s*import\s+proctoriq_rag.*$")
FUTURE_LINE = re.compile(r"^\s*from __future__ import .*$")
MULTILINE_PKG_IMPORT = re.compile(
    r"^from proctoriq_rag[\w.]* import \($.*?^\)$", re.MULTILINE | re.DOTALL
)


def inline_source(relative: str, replacements: list[tuple[str, str]]) -> str:
    """Read a module and strip what cannot survive outside the package."""
    text = (SRC / relative).read_text(encoding="utf-8")
    text = MULTILINE_PKG_IMPORT.sub("", text)
    kept = [
        line for line in text.splitlines()
        if not IMPORT_LINE.match(line) and not FUTURE_LINE.match(line)
    ]
    text = "\n".join(kept)
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

    # Postponed annotations are re-added as belt-and-braces against forward
    # references between inlined modules. It must go AFTER the module docstring:
    # putting it first displaces the docstring from body[0], which stops it being
    # recognised as a docstring — and the export-boundary check exempts docstrings
    # but not bare strings, so the build would fail on prose it should allow.
    text = text.strip()
    insert_at = 0
    try:
        tree = ast.parse(text)
        first = tree.body[0] if tree.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            insert_at = first.end_lineno or 0
    except SyntaxError:
        insert_at = 0

    lines = text.splitlines()
    lines.insert(insert_at, "\nfrom __future__ import annotations")
    return (
        f"# ── inlined from src/proctoriq_rag/{relative} ──\n"
        + "\n".join(lines).strip()
        + "\n"
    )


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def build_notebook() -> nbf.NotebookNode:
    cells: list[nbf.NotebookNode] = []

    cells.append(md("""
# ProctorIQ Assessment Journey — RAG Submission

A retrieval pipeline for proctored-assessment support. Given a student's question it retrieves
from a 10-document knowledge base, cites the document and section it used, and answers without
assisting in evading proctoring integrity checks.

**Pipeline:** parse markdown into `##` sections → cross-encoder reranks all sections against the
question → citation strategy picks how many to cite → answer generated from the top section →
validated `submission.csv`.

**Design decisions this notebook encodes** (full reasoning in the project's decision log):

- `###` subheadings are **content, not section boundaries**. Doc 01 §2 holds three distinct
  installation errors as `###` blocks and all three cite that one section — splitting on `###`
  would invent sections the grader has never seen.
- The cross-encoder scores **all 53 sections exhaustively**. Measured against retrieve-then-rerank
  with pools of 10/20/30 it tied to four decimal places, so this path is preferred purely for
  having fewer moving parts and no recall ceiling.
- Answer text is built from the **top-ranked section only**, independently of how many sections are
  cited. That keeps `answer_text` identical when only the citation configuration changes, which is
  what made the leaderboard probe sequence interpretable.
- The **submission format was resolved by probe**, not by reasoning: no `.md` extension, sections as
  `Section N`. Probe 4 settled the grader model — `title_only` scored *identically* to
  `full_header`, to the cent, which is only possible under exact string matching.

**Measured negative, and kept in the code as evidence rather than deleted:** HyDE (recall@10
−0.078), RAG Fusion (below the cross-encoder at 4 LLM calls per question), router score biasing
(−1.40 out of 35, and exactly 0.0000 gain on the adversarial class it was built for), and router
cardinality routing (neutral). The router ships for **refusal firing only**.

**Every failure writes nothing.** A missing key, a deprecated model, a router that fell back to
keywords, a truncated refusal, or a flag/arm mismatch each raises before `submission.csv` is
written. Four separate wrong-arm mechanisms were caught this way during development, each of which
had produced a valid-looking 50-row file that was not the arm it claimed to be.
"""))

    cells.append(md("""
## Step 0 — Install dependencies

Requires **Internet: On** in the notebook settings (Settings → Internet). If internet must be
disabled, attach the reranker model as a Kaggle Dataset — `resolve_model_source` below searches
`/kaggle/input` for a local copy before falling back to the hub.
"""))
    cells.append(code("""
!pip install -U -q groq sentence-transformers rank-bm25
"""))

    cells.append(md("""
## Step 1 — Configuration

**Every probe variable lives in this one cell.** Changing a submission is a one-line edit here.
"""))
    cells.append(code('''
# ══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — the only cell a probe needs to touch
# ══════════════════════════════════════════════════════════════════════════

# --- submission format: RESOLVED BY PROBE, do not change ---
# Probe 2: adding .md cost 15.66 points (= 20 x doc_F1 0.783).
# Probe 3: number_only gained 11.25 (= 15 x cite_F1 0.750) over full_header.
# Probe 4: title_only tied full_header to the cent -> the grader matches EXACTLY.
DOC_EXTENSION   = False          # False -> "01_windows_..."   True -> "01_windows_....md"
SECTION_FORMAT  = "number_only"  # "full_header" | "number_only" | "title_only"

# --- how many sections to cite ---
# Probe 5: blanket topk-2 cost 2.37 points. The grader gives F1 partial credit.
CITATION_STRATEGY = "topk-1"     # "topk-1" | "topk-2" | "gap-0.95" | "thresh-0.9" ...

# --- answer generation ---
GENERATION_MODE = "extractive"   # "extractive" (deterministic, no API key) | "generative"
TEMPLATE_NAME   = "answer-first-explained"
ANSWER_FROM     = "top1"         # "top1" | "top2" | "top3" | "cited"
                                 # top2/top3 widen the ANSWER's context to the top
                                 # N reranked sections while citations stay exactly
                                 # as the citation strategy chose. Measured on the
                                 # RAG Triad: answer relevancy 0.634 -> 0.918 with
                                 # generative + top2.
MAX_ANSWER_CHARS = 700

# --- router (ships for REFUSAL FIRING ONLY) ---
# Both retrieval-side uses measured negative and are off by default:
#   score biasing        26.93 -> 25.53 out of 35; adversarial gain EXACTLY 0.0000
#   cardinality routing  26.88 vs 26.93 off — neutral
# Refusal firing is what the router is for. See docs/DECISIONS.md D-034..D-037.
ROUTER_ENABLED      = False      # classify intent before answering
REFUSAL_ENABLED     = False      # reasoned refusals on policy-boundary questions
SCORE_BIAS          = "off"      # "off" | "bias" | "filter"   (measured negative)
CARDINALITY_ROUTING = False      # let intent set citation count (measured neutral)

# --- subsection selection ---
# Cross-encoder subsection selection instead of lexical overlap. Fixes one
# question whose answer came from the wrong error passage: it asked about
# "Unspecified Error" and received the "Session Start Error" text. The citation
# was correct either way, so this costs nothing on the 35% citation half and
# affects only the other 65%. Off by default so the locked baseline stays
# reproducible; probe 7 turns it on.
SUBSECTION_FIX      = False

# ══════════════════════════════════════════════════════════════════════════
#  WHICH ARM AM I RUNNING?
#
#  Set EXPECTED_ARM to the arm you intend. The pipeline derives the arm from
#  the flags actually in effect and RAISES before writing if they disagree.
#  A printed description is not a check — four wrong-arm submissions in four
#  days went out under correct-looking summaries.
#
#    "extractive-locked"         all flags off  -> reproduces v1.0-locked-79.27
#    "extractive-subsection"     SUBSECTION_FIX only
#    "...+inert-subsection"      SUBSECTION_FIX set where it has no effect. The
#                                suffix is deliberate: acknowledge it in
#                                EXPECTED_ARM or clear the flag.
#    "refusals-only"             ROUTER + REFUSAL
#    "refusals-plus-subsection"  ROUTER + REFUSAL + SUBSECTION_FIX   <- PROBE 7
#    "generative"                GENERATION_MODE = "generative", ANSWER_FROM = "top1"
#    "generative-top2"           GENERATION_MODE = "generative", ANSWER_FROM = "top2"
#
#  PROBE 7 — subsection fix on top of refusals. Set exactly:
#      ROUTER_ENABLED  = True
#      REFUSAL_ENABLED = True
#      SUBSECTION_FIX  = True
#      EXPECTED_ARM    = "refusals-plus-subsection"
#  Refusal fires on 17 questions under gpt-oss-120b (the earlier count of 19 was
#  llama-3.1-8b, which Groq has since removed). Requires GROQ_API_KEY — or any of
#  GROQ_API_KEY_1..9 — in Kaggle Secrets, and Internet: On.
# ══════════════════════════════════════════════════════════════════════════
EXPECTED_ARM = "extractive-locked"

# --- models ---
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TEXT_VARIANT = "body"            # what the cross-encoder scores: "body" | "titled"
GROQ_MODEL   = "openai/gpt-oss-120b"   # llama-3.1-8b-instant was removed by Groq (D-041)

# --- paths (overridable so the notebook can be executed and tested off-Kaggle) ---
import os

# Kaggle images ship TensorFlow. `transformers` probes for it at import time and
# raises on Keras 3 without the `tf-keras` shim — a failure with nothing to do
# with this pipeline, which is PyTorch end to end. Set before any HuggingFace
# import; transformers caches the result on first read.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

INPUT_ROOT   = os.environ.get("PROCTORIQ_INPUT_ROOT", "/kaggle/input")
WORKING_ROOT = os.environ.get("PROCTORIQ_WORKING_ROOT", "/kaggle/working")
'''))

    cells.append(md("""
## Step 2 — Locate the competition data

Paths are discovered by walking the input tree. The dataset directory name is not knowable in
advance, and document IDs are always derived from the filesystem — never a hardcoded list.
"""))
    cells.append(code('''
import glob, json, re, time, hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd


def find_file_dir(filename_pattern, root=None):
    """Directory containing the first file matching `filename_pattern`."""
    root = root or INPUT_ROOT
    for current, _dirs, files in os.walk(root):
        for name in files:
            if Path(name).match(filename_pattern):
                return current
    return None


KB_DIR = find_file_dir("*.md")
TEST_CSV_DIR = find_file_dir("test.csv")
SAMPLE_DIR = find_file_dir("sample_submission.csv")

if KB_DIR is None:
    raise FileNotFoundError(f"Knowledge base .md files not found under {INPUT_ROOT}")
if TEST_CSV_DIR is None:
    raise FileNotFoundError(f"test.csv not found under {INPUT_ROOT}")

TEST_CSV_PATH = os.path.join(TEST_CSV_DIR, "test.csv")
SAMPLE_PATH = os.path.join(SAMPLE_DIR, "sample_submission.csv") if SAMPLE_DIR else None
SUBMISSION_PATH = os.path.join(WORKING_ROOT, "submission.csv")
os.makedirs(WORKING_ROOT, exist_ok=True)

print(f"knowledge base : {KB_DIR}")
print(f"test.csv       : {TEST_CSV_PATH}")
print(f"submission     : {SUBMISSION_PATH}")
'''))

    cells.append(md("""
## Step 3 — Corpus, chunking, citation and reranking

Inlined from the project source so the notebook is self-contained. The corpus loader splits on
`##` only; `###` stays inside the body of its parent section.
"""))
    for relative, replacements in INLINE_MODULES:
        cells.append(code(inline_source(relative, replacements)))

    cells.append(md("""
## Step 4 — Submission format

The one place the format flags are applied. Everything upstream works in canonical form — bare
document stems and verbatim `##` header text — and translation happens here at serialization.
"""))
    cells.append(code('''
from enum import Enum


class SectionFormat(str, Enum):
    FULL_HEADER = "full_header"
    NUMBER_ONLY = "number_only"
    TITLE_ONLY = "title_only"


SUBMISSION_COLUMNS = ("question_id", "answer_text", "cited_docs", "cited_sections")
SECTION_PATTERN = re.compile(r"^Section\\s+(\\d+)\\s*:\\s*(.+)$")


def format_doc(doc_id, extension):
    stem = doc_id[:-3] if doc_id.endswith(".md") else doc_id
    return f"{stem}.md" if extension else stem


def format_section(section_title, fmt):
    fmt = SectionFormat(fmt)
    title = section_title.strip()
    if fmt is SectionFormat.FULL_HEADER:
        return title
    match = SECTION_PATTERN.match(title)
    if match is None:
        return title
    number, label = match.group(1), match.group(2).strip()
    return f"Section {number}" if fmt is SectionFormat.NUMBER_ONLY else label


class SubmissionValidationError(Exception):
    pass


def validate_rows(rows, question_ids, expected_header=SUBMISSION_COLUMNS, separator="|"):
    """Refuse to write anything defective. Nothing is written unless this passes."""
    problems = []
    if len(rows) != len(question_ids):
        problems.append(f"expected {len(question_ids)} rows, got {len(rows)}")
    if [r.get("question_id") for r in rows] != list(question_ids):
        problems.append("question_id order does not match test.csv")

    empty_answers, empty_cites, misaligned = [], [], []
    for row in rows:
        qid = row.get("question_id", "?")
        if any(row.get(c) is None for c in expected_header):
            problems.append(f"{qid}: null field")
        # An empty answer looks structurally valid but scores zero on four of five
        # dimensions — the failure mode most likely to survive a visual check.
        if not (row.get("answer_text") or "").strip():
            empty_answers.append(qid)
        docs = (row.get("cited_docs") or "").split(separator) if row.get("cited_docs") else []
        secs = (row.get("cited_sections") or "").split(separator) if row.get("cited_sections") else []
        if not docs or any(not d.strip() for d in docs):
            empty_cites.append(qid)
        if not secs or any(not s.strip() for s in secs):
            empty_cites.append(qid)
        if len(docs) != len(secs):
            misaligned.append(qid)

    if empty_answers:
        problems.append(f"empty answer_text: {', '.join(empty_answers)}")
    if empty_cites:
        problems.append(f"empty citations: {', '.join(sorted(set(empty_cites)))}")
    if misaligned:
        problems.append(f"docs/sections misaligned: {', '.join(misaligned)}")
    if problems:
        raise SubmissionValidationError("; ".join(problems))
'''))

    cells.append(md("""
## Step 5 — Build the pipeline
"""))
    cells.append(code('''
documents = load_corpus(KB_DIR)
print(f"{len(documents)} documents, {len(documents.sections())} sections")

test_df = pd.read_csv(TEST_CSV_PATH)
question_ids = test_df["question_id"].astype(str).tolist()
question_texts = test_df["question"].astype(str).tolist()
print(f"{len(test_df)} questions")

chunks = SectionChunker(include_header_in_text=True).chunk(documents)

reranker = CrossEncoderReranker(
    RERANK_MODEL, score_transform="auto", text_variant=TEXT_VARIANT,
)
reranker.fit(question_texts, chunks, documents)
print("reranking complete")


def build_strategy(spec):
    spec = spec.strip().lower()
    if spec.startswith("topk-"):
        return TopK(int(spec.split("-", 1)[1]))
    if spec.startswith("gap-"):
        return RelativeGap(float(spec.split("-", 1)[1]))
    if spec.startswith("thresh-"):
        return ScoreThreshold(float(spec.split("-", 1)[1]))
    raise ValueError(f"unknown citation strategy {spec!r}")


strategy = build_strategy(CITATION_STRATEGY)
'''))

    cells.append(md("""
## Step 6 — Answer generation

Extractive mode is deterministic and needs no API key. Generative and refusal modes use Groq,
reading the key from Kaggle Secrets.

Answer style is worth real points: accuracy and groundedness are both similarity against text
derived from the source documents, with no LLM judge, so conversational padding actively lowers the
score. The prompts suppress it explicitly — zero padding phrases across 150 generated answers.

**Refusals are the one generated path that ships.** Extractive mode pastes policy prose on
adversarial questions: the right source text, but it does not decline, address the student, or
explain. Integrity-refusal is 15% and is scored against a *correct refusal*. Measured on the 14
adversarial questions: **8/14 declining extractive versus 13/14 with the refusal template**, and the
RAG Triad — which never sees the answer key — scores answer-relevancy **0.493 → 0.871** on that
subset.
"""))
    cells.append(code('''
# A Groq key is needed for generative mode AND for the router/refusal path.
#
# Resolution order: GROQ_API_KEY first, then GROQ_API_KEY_1..9. A missing
# unsuffixed secret once caused a run to silently produce the WRONG ARM.
NEEDS_KEY = GENERATION_MODE == "generative" or ROUTER_ENABLED or REFUSAL_ENABLED
KEY_NAMES = ["GROQ_API_KEY"] + [f"GROQ_API_KEY_{i}" for i in range(1, 10)]

api_key, key_source = "", "none"
if NEEDS_KEY:
    for name in KEY_NAMES:
        value = os.environ.get(name, "")
        if value:
            api_key, key_source = value, f"env:{name}"
            break
    if not api_key:
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            for name in KEY_NAMES:
                try:
                    value = secrets.get_secret(name)
                except Exception:
                    continue
                if value:
                    api_key, key_source = value, f"kaggle_secrets:{name}"
                    break
        except Exception as exc:
            print(f"Kaggle Secrets unavailable ({exc}).")
    print(f"Groq key resolved from: {key_source}")

answerer = None
if GENERATION_MODE == "generative":
    if api_key:
        from groq import Groq

        class NotebookGroqClient:
            """Minimal client with backoff. Temperature 0 for reproducibility."""

            def __init__(self, api_key, model=GROQ_MODEL, max_attempts=5):
                self.client = Groq(api_key=api_key)
                self.model = model
                self.max_attempts = max_attempts

            def complete(self, prompt, **kwargs):
                last = None
                for attempt in range(self.max_attempts):
                    try:
                        done = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,
                            max_tokens=1200,
                        )
                        return (done.choices[0].message.content or "").strip()
                    except Exception as error:
                        last = error
                        time.sleep(min(2 ** attempt, 30))
                raise RuntimeError(f"Groq failed after {self.max_attempts} attempts: {last}")

        answerer = GroqAnswerer(
            documents, client=NotebookGroqClient(api_key),
            template_name=TEMPLATE_NAME, max_chars=MAX_ANSWER_CHARS,
            answer_from=ANSWER_FROM,
        )

if answerer is None:
    answerer = ExtractiveAnswerer(
        documents, max_chars=MAX_ANSWER_CHARS, answer_from=ANSWER_FROM
    )

# The subsection fix is a flag, not a default, so probe 7 can isolate it.
#
# It only has an effect on the EXTRACTIVE path: the focuser lives inside
# ExtractiveAnswerer, and GroqAnswerer has no `fit_focuser` at all. Probe 6b was
# submitted with SUBSECTION_FIX = True in generative mode, where it did nothing,
# and the run summary printed a bare "off" that read as a configuration choice
# rather than an ignored flag. A True flag that silently does nothing is the same
# class of defect as the four wrong-arm submissions, so it is now reported as an
# ignored flag and is carried into the arm name, forcing EXPECTED_ARM to say so.
SUBSECTION_ACTIVE = SUBSECTION_FIX and hasattr(answerer, "fit_focuser")
if SUBSECTION_ACTIVE:
    answerer.fit_focuser(question_texts, reranker)
    print("subsection fix: ON (cross-encoder subsection selection)")
elif SUBSECTION_FIX:
    print("subsection fix: REQUESTED BUT INERT — the subsection focuser is part of")
    print(f"                the extractive answerer; {answerer.name!r} has none.")
    print("                Reflected in the arm name; not a silent no-op.")
else:
    print("subsection fix: off (lexical overlap)")

print(f"answerer: {answerer.name}")

# ── router: classify before answering ─────────────────────────────────────
groq_client = None
if api_key and (ROUTER_ENABLED or REFUSAL_ENABLED):
    from groq import Groq

    class RouterClient:
        """Temperature 0. Measured 100% stable across 5 runs on all four axes."""

        def __init__(self, api_key, model=GROQ_MODEL, max_attempts=4):
            self.client = Groq(api_key=api_key)
            self.model = model
            self.max_attempts = max_attempts

        def complete(self, prompt, **kwargs):
            last = None
            for attempt in range(self.max_attempts):
                try:
                    done = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0, max_tokens=1200,
                    )
                    return (done.choices[0].message.content or "").strip()
                except Exception as error:
                    last = error
                    time.sleep(min(2 ** attempt, 20))
            raise RuntimeError(f"Groq failed after {self.max_attempts} attempts: {last}")

    groq_client = RouterClient(api_key)

    # ── PREFLIGHT: confirm the model exists before processing 50 questions ──
    # llama-3.1-8b-instant was removed by Groq mid-competition. Every call 404'd,
    # the router silently fell back on all 50, and only a post-hoc guard stopped a
    # wrong arm being written. Failing at question 0 instead of after a full run
    # costs one call and turns a 6-minute wrong-arm run into an instant error.
    try:
        probe = groq_client.complete("Reply with exactly: ok")
        print(f"model preflight OK: {GROQ_MODEL} -> {probe[:40]!r}")
    except Exception as exc:
        available = ""
        try:
            from groq import Groq as _G
            available = ", ".join(sorted(m.id for m in _G(api_key=api_key).models.list().data))
        except Exception:
            available = "(could not list models)"
        raise RuntimeError(
            f"Groq model {GROQ_MODEL!r} is not usable: {exc}. "
            f"Available models: {available}. "
            "Set GROQ_MODEL in the configuration cell to one of these, or set "
            "ROUTER_ENABLED/REFUSAL_ENABLED to False to run the extractive arm."
        ) from exc

router = QueryRouter(client=groq_client, cache_path=None, use_llm=ROUTER_ENABLED)
decisions = {d.question_id: d
             for d in router.classify_all(question_ids, question_texts)}
print(f"router: {router.stats}")

refuser = None
fires = []
if REFUSAL_ENABLED:
    refuser = RefusalAnswerer(documents, base=answerer, client=groq_client,
                              max_chars=MAX_ANSWER_CHARS, answer_from=ANSWER_FROM)
    fires = [q for q in question_ids if decisions[q].is_policy_boundary]
    print(f"refusal fires on {len(fires)} questions: {', '.join(fires)}")


# ══════════════════════════════════════════════════════════════════════════
#  FAIL LOUDLY — never write a wrong arm that looks correct
# ══════════════════════════════════════════════════════════════════════════
# A run with ROUTER_ENABLED/REFUSAL_ENABLED but no resolvable key silently
# degrades to the extractive baseline and writes a valid-looking 50-row
# submission. That is the most dangerous failure in this project: submitting it
# would have scored ~79.27 and read as "refusals do nothing".
#
# Silent degradation is acceptable ONLY when those flags are False.
if NEEDS_KEY and not api_key:
    raise RuntimeError(
        "ROUTER_ENABLED/REFUSAL_ENABLED/generative is set but no Groq key resolved "
        f"(tried {', '.join(KEY_NAMES)} in env and Kaggle Secrets). "
        "Refusing to write a submission that would silently be the extractive "
        "baseline. Add the secret, or set the flags to False deliberately."
    )
if ROUTER_ENABLED and router.stats["llm"] == 0:
    raise RuntimeError(
        f"ROUTER_ENABLED is True but the LLM classified nothing "
        f"({router.stats}). Every decision came from the keyword fallback, so "
        "this is NOT the routed arm. Refusing to write."
    )
if REFUSAL_ENABLED and not fires:
    raise RuntimeError(
        "REFUSAL_ENABLED is True but the router classified zero questions as "
        "policy_boundary or compound. Refusing to write what would be an "
        "extractive run wearing a refusal label."
    )
'''))

    cells.append(md("""
## Step 7 — Generate and write `submission.csv`
"""))
    cells.append(code('''
import csv

rows = []
for index, (qid, question) in enumerate(zip(question_ids, question_texts)):
    decision = decisions[qid]

    ranked = reranker.as_ranked_sections(reranker.rerank(index))
    ranked = apply_routing(ranked, decision, RoutingWeights(mode=SCORE_BIAS))

    active = cardinality_strategy(decision) if CARDINALITY_ROUTING else strategy
    chosen = active.select(ranked)
    citations = [s.citation for s in chosen]

    # Widened answer context: the top-N reranked sections. Citations are NOT
    # touched — they remain whatever `strategy` chose above.
    context_sections = [s.citation for s in ranked[:3]]

    if refuser is not None and decision.is_policy_boundary:
        variant = "compound" if decision.intent == "compound" else "pure_boundary"
        answer = refuser.answer_with_variant(question, citations, variant, question_id=qid)
    else:
        try:
            answer = answerer.answer(question, citations, context_sections)
        except TypeError:
            answer = answerer.answer(question, citations)
    docs = [format_doc(d, DOC_EXTENSION) for d, _ in citations]
    sections = [format_section(s, SECTION_FORMAT) for _, s in citations]

    rows.append({
        "question_id": qid,
        "answer_text": answer,
        "cited_docs": "|".join(docs),
        "cited_sections": "|".join(sections),
    })

    if index < 3 or index == len(question_ids) - 1:
        print(f"[{qid}] {question[:70]}")
        print(f"     -> {docs} | {sections}")
        print(f"     -> {answer[:160]}{'...' if len(answer) > 160 else ''}\\n")

expected_header = SUBMISSION_COLUMNS
if SAMPLE_PATH and os.path.exists(SAMPLE_PATH):
    with open(SAMPLE_PATH, "r", encoding="utf-8", newline="") as handle:
        expected_header = tuple(h.strip() for h in next(csv.reader(handle)))

# ── run summary: the Logs tab must answer "did my intended arm run?" ──────
refusals_written = sum(1 for r in rows if r["question_id"] in fires)
print()
print("=" * 70)
print("RUN SUMMARY")
print("=" * 70)
print(f"  groq key resolved     : {key_source}")
print(f"  router enabled        : {ROUTER_ENABLED}   classifications: {router.stats}")
print(f"  refusal enabled       : {REFUSAL_ENABLED}  fired on {len(fires)} questions")
print(f"  refusal rows written  : {refusals_written}")
print(f"  score bias            : {SCORE_BIAS}")
print(f"  cardinality routing   : {CARDINALITY_ROUTING}")
print(f"  doc extension         : {DOC_EXTENSION}")
print(f"  section format        : {SECTION_FORMAT}")
print(f"  citation strategy     : {CITATION_STRATEGY}")
print(f"  generation mode       : {GENERATION_MODE}   answerer: {answerer.name}")
print(f"  mean citations        : "
      f"{sum(len(r['cited_docs'].split('|')) for r in rows) / len(rows):.2f}")
print("=" * 70)

# ── derive the arm from the flags IN EFFECT, then assert it ────────────────
if GENERATION_MODE == "generative":
    ARM = "generative" if ANSWER_FROM == "top1" else f"generative-{ANSWER_FROM}"
    if SUBSECTION_FIX:
        ARM += "+inert-subsection"
elif ROUTER_ENABLED and REFUSAL_ENABLED:
    ARM = "refusals-plus-subsection" if SUBSECTION_ACTIVE else "refusals-only"
elif SUBSECTION_ACTIVE:
    ARM = "extractive-subsection"
else:
    ARM = "extractive-locked"

print(f"  DERIVED ARM : {ARM}")
print(f"  EXPECTED ARM: {EXPECTED_ARM}")
print()

if ARM != EXPECTED_ARM:
    raise RuntimeError(
        f"ARM MISMATCH — you asked for {EXPECTED_ARM!r} but the flags produce "
        f"{ARM!r}. Nothing has been written. Either set EXPECTED_ARM to {ARM!r}, "
        f"or fix the flags: ROUTER_ENABLED={ROUTER_ENABLED}, "
        f"REFUSAL_ENABLED={REFUSAL_ENABLED}, SUBSECTION_FIX={SUBSECTION_FIX}, "
        f"GENERATION_MODE={GENERATION_MODE!r}."
    )

# A truncated refusal falls back to extractive text and writes a valid-looking
# row that is NOT the refusal arm. Hard stop, same as the key and router guards.
if refuser is not None and getattr(refuser, "truncated", []):
    raise RuntimeError(
        f"{len(refuser.truncated)} refusal call(s) TRUNCATED "
        f"({', '.join(refuser.truncated)}): the reasoning trace exhausted "
        f"max_tokens and returned empty content, so those rows silently fell back "
        f"to extractive text. Refusing to write an arm that is not what it claims. "
        f"Raise max_tokens in RouterClient and re-run."
    )

validate_rows(rows, question_ids, expected_header)

with open(SUBMISSION_PATH, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(expected_header))
    writer.writeheader()
    writer.writerows(rows)

print(f"submission.csv written to {SUBMISSION_PATH}")
pd.read_csv(SUBMISSION_PATH).head()
'''))

    cells.append(md("""
## Before submitting

- [ ] **Internet: On** (needed for `pip install` and the model download), or the reranker attached
      as a Dataset
- [ ] Ran top to bottom with **Save & Run All (Commit)**
- [ ] `submission.csv` has 50 rows in `test.csv` order and passed `validate_rows`
- [ ] Submitted from this notebook's committed Output, not an uploaded CSV
- [ ] Configuration cell records which probe this run is
"""))

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return notebook


def _docstring_ids(tree: ast.AST) -> set[int]:
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, owners):
            continue
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def _executable_strings_and_names(tree: ast.AST) -> list[str]:
    """String literals and identifiers, excluding docstrings."""
    skip = _docstring_ids(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in skip:
                found.append(node.value)
        elif isinstance(node, ast.Name):
            found.append(node.id)
        elif isinstance(node, ast.Attribute):
            found.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.append(node.name)
    return found


def _imported_modules(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def check_no_leakage(notebook: nbf.NotebookNode) -> list[str]:
    """The export boundary is a leakage boundary. Fail the build, not the review.

    Checks *executable code*, not prose — the same distinction D-003 settled for
    the repo-side guard. The inlined loader's docstring legitimately explains why
    three questions share one section, and an explanation cannot branch. What is
    forbidden is a literal in code, or an actual import of the package.
    """
    problems: list[str] = []
    for index, cell in enumerate(notebook.cells):
        source = cell.get("source", "")

        if cell.cell_type != "code":
            continue

        # IPython magics and shell escapes are not Python; drop them before parsing.
        python_only = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith(("!", "%"))
        )
        if not python_only.strip():
            continue

        try:
            tree = ast.parse(python_only)
        except SyntaxError as error:
            problems.append(f"cell {index}: does not parse as Python ({error})")
            continue

        for module in _imported_modules(tree):
            if module.startswith("proctoriq_rag"):
                problems.append(f"cell {index}: imports {module!r} — notebook must be self-contained")

        for value in _executable_strings_and_names(tree):
            for token in FORBIDDEN_TOKENS:
                if token in value:
                    problems.append(f"cell {index}: code references {token!r} in {value[:60]!r}")
            for match in set(QUESTION_ID_LITERAL.findall(value)):
                problems.append(f"cell {index}: hardcodes question id {match!r} in code")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "notebooks" / "proctoriq_rag_submission.ipynb",
    )
    args = parser.parse_args()

    notebook = build_notebook()

    problems = check_no_leakage(notebook)
    if problems:
        print("EXPORT FAILED — the notebook must contain no key content and no question ids:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, str(args.out))

    json.loads(args.out.read_text(encoding="utf-8"))  # must be valid JSON
    code_cells = sum(1 for c in notebook.cells if c.cell_type == "code")
    print(f"wrote {args.out}")
    print(f"  {len(notebook.cells)} cells ({code_cells} code), "
          f"{args.out.stat().st_size / 1024:.0f} KB")
    print("  leakage check: PASS (no key content, no question-id literals, no src imports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
