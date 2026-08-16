"""The holdout boundary, extended to ``scripts/``.

Why a second guard was needed
-----------------------------
``tests/test_no_key_leakage.py`` walks ``src/`` and never sees measurement
harnesses. So this leaked through it:

    adversarial = {e.id for e in key if e.kind == "adversarial"}
    ...
    fire = decision.is_policy_boundary or qid in adversarial
    if arm == "refusal" and fire:
        texts.append(refuser.refuse(...))      # <- produces answer_text

The answer key decided which questions got a refusal. That is the key
determining what the *pipeline does*, not what we measure — and it is how a
router misclassification on Q33 got silently corrected in the measurement while
the real pipeline would have kept it.

The rule, which cannot be a blanket ban
---------------------------------------
Scripts legitimately read the key: that is what scoring *is*. So the rule is
directional:

    The key may determine what we MEASURE. It may never determine what the
    PIPELINE DOES.

Operationally: no key-derived value may reach a function that produces
``answer_text``, ``cited_docs`` or ``cited_sections`` — neither as an argument
nor as a condition guarding the call.

This is a deliberately simple taint analysis. It will not catch every possible
laundering of a key-derived value through three intermediate helpers, and it is
not trying to: it catches the shape that actually occurred, cheaply, on every
test run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

SCRIPTS = REPO_ROOT / "scripts"

#: Expressions that introduce key-derived data.
KEY_SOURCES: tuple[str, ...] = (
    "load_answer_key", "answer_key", "load_alternates", "alternates",
)

#: Attributes of a key entry. Reading any of these taints the result.
KEY_ATTRIBUTES: tuple[str, ...] = (
    "pairs", "docs", "sections", "kind", "confidence", "doc_set",
    "by_kind", "by_confidence", "adversarial_ids",
)

#: Names whose results are MEASUREMENT output, not key-derived pipeline input.
#: Scoring reads the key by definition; propagating taint through it would make
#: every downstream variable in a scoring script look like a leak.
MEASUREMENT_SINKS: tuple[str, ...] = (
    "Scorer", "score", "score_question", "breakdown", "report", "per_question",
    "section_ranks", "dimension_means",
)

#: Functions that PRODUCE submission content. Tainting one of these means the
#: key changed the pipeline's output.
PRODUCERS: tuple[str, ...] = (
    "answer", "answer_with_variant", "refuse", "run", "citations_for",
    "generate", "build_prompt", "write_submission", "build_rows",
)


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _mentions_key(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and any(s in sub.id for s in KEY_SOURCES):
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in KEY_ATTRIBUTES:
            return True
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name) and any(s in func.id for s in KEY_SOURCES):
                return True
            if isinstance(func, ast.Attribute) and func.attr in KEY_ATTRIBUTES:
                return True
    return False


def _is_measurement(node: ast.AST) -> bool:
    """True when the expression is a scoring/measurement result."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr in MEASUREMENT_SINKS:
            return True
        if isinstance(sub, ast.Name) and sub.id in MEASUREMENT_SINKS:
            return True
    return False


def tainted_names(tree: ast.AST) -> set[str]:
    """Names bound to key-derived expressions.

    Deliberately shallow — two passes, and never through a measurement result.
    An earlier version propagated transitively to a fixed point and tainted
    everything: ``Scorer(corpus, key)`` tainted ``scorer``, which tainted
    ``report``, which tainted every loop variable named ``q`` or ``i`` anywhere
    in the module, because ``ast.walk`` has no notion of scope. A guard that
    flags every line teaches nothing.

    Two passes catch ``adversarial = {e.id for e in key ...}`` and one alias of
    it, which is the shape that actually occurred.
    """
    tainted: set[str] = set()
    for _ in range(2):
        for node in ast.walk(tree):
            targets: list[ast.AST] = []
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets, value = [node.target], node.value
            if value is None or _is_measurement(value):
                continue
            if _mentions_key(value) or (_names_in(value) & tainted):
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            tainted.add(name.id)
    return tainted


def _producer_calls(tree: ast.AST) -> list[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in PRODUCERS:
            calls.append(node)
    return calls


def violations(source: str) -> list[str]:
    """Key-derived values reaching a producer, as arguments or as guards."""
    tree = ast.parse(source)
    tainted = tainted_names(tree)
    if not tainted:
        return []

    parents = _parents(tree)
    problems: list[str] = []

    for call in _producer_calls(tree):
        name = (call.func.attr if isinstance(call.func, ast.Attribute)
                else getattr(call.func, "id", "?"))

        for argument in list(call.args) + [k.value for k in call.keywords]:
            hit = _names_in(argument) & tainted
            if hit:
                problems.append(
                    f"line {call.lineno}: key-derived {sorted(hit)} passed to {name}()"
                )

        # Walk enclosing scope for an `if` whose test is key-derived.
        node: ast.AST | None = call
        while node is not None:
            parent = parents.get(node)
            if isinstance(parent, ast.If) and node in parent.body:
                hit = _names_in(parent.test) & tainted
                if hit:
                    problems.append(
                        f"line {call.lineno}: {name}() is guarded by key-derived "
                        f"{sorted(hit)} — the key must not decide what the pipeline does"
                    )
            node = parent
    return problems


SCRIPT_FILES = sorted(SCRIPTS.glob("*.py")) if SCRIPTS.is_dir() else []


def test_there_are_scripts_to_check():
    assert SCRIPT_FILES, "no scripts found — this guard would pass vacuously"


@pytest.mark.parametrize("path", SCRIPT_FILES, ids=lambda p: p.name)
def test_no_key_derived_value_reaches_a_producer(path: Path):
    problems = violations(path.read_text(encoding="utf-8"))
    assert not problems, (
        f"{path.name}: the answer key must determine what we MEASURE, never what "
        f"the PIPELINE DOES.\n" + "\n".join(f"  - {p}" for p in problems)
    )


# ── the guard must fail on the exact pattern that was removed ──────────────
REMOVED_PATTERN = '''
key = load_answer_key(KEY_PATH, corpus)
adversarial = {e.id for e in key if e.kind == "adversarial"}
for i, qid in enumerate(qids):
    fire = decision.is_policy_boundary or qid in adversarial
    if arm == "refusal" and fire:
        texts.append(refuser.refuse(qtexts[i], citations[i]))
    else:
        texts.append(extractive.answer(qtexts[i], citations[i]))
'''

CURRENT_PATTERN = '''
key = load_answer_key(KEY_PATH, corpus)
adversarial = {e.id for e in key if e.kind == "adversarial"}
for i, qid in enumerate(qids):
    if arm == "refusal" and decision.is_policy_boundary:
        texts.append(refuser.answer_with_variant(qtexts[i], citations[i], "compound"))
    else:
        texts.append(extractive.answer(qtexts[i], citations[i]))
scores = [q for q in report.per_question if q.question_id in adversarial]
'''


def test_guard_catches_the_pattern_that_was_removed():
    """A guard that cannot fail is not a guard."""
    problems = violations(REMOVED_PATTERN)
    assert problems, "the guard missed the exact leak it was written for"
    assert any("guarded by key-derived" in p for p in problems)
    # Reported as `fire`, the alias the guard actually sees in the `if` test —
    # `adversarial` reaches the producer one hop back through that assignment.
    assert any("fire" in p or "adversarial" in p for p in problems)


def test_guard_permits_using_the_key_to_measure():
    """Scoring against the key is the whole point — that must stay legal."""
    assert violations(CURRENT_PATTERN) == []


def test_guard_catches_a_key_value_passed_as_an_argument():
    source = '''
key = load_answer_key(p, corpus)
expected = key["x"].pairs
text = answerer.answer(question, expected)
'''
    problems = violations(source)
    assert any("passed to answer()" in p for p in problems)


def test_guard_ignores_scripts_that_never_touch_the_key():
    source = '''
citations = pipeline.citations_for(0)
text = answerer.answer(question, citations)
'''
    assert violations(source) == []
