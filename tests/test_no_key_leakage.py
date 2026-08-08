"""The guardrail that keeps the holdout honest.

``data/validation/answer_key.yaml`` is the only independent notion of correctness
we have. The moment it influences pipeline behaviour it stops being a
measurement, and the competition rules explicitly forbid hardcoded answers. So
the boundary is enforced structurally — by walking the AST of every module under
``src/`` — rather than by convention.

**The rule, stated precisely.** Taken literally, "no module under ``src/`` may
reference the answer key" would fail ``evaluation/scorer.py``, whose entire job
is to compare against the key. The enforceable form of the same guarantee is:

- every module under ``src/proctoriq_rag/`` **except** ``evaluation/`` may not
  import ``proctoriq_rag.evaluation.*``, name ``answer_key``, or read
  ``data/validation/``;
- ``evaluation/`` may import the pipeline, never the reverse.

The pipeline therefore cannot see the key even transitively, because it cannot
see the only package that can.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests.conftest import SRC_ROOT

#: The one subpackage allowed to know the key exists.
EVALUATION_PACKAGE = "evaluation"

FORBIDDEN_NAME = "answer_key"
FORBIDDEN_PATH_FRAGMENTS = ("data/validation", "data\\validation", "answer_key.yaml")

#: Matches a hardcoded competition question ID, Q01..Q50.
QUESTION_ID_LITERAL = re.compile(r"\bQ(?:0[1-9]|[1-4][0-9]|50)\b")


def pipeline_modules() -> list[Path]:
    """Every module under src/ that is NOT part of the evaluation package."""
    return sorted(
        path
        for path in SRC_ROOT.rglob("*.py")
        if EVALUATION_PACKAGE not in path.relative_to(SRC_ROOT).parts
    )


def all_modules() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def imported_modules(tree: ast.AST) -> list[str]:
    """Every dotted module name imported by this tree."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.append(base)
            names.extend(f"{base}.{alias.name}" for alias in node.names)
    return names


DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """Identify the string constants that are docstrings, so they can be excluded.

    Prose is allowed to discuss the answer key and to name specific questions —
    the docstrings in this repo do both, deliberately. What is forbidden is *code*
    that acts on them.
    """
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, DOCSTRING_OWNERS):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def string_literals(tree: ast.AST, include_docstrings: bool = False) -> list[str]:
    skip = set() if include_docstrings else _docstring_node_ids(tree)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in skip
    ]


def identifiers(tree: ast.AST) -> list[str]:
    """Every name that appears in executable position, imports included."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.extend(alias.name.split("."))
                if alias.asname:
                    names.append(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            names.extend((node.module or "").split("."))
            for alias in node.names:
                names.append(alias.name)
                if alias.asname:
                    names.append(alias.asname)
    return names


def parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# ── sanity: the walker actually finds something to check ───────────────────
def test_there_are_pipeline_modules_to_check():
    modules = pipeline_modules()
    assert modules, "found no modules under src/ — the guard would pass vacuously"
    assert any(m.name == "config.py" for m in modules)
    assert any("corpus" in m.parts for m in modules)


def test_the_evaluation_package_exists_and_is_excluded():
    assert (SRC_ROOT / EVALUATION_PACKAGE / "answer_key.py").exists()
    assert all(
        EVALUATION_PACKAGE not in m.relative_to(SRC_ROOT).parts
        for m in pipeline_modules()
    )


# ── the guard ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("path", pipeline_modules(), ids=lambda p: p.name)
def test_pipeline_module_does_not_import_evaluation(path: Path):
    for name in imported_modules(parse(path)):
        assert EVALUATION_PACKAGE not in name.split("."), (
            f"{path.relative_to(SRC_ROOT)} imports {name!r}. Pipeline code may not "
            "depend on the evaluation package — that is how the holdout stays a holdout."
        )


@pytest.mark.parametrize("path", pipeline_modules(), ids=lambda p: p.name)
def test_pipeline_module_does_not_name_the_answer_key(path: Path):
    tree = parse(path)
    offenders = [n for n in identifiers(tree) if FORBIDDEN_NAME in n.lower()]
    assert not offenders, (
        f"{path.relative_to(SRC_ROOT)} references {offenders} — pipeline code may "
        "not know the answer key exists."
    )


@pytest.mark.parametrize("path", pipeline_modules(), ids=lambda p: p.name)
def test_pipeline_module_does_not_read_the_validation_directory(path: Path):
    for literal in string_literals(parse(path)):
        lowered = literal.lower()
        for fragment in FORBIDDEN_PATH_FRAGMENTS:
            assert fragment not in lowered, (
                f"{path.relative_to(SRC_ROOT)} contains the string {literal!r}. "
                "Only tests/ and scripts/ may read data/validation/."
            )


@pytest.mark.parametrize("path", all_modules(), ids=lambda p: p.name)
def test_no_module_hardcodes_a_question_id(path: Path):
    """No `Q01`..`Q50` literal in executable code anywhere under src/.

    The competition forbids hardcoded answers, and question-specific branching is
    the first step towards them. ``EXPECTED_QUESTION_IDS`` is built with a
    comprehension for exactly this reason.

    Docstrings and comments are exempt: explaining *why* the Q44 duplicate-document
    case is handled the way it is makes the code better, and prose cannot branch.
    What this catches is a literal in code — ``{"Q28": ...}``, ``if qid == "Q14"``.
    """
    tree = parse(path)
    candidates = string_literals(tree) + identifiers(tree)
    offenders = sorted(
        {
            match
            for candidate in candidates
            for match in QUESTION_ID_LITERAL.findall(candidate)
        }
    )
    assert not offenders, (
        f"{path.relative_to(SRC_ROOT)} hardcodes question id(s) {offenders} in "
        "executable code. Derive them from test.csv instead."
    )


# ── the guard can fail ─────────────────────────────────────────────────────
def test_guard_detects_a_violating_module(tmp_path):
    """A guard that cannot fail is not a guard. Prove each check catches its case."""
    violation = tmp_path / "leaky.py"

    violation.write_text(
        "from proctoriq_rag.evaluation.answer_key import load_answer_key\n",
        encoding="utf-8",
    )
    tree = parse(violation)
    assert any(EVALUATION_PACKAGE in n.split(".") for n in imported_modules(tree))
    assert any(FORBIDDEN_NAME in n.lower() for n in identifiers(tree))

    violation.write_text('KEY = "data/validation/answer_key.yaml"\n', encoding="utf-8")
    literals = [s.lower() for s in string_literals(parse(violation))]
    assert any(
        fragment in literal
        for literal in literals
        for fragment in FORBIDDEN_PATH_FRAGMENTS
    )

    violation.write_text('SPECIAL_CASES = {"Q28": "terminate"}\n', encoding="utf-8")
    code_strings = string_literals(parse(violation))
    assert any(QUESTION_ID_LITERAL.search(s) for s in code_strings)


def test_docstrings_may_discuss_question_ids(tmp_path):
    """Prose about a question is fine; a literal in code is not."""
    module = tmp_path / "documented.py"
    module.write_text(
        '"""Handles the Q44 duplicate-document case."""\n\nVALUE = 1\n',
        encoding="utf-8",
    )
    tree = parse(module)
    assert not [
        m for s in string_literals(tree) for m in QUESTION_ID_LITERAL.findall(s)
    ]
    assert [
        m
        for s in string_literals(tree, include_docstrings=True)
        for m in QUESTION_ID_LITERAL.findall(s)
    ] == ["Q44"]


def test_question_id_pattern_boundaries():
    """Q01-Q50 only — not Q00, not Q51, not a substring of a longer token."""
    assert QUESTION_ID_LITERAL.findall("Q01 Q50 Q25") == ["Q01", "Q50", "Q25"]
    assert QUESTION_ID_LITERAL.findall("Q00 Q51 Q99") == []
    assert QUESTION_ID_LITERAL.findall("QQ01x ABQ01") == []
