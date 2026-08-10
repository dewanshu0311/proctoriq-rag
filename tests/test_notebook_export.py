"""Notebook export tests.

Structural checks run always. Full execution is opt-in (~3 minutes) via
``pytest -m slow`` because it loads a cross-encoder and scores 2,650 pairs.

The guarantee that matters is **behavioural equivalence**: the notebook carries
its own copy of the formatting logic, so a test asserts that copy produces byte-
identical output to the library's across the whole corpus. Textual identity is not
required and not checked — behaviour is.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys

import pytest

from proctoriq_rag.config import SectionFormat
from proctoriq_rag.submission.writer import format_doc, format_section
from tests.conftest import REPO_ROOT, requires_competition_data

NOTEBOOK = REPO_ROOT / "notebooks" / "proctoriq_rag_submission.ipynb"
EXPORTER = REPO_ROOT / "scripts" / "export_notebook.py"
QUESTION_ID = re.compile(r"\bQ(?:0[1-9]|[1-4][0-9]|50)\b")


@pytest.fixture(scope="module")
def notebook() -> dict:
    if not NOTEBOOK.exists():
        subprocess.run([sys.executable, str(EXPORTER)], check=True, cwd=REPO_ROOT)
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code_cells(notebook) -> list[str]:
    return [
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    ]


def python_only(source: str) -> str:
    return "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith(("!", "%"))
    )


# ── structure ──────────────────────────────────────────────────────────────
def test_notebook_is_valid_json_and_nbformat(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["cells"]


def test_every_code_cell_parses(code_cells):
    for index, source in enumerate(code_cells):
        body = python_only(source)
        if body.strip():
            ast.parse(body)  # raises on syntax error


def test_has_markdown_explaining_each_stage(notebook):
    """Graded work walked through on video — the stages must be narrated."""
    markdown = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    for heading in ("Configuration", "competition data", "generation", "submission"):
        assert heading.lower() in markdown.lower()


def test_config_cell_exists_near_the_top(code_cells):
    """A probe must be a one-line edit, not a hunt."""
    config_index = next(
        i for i, s in enumerate(code_cells) if "CONFIGURATION" in s
    )
    assert config_index <= 2
    cell = code_cells[config_index]
    for name in ("DOC_EXTENSION", "SECTION_FORMAT", "CITATION_STRATEGY",
                 "GENERATION_MODE", "ANSWER_FROM"):
        assert name in cell


# ── the leakage boundary ───────────────────────────────────────────────────
def test_no_package_imports(code_cells):
    """Kaggle cannot import our src package; the notebook must be self-contained."""
    for source in code_cells:
        body = python_only(source)
        if not body.strip():
            continue
        for node in ast.walk(ast.parse(body)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("proctoriq_rag")
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("proctoriq_rag")


def test_no_answer_key_content_anywhere(notebook):
    whole = json.dumps(notebook)
    assert "answer_key" not in whole
    assert "data/validation" not in whole


def test_no_question_id_literals_in_code(code_cells):
    """Same rule as D-003: prose may explain, code may not branch."""
    from scripts.export_notebook import _executable_strings_and_names

    for source in code_cells:
        body = python_only(source)
        if not body.strip():
            continue
        for value in _executable_strings_and_names(ast.parse(body)):
            assert not QUESTION_ID.findall(value), f"hardcoded id in {value[:60]!r}"


def test_exporter_rejects_a_leaking_notebook():
    """A guard that cannot fail is not a guard."""
    import nbformat as nbf

    from scripts.export_notebook import check_no_leakage

    bad = nbf.v4.new_notebook(cells=[
        nbf.v4.new_code_cell('SPECIAL = {"Q28": "terminate"}'),
        nbf.v4.new_code_cell('KEY = "data/validation/answer_key.yaml"'),
        nbf.v4.new_code_cell("from proctoriq_rag.corpus.loader import load_corpus"),
    ])
    problems = check_no_leakage(bad)
    assert any("question id" in p for p in problems)
    assert any("answer_key" in p for p in problems)
    assert any("self-contained" in p for p in problems)


def test_exporter_allows_docstrings_that_mention_questions():
    """Prose explaining why three questions share a section is not a violation."""
    import nbformat as nbf

    from scripts.export_notebook import check_no_leakage

    ok = nbf.v4.new_notebook(cells=[
        nbf.v4.new_code_cell('"""Q01, Q02 and Q03 all cite one section."""\nVALUE = 1'),
        nbf.v4.new_code_cell("!pip install -q groq"),
    ])
    assert check_no_leakage(ok) == []


# ── behavioural equivalence with the library ───────────────────────────────
@requires_competition_data
def test_notebook_formatting_matches_the_library(code_cells, corpus):
    """The whole point of inlining: same logic, verified, not assumed.

    All 53 headers x 3 section formats x 2 extension settings.
    """
    # `re` is imported in an earlier notebook cell; supply it the way the running
    # notebook would rather than reordering the export for the test's convenience.
    namespace: dict = {"re": re}
    format_cell = next(c for c in code_cells if "def format_section" in c)
    exec(compile(python_only(format_cell), "<notebook>", "exec"), namespace)  # noqa: S102

    for section in corpus.sections():
        for fmt in SectionFormat:
            assert namespace["format_section"](section.section_title, fmt.value) == \
                format_section(section.section_title, fmt)
        for extension in (True, False):
            assert namespace["format_doc"](section.doc_id, extension) == \
                format_doc(section.doc_id, extension=extension)


@requires_competition_data
def test_notebook_validation_rejects_the_same_defects(code_cells):
    """The inlined validator must refuse what the library's refuses."""
    namespace: dict = {"re": re}
    cell = next(c for c in code_cells if "def validate_rows" in c)
    exec(compile(python_only(cell), "<notebook>", "exec"), namespace)  # noqa: S102
    validate = namespace["validate_rows"]
    error = namespace["SubmissionValidationError"]

    good = [{"question_id": "A", "answer_text": "text",
             "cited_docs": "d", "cited_sections": "s"}]
    validate(good, ["A"])

    for broken in (
        [{**good[0], "answer_text": "   "}],          # blank answer
        [{**good[0], "cited_docs": ""}],              # empty citation
        [{**good[0], "cited_docs": "a|b"}],           # misaligned
    ):
        with pytest.raises(error):
            validate(broken, ["A"])

    with pytest.raises(error):
        validate(good, ["A", "B"])                    # wrong row count


# ── full execution, opt-in ─────────────────────────────────────────────────
@pytest.mark.slow
@requires_competition_data
def test_notebook_executes_and_scores(tmp_path, corpus, answer_key):
    """Export that does not run is worthless. ~3 minutes."""
    import shutil

    import nbformat
    import pandas as pd
    from nbclient import NotebookClient

    from proctoriq_rag.evaluation.scorer import Scorer
    from proctoriq_rag.submission.writer import Prediction

    data = tmp_path / "input" / "comp"
    (data / "kb").mkdir(parents=True)
    for path in (REPO_ROOT / "data" / "raw" / "kb").glob("*.md"):
        shutil.copy(path, data / "kb" / path.name)
    for name in ("test.csv", "sample_submission.csv"):
        shutil.copy(REPO_ROOT / "data" / "raw" / name, data / name)
    working = tmp_path / "working"
    working.mkdir()

    import os

    os.environ["PROCTORIQ_INPUT_ROOT"] = str(tmp_path / "input")
    os.environ["PROCTORIQ_WORKING_ROOT"] = str(working)

    nb = nbformat.read(str(NOTEBOOK), as_version=4)
    nb.cells = [
        c for c in nb.cells
        if not (c.cell_type == "code" and c.source.strip().startswith("!pip"))
    ]
    NotebookClient(nb, timeout=1800, kernel_name="python3").execute()

    frame = pd.read_csv(working / "submission.csv")
    assert len(frame) == 50
    assert frame.notna().all().all()
    assert (frame["answer_text"].astype(str).str.strip() != "").all()

    predictions = [
        Prediction(r.question_id, str(r.answer_text),
                   str(r.cited_docs).split("|"), str(r.cited_sections).split("|"))
        for r in frame.itertuples()
    ]
    report = Scorer(corpus, answer_key).score(predictions)
    # Must reproduce the Phase 2 library baseline exactly.
    assert report.dimension_means["citation"] == pytest.approx(0.6667, abs=1e-3)
    assert report.dimension_means["retrieval"] == pytest.approx(0.8467, abs=1e-3)
