"""Submission writer tests.

Two things matter most here: the format flags must produce every variant we might
need to probe, and validation must refuse to write anything defective — including
the blank-answer case the starter notebook produces when generation throws.
"""

from __future__ import annotations

import csv

import pandas as pd
import pytest

from proctoriq_rag.config import SectionFormat, SubmissionConfig
from proctoriq_rag.submission.writer import (
    SUBMISSION_COLUMNS,
    Prediction,
    SubmissionValidationError,
    build_rows,
    format_doc,
    format_section,
    read_expected_header,
    validate_rows,
    write_submission,
)
from tests.conftest import SAMPLE_SUBMISSION, TEST_CSV, requires_competition_data

IDS = ["Q01", "Q02", "Q03"]
DOC = "01_windows_installation_login_guide"
SECTION = "Section 2: Common Installation Errors"


def simple(question_id="Q01", docs=(DOC,), sections=(SECTION,), answer="An answer."):
    return Prediction(question_id, answer, list(docs), list(sections))


@pytest.fixture
def test_csv(tmp_path):
    path = tmp_path / "test.csv"
    pd.DataFrame({"question_id": IDS, "question": ["q"] * len(IDS)}).to_csv(
        path, index=False
    )
    return path


@pytest.fixture
def three_predictions():
    return [simple(q) for q in IDS]


# ── format_doc ─────────────────────────────────────────────────────────────
def test_format_doc_without_extension():
    assert format_doc(DOC, extension=False) == DOC


def test_format_doc_with_extension():
    assert format_doc(DOC, extension=True) == f"{DOC}.md"


def test_format_doc_is_idempotent():
    assert format_doc(f"{DOC}.md", extension=True) == f"{DOC}.md"
    assert format_doc(f"{DOC}.md", extension=False) == DOC


# ── format_section ─────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "fmt,expected",
    [
        (SectionFormat.FULL_HEADER, "Section 2: Common Installation Errors"),
        (SectionFormat.NUMBER_ONLY, "Section 2"),
        (SectionFormat.TITLE_ONLY, "Common Installation Errors"),
    ],
)
def test_all_three_section_variants(fmt, expected):
    assert format_section(SECTION, fmt) == expected


@pytest.mark.parametrize("fmt", list(SectionFormat))
def test_headers_without_a_section_number_are_verbatim(fmt):
    """'Overview' has no number to extract and no prefix to strip."""
    assert format_section("Overview", fmt) == "Overview"


@pytest.mark.parametrize("fmt", list(SectionFormat))
def test_apple_silicon_style_header(fmt):
    result = format_section("Section 4: Apple Silicon Note", fmt)
    assert result in {
        "Section 4: Apple Silicon Note",
        "Section 4",
        "Apple Silicon Note",
    }


def test_section_format_accepts_a_string():
    assert format_section(SECTION, "number_only") == "Section 2"


def test_titles_containing_colons_keep_the_tail():
    assert (
        format_section("Section 3: Automatic Termination vs. Logged Flag",
                       SectionFormat.TITLE_ONLY)
        == "Automatic Termination vs. Logged Flag"
    )


# ── build_rows ─────────────────────────────────────────────────────────────
def test_rows_follow_test_csv_order(three_predictions):
    shuffled = list(reversed(three_predictions))
    rows = build_rows(shuffled, IDS, SubmissionConfig())
    assert [r["question_id"] for r in rows] == IDS


def test_missing_prediction_yields_an_empty_row_not_a_dropped_row():
    rows = build_rows([simple("Q01")], IDS, SubmissionConfig())
    assert len(rows) == 3
    assert rows[1]["answer_text"] == ""


def test_pipe_join_preserves_positional_alignment():
    prediction = simple(
        docs=["doc_a", "doc_b"], sections=["Section 1: A", "Section 2: B"]
    )
    row = build_rows([prediction], ["Q01"], SubmissionConfig())[0]
    assert row["cited_docs"] == "doc_a|doc_b"
    assert row["cited_sections"] == "Section 1: A|Section 2: B"
    assert row["cited_docs"].split("|")[1] == "doc_b"
    assert row["cited_sections"].split("|")[1] == "Section 2: B"


def test_duplicate_document_is_not_collapsed():
    """The Q44 case: one document, two sections. Both must survive."""
    prediction = simple(
        docs=["09_status", "09_status"],
        sections=["Section 1: Where to Check Status", "Section 2: Possible Statuses"],
    )
    row = build_rows([prediction], ["Q01"], SubmissionConfig())[0]
    assert row["cited_docs"] == "09_status|09_status"
    assert len(row["cited_docs"].split("|")) == 2
    assert len(row["cited_sections"].split("|")) == 2


def test_config_flags_are_applied_at_serialization():
    config = SubmissionConfig(
        doc_extension=True, section_format=SectionFormat.NUMBER_ONLY
    )
    row = build_rows([simple()], ["Q01"], config)[0]
    assert row["cited_docs"] == f"{DOC}.md"
    assert row["cited_sections"] == "Section 2"


# ── validation ─────────────────────────────────────────────────────────────
def test_valid_rows_pass():
    validate_rows(build_rows([simple()], ["Q01"], SubmissionConfig()), ["Q01"])


def test_row_count_mismatch_raises():
    rows = build_rows([simple()], ["Q01"], SubmissionConfig())
    with pytest.raises(SubmissionValidationError, match="expected 3 rows"):
        validate_rows(rows, IDS)


def test_empty_answer_text_raises_and_names_the_questions():
    """The starter notebook writes "" on exception — that must not pass silently."""
    predictions = [simple("Q01"), simple("Q02", answer=""), simple("Q03", answer="   ")]
    rows = build_rows(predictions, IDS, SubmissionConfig())
    with pytest.raises(SubmissionValidationError) as excinfo:
        validate_rows(rows, IDS)
    message = str(excinfo.value)
    assert "empty or whitespace-only answer_text" in message
    assert "Q02" in message and "Q03" in message
    assert "Q01" not in message.split("answer_text")[1].split("\n")[0]


def test_empty_citations_raise():
    rows = build_rows([simple(docs=[], sections=[])], ["Q01"], SubmissionConfig())
    with pytest.raises(SubmissionValidationError, match="empty cited_docs"):
        validate_rows(rows, ["Q01"])


def test_misaligned_citations_raise():
    rows = [
        {
            "question_id": "Q01",
            "answer_text": "an answer",
            "cited_docs": "a|b",
            "cited_sections": "Section 1: X",
        }
    ]
    with pytest.raises(SubmissionValidationError, match="positional misalignment"):
        validate_rows(rows, ["Q01"])


def test_wrong_question_order_raises():
    rows = build_rows([simple(q) for q in IDS], IDS, SubmissionConfig())
    rows.reverse()
    with pytest.raises(SubmissionValidationError, match="order does not match"):
        validate_rows(rows, IDS)


def test_wrong_header_raises():
    rows = [{"qid": "Q01", "answer_text": "a", "cited_docs": "d", "cited_sections": "s"}]
    with pytest.raises(SubmissionValidationError):
        validate_rows(rows, ["Q01"])


def test_null_field_raises():
    rows = [
        {
            "question_id": "Q01",
            "answer_text": None,
            "cited_docs": "d",
            "cited_sections": "s",
        }
    ]
    with pytest.raises(SubmissionValidationError, match="null field"):
        validate_rows(rows, ["Q01"])


# ── write_submission ───────────────────────────────────────────────────────
def test_writes_a_readable_csv(tmp_path, test_csv, three_predictions):
    out = write_submission(
        three_predictions, test_csv, tmp_path / "submission.csv", SubmissionConfig()
    )
    frame = pd.read_csv(out)
    assert list(frame.columns) == list(SUBMISSION_COLUMNS)
    assert frame["question_id"].tolist() == IDS


def test_nothing_is_written_when_validation_fails(tmp_path, test_csv):
    out = tmp_path / "submission.csv"
    with pytest.raises(SubmissionValidationError):
        write_submission([simple("Q01", answer="")], test_csv, out, SubmissionConfig())
    assert not out.exists()


@pytest.mark.parametrize(
    "answer",
    [
        'He said "restart it", then left',
        "Comma, separated, answer",
        "Line one\nLine two",
        'Mixed: "quotes", commas,\nand a newline',
    ],
)
def test_awkward_answers_round_trip(tmp_path, test_csv, answer):
    predictions = [simple(q, answer=answer) for q in IDS]
    out = write_submission(
        predictions, test_csv, tmp_path / "submission.csv", SubmissionConfig()
    )
    frame = pd.read_csv(out)
    assert frame["answer_text"].tolist() == [answer] * len(IDS)


def test_separator_is_configurable(tmp_path, test_csv):
    config = SubmissionConfig(separator="|")
    prediction = simple("Q01", docs=["a", "b"], sections=["Section 1: X", "Section 2: Y"])
    predictions = [prediction] + [simple(q) for q in IDS[1:]]
    out = write_submission(predictions, test_csv, tmp_path / "s.csv", config)
    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["cited_docs"] == "a|b"


# ── against the real competition files ─────────────────────────────────────
@requires_competition_data
def test_header_matches_sample_submission():
    assert read_expected_header(SAMPLE_SUBMISSION) == SUBMISSION_COLUMNS


@requires_competition_data
def test_full_50_row_submission(tmp_path):
    question_ids = pd.read_csv(TEST_CSV)["question_id"].tolist()
    predictions = [simple(q) for q in question_ids]
    out = write_submission(
        predictions,
        TEST_CSV,
        tmp_path / "submission.csv",
        SubmissionConfig(),
        sample_submission=SAMPLE_SUBMISSION,
    )
    frame = pd.read_csv(out)
    assert len(frame) == 50
    assert frame["question_id"].tolist() == question_ids
    assert list(frame.columns) == list(read_expected_header(SAMPLE_SUBMISSION))
    assert frame.notna().all().all()
