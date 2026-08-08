"""Answer key loader and validator tests.

The real key is asserted to load cleanly. Every failure mode is exercised with a
synthetic key written to tmp_path, so a broken validator cannot pass by accident.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from proctoriq_rag.evaluation.answer_key import (
    EXPECTED_QUESTION_IDS,
    AnswerKeyValidationError,
    load_answer_key,
)
from tests.conftest import ANSWER_KEY_PATH, requires_competition_data


def write_key(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "key.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


GOOD_ENTRY = """
meta:
  version: 1
questions:
  - id: Q01
    docs: [01_alpha_guide]
    sections: ["Section 1: Installing Alpha"]
    kind: lookup
    confidence: high
"""


# ── the real key ───────────────────────────────────────────────────────────
@requires_competition_data
def test_real_key_loads_cleanly(corpus):
    """If this fails, the parser is wrong — the key was verified by hand."""
    key = load_answer_key(ANSWER_KEY_PATH, corpus)
    assert len(key) == 50
    assert key.ids() == list(EXPECTED_QUESTION_IDS)


@requires_competition_data
def test_real_key_composition(answer_key):
    assert answer_key.counts("kind") == {
        "lookup": 29,
        "adversarial": 14,
        "multi_doc": 5,
        "multi_section": 1,
        "trap": 1,
    }
    assert answer_key.counts("confidence") == {"high": 36, "medium": 11, "low": 3}
    assert len(answer_key.adversarial_ids()) == 14


@requires_competition_data
def test_every_entry_is_positionally_aligned(answer_key):
    for entry in answer_key:
        assert len(entry.docs) == len(entry.sections)
        assert len(entry.pairs) == len(entry.docs)


@requires_competition_data
def test_q44_cites_one_document_twice(answer_key):
    """The only same-document two-section entry — must not collapse."""
    entry = answer_key["Q44"]
    assert len(entry.docs) == 2
    assert len(entry.doc_set) == 1
    assert len(set(entry.pairs)) == 2


@requires_competition_data
def test_low_confidence_entries_are_the_three_we_expect(answer_key):
    assert [e.id for e in answer_key.by_confidence("low")] == ["Q14", "Q28", "Q35"]


# ── failure modes ──────────────────────────────────────────────────────────
def test_length_mismatch_is_rejected(tiny_corpus, tmp_path):
    path = write_key(
        tmp_path,
        """
        questions:
          - id: Q01
            docs: [01_alpha_guide, 02_beta_guide]
            sections: ["Section 1: Installing Alpha"]
            kind: lookup
            confidence: high
        """,
    )
    with pytest.raises(AnswerKeyValidationError, match="length mismatch"):
        load_answer_key(path, tiny_corpus, expected_ids=["Q01"])


def test_unknown_document_is_rejected(tiny_corpus, tmp_path):
    path = write_key(
        tmp_path,
        """
        questions:
          - id: Q01
            docs: [99_nonexistent_guide]
            sections: ["Section 1: Installing Alpha"]
            kind: lookup
            confidence: high
        """,
    )
    with pytest.raises(AnswerKeyValidationError, match="unknown document"):
        load_answer_key(path, tiny_corpus, expected_ids=["Q01"])


def test_fabricated_section_is_rejected(tiny_corpus, tmp_path):
    path = write_key(
        tmp_path,
        """
        questions:
          - id: Q01
            docs: [01_alpha_guide]
            sections: ["Section 9: Does Not Exist"]
            kind: lookup
            confidence: high
        """,
    )
    with pytest.raises(AnswerKeyValidationError, match="is not a ## header"):
        load_answer_key(path, tiny_corpus, expected_ids=["Q01"])


def test_right_section_wrong_document_is_rejected(tiny_corpus, tmp_path):
    """'Section 1: Installing Alpha' is real, but not in the beta guide."""
    path = write_key(
        tmp_path,
        """
        questions:
          - id: Q01
            docs: [02_beta_guide]
            sections: ["Section 1: Installing Alpha"]
            kind: lookup
            confidence: high
        """,
    )
    with pytest.raises(AnswerKeyValidationError, match="is not a ## header"):
        load_answer_key(path, tiny_corpus, expected_ids=["Q01"])


def test_duplicate_question_id_is_rejected(tiny_corpus, tmp_path):
    path = write_key(
        tmp_path,
        """
        questions:
          - id: Q01
            docs: [01_alpha_guide]
            sections: ["Section 1: Installing Alpha"]
            kind: lookup
            confidence: high
          - id: Q01
            docs: [02_beta_guide]
            sections: ["Section 1: Installing Beta"]
            kind: lookup
            confidence: high
        """,
    )
    with pytest.raises(AnswerKeyValidationError, match="duplicated question id"):
        load_answer_key(path, tiny_corpus, expected_ids=["Q01"])


def test_missing_question_id_is_rejected(tiny_corpus, tmp_path):
    path = write_key(tmp_path, GOOD_ENTRY)
    with pytest.raises(AnswerKeyValidationError, match="missing question id"):
        load_answer_key(path, tiny_corpus, expected_ids=["Q01", "Q02"])


def test_unexpected_question_id_is_rejected(tiny_corpus, tmp_path):
    path = write_key(tmp_path, GOOD_ENTRY)
    with pytest.raises(AnswerKeyValidationError, match="unexpected question id"):
        load_answer_key(path, tiny_corpus, expected_ids=[])


def test_unknown_kind_and_confidence_are_rejected(tiny_corpus, tmp_path):
    path = write_key(
        tmp_path,
        """
        questions:
          - id: Q01
            docs: [01_alpha_guide]
            sections: ["Section 1: Installing Alpha"]
            kind: banana
            confidence: extremely
        """,
    )
    with pytest.raises(AnswerKeyValidationError) as excinfo:
        load_answer_key(path, tiny_corpus, expected_ids=["Q01"])
    assert any("unknown kind" in p for p in excinfo.value.problems)
    assert any("unknown confidence" in p for p in excinfo.value.problems)


def test_all_problems_are_reported_together(tiny_corpus, tmp_path):
    """One run should tell you everything wrong, not just the first thing."""
    path = write_key(
        tmp_path,
        """
        questions:
          - id: Q01
            docs: [99_nonexistent_guide, 01_alpha_guide]
            sections: ["Section 1: Installing Alpha"]
            kind: banana
            confidence: high
        """,
    )
    with pytest.raises(AnswerKeyValidationError) as excinfo:
        load_answer_key(path, tiny_corpus, expected_ids=["Q01"])
    assert len(excinfo.value.problems) >= 3


def test_strict_false_collects_without_raising(tiny_corpus, tmp_path):
    path = write_key(
        tmp_path,
        """
        questions:
          - id: Q01
            docs: [99_nonexistent_guide]
            sections: ["Section 1: Installing Alpha"]
            kind: lookup
            confidence: high
        """,
    )
    key = load_answer_key(path, tiny_corpus, expected_ids=["Q01"], strict=False)
    assert key.problems  # type: ignore[attr-defined]


def test_missing_file_raises(tiny_corpus, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_answer_key(tmp_path / "absent.yaml", tiny_corpus)


def test_expected_ids_are_generated_not_typed():
    assert EXPECTED_QUESTION_IDS[0] == "Q01"
    assert EXPECTED_QUESTION_IDS[-1] == "Q50"
    assert len(EXPECTED_QUESTION_IDS) == 50
