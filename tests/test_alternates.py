"""Alternate-key loader tests.

Alternates are held to exactly the primary key's standards. An alternate reading
that cannot be validated is not evidence about anything — it is a typo.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from proctoriq_rag.evaluation.alternates import load_alternates
from proctoriq_rag.evaluation.answer_key import AnswerKeyValidationError
from tests.conftest import REPO_ROOT, requires_competition_data

ALT_PATH = REPO_ROOT / "data" / "validation" / "answer_key_alternates.yaml"
FLAGGED = ("Q14", "Q28", "Q31", "Q35")


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "alt.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


# ── the real file ──────────────────────────────────────────────────────────
@requires_competition_data
def test_real_alternates_file_validates(corpus, answer_key):
    alternates = load_alternates(ALT_PATH, corpus, answer_key)
    assert set(alternates) == set(FLAGGED)


@requires_competition_data
def test_covers_exactly_the_four_flagged_questions(corpus, answer_key):
    """The four Phase 1 singled out — the only unretrievable expected sections."""
    alternates = load_alternates(ALT_PATH, corpus, answer_key)
    for qid in FLAGGED:
        assert alternates[qid], f"{qid} has no alternate reading"


@requires_competition_data
def test_every_alternate_is_positionally_aligned(corpus, answer_key):
    for readings in load_alternates(ALT_PATH, corpus, answer_key).values():
        for reading in readings:
            assert len(reading.docs) == len(reading.sections)
            assert len(reading.pairs) == len(reading.docs)


@requires_competition_data
def test_every_alternate_section_is_real(corpus, answer_key):
    for readings in load_alternates(ALT_PATH, corpus, answer_key).values():
        for reading in readings:
            for doc_id, section in reading.pairs:
                assert corpus.has_section(doc_id, section)


@requires_competition_data
def test_every_alternate_differs_from_the_primary(corpus, answer_key):
    """An alternate identical to the primary would be vacuous evidence."""
    alternates = load_alternates(ALT_PATH, corpus, answer_key)
    for qid, readings in alternates.items():
        primary = set(answer_key[qid].pairs)
        for reading in readings:
            assert set(reading.pairs) != primary


@requires_competition_data
def test_alternates_carry_a_rationale(corpus, answer_key):
    for readings in load_alternates(ALT_PATH, corpus, answer_key).values():
        for reading in readings:
            assert reading.rationale


@requires_competition_data
def test_loading_alternates_does_not_mutate_the_primary_key(corpus, answer_key):
    before = {e.id: e.pairs for e in answer_key}
    load_alternates(ALT_PATH, corpus, answer_key)
    assert {e.id: e.pairs for e in answer_key} == before


# ── failure modes, same strictness as the primary key ──────────────────────
def test_missing_file_returns_empty(corpus_or_tiny, tmp_path):
    assert load_alternates(tmp_path / "absent.yaml", corpus_or_tiny) == {}


def test_fabricated_section_is_rejected(tiny_corpus, tmp_path):
    path = write(tmp_path, """
        alternates:
          Q01:
            - label: bogus
              docs: [01_alpha_guide]
              sections: ["Section 9: Does Not Exist"]
        """)
    with pytest.raises(AnswerKeyValidationError, match="is not a ## header"):
        load_alternates(path, tiny_corpus)


def test_unknown_document_is_rejected(tiny_corpus, tmp_path):
    path = write(tmp_path, """
        alternates:
          Q01:
            - label: bogus
              docs: [99_nope]
              sections: ["Section 1: Installing Alpha"]
        """)
    with pytest.raises(AnswerKeyValidationError, match="unknown document"):
        load_alternates(path, tiny_corpus)


def test_right_section_wrong_document_is_rejected(tiny_corpus, tmp_path):
    path = write(tmp_path, """
        alternates:
          Q01:
            - label: mispaired
              docs: [02_beta_guide]
              sections: ["Section 1: Installing Alpha"]
        """)
    with pytest.raises(AnswerKeyValidationError, match="is not a ## header"):
        load_alternates(path, tiny_corpus)


def test_length_mismatch_is_rejected(tiny_corpus, tmp_path):
    path = write(tmp_path, """
        alternates:
          Q01:
            - label: ragged
              docs: [01_alpha_guide, 02_beta_guide]
              sections: ["Section 1: Installing Alpha"]
        """)
    with pytest.raises(AnswerKeyValidationError, match="length mismatch"):
        load_alternates(path, tiny_corpus)


def test_duplicate_label_is_rejected(tiny_corpus, tmp_path):
    path = write(tmp_path, """
        alternates:
          Q01:
            - label: same
              docs: [01_alpha_guide]
              sections: ["Overview"]
            - label: same
              docs: [02_beta_guide]
              sections: ["Overview"]
        """)
    with pytest.raises(AnswerKeyValidationError, match="duplicate label"):
        load_alternates(path, tiny_corpus)


def test_non_list_readings_rejected(tiny_corpus, tmp_path):
    path = write(tmp_path, """
        alternates:
          Q01:
            label: not-a-list
        """)
    with pytest.raises(AnswerKeyValidationError, match="expected a list"):
        load_alternates(path, tiny_corpus)


def test_strict_false_collects_without_raising(tiny_corpus, tmp_path):
    path = write(tmp_path, """
        alternates:
          Q01:
            - label: bogus
              docs: [99_nope]
              sections: ["Overview"]
        """)
    assert load_alternates(path, tiny_corpus, strict=False) is not None


@pytest.fixture
def corpus_or_tiny(tiny_corpus):
    return tiny_corpus
