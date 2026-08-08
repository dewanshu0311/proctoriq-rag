"""Shared fixtures.

``tests/`` is one of the few places permitted to touch the answer key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from proctoriq_rag.corpus.loader import Corpus, load_corpus
from proctoriq_rag.evaluation.answer_key import AnswerKey, load_answer_key
from proctoriq_rag.submission.writer import Prediction

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "proctoriq_rag"
KB_DIR = REPO_ROOT / "data" / "raw" / "kb"
TEST_CSV = REPO_ROOT / "data" / "raw" / "test.csv"
SAMPLE_SUBMISSION = REPO_ROOT / "data" / "raw" / "sample_submission.csv"
ANSWER_KEY_PATH = REPO_ROOT / "data" / "validation" / "answer_key.yaml"

#: Expected ## section counts per document, in sorted doc order. Hardcoded on
#: purpose: this is a regression fixture, and a change here should be a
#: deliberate edit rather than a silently-passing test.
EXPECTED_SECTIONS_PER_DOC = [5, 6, 5, 6, 5, 5, 6, 5, 5, 5]
EXPECTED_TOTAL_SECTIONS = 53
EXPECTED_DOC_COUNT = 10

requires_competition_data = pytest.mark.skipif(
    not KB_DIR.is_dir() or not any(KB_DIR.glob("*.md")),
    reason="competition data absent (data/raw/ is gitignored)",
)


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return load_corpus(KB_DIR)


@pytest.fixture(scope="session")
def answer_key(corpus: Corpus) -> AnswerKey:
    return load_answer_key(ANSWER_KEY_PATH, corpus)


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Corpus:
    """A two-document synthetic corpus.

    Deliberately mirrors the real corpus's awkward shapes: an ``## Overview``
    with no ``Section N:`` prefix, and a section containing ``###`` subheadings
    that must NOT become section boundaries.
    """
    kb = tmp_path / "kb"
    kb.mkdir()

    (kb / "01_alpha_guide.md").write_text(
        "# Alpha Guide\n"
        "\n"
        "## Overview\n"
        "Alpha overview body.\n"
        "\n"
        "## Section 1: Installing Alpha\n"
        "Install steps for alpha.\n"
        "\n"
        "## Section 2: Common Alpha Errors\n"
        "Lead line for alpha errors.\n"
        "\n"
        '### "First Error"\n'
        "Fix the first error.\n"
        "\n"
        '### "Second Error"\n'
        "Fix the second error.\n",
        encoding="utf-8",
    )
    (kb / "02_beta_guide.md").write_text(
        "# Beta Guide\n"
        "\n"
        "## Overview\n"
        "Beta overview body.\n"
        "\n"
        "## Section 1: Installing Beta\n"
        "Install steps for beta.\n",
        encoding="utf-8",
    )
    return load_corpus(kb)


@pytest.fixture
def make_prediction():
    """Factory for building a :class:`Prediction` tersely in tests."""

    def _make(
        question_id: str,
        docs: list[str] | None = None,
        sections: list[str] | None = None,
        answer: str = "A non-empty answer.",
    ) -> Prediction:
        return Prediction(
            question_id=question_id,
            answer_text=answer,
            cited_docs=list(docs or []),
            cited_sections=list(sections or []),
        )

    return _make
