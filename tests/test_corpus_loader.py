"""Corpus loader tests.

The `###`-is-content guarantee is the load-bearing one — if it regresses,
citations break on roughly a third of the test set and every downstream score
degrades without any obvious error.
"""

from __future__ import annotations

import pytest

from proctoriq_rag.corpus.loader import load_corpus, split_sections
from tests.conftest import (
    EXPECTED_DOC_COUNT,
    EXPECTED_SECTIONS_PER_DOC,
    EXPECTED_TOTAL_SECTIONS,
    KB_DIR,
    requires_competition_data,
)

pytestmark = requires_competition_data


def test_loads_all_documents(corpus):
    assert len(corpus) == EXPECTED_DOC_COUNT


def test_total_section_count_is_53(corpus):
    """A parsing regression must fail loudly, not quietly shift citations."""
    assert len(corpus.sections()) == EXPECTED_TOTAL_SECTIONS


def test_sections_per_document(corpus):
    counts = [len(corpus.section_titles(d)) for d in corpus.doc_ids()]
    assert counts == EXPECTED_SECTIONS_PER_DOC


def test_doc_id_is_file_stem(corpus):
    for doc_id, document in corpus.documents.items():
        assert doc_id == document.path.stem
        assert not doc_id.endswith(".md")


def test_document_names_are_read_from_disk():
    """No document name is hardcoded — the loader mirrors whatever is on disk."""
    on_disk = sorted(p.stem for p in KB_DIR.glob("*.md"))
    assert load_corpus(KB_DIR).doc_ids() == on_disk


def test_h3_subheadings_are_content_not_boundaries(corpus):
    """Doc 01 Section 2 holds three ### errors; all three live in one section.

    This is exactly why Q01, Q02 and Q03 all cite the same section string.
    """
    section = corpus.get_section(
        "01_windows_installation_login_guide", "Section 2: Common Installation Errors"
    )
    assert section is not None
    assert "Element not found" in section.body
    assert "Session Start Error" in section.body
    assert "Unspecified Error" in section.body
    assert "###" in section.body


def test_no_section_title_starts_with_a_hash(corpus):
    for section in corpus.sections():
        assert not section.section_title.startswith("#")
        assert section.section_title == section.section_title.strip()


def test_section_titles_are_verbatim_header_text(corpus):
    titles = corpus.section_titles("01_windows_installation_login_guide")
    assert titles[0] == "Overview"
    assert "Section 2: Common Installation Errors" in titles


def test_section_index_is_sequential(corpus):
    for doc_id in corpus.doc_ids():
        indices = [s.section_index for s in corpus[doc_id].sections]
        assert indices == list(range(len(indices)))


def test_has_section_is_document_scoped(corpus):
    """A real section title paired with the wrong document is not a match."""
    assert corpus.has_section(
        "01_windows_installation_login_guide", "Section 3: Login Issues"
    )
    assert not corpus.has_section(
        "04_network_connectivity_issue_playbook", "Section 3: Login Issues"
    )


def test_section_text_concatenates_bodies(corpus):
    pairs = [
        ("01_windows_installation_login_guide", "Section 3: Login Issues"),
        ("02_mac_installation_login_guide", "Section 3: Login Issues"),
    ]
    text = corpus.section_text(pairs)
    assert "PSB requires a full account login" in text
    assert "Quit all browser windows" in text


def test_section_text_skips_unknown_pairs(corpus):
    assert corpus.section_text([("nope", "Section 1: Nothing")]) == ""


def test_missing_kb_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "does_not_exist")


def test_empty_kb_dir_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No .md files"):
        load_corpus(tmp_path / "empty")


# ── synthetic fixtures ─────────────────────────────────────────────────────
def test_split_sections_drops_preamble():
    text = "# Title\n\nPreamble line.\n\n## Section 1: One\nBody one.\n"
    sections = split_sections("doc", text)
    assert [s.section_title for s in sections] == ["Section 1: One"]
    assert "Preamble" not in sections[0].body


def test_tiny_corpus_h3_stays_inside_section(tiny_corpus):
    section = tiny_corpus.get_section("01_alpha_guide", "Section 2: Common Alpha Errors")
    assert section is not None
    assert "First Error" in section.body
    assert "Second Error" in section.body
    assert len(tiny_corpus.section_titles("01_alpha_guide")) == 3
