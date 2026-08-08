"""Chunking tests.

The invariant under test everywhere: a chunk's citation metadata is always a real
`##` header in its own document, no matter how the text was split.
"""

from __future__ import annotations

import pytest

from proctoriq_rag.retrieval.chunking import (
    RecursiveChunker,
    SectionChunker,
    SubsectionChunker,
    build_chunkers,
    validate_chunks,
)
from tests.conftest import EXPECTED_TOTAL_SECTIONS, requires_competition_data

DOC01 = "01_windows_installation_login_guide"
INSTALL_ERRORS = "Section 2: Common Installation Errors"

ALL_CHUNKERS = [
    SectionChunker(),
    SectionChunker(include_header_in_text=False),
    RecursiveChunker(256),
    RecursiveChunker(400),
    RecursiveChunker(512),
    RecursiveChunker(768),
    SubsectionChunker(),
    SubsectionChunker(include_header_in_text=False),
]


# ── the shared invariant ───────────────────────────────────────────────────
@requires_competition_data
@pytest.mark.parametrize("chunker", ALL_CHUNKERS, ids=lambda c: c.fingerprint)
def test_every_chunk_cites_a_real_header(chunker, corpus):
    chunks = chunker.chunk(corpus)
    assert chunks
    validate_chunks(chunks, corpus)  # raises on a fabricated citation
    for chunk in chunks:
        assert corpus.has_section(chunk.doc_id, chunk.section_title)


@requires_competition_data
@pytest.mark.parametrize("chunker", ALL_CHUNKERS, ids=lambda c: c.fingerprint)
def test_chunk_ids_are_unique(chunker, corpus):
    chunks = chunker.chunk(corpus)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


@requires_competition_data
@pytest.mark.parametrize("chunker", ALL_CHUNKERS, ids=lambda c: c.fingerprint)
def test_no_chunk_text_is_empty(chunker, corpus):
    assert all(c.text.strip() for c in chunker.chunk(corpus))


@requires_competition_data
@pytest.mark.parametrize("chunker", ALL_CHUNKERS, ids=lambda c: c.fingerprint)
def test_section_title_never_carries_hash_markers(chunker, corpus):
    """`###` text must never leak into a citation — it is not a section."""
    for chunk in chunker.chunk(corpus):
        assert "#" not in chunk.section_title


# ── SectionChunker ─────────────────────────────────────────────────────────
@requires_competition_data
def test_section_chunker_produces_exactly_53(corpus):
    assert len(SectionChunker().chunk(corpus)) == EXPECTED_TOTAL_SECTIONS


@requires_competition_data
def test_section_chunker_is_one_chunk_per_section(corpus):
    chunks = SectionChunker().chunk(corpus)
    assert len({c.citation for c in chunks}) == len(chunks)


# ── SubsectionChunker: the load-bearing case ───────────────────────────────
@requires_competition_data
def test_subsection_splits_install_errors_but_cites_one_section(corpus):
    """Doc 01's three ### errors become separate chunks sharing one citation.

    This is the whole hypothesis of the chunker: sharpen retrieval without
    changing what gets cited.
    """
    chunks = [
        c
        for c in SubsectionChunker().chunk(corpus)
        if c.doc_id == DOC01 and c.section_title == INSTALL_ERRORS
    ]
    assert len(chunks) >= 3
    assert all(c.section_title == INSTALL_ERRORS for c in chunks)
    assert len({c.citation for c in chunks}) == 1

    subsections = {c.subsection_title for c in chunks}
    assert '"Element not found"' in subsections
    assert '"Session Start Error"' in subsections
    assert '"Unspecified Error"' in subsections


@requires_competition_data
def test_subsection_separates_the_three_errors(corpus):
    """Each error's fix must land in its own chunk, not be smeared across all three."""
    chunks = {
        c.subsection_title: c.text
        for c in SubsectionChunker().chunk(corpus)
        if c.doc_id == DOC01 and c.section_title == INSTALL_ERRORS
    }
    assert "Control Panel" in chunks['"Element not found"']
    assert "antivirus" in chunks['"Session Start Error"'].lower()
    assert "Task Manager" in chunks['"Unspecified Error"']
    assert "Task Manager" not in chunks['"Element not found"']


@requires_competition_data
def test_subsection_leaves_sections_without_h3_whole(corpus):
    """A section with no ### must stay a single chunk."""
    chunks = [
        c
        for c in SubsectionChunker().chunk(corpus)
        if c.doc_id == "07_permitted_prohibited_actions_policy"
        and c.section_title == "Section 1: Prohibited Actions"
    ]
    assert len(chunks) == 1
    assert chunks[0].subsection_title is None


@requires_competition_data
def test_subsection_yields_more_chunks_than_sections(corpus):
    chunks = SubsectionChunker().chunk(corpus)
    assert len(chunks) > EXPECTED_TOTAL_SECTIONS
    assert len(chunks) == EXPECTED_TOTAL_SECTIONS - 5 + 13  # 5 split into 13


def test_split_body_without_h3_returns_one_part():
    parts = SubsectionChunker.split_body("Just a body with no subheadings.")
    assert len(parts) == 1
    assert parts[0][0] is None


def test_split_body_captures_preamble():
    body = "Lead line before any sub.\n\n### First\nAlpha.\n\n### Second\nBeta.\n"
    parts = SubsectionChunker.split_body(body)
    assert parts[0] == (None, "Lead line before any sub.")
    assert parts[1][0] == "First"
    assert parts[2][0] == "Second"


def test_split_body_ignores_h4():
    """Only ### is a subsection boundary; #### is content."""
    parts = SubsectionChunker.split_body("### Real\nA.\n\n#### Deeper\nB.\n")
    assert len(parts) == 1
    assert parts[0][0] == "Real"
    assert "Deeper" in parts[0][1]


# ── RecursiveChunker ───────────────────────────────────────────────────────
@requires_competition_data
def test_recursive_preserves_metadata_on_every_piece(corpus):
    chunks = RecursiveChunker(256).chunk(corpus)
    for chunk in chunks:
        assert corpus.has_section(chunk.doc_id, chunk.section_title)


@requires_competition_data
@pytest.mark.parametrize("size", [256, 400, 512, 768])
def test_recursive_smaller_size_yields_more_chunks(size, corpus):
    """Monotonic: shrinking the window cannot reduce the chunk count."""
    smaller = len(RecursiveChunker(size).chunk(corpus))
    bigger = len(RecursiveChunker(size * 2).chunk(corpus))
    assert smaller >= bigger


@requires_competition_data
def test_recursive_768_is_nearly_a_no_op(corpus):
    """Only 5 of 53 sections exceed 768 chars, so this must stay close to 53.

    Documents the flat top end of the size sweep as a property, not a surprise.
    """
    assert 53 <= len(RecursiveChunker(768).chunk(corpus)) <= 62


@requires_competition_data
def test_recursive_respects_chunk_size_on_bodies(corpus):
    """Allow header overhead, but the body portion must respect the window."""
    for chunk in RecursiveChunker(256, chunk_overlap=0,
                                  include_header_in_text=False).chunk(corpus):
        assert len(chunk.text) <= 256


# ── the header flag ────────────────────────────────────────────────────────
@requires_competition_data
def test_header_flag_changes_text_not_metadata(corpus):
    with_header = SectionChunker(include_header_in_text=True).chunk(corpus)
    without = SectionChunker(include_header_in_text=False).chunk(corpus)

    assert [c.citation for c in with_header] == [c.citation for c in without]
    assert with_header[5].text != without[5].text
    assert with_header[5].text.startswith(with_header[5].section_title)


@requires_competition_data
def test_subsection_header_text_includes_subsection_title(corpus):
    chunk = next(
        c
        for c in SubsectionChunker(include_header_in_text=True).chunk(corpus)
        if c.subsection_title == '"Session Start Error"'
    )
    assert chunk.section_title in chunk.text
    assert '"Session Start Error"' in chunk.text


# ── fingerprints and the grid ──────────────────────────────────────────────
def test_fingerprints_are_distinct_across_the_grid():
    fingerprints = [c.fingerprint for c in build_chunkers()]
    assert len(set(fingerprints)) == len(fingerprints)


def test_build_chunkers_covers_the_required_sweep():
    """The competition requires >=2 chunk sizes; we sweep four plus no-split."""
    fingerprints = {c.fingerprint for c in build_chunkers(header_variants=(True,))}
    assert "section-hdr" in fingerprints
    assert "subsection-hdr" in fingerprints
    for size in (256, 400, 512, 768):
        assert any(f.startswith(f"recursive-{size}-") for f in fingerprints)


def test_fingerprint_encodes_the_header_flag():
    assert SectionChunker(True).fingerprint.endswith("-hdr")
    assert SectionChunker(False).fingerprint.endswith("-nohdr")


# ── validation ─────────────────────────────────────────────────────────────
def test_validate_chunks_rejects_a_fabricated_citation(tiny_corpus):
    from proctoriq_rag.retrieval.chunking import Chunk

    bad = [Chunk("x::0::0", "01_alpha_guide", "Section 9: Invented", "text", 0)]
    with pytest.raises(ValueError, match="is not a ## header"):
        validate_chunks(bad, tiny_corpus)


def test_validate_chunks_rejects_duplicate_ids(tiny_corpus):
    from proctoriq_rag.retrieval.chunking import Chunk

    dup = [
        Chunk("same", "01_alpha_guide", "Overview", "a", 0),
        Chunk("same", "01_alpha_guide", "Overview", "b", 1),
    ]
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        validate_chunks(dup, tiny_corpus)


def test_tiny_corpus_subsection_keeps_parent_title(tiny_corpus):
    """Preamble + two subsections, all three citing the same parent `##`."""
    chunks = [
        c
        for c in SubsectionChunker().chunk(tiny_corpus)
        if c.section_title == "Section 2: Common Alpha Errors"
    ]
    assert len(chunks) == 3
    assert {c.subsection_title for c in chunks} == {
        None, '"First Error"', '"Second Error"'
    }
    assert all(c.section_title == "Section 2: Common Alpha Errors" for c in chunks)
    assert len({c.citation for c in chunks}) == 1

    preamble = next(c for c in chunks if c.subsection_title is None)
    assert "Lead line for alpha errors." in preamble.text
