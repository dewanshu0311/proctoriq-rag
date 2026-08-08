"""Chunking strategies.

**The invariant.** Every chunk carries the `##` section it came from as metadata,
and citations are derived from that metadata — never from the chunk's text. A
chunk may be a fragment of a section, a `###` subsection, or the whole thing; it
always knows which `##` header it belongs to, because that is what the grader
scores citations against.

Why the corpus shape matters here
---------------------------------
Section bodies are small: min 134 characters, median 347, mean 432, max 1317.
Only 5 of 53 sections exceed 768 characters. So a size-based splitter set to 768
leaves 48 sections untouched and is very nearly `SectionChunker` — the top of the
size sweep is close to a no-op, and that flat region is a finding rather than a
disappointment.

The interesting structure is not size but nesting. All 13 `###` subsections live
in those same 5 large sections:

    01_windows  Section 2: Common Installation Errors    3 subsections
    01_windows  Section 3: Login Issues                  2
    02_mac      Section 2: Common Installation Errors    3
    02_mac      Section 3: Login Issues                  2
    03_mock     Section 3: Common Mock-Test Failures     3

Those five sections carry the densest lookup region in the test set.
`SubsectionChunker` splits precisely there while still citing the parent `##`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from proctoriq_rag.corpus.loader import Corpus, Section

SUBSECTION_HEADER_RE = re.compile(r"^###(?!#)\s+(.*\S)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit.

    ``doc_id`` and ``section_title`` are the citation. ``subsection_title`` is
    provenance only — it is never cited, because `###` headers are not sections
    (see DECISIONS D-001).
    """

    chunk_id: str
    doc_id: str
    section_title: str
    text: str
    chunk_index: int
    subsection_title: str | None = None

    @property
    def citation(self) -> tuple[str, str]:
        """The ``(doc_id, section_title)`` pair this chunk would cite."""
        return (self.doc_id, self.section_title)


@runtime_checkable
class Chunker(Protocol):
    """Common interface. ``fingerprint`` identifies the config for cache keying."""

    @property
    def fingerprint(self) -> str: ...

    def chunk(self, corpus: Corpus) -> list[Chunk]: ...


def _compose_text(
    section_title: str,
    subsection_title: str | None,
    body: str,
    include_header: bool,
) -> str:
    """Assemble the text that actually gets embedded.

    With ``include_header``, the section title (and subsection title when there is
    one) is prepended. Phase 0's diagnostic did this implicitly; here it is an
    explicit, sweepable variable. Headers carry real signal — "Common Installation
    Errors" says more about a passage than the numbered steps beneath it — but a
    repeated header can also flatten the distinction between sections of the same
    document, so it is measured rather than assumed.
    """
    if not include_header:
        return body.strip()
    parts = [section_title]
    if subsection_title:
        parts.append(subsection_title)
    parts.append(body.strip())
    return "\n".join(p for p in parts if p)


class SectionChunker:
    """One chunk per `##` section. 53 chunks, no splitting.

    The natural unit: citation is graded at section level, so this is the
    configuration with zero mismatch between what is retrieved and what is cited.
    It is the baseline everything else has to beat.
    """

    def __init__(self, include_header_in_text: bool = True) -> None:
        self.include_header_in_text = include_header_in_text

    @property
    def fingerprint(self) -> str:
        return f"section-{'hdr' if self.include_header_in_text else 'nohdr'}"

    def chunk(self, corpus: Corpus) -> list[Chunk]:
        chunks: list[Chunk] = []
        for index, section in enumerate(corpus.sections()):
            chunks.append(
                Chunk(
                    chunk_id=f"{section.doc_id}::{section.section_index}::0",
                    doc_id=section.doc_id,
                    section_title=section.section_title,
                    text=_compose_text(
                        section.section_title, None, section.body,
                        self.include_header_in_text,
                    ),
                    chunk_index=index,
                )
            )
        return chunks


class RecursiveChunker:
    """Split section bodies with ``RecursiveCharacterTextSplitter``.

    Section metadata is copied onto every piece, so a section split into four
    fragments still produces exactly one citation string.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int | None = None,
        include_header_in_text: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = (
            chunk_size // 8 if chunk_overlap is None else chunk_overlap
        )
        self.include_header_in_text = include_header_in_text

    @property
    def fingerprint(self) -> str:
        suffix = "hdr" if self.include_header_in_text else "nohdr"
        return f"recursive-{self.chunk_size}-{self.chunk_overlap}-{suffix}"

    def _splitter(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def chunk(self, corpus: Corpus) -> list[Chunk]:
        splitter = self._splitter()
        chunks: list[Chunk] = []
        running = 0
        for section in corpus.sections():
            pieces = splitter.split_text(section.body) or [section.body]
            for local_index, piece in enumerate(pieces):
                if not piece.strip():
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=(
                            f"{section.doc_id}::{section.section_index}::{local_index}"
                        ),
                        doc_id=section.doc_id,
                        section_title=section.section_title,
                        text=_compose_text(
                            section.section_title, None, piece,
                            self.include_header_in_text,
                        ),
                        chunk_index=running,
                    )
                )
                running += 1
        return chunks


class SubsectionChunker:
    """Split on `###` boundaries, cite the parent `##`.

    Sections without `###` stay whole. Any preamble before the first `###` becomes
    its own chunk when it holds content.

    The hypothesis this exists to test: the three Windows install errors are
    distinct problems sharing one section, so a query about "Session Start Error"
    currently has to match a 1317-character passage that is two-thirds about other
    errors. Splitting should sharpen retrieval without changing the citation,
    since all three pieces still carry "Section 2: Common Installation Errors".
    """

    def __init__(self, include_header_in_text: bool = True) -> None:
        self.include_header_in_text = include_header_in_text

    @property
    def fingerprint(self) -> str:
        return f"subsection-{'hdr' if self.include_header_in_text else 'nohdr'}"

    @staticmethod
    def split_body(body: str) -> list[tuple[str | None, str]]:
        """Split a section body into ``(subsection_title, text)`` parts.

        A body with no `###` returns a single ``(None, body)`` part.
        """
        matches = list(SUBSECTION_HEADER_RE.finditer(body))
        if not matches:
            return [(None, body)]

        parts: list[tuple[str | None, str]] = []
        preamble = body[: matches[0].start()].strip()
        if preamble:
            parts.append((None, preamble))

        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            text = body[start:end].strip()
            if text:
                parts.append((match.group(1).strip(), text))
        return parts

    def chunk(self, corpus: Corpus) -> list[Chunk]:
        chunks: list[Chunk] = []
        running = 0
        for section in corpus.sections():
            for local_index, (subtitle, text) in enumerate(
                self.split_body(section.body)
            ):
                chunks.append(
                    Chunk(
                        chunk_id=(
                            f"{section.doc_id}::{section.section_index}::{local_index}"
                        ),
                        doc_id=section.doc_id,
                        section_title=section.section_title,
                        text=_compose_text(
                            section.section_title, subtitle, text,
                            self.include_header_in_text,
                        ),
                        chunk_index=running,
                        subsection_title=subtitle,
                    )
                )
                running += 1
        return chunks


def validate_chunks(chunks: Sequence[Chunk], corpus: Corpus) -> None:
    """Assert every chunk cites a real `##` header in its own document.

    Cheap enough to run on every sweep configuration. A chunker that invents a
    citation would otherwise fail silently and score zero for a reason that looks
    like bad retrieval.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            problems.append(f"duplicate chunk_id {chunk.chunk_id!r}")
        seen.add(chunk.chunk_id)
        if not corpus.has_section(chunk.doc_id, chunk.section_title):
            problems.append(
                f"{chunk.chunk_id}: {chunk.section_title!r} is not a ## header "
                f"in {chunk.doc_id!r}"
            )
    if problems:
        raise ValueError(
            "chunk validation failed:\n" + "\n".join(f"  - {p}" for p in problems)
        )


def build_chunkers(
    chunk_sizes: Sequence[int] = (256, 400, 512, 768),
    header_variants: Sequence[bool] = (True, False),
) -> list[Chunker]:
    """The chunker grid for the sweep: no-split, four sizes, and subsection."""
    chunkers: list[Chunker] = []
    for include_header in header_variants:
        chunkers.append(SectionChunker(include_header_in_text=include_header))
        for size in chunk_sizes:
            chunkers.append(
                RecursiveChunker(chunk_size=size, include_header_in_text=include_header)
            )
        chunkers.append(SubsectionChunker(include_header_in_text=include_header))
    return chunkers
