"""Parse the markdown knowledge base into a structured, queryable corpus.

The single load-bearing decision in this module
----------------------------------------------
Sections are split on ``##`` headers **only**. ``###`` subheadings are content,
not section boundaries.

This is not a style preference, it is what makes citations correct. Document 01,
"Section 2: Common Installation Errors", contains three distinct errors as
``###`` subheadings — "Element not found", "Session Start Error" and
"Unspecified Error". The answer key cites all three questions (Q01, Q02, Q03) to
the *same* section string, because that is the section they live in. Splitting on
``###`` would produce three sections where the ground truth has one, breaking
citations on roughly a third of the test set.

Document IDs are always derived from the filesystem (the file stem). No document
name is hardcoded anywhere in this package.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# A line that is exactly a level-2 ATX header. `^##` followed by a space excludes
# `###` (three hashes then space fails the `\s` after exactly two) — the negative
# lookahead makes that explicit rather than incidental.
SECTION_HEADER_RE = re.compile(r"^##(?!#)\s+(.*\S)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Section:
    """One ``##`` section of one document.

    ``section_title`` is the verbatim header text with the leading ``"## "``
    removed — e.g. ``"Section 2: Common Installation Errors"``. This is the
    string the answer key matches against, and the string the submission writer
    reformats. It is never normalized, lowercased or trimmed beyond surrounding
    whitespace.

    ``body`` is everything between this header and the next ``##`` header,
    including any ``###`` subheadings.
    """

    doc_id: str
    section_title: str
    section_index: int
    body: str

    @property
    def key(self) -> tuple[str, str]:
        """The ``(doc_id, section_title)`` pair used for citation scoring."""
        return (self.doc_id, self.section_title)

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.doc_id} :: {self.section_title}"


@dataclass(frozen=True)
class Document:
    """One markdown file from the knowledge base."""

    doc_id: str
    path: Path
    raw_text: str
    sections: tuple[Section, ...] = field(default_factory=tuple)

    @property
    def title(self) -> str:
        """The ``#`` level-1 title, or the doc_id if the file has none."""
        for line in self.raw_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        return self.doc_id


def split_sections(doc_id: str, text: str) -> list[Section]:
    """Split one document's text into ``##`` sections.

    Any preamble before the first ``##`` header (the ``#`` title line) is not a
    section and is dropped — it carries no citable content.
    """
    matches = list(SECTION_HEADER_RE.finditer(text))
    sections: list[Section] = []

    for index, match in enumerate(matches):
        title = match.group(1).strip()
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip("\n")
        sections.append(
            Section(
                doc_id=doc_id,
                section_title=title,
                section_index=index,
                body=body,
            )
        )

    return sections


class Corpus:
    """The full knowledge base, indexed for lookup by document and section."""

    def __init__(self, documents: dict[str, Document]) -> None:
        self._documents = dict(documents)
        self._by_key: dict[tuple[str, str], Section] = {
            section.key: section
            for document in self._documents.values()
            for section in document.sections
        }

    # ── access ─────────────────────────────────────────────────────────────
    @property
    def documents(self) -> dict[str, Document]:
        return dict(self._documents)

    def doc_ids(self) -> list[str]:
        """Document IDs in sorted (i.e. numeric-prefix) order."""
        return sorted(self._documents)

    def sections(self) -> list[Section]:
        """Every section in the corpus, document order then section order."""
        return [
            section
            for doc_id in self.doc_ids()
            for section in self._documents[doc_id].sections
        ]

    def section_titles(self, doc_id: str) -> list[str]:
        if doc_id not in self._documents:
            raise KeyError(f"Unknown document {doc_id!r}")
        return [s.section_title for s in self._documents[doc_id].sections]

    def get_section(self, doc_id: str, section_title: str) -> Section | None:
        return self._by_key.get((doc_id, section_title))

    def has_document(self, doc_id: str) -> bool:
        return doc_id in self._documents

    def has_section(self, doc_id: str, section_title: str) -> bool:
        return (doc_id, section_title) in self._by_key

    def section_text(self, pairs: list[tuple[str, str]]) -> str:
        """Concatenate the bodies of the given ``(doc, section)`` pairs.

        Used by the groundedness proxy, which compares a generated answer
        against the text it was supposed to be grounded in. Unknown pairs are
        skipped rather than raising — the caller is scoring, not validating.
        """
        bodies = [
            section.body
            for pair in pairs
            if (section := self._by_key.get(pair)) is not None
        ]
        return "\n\n".join(bodies)

    # ── dunder ─────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        """Number of documents. Use ``len(corpus.sections())`` for sections."""
        return len(self._documents)

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._documents

    def __getitem__(self, doc_id: str) -> Document:
        return self._documents[doc_id]

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"Corpus({len(self._documents)} documents, {len(self._by_key)} sections)"


def load_corpus(kb_dir: Path | str) -> Corpus:
    """Load every ``*.md`` file in ``kb_dir`` into a :class:`Corpus`.

    Document IDs are file stems, read off disk. Nothing here knows the names of
    the ProctorIQ documents, so adding or renaming a document requires no code
    change.
    """
    kb_dir = Path(kb_dir)
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    paths = sorted(kb_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No .md files found in {kb_dir}")

    documents: dict[str, Document] = {}
    for path in paths:
        doc_id = path.stem
        if doc_id in documents:
            raise ValueError(f"Duplicate document id {doc_id!r} in {kb_dir}")
        text = path.read_text(encoding="utf-8")
        documents[doc_id] = Document(
            doc_id=doc_id,
            path=path,
            raw_text=text,
            sections=tuple(split_sections(doc_id, text)),
        )

    return Corpus(documents)
