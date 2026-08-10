"""Turning retrieved sections into ``answer_text``.

Two modes behind one interface.

**Extractive** is deterministic and model-free. It is what the probe submissions
use, and the reason is structural: probes require ``answer_text`` to be
byte-identical across submissions so that when only the citation columns change,
the entire score delta is attributable to the 35% citation half. An LLM cannot
guarantee that across separate Kaggle runs even at temperature 0.

It is also not a throwaway. Groundedness is 25% and is measured as similarity to
the source excerpt — an extractive answer *is* close to the source excerpt.

**Generative** uses Groq, as the competition rules require.

Answer sourcing is decoupled from citation cardinality
------------------------------------------------------
``answer_from="top1"`` builds the answer from the top-ranked section only,
regardless of how many sections end up cited. Without this, probe 5
(``topk-1`` -> ``topk-2``) would change the cited set, change the answer, and
stop being a clean citation-only experiment. ``answer_from="cited"`` is
implemented and tested for later phases, once the format questions are settled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, runtime_checkable

from proctoriq_rag.corpus.loader import Corpus
from proctoriq_rag.generation.cleaning import clean, trim_to_budget
from proctoriq_rag.generation.prompts import (
    DEFAULT_TEMPLATE,
    TEMPLATES_BY_NAME,
    PromptTemplate,
)
from proctoriq_rag.retrieval.chunking import SubsectionChunker

AnswerFrom = Literal["top1", "cited"]

DEFAULT_MAX_CHARS = 700


@runtime_checkable
class Answerer(Protocol):
    """Question plus ranked citations in, ``answer_text`` out."""

    @property
    def name(self) -> str: ...

    def answer(self, question: str, citations: Sequence[tuple[str, str]]) -> str: ...


def select_source_citations(
    citations: Sequence[tuple[str, str]], answer_from: AnswerFrom
) -> list[tuple[str, str]]:
    """Which cited sections the answer is allowed to draw on."""
    if not citations:
        return []
    return [citations[0]] if answer_from == "top1" else list(citations)


class SubsectionFocuser:
    """Picks the most relevant ``###`` block inside a section, by lexical overlap.

    Five sections in this corpus hold 13 ``###`` subsections between them, and
    they are the largest sections — doc 01 §2 is 1,317 characters covering three
    unrelated installation errors. Answering a question about "Session Start
    Error" with all three dilutes similarity against a golden answer about one.

    Selection is lexical rather than model-based on purpose: it must be
    deterministic and dependency-free so the extractive path stays byte-identical
    across runs and inlines cleanly into the Kaggle notebook. The citation is
    always the parent ``##`` regardless of which subsection is chosen.
    """

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self._by_section: dict[tuple[str, str], list[tuple[str | None, str]]] = {}
        for chunk in SubsectionChunker(include_header_in_text=False).chunk(corpus):
            self._by_section.setdefault(chunk.citation, []).append(
                (chunk.subsection_title, chunk.text)
            )

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in "".join(
            c.lower() if c.isalnum() else " " for c in text
        ).split() if len(t) > 2}

    def focus(self, question: str, doc_id: str, section_title: str) -> str:
        """Return the best-matching subsection body, or the whole section."""
        parts = self._by_section.get((doc_id, section_title), [])
        if len(parts) <= 1:
            section = self.corpus.get_section(doc_id, section_title)
            return section.body if section else (parts[0][1] if parts else "")

        question_tokens = self._tokens(question)
        best_text, best_score = parts[0][1], -1.0
        for subsection_title, text in parts:
            haystack = self._tokens(f"{subsection_title or ''} {text}")
            overlap = len(question_tokens & haystack)
            # Length-normalised so a long subsection does not win on volume.
            score = overlap / (len(haystack) ** 0.5 + 1e-9)
            if score > best_score:
                best_text, best_score = text, score
        return best_text


@dataclass
class ExtractiveAnswerer:
    """Deterministic answer built straight from the source text."""

    corpus: Corpus
    max_chars: int = DEFAULT_MAX_CHARS
    answer_from: AnswerFrom = "top1"
    focus_subsections: bool = True

    def __post_init__(self) -> None:
        self._focuser = SubsectionFocuser(self.corpus) if self.focus_subsections else None

    @property
    def name(self) -> str:
        return "extractive"

    def _source_text(self, question: str, doc_id: str, section_title: str) -> str:
        if self._focuser is not None:
            return self._focuser.focus(question, doc_id, section_title)
        section = self.corpus.get_section(doc_id, section_title)
        return section.body if section else ""

    def answer(self, question: str, citations: Sequence[tuple[str, str]]) -> str:
        sources = select_source_citations(citations, self.answer_from)
        bodies = [
            self._source_text(question, doc_id, section_title)
            for doc_id, section_title in sources
        ]
        combined = clean("\n".join(b for b in bodies if b))
        if not combined:
            return ""
        return trim_to_budget(combined, self.max_chars)


@dataclass
class GroqAnswerer:
    """Generative answers via Groq, as the competition rules require."""

    corpus: Corpus
    client: object | None = None
    template_name: str = DEFAULT_TEMPLATE
    max_chars: int = DEFAULT_MAX_CHARS
    answer_from: AnswerFrom = "top1"
    fallback: ExtractiveAnswerer | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            from proctoriq_rag.generation.groq_client import GroqChatClient

            self.client = GroqChatClient()
        if self.fallback is None:
            self.fallback = ExtractiveAnswerer(
                self.corpus, max_chars=self.max_chars, answer_from=self.answer_from
            )

    @property
    def name(self) -> str:
        return f"groq:{self.template_name}"

    @property
    def template(self) -> PromptTemplate:
        return TEMPLATES_BY_NAME[self.template_name]

    def build_prompt(self, question: str, citations: Sequence[tuple[str, str]]) -> str:
        passages = []
        for doc_id, section_title in select_source_citations(citations, self.answer_from):
            section = self.corpus.get_section(doc_id, section_title)
            if section is None:
                continue
            doc_title = self.corpus[doc_id].title if doc_id in self.corpus else doc_id
            passages.append((doc_title, section_title, clean(section.body)))
        return self.template.render(question, passages)

    def answer(self, question: str, citations: Sequence[tuple[str, str]]) -> str:
        """Generate, falling back to extractive rather than emitting an empty row.

        An empty ``answer_text`` fails submission validation and scores zero on
        four of five dimensions, so a degraded answer beats no answer.
        """
        try:
            text = self.client.complete(self.build_prompt(question, citations))
        except Exception:  # noqa: BLE001 - any API failure degrades, never crashes
            text = ""
        text = " ".join(text.split()).strip()
        if not text:
            return self.fallback.answer(question, citations)
        return trim_to_budget(text, self.max_chars)


def build_answerer(
    corpus: Corpus,
    mode: str = "extractive",
    template_name: str = DEFAULT_TEMPLATE,
    max_chars: int = DEFAULT_MAX_CHARS,
    answer_from: AnswerFrom = "top1",
    client: object | None = None,
) -> Answerer:
    """Config-driven construction, so the notebook can switch by flag."""
    if mode == "extractive":
        return ExtractiveAnswerer(corpus, max_chars=max_chars, answer_from=answer_from)
    if mode == "generative":
        return GroqAnswerer(
            corpus, client=client, template_name=template_name,
            max_chars=max_chars, answer_from=answer_from,
        )
    raise ValueError(f"unknown generation mode {mode!r} (expected extractive|generative)")
