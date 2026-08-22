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

AnswerFrom = Literal["top1", "cited", "top2", "top3"]

DEFAULT_MAX_CHARS = 700


@runtime_checkable
class Answerer(Protocol):
    """Question plus ranked citations in, ``answer_text`` out."""

    @property
    def name(self) -> str: ...

    def answer(self, question: str, citations: Sequence[tuple[str, str]]) -> str: ...


def select_source_citations(
    citations: Sequence[tuple[str, str]],
    answer_from: AnswerFrom,
    context_sections: Sequence[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Which sections the answer may draw on — NOT necessarily the cited ones.

    ``top1``  the top cited section only. Probe-safe and the Phase 3 default.
    ``cited`` every cited section.
    ``top2`` / ``top3``
              the top 2 or 3 **reranked** sections, which may exceed what is
              cited. This is the fix for multi-part questions.

    Why the last option exists. The RAG Triad measured answer relevancy at
    **0.360 on multi_doc and 0.200 on multi_section** — by far the worst
    per-question failure in the system — against 0.776 on lookup. The cause is
    structural: ``top1`` sourcing answers one half of a two-part question. That
    was frozen deliberately in Phase 3 so probes 5 and 6a could vary citations
    without also varying the answer.

    Widening the *context* while leaving *citations* untouched fixes the answer
    without breaking probe hygiene: the citation columns stay byte-identical to
    the locked baseline, and only ``answer_text`` changes.
    """
    if not citations and not context_sections:
        return []
    if answer_from == "top1":
        return [citations[0]] if citations else []
    if answer_from == "cited":
        return list(citations)

    depth = 2 if answer_from == "top2" else 3
    pool = list(context_sections) if context_sections else list(citations)
    seen, out = set(), []
    for pair in list(citations) + pool:
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
        if len(out) >= depth:
            break
    return out


class SubsectionFocuser:
    """Picks the most relevant ``###`` block inside a section.

    Five sections in this corpus hold 13 ``###`` subsections between them, and
    they are the largest sections — doc 01 §2 is 1,317 characters covering three
    unrelated installation errors. Answering a question about "Session Start
    Error" with all three dilutes similarity against a golden answer about one.

    Two scorers, same interface
    ---------------------------
    The default is lexical token overlap: deterministic, dependency-free, and it
    keeps the extractive path runnable with no model at all.

    It is also **measurably too weak**. Audited across the `###`-dense sections it
    gave Q02 and Q03 the same passage — both received the Session Start Error
    text though Q03 asks about Unspecified Error. Citation stays correct in that
    case, so it costs nothing on the 35% citation half and everything on the
    other 65%.

    So when a cross-encoder is available, :meth:`fit` precomputes scores for every
    (question, subsection) pair and selection uses those instead. It is the same
    model already ranking sections, so this adds no new dependency and stays
    deterministic. The citation is always the parent ``##`` either way.
    """

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self._by_section: dict[tuple[str, str], list[tuple[str | None, str]]] = {}

        # Two parallel views of the same subsections, in the same order.
        # `_chunks` (no header) is what gets returned as answer text.
        # `_scoring_chunks` (header included) is what gets scored — the ### title
        # IS the discriminating signal here. The error names "Element not found",
        # "Session Start Error" and "Unspecified Error" appear ONLY in the
        # headers; the bodies describe fixes without ever naming the error. Score
        # the bodies alone and the three become nearly indistinguishable, which is
        # exactly the collision this class exists to fix.
        self._chunks = SubsectionChunker(include_header_in_text=False).chunk(corpus)
        self._scoring_chunks = SubsectionChunker(include_header_in_text=True).chunk(corpus)

        for chunk in self._chunks:
            self._by_section.setdefault(chunk.citation, []).append(
                (chunk.subsection_title, chunk.text)
            )
        self._scores: dict[str, dict[tuple[str | None, str], float]] = {}

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in "".join(
            c.lower() if c.isalnum() else " " for c in text
        ).split() if len(t) > 2}

    # ── optional cross-encoder scoring ─────────────────────────────────────
    def fit(self, questions: Sequence[str], reranker) -> "SubsectionFocuser":
        """Precompute cross-encoder scores for every (question, subsection) pair.

        ``reranker`` is any :class:`CrossEncoderReranker`-shaped object. Roughly
        50 x 61 = 3,050 pairs, cached on disk like every other scoring pass.
        """
        from proctoriq_rag.retrieval.reranker import CrossEncoderReranker

        scorer = CrossEncoderReranker(
            reranker.model_name,
            score_transform=reranker.score_transform,
            text_variant="body",
            cache=reranker.cache,
            model=reranker.model,
        )
        scorer.fit(list(questions), self._scoring_chunks, self.corpus)
        matrix = scorer.scores()

        # Score with headers, key by the header-free text that will be returned.
        for row, question in enumerate(questions):
            self._scores[question] = {
                (chunk.subsection_title, chunk.text): float(matrix[row][column])
                for column, chunk in enumerate(self._chunks)
            }
        return self

    def focus(self, question: str, doc_id: str, section_title: str) -> str:
        """Return the best-matching subsection body, or the whole section."""
        parts = self._by_section.get((doc_id, section_title), [])
        if len(parts) <= 1:
            section = self.corpus.get_section(doc_id, section_title)
            return section.body if section else (parts[0][1] if parts else "")

        scored = self._scores.get(question)
        if scored is not None:
            return max(parts, key=lambda part: scored.get(part, float("-inf")))[1]

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

    def fit_focuser(self, questions: Sequence[str], reranker) -> "ExtractiveAnswerer":
        """Upgrade subsection selection from lexical overlap to the cross-encoder."""
        if self._focuser is not None and reranker is not None:
            self._focuser.fit(questions, reranker)
        return self

    @property
    def name(self) -> str:
        return "extractive"

    def _source_text(self, question: str, doc_id: str, section_title: str) -> str:
        if self._focuser is not None:
            return self._focuser.focus(question, doc_id, section_title)
        section = self.corpus.get_section(doc_id, section_title)
        return section.body if section else ""

    def answer(
        self,
        question: str,
        citations: Sequence[tuple[str, str]],
        context_sections: Sequence[tuple[str, str]] | None = None,
    ) -> str:
        sources = select_source_citations(citations, self.answer_from, context_sections)
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

    def build_prompt(
        self,
        question: str,
        citations: Sequence[tuple[str, str]],
        context_sections: Sequence[tuple[str, str]] | None = None,
    ) -> str:
        passages = []
        for doc_id, section_title in select_source_citations(
            citations, self.answer_from, context_sections
        ):
            section = self.corpus.get_section(doc_id, section_title)
            if section is None:
                continue
            doc_title = self.corpus[doc_id].title if doc_id in self.corpus else doc_id
            passages.append((doc_title, section_title, clean(section.body)))
        return self.template.render(question, passages)

    def answer(
        self,
        question: str,
        citations: Sequence[tuple[str, str]],
        context_sections: Sequence[tuple[str, str]] | None = None,
    ) -> str:
        """Generate, falling back to extractive rather than emitting an empty row.

        An empty ``answer_text`` fails submission validation and scores zero on
        four of five dimensions, so a degraded answer beats no answer.
        """
        try:
            text = self.client.complete(
                self.build_prompt(question, citations, context_sections)
            )
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
