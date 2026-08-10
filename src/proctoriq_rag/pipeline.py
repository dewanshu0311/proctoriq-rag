"""End-to-end pipeline: questions in, predictions out.

One place that wires corpus -> chunks -> reranker -> citations -> answers, used
by the local runner, the template comparison, and mirrored by the exported Kaggle
notebook. Having a single assembly point is what makes "does the notebook do the
same thing as the library?" a testable question rather than a hopeful one.

No evaluation imports. This is pipeline code and the holdout boundary applies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from proctoriq_rag.config import Config, SectionFormat
from proctoriq_rag.corpus.loader import Corpus, load_corpus
from proctoriq_rag.generation.answerer import Answerer, build_answerer
from proctoriq_rag.retrieval.chunking import Chunk, SectionChunker
from proctoriq_rag.retrieval.citation import (
    CitationStrategy,
    RelativeGap,
    ScoreThreshold,
    TopK,
)
from proctoriq_rag.retrieval.reranker import CrossEncoderReranker
from proctoriq_rag.submission.writer import Prediction

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_strategy(spec: str) -> CitationStrategy:
    """Parse a citation-strategy spec string such as ``topk-1`` or ``gap-0.95``."""
    spec = (spec or "topk-1").strip().lower()
    if spec.startswith("topk-"):
        return TopK(int(spec.split("-", 1)[1]))
    if spec.startswith("gap-"):
        return RelativeGap(float(spec.split("-", 1)[1]))
    if spec.startswith("thresh-"):
        return ScoreThreshold(float(spec.split("-", 1)[1]))
    raise ValueError(f"unknown citation strategy {spec!r}")


@dataclass
class PipelineConfig:
    """Everything a probe might vary, in one place."""

    rerank_model: str = DEFAULT_RERANK_MODEL
    text_variant: str = "body"
    score_transform: str = "auto"
    pool_size: int | None = None          # None = exhaustive over all sections
    citation_strategy: str = "topk-1"
    generation_mode: str = "extractive"
    template_name: str = "answer-first-explained"
    answer_from: str = "top1"
    max_answer_chars: int = 700
    doc_extension: bool = False
    section_format: SectionFormat = SectionFormat.FULL_HEADER

    def describe(self) -> str:
        return (
            f"{self.rerank_model.rsplit('/', 1)[-1]}/{self.text_variant}"
            f" · {self.citation_strategy} · {self.generation_mode}"
            f" · ext={self.doc_extension} · fmt={self.section_format.value}"
        )


@dataclass
class Pipeline:
    """Reranks sections, selects citations, writes answers."""

    corpus: Corpus
    config: PipelineConfig = field(default_factory=PipelineConfig)
    reranker: CrossEncoderReranker | None = None
    answerer: Answerer | None = None
    chunks: list[Chunk] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.chunks:
            self.chunks = SectionChunker(include_header_in_text=True).chunk(self.corpus)
        if self.reranker is None:
            self.reranker = CrossEncoderReranker(
                self.config.rerank_model,
                score_transform=self.config.score_transform,
                text_variant=self.config.text_variant,
            )
        if self.answerer is None:
            self.answerer = build_answerer(
                self.corpus,
                mode=self.config.generation_mode,
                template_name=self.config.template_name,
                max_chars=self.config.max_answer_chars,
                answer_from=self.config.answer_from,
            )
        self._strategy = build_strategy(self.config.citation_strategy)

    def fit(self, questions: Sequence[str]) -> "Pipeline":
        self.reranker.fit(questions, self.chunks, self.corpus)
        return self

    def citations_for(self, query_index: int) -> list[tuple[str, str]]:
        """Ranked, then cardinality-filtered, ``(doc, section)`` citations."""
        ranked = self.reranker.as_ranked_sections(
            self.reranker.rerank(query_index, top_k=self.config.pool_size)
        )
        return [s.citation for s in self._strategy.select(ranked)]

    def run(
        self, question_ids: Sequence[str], questions: Sequence[str]
    ) -> list[Prediction]:
        self.fit(questions)
        predictions: list[Prediction] = []
        for index, (question_id, question) in enumerate(zip(question_ids, questions)):
            citations = self.citations_for(index)
            predictions.append(
                Prediction(
                    question_id=question_id,
                    answer_text=self.answerer.answer(question, citations),
                    cited_docs=[doc for doc, _ in citations],
                    cited_sections=[section for _, section in citations],
                )
            )
        return predictions


def pipeline_from_config(
    config: Config, overrides: PipelineConfig | None = None
) -> Pipeline:
    """Build a pipeline from the repo config, with optional probe overrides."""
    corpus = load_corpus(config.paths.kb_dir)
    pipeline_config = overrides or PipelineConfig(
        doc_extension=config.submission.doc_extension,
        section_format=config.submission.section_format,
    )
    return Pipeline(corpus=corpus, config=pipeline_config)
