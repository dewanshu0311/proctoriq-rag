"""HyDE — Hypothetical Document Embeddings.

The idea: a question and its answer occupy different regions of embedding space.
"How do I turn off face verification?" shares little surface vocabulary with
"There is no setting, workaround, or support-assisted method to disable face
verification." So instead of embedding the question, ask an LLM to *write* a
plausible answer, embed that, and retrieve against it — the hypothetical answer
lives nearer the real one than the question does.

The prediction on record, before measuring
------------------------------------------
On a 4,000-word corpus where recall@10 is already 93.75%, this should not move
retrieval much. There is very little room above the ceiling and the gap HyDE
closes — question-vocabulary versus answer-vocabulary — is exactly what the
cross-encoder already handles by scoring the pair jointly.

The one place it plausibly helps is **policy_boundary** questions, where that
vocabulary gap is widest. Measured as its own row rather than folded into the
average, because an effect on 14 questions can vanish inside a mean over 50.

If it does not help, that is reported as a measured negative with numbers. A
documented "implemented, measured, did not help here" is a stronger result than a
contrived routing rule, and it is honest about a mandated component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from proctoriq_rag.retrieval.chunking import Chunk
from proctoriq_rag.retrieval.retriever import ScoredChunk

HYDE_PROMPT = """\
Write a short passage from a support knowledge base that would answer this student's
question about a proctored-exam platform.

Question: {question}

Write it as documentation, not as a reply to the student: no greeting, no "you asked",
no sign-off. Three to four sentences. State what the policy or procedure is, in the
register a support document would use. If the question asks for something a proctoring
system would not allow, write the passage that states that boundary.

Passage:"""


@dataclass
class HyDERetriever:
    """Embed a generated hypothetical answer instead of the question.

    ``client`` is any object with ``complete(prompt) -> str``. ``embedder`` is a
    Phase 1 embedding backend. Falls back to embedding the raw question whenever
    generation fails, so a Groq outage degrades HyDE to plain dense retrieval
    rather than breaking the pipeline.
    """

    client: object
    embedder: object
    chunks: Sequence[Chunk]
    chunk_vectors: np.ndarray
    hypotheticals: dict[str, str] = field(default_factory=dict)
    fallbacks: int = 0

    def generate(self, question: str) -> str:
        """Write the hypothetical answer. Cached per question."""
        if question in self.hypotheticals:
            return self.hypotheticals[question]
        try:
            text = self.client.complete(HYDE_PROMPT.format(question=question))
        except Exception:  # noqa: BLE001 - degrade to the question itself
            text = ""
        text = " ".join((text or "").split()).strip()
        if not text:
            self.fallbacks += 1
            text = question
        self.hypotheticals[question] = text
        return text

    def scores(self, question: str) -> np.ndarray:
        """Cosine similarity of every chunk against the hypothetical answer."""
        hypothetical = self.generate(question)
        vector = np.asarray(self.embedder.encode_queries([hypothetical]))[0]
        return np.asarray(self.chunk_vectors) @ vector

    def retrieve(self, question: str, k: int = 10) -> list[ScoredChunk]:
        scores = self.scores(question)
        order = np.argsort(-scores)[:k]
        return [ScoredChunk(self.chunks[i], float(scores[i])) for i in order]
