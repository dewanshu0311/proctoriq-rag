"""RAG Triad evaluator — context relevancy, faithfulness, answer relevancy.

The three questions the triad asks, and why each is distinct:

**Context relevancy** — did retrieval fetch passages that bear on the question?
Scores the retriever, independently of what was written afterwards.

**Faithfulness** — is every claim in the answer supported by those passages?
Scores hallucination specifically. An answer can be perfectly relevant and still
invent a step, which is exactly the Q33 failure that produced *"Restarting your
laptop is not permitted"* against a passage instructing a restart.

**Answer relevancy** — does the answer actually address what was asked? Catches
the opposite failure: text that is faithful to the passages but answers a
different question, which is what a policy dump does on an adversarial ask.

Why this exists beyond being mandated
-------------------------------------
It is the only evaluation in this project that does not consult the answer key.
Every other measurement scores against a hand-built holdout, so it inherits that
holdout's opinions — including on the four contested entries. The triad judges the
pipeline on its own terms, which makes it the right instrument for justifying
design choices on camera without appealing to a leaderboard number.

**It is an LLM judge, and inherits the biases of one.** Scores are ordinal, not
calibrated: useful for comparing arms and for finding the worst questions, not as
an absolute quality measure. Same caution as D-007 on the groundedness proxy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

CONTEXT_RELEVANCY = """\
Rate how relevant these retrieved passages are to the question.

Question: {question}

Passages:
{context}

Score 0 to 10, where 10 means the passages contain what is needed to answer, and 0
means they are about something else entirely. Judge only relevance to the question,
not whether an answer was written well.

Reply with a single integer and nothing else."""

FAITHFULNESS = """\
Check whether the answer is supported by the passages.

Passages:
{context}

Answer:
{answer}

Score 0 to 10, where 10 means every claim in the answer is stated in or directly
follows from the passages, and 0 means the answer asserts things the passages do not
support. An answer that declines to do something IS supported if the passages say it
is not permitted.

Reply with a single integer and nothing else."""

ANSWER_RELEVANCY = """\
Rate how well the answer addresses the question that was asked.

Question: {question}

Answer:
{answer}

Score 0 to 10, where 10 means it directly answers what was asked, and 0 means it is
about something else. An answer that correctly refuses a request the policy forbids
IS addressing the question, provided it explains why. Quoting policy without
responding to the student is NOT.

Reply with a single integer and nothing else."""


def parse_score(text: str) -> float | None:
    """First integer 0-10 in the reply, or None if the judge did not answer."""
    for token in re.findall(r"\d+", text or ""):
        value = int(token)
        if 0 <= value <= 10:
            return value / 10.0
    return None


@dataclass
class TriadScore:
    question_id: str
    context_relevancy: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None

    @property
    def mean(self) -> float | None:
        present = [
            v for v in (self.context_relevancy, self.faithfulness, self.answer_relevancy)
            if v is not None
        ]
        return sum(present) / len(present) if present else None


@dataclass
class Eval:
    """The RAG Triad. ``client`` is anything with ``complete(prompt) -> str``."""

    client: object
    unparseable: list[str] = field(default_factory=list)

    def _score(self, prompt: str, question_id: str) -> float | None:
        try:
            raw = self.client.complete(prompt)
        except Exception:  # noqa: BLE001 - a judge failure is a missing score, not a crash
            self.unparseable.append(question_id)
            return None
        value = parse_score(raw)
        if value is None:
            self.unparseable.append(question_id)
        return value

    def get_context_relevancy(
        self, question: str, context: str, question_id: str = ""
    ) -> float | None:
        return self._score(
            CONTEXT_RELEVANCY.format(question=question, context=context), question_id
        )

    def get_faithfulness_score(
        self, context: str, answer: str, question_id: str = ""
    ) -> float | None:
        return self._score(
            FAITHFULNESS.format(context=context, answer=answer), question_id
        )

    def get_answer_relevancy(
        self, question: str, answer: str, question_id: str = ""
    ) -> float | None:
        return self._score(
            ANSWER_RELEVANCY.format(question=question, answer=answer), question_id
        )

    def evaluate(
        self, question_id: str, question: str, context: str, answer: str
    ) -> TriadScore:
        return TriadScore(
            question_id=question_id,
            context_relevancy=self.get_context_relevancy(question, context, question_id),
            faithfulness=self.get_faithfulness_score(context, answer, question_id),
            answer_relevancy=self.get_answer_relevancy(question, answer, question_id),
        )

    def evaluate_all(
        self,
        question_ids: Sequence[str],
        questions: Sequence[str],
        contexts: Sequence[str],
        answers: Sequence[str],
    ) -> list[TriadScore]:
        return [
            self.evaluate(qid, q, c, a)
            for qid, q, c, a in zip(question_ids, questions, contexts, answers)
        ]
