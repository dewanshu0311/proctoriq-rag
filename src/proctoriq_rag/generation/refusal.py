"""Reasoned refusals for policy-boundary questions.

Why this is the real test, not a side quest
-------------------------------------------
Phase 2 hypothesised that the adversarial gap was a retrieval problem and that
intent classification would fix it. The Phase 4 router A/B **falsified the
retrieval half of that**: routing moved adversarial document-F1 by exactly
0.0000 — not marginally, zero. The policy-document boost changed no rankings at
all.

What that does *not* falsify is the other half: we may already be retrieving the
correct policy section and then **pasting it verbatim instead of refusing**.
Extractive mode currently emits policy prose — "There is no setting, workaround,
or support-assisted method to disable face verification..." — which is the right
source text and is not a refusal. It does not decline, does not address the
student, does not explain. Integrity-refusal is 15% and is scored against a
*correct refusal*.

So this module is where the adversarial hypothesis actually gets tested.

What a refusal has to contain
-----------------------------
The competition states a reasoned refusal outscores a bare one, so "I can't help
with that" is doubly penalised — unhelpful *and* short against a reference that
explains. Three required moves:

1. **State the boundary.** Plainly, first, no hedging or apology.
2. **Explain it in the policy's own vocabulary.** The grader compares against
   text derived from these documents, so borrowed phrasing scores and invented
   phrasing does not.
3. **Say what the student can do.** Only where the source supports it — a
   fabricated remedy is worse than none.

The compound case
-----------------
Some questions smuggle a boundary request alongside a legitimate one. Q33 is the
acceptance test: the page froze and the laptop was restarted (legitimate, doc 05)
*and* "is it okay to quickly alt-tab to check my notes" (prohibited, doc 07). An
answer that only refuses abandons a real problem; one that only helps endorses a
violation by silence. Both halves must land.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from proctoriq_rag.corpus.loader import Corpus
from proctoriq_rag.generation.answerer import (
    AnswerFrom,
    ExtractiveAnswerer,
    select_source_citations,
)
from proctoriq_rag.generation.cleaning import clean, trim_to_budget

REFUSAL_PROMPT = """\
A student has asked a support question that requests something the proctoring policy does not
permit. Write the reply.

Policy passages (the only source you may use):
{context}

Student's question: {question}

Your reply must, in this order:
1. State plainly that this is not something that can be done. First sentence. No apology, no
   "I understand", no softening preamble.
2. Explain WHY, using the wording of the passages above. Reuse their phrasing rather than
   rewriting it.
3. If the passages describe what the student CAN do instead, say so. If they do not, say nothing
   about alternatives — do not invent a remedy.

Rules:
- No greeting, no sign-off, no "hope this helps", no offer of further assistance, no emoji.
- Nothing that is not stated in the passages.
- Do not soften a fixed boundary into a suggestion, and do not harden a documented
  recommendation into a prohibition.
- Three to five sentences. Prose, not a list.

Reply:"""

COMPOUND_PROMPT = """\
A student has asked a support question with TWO parts: a legitimate technical problem, and a
request that the proctoring policy does not permit. Both parts must be answered.

Passages (the only source you may use):
{context}

Student's question: {question}

Your reply must:
1. Answer the legitimate technical part first, concretely, using the passages' own wording.
2. Then address the part the policy does not permit — state plainly that it is not allowed and
   explain why in the passages' terms.

Do not skip either part. Answering only the technical half endorses the prohibited request by
saying nothing about it; answering only the prohibited half abandons a real problem.

Rules:
- No greeting, no sign-off, no offer of further assistance, no emoji.
- Nothing that is not stated in the passages.
- Four to six sentences. Prose, not a list.

Reply:"""


@dataclass
class RefusalAnswerer:
    """Generates reasoned refusals; delegates everything else to the wrapped answerer.

    Firing is driven by the router's classification rather than by keywords, so
    the decision is made once, measurably, in one place.
    """

    corpus: Corpus
    base: ExtractiveAnswerer
    client: object | None = None
    max_chars: int = 700
    answer_from: AnswerFrom = "top1"

    @property
    def name(self) -> str:
        return "refusal+extractive"

    def _passages(self, citations: Sequence[tuple[str, str]]) -> str:
        """Refusals use EVERY cited section, not just the top one.

        Deliberately different from the default ``top1`` sourcing. The two-source
        adversarial questions pair a factual section with a policy-boundary
        section, and a refusal built from only one of them answers only half the
        question. Citations are unchanged by this — only the text the model sees.
        """
        parts = []
        for doc_id, section_title in citations or []:
            section = self.corpus.get_section(doc_id, section_title)
            if section is None:
                continue
            title = self.corpus[doc_id].title if doc_id in self.corpus else doc_id
            parts.append(f"[{title} — {section_title}]\n{clean(section.body)}")
        return "\n\n".join(parts)

    def refuse(
        self,
        question: str,
        citations: Sequence[tuple[str, str]],
        compound: bool = False,
    ) -> str:
        """Generate a refusal, falling back to extractive rather than emitting nothing."""
        if self.client is None:
            return self.base.answer(question, citations)

        template = COMPOUND_PROMPT if compound else REFUSAL_PROMPT
        prompt = template.format(
            context=self._passages(citations) or "(no passages)", question=question
        )
        try:
            text = self.client.complete(prompt)
        except Exception:  # noqa: BLE001 - degrade, never crash a submission
            text = ""

        text = " ".join((text or "").split()).strip()
        if not text:
            return self.base.answer(question, citations)
        return trim_to_budget(text, self.max_chars)

    def answer(
        self,
        question: str,
        citations: Sequence[tuple[str, str]],
        is_policy_boundary: bool = False,
        compound: bool = False,
    ) -> str:
        if is_policy_boundary or compound:
            return self.refuse(question, citations, compound=compound)
        return self.base.answer(question, citations)


#: Markers a genuine refusal should contain. Used to measure whether the template
#: actually declines rather than merely reciting policy — the exact failure the
#: extractive baseline has.
REFUSAL_MARKERS: tuple[str, ...] = (
    "cannot", "can not", "can't", "not permitted", "not allowed", "no setting",
    "not something", "unable", "is not possible", "will not", "won't",
    "no workaround", "not available", "outside what",
)


def looks_like_refusal(text: str) -> bool:
    """True when the text actually declines, rather than describing a policy."""
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def addresses_student(text: str) -> bool:
    """True when the reply speaks to the student rather than reciting documentation."""
    lowered = text.lower()
    return any(token in lowered for token in (" you ", " your ", "you ", "your "))
