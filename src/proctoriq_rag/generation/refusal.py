"""Reasoned refusals for policy-boundary questions.

Why this is the real test, not a side quest
-------------------------------------------
Phase 2 hypothesised the adversarial gap was a retrieval problem that intent
classification would fix. The Phase 4 router A/B **falsified the retrieval half**:
routing moved adversarial document-F1 by exactly 0.0000. The policy-document
boost changed no rankings at all.

What survives is the generation half: we may already retrieve the correct policy
section and then **paste it verbatim instead of refusing**. Extractive mode emits
policy prose — "There is no setting, workaround, or support-assisted method..." —
which is the right source text and is not a refusal. It does not decline, does
not address the student, does not explain.

Three intent-conditioned variants, not one
------------------------------------------
The first version of this module used a single prompt that **mandated** a
prohibition: "state plainly that this is not something that can be done." On Q33
that produced a confidently wrong answer — *"Restarting your laptop is not
permitted when the assessment page freezes"* — which contradicts doc 05 §1, the
very section it was given, which instructs the student to press and hold the
power button. It then ignored the alt-tab request, the one thing actually
prohibited.

**No prompt may mandate a conclusion the retrieved passage does not support.**
That class of error is worse than a weak answer: it is confidently wrong against
the grader's own source text. So the refusal is now *conditional* — every variant
instructs the model to check what the passages actually say, and if a passage
tells the student to do something, the answer must not call it forbidden.

``PURE_BOUNDARY``   the question asks only for something policy forbids
``COMPOUND``        a legitimate question wrapped around a boundary request;
                    answer the legitimate half from its passage, then decline the
                    other half from the policy passage
``PLAIN``           not adversarial; answer normally, no refusal language

Refusals always receive **every** cited passage, never just the top one. The
two-source adversarial questions pair a factual section with a policy section,
and a refusal built from one of them answers half the question — which is
precisely how Q33 came to invent a prohibition it could not quote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from proctoriq_rag.corpus.loader import Corpus
from proctoriq_rag.generation.answerer import AnswerFrom, ExtractiveAnswerer
from proctoriq_rag.generation.cleaning import clean, trim_to_budget

#: Shared by every variant. This is the guard against manufacturing prohibitions.
GROUNDING_RULE = """\
Ground every claim in the passages:
- Only call something forbidden if a passage actually says it is not permitted, not possible,
  or cannot be configured. Quote the passage's own wording for the reason.
- If a passage INSTRUCTS the student to do the thing they asked about, then it is permitted.
  Say so and give the instruction. Never describe an instructed action as prohibited.
- If a passage describes something as advisable or not strictly required, keep it that way.
  Do not turn a recommendation into a rule.
- Say nothing the passages do not support. No invented steps, no invented remedies."""

STYLE_RULE = """\
Style: no greeting, no apology, no "I understand", no sign-off, no offer of further help,
no emoji. Prose, not a list. Reuse the passages' wording rather than paraphrasing."""


PURE_BOUNDARY = """\
A student has asked for something the proctoring policy does not allow. Write the reply.

Policy passages (your only source):
{context}

Student's question: {question}

Your reply must:
1. Say plainly, in the first sentence, that this is not something that can be done — but ONLY if
   the passages say so. If they do not, answer the question straightforwardly instead.
2. Explain why, in the passages' own terms.
3. If the passages say what the student CAN do, say it. If they do not, say nothing about
   alternatives.

{grounding}

{style}
Three to five sentences.

Reply:"""


COMPOUND = """\
A student has asked TWO things at once: something the documentation answers normally, and
something the proctoring policy does not allow. Both parts must be answered.

Passages (your only source):
{context}

Student's question: {question}

Your reply must:
1. FIRST answer the legitimate part, concretely, using the passage that covers it. If a passage
   tells the student to take an action, tell them to take it — do not hedge it or call it
   forbidden.
2. THEN address the part policy does not allow: say plainly that it is not permitted and explain
   why, using the policy passage's wording.

Do not skip either part. Answering only the technical half endorses the prohibited request by
silence; answering only the prohibited half abandons a real problem.

{grounding}

{style}
Four to six sentences.

Reply:"""


PLAIN = """\
Answer the student's question using only the passages below.

Passages:
{context}

Student's question: {question}

{grounding}

{style}
Two to four sentences. Do not refuse — this question asks for nothing prohibited.

Reply:"""


VARIANTS: dict[str, str] = {
    "pure_boundary": PURE_BOUNDARY,
    "compound": COMPOUND,
    "plain": PLAIN,
}


def render(variant: str, question: str, context: str) -> str:
    template = VARIANTS.get(variant, PLAIN)
    return template.format(
        context=context or "(no passages)",
        question=question,
        grounding=GROUNDING_RULE,
        style=STYLE_RULE,
    )


@dataclass
class RefusalAnswerer:
    """Generates intent-conditioned answers; delegates the rest to the base answerer."""

    corpus: Corpus
    base: ExtractiveAnswerer
    client: object | None = None
    max_chars: int = 700
    answer_from: AnswerFrom = "top1"

    #: Questions whose refusal was TRUNCATED rather than generated. A truncated
    #: refusal falls back to extractive text and produces a valid-looking row that
    #: is silently the wrong arm — the third occurrence of that shape in this
    #: project. Callers must treat a non-empty list as fatal, not informational.
    truncated: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "refusal+extractive"

    def passages(self, citations: Sequence[tuple[str, str]]) -> str:
        """Every cited section, always — not just the top one.

        Deliberately different from the default ``top1`` sourcing. Citations are
        unchanged by this; only the text the model sees.
        """
        parts = []
        for doc_id, section_title in citations or []:
            section = self.corpus.get_section(doc_id, section_title)
            if section is None:
                continue
            title = self.corpus[doc_id].title if doc_id in self.corpus else doc_id
            parts.append(f"[{title} — {section_title}]\n{clean(section.body)}")
        return "\n\n".join(parts)

    def answer_with_variant(
        self, question: str, citations: Sequence[tuple[str, str]], variant: str,
        question_id: str = "",
    ) -> str:
        """Generate under the named variant, falling back rather than emitting nothing.

        Records truncation separately from failure. An exhausted token budget on a
        reasoning model returns ``finish_reason="length"`` with empty content, which
        looks identical to an API error here but means something different: the
        model was working and ran out of room. Callers raise on it.
        """
        if self.client is None:
            return self.base.answer(question, citations)

        try:
            text = self.client.complete(
                render(variant, question, self.passages(citations))
            )
        except Exception:  # noqa: BLE001 - degrade, never crash a submission
            text = ""

        if getattr(self.client, "last_finish_reason", "") == "length" and not text:
            self.truncated.append(question_id or question[:40])

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
        if compound:
            return self.answer_with_variant(question, citations, "compound")
        if is_policy_boundary:
            return self.answer_with_variant(question, citations, "pure_boundary")
        return self.base.answer(question, citations)


# ── structural checks ──────────────────────────────────────────────────────
#: Markers a genuine refusal should contain. Used to measure whether the template
#: actually declines rather than merely reciting policy.
REFUSAL_MARKERS: tuple[str, ...] = (
    "cannot", "can not", "can't", "not permitted", "not allowed", "no setting",
    "not something", "unable", "is not possible", "will not", "won't",
    "no workaround", "not available", "outside what",
)

#: Distinctive actions the corpus INSTRUCTS. If a passage instructs one and the
#: answer calls that same action prohibited, the answer contradicts its own source.
#:
#: Deliberately narrow. An earlier version included generic verbs — check, confirm,
#: close, wait, contact — and they collide constantly on ordinary English: "checking
#: your notes is not permitted" tripped on the source's "check whether the page
#: resumes", flagging a CORRECT refusal as a contradiction. This guard exists to
#: catch confidently-wrong answers, so a false positive here is expensive: it would
#: block a good answer while teaching us nothing.
INSTRUCTION_VERBS: tuple[str, ...] = (
    "press and hold", "restart", "reopen", "reconnect", "uninstall", "reinstall",
    "force-quit", "force quit", "move closer", "relaunch",
)

PROHIBITION_PATTERNS: tuple[str, ...] = (
    "is not permitted", "are not permitted", "is not allowed", "are not allowed",
    "is prohibited", "are prohibited", "you cannot", "you can not", "you can't",
    "not something you can", "is forbidden",
)


def looks_like_refusal(text: str) -> bool:
    """True when the text actually declines, rather than describing a policy."""
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def addresses_student(text: str) -> bool:
    """True when the reply speaks to the student rather than reciting documentation."""
    lowered = text.lower()
    return any(token in lowered for token in (" you ", " your ", "you ", "your "))


def contradicts_source(text: str, source: str) -> list[str]:
    """Actions the answer calls prohibited that the source explicitly instructs.

    This is the Q33 failure mode, and it is worse than a weak answer: it is
    confidently wrong against the grader's own source text. Returns the offending
    action verbs, empty when clean.
    """
    from proctoriq_rag.generation.cleaning import split_sentences

    source_lower = source.lower()
    instructed = [v for v in INSTRUCTION_VERBS if v in source_lower]
    if not instructed:
        return []

    offenders: list[str] = []
    # Scope to the SENTENCE containing the prohibition. A character window spans
    # sentence boundaries, which flagged the corrected Q33 answer as a
    # contradiction: it instructs a restart in one sentence and refuses alt-tab in
    # the next, and those are 60 characters apart. Two separate claims in two
    # separate sentences is exactly what a correct compound answer looks like.
    # Split on the ORIGINAL case — the sentence splitter keys on a capital letter
    # after the period, so lowercasing first silently yields one giant "sentence"
    # and the scoping does nothing.
    for sentence in split_sentences(text):
        lowered = sentence.lower()
        if not any(pattern in lowered for pattern in PROHIBITION_PATTERNS):
            continue
        offenders.extend(verb for verb in instructed if verb in lowered)
    return sorted(set(offenders))
