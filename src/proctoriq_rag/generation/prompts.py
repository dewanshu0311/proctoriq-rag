"""Prompt templates, and the style contract they all enforce.

The scoring reality this is built around
----------------------------------------
Answer accuracy (25%) and groundedness (25%) are cosine similarity against text
written from the source documents. **There is no LLM judge.** Nothing rewards
helpfulness, warmth, or good manners, and every token spent on them is
similarity diluted:

    "I'm sorry to hear that! Here's what you can do:"   <- ~10 tokens of pure noise
    "Hope this helps! Let me know if you need anything else."

None of that appears in a golden answer or a source excerpt. So the templates
are written to suppress it explicitly rather than hoping the model stays terse.

The refusal requirement
-----------------------
Integrity-refusal is 15%, and a bare refusal scores lower than a reasoned one.
"I can't help with that" is both unhelpful and short — doubly bad against a
reference refusal that explains the boundary. So refusals must state the
boundary, explain it in the policy's own terms, and where the source supports it,
say what the student *can* do.

The nuance trap
---------------
Doc 03 §2 says the 24-hour mock-test window is "not a hard technical requirement"
but is strongly advised. An answer that flattens that into "no, you must" both
contradicts its own cited passage and scores badly against it. Every template
carries an explicit instruction not to convert documented nuance into a flat
prohibition — this is checked by hand on that question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

#: Shared by every template. Kept separate so a template comparison varies the
#: framing, not the ground rules.
STYLE_CONTRACT = """\
Rules for your answer:
- Answer immediately. No greeting, no restating the question, no acknowledgement.
- Use the wording of the source passages wherever it fits. Do not paraphrase for variety.
- Include nothing that is not stated in the passages. No general troubleshooting advice,
  no invented steps, no guesses about causes.
- No sign-off, no "hope this helps", no offer of further assistance, no emoji.
- Match the length and register of the passages. Prose, not bullet lists.
- If the passages describe something as advisable, recommended, or not strictly required,
  say it that way. Do not turn a recommendation into a rule or a rule into a suggestion.
- If the question asks for something the passages say is not permitted or not possible,
  say so plainly, explain why using the passages' own reasoning, and state what the
  student can do instead if the passages say."""


@dataclass(frozen=True)
class PromptTemplate:
    """One candidate prompt, identified for the comparison report."""

    name: str
    description: str
    body: str

    def render(self, question: str, passages: Sequence[tuple[str, str, str]]) -> str:
        """``passages`` is a sequence of ``(doc_title, section_title, text)``."""
        context = "\n\n".join(
            f"[{doc_title} — {section_title}]\n{text}"
            for doc_title, section_title, text in passages
        )
        return self.body.format(
            question=question, context=context, style=STYLE_CONTRACT
        )


TERSE_EXTRACTIVE = PromptTemplate(
    name="terse-extractive",
    description=(
        "Maximum source echo. Instructs the model to behave almost like an "
        "extractive summariser, on the theory that similarity scoring rewards "
        "reusing the source's exact wording."
    ),
    body="""\
You are answering a student's question about the ProctorIQ assessment platform using only the
support documentation below.

Passages:
{context}

Question: {question}

{style}

Answer using the passages' own sentences wherever possible, lightly edited only where needed to
read as a direct answer. Aim for two to four sentences.

Answer:""",
)


STRUCTURED_STEPS = PromptTemplate(
    name="structured-steps",
    description=(
        "Preserves the source's step ordering as flowing prose. Most of this "
        "corpus is procedural, so the ordered steps may be the substance a "
        "golden answer is built from."
    ),
    body="""\
You are a support assistant for the ProctorIQ assessment platform. Answer the student's question
using only the documentation passages below.

Passages:
{context}

Question: {question}

{style}

If the passages give a sequence of steps, keep them in the same order and express them as prose
sentences rather than a list. If the passages give a fact or a policy rather than steps, state it
directly.

Answer:""",
)


ANSWER_FIRST_EXPLAINED = PromptTemplate(
    name="answer-first-explained",
    description=(
        "Direct answer first, then the source's reasoning. Designed for the "
        "adversarial third of the test set, where a reasoned refusal scores "
        "higher than a bare one."
    ),
    body="""\
You are a support assistant for the ProctorIQ assessment platform. Answer the student's question
using only the documentation passages below.

Passages:
{context}

Question: {question}

{style}

Structure: first sentence gives the direct answer — yes, no, or the specific action to take. The
sentences after it give the reason and the detail, drawn from the passages. If the question mixes a
legitimate request with one the passages do not permit, answer the legitimate part fully and address
the other part explicitly; do not ignore either.

Answer:""",
)


TEMPLATES: tuple[PromptTemplate, ...] = (
    TERSE_EXTRACTIVE,
    STRUCTURED_STEPS,
    ANSWER_FIRST_EXPLAINED,
)

TEMPLATES_BY_NAME = {template.name: template for template in TEMPLATES}

DEFAULT_TEMPLATE = ANSWER_FIRST_EXPLAINED.name

#: Phrases that dilute similarity. Counted in the template comparison so the
#: style contract's effectiveness is measured rather than assumed.
PADDING_PHRASES: tuple[str, ...] = (
    "i'm sorry", "i am sorry", "sorry to hear", "hope this helps",
    "let me know", "feel free", "here's what", "here is what",
    "great question", "certainly", "of course", "i'd be happy",
    "i would be happy", "as an ai", "please note that", "don't worry",
    "no problem", "happy to help", "thanks for", "thank you for",
)


def count_padding(text: str) -> int:
    """How many padding phrases appear in an answer."""
    lowered = text.lower()
    return sum(1 for phrase in PADDING_PHRASES if phrase in lowered)
