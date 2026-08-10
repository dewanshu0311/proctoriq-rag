"""Turn markdown source text into plain prose suitable for ``answer_text``.

Deterministic and pure — no model, no randomness. The probe design depends on
extractive answers being byte-identical across runs, and this module is where
that property is actually established.

Why this matters more than it looks
-----------------------------------
Answer accuracy (25%) and groundedness (25%) are both cosine similarity against
text derived from the source documents, scored by machine with no LLM judge. So
markdown syntax is pure noise: a golden answer contains no ``- `` bullet glyphs
and no ``**bold**`` markers, and every such token spent is similarity diluted.
Stripping them is not cosmetic.
"""

from __future__ import annotations

import re

# ── markdown constructs actually present in this corpus ────────────────────
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BULLET = re.compile(r"^\s{0,4}[-*+]\s+", re.MULTILINE)
_ORDERED = re.compile(r"^\s{0,4}\d+[.)]\s+", re.MULTILINE)
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_ITALIC = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", re.DOTALL)
_CODE = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_HRULE = re.compile(r"^\s{0,3}([-*_])\s*(\1\s*){2,}$", re.MULTILINE)
_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{2,}")

#: Sentence-ish boundary. Deliberately conservative: it must not split on the
#: abbreviations this corpus actually contains ("vs.", "e.g.", "Wi-Fi").
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")

_ABBREVIATIONS = ("vs.", "e.g.", "i.e.", "etc.", "Dr.", "Mr.", "Ms.", "approx.")


def strip_markdown(text: str) -> str:
    """Remove markdown syntax, preserving the words and their order."""
    text = _HRULE.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _HEADING.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _BOLD.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)
    text = _CODE.sub(r"\1", text)
    text = _BULLET.sub("", text)
    text = _ORDERED.sub("", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces and blank lines; join lines into flowing prose.

    Bullet lists become sentences separated by ``. `` where they do not already
    end in punctuation, because a golden answer is prose rather than a list.
    """
    text = _MULTISPACE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    joined: list[str] = []
    for line in lines:
        if joined and not joined[-1].endswith((".", "!", "?", ":", ";", ",")):
            joined[-1] = joined[-1] + "."
        joined.append(line)

    result = " ".join(joined)
    result = _MULTISPACE.sub(" ", result)
    return _MULTINEWLINE.sub("\n", result).strip()


def clean(text: str) -> str:
    """Full markdown -> prose pipeline. Idempotent."""
    return normalize_whitespace(strip_markdown(text))


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences without breaking on known abbreviations."""
    guarded = text
    for index, abbreviation in enumerate(_ABBREVIATIONS):
        guarded = guarded.replace(abbreviation, f"\x00{index}\x00")

    parts = _SENTENCE_END.split(guarded)

    restored: list[str] = []
    for part in parts:
        for index, abbreviation in enumerate(_ABBREVIATIONS):
            part = part.replace(f"\x00{index}\x00", abbreviation)
        part = part.strip()
        if part:
            restored.append(part)
    return restored


def trim_to_budget(text: str, max_chars: int) -> str:
    """Trim to ``max_chars`` on a sentence boundary, never mid-word.

    Always returns at least the first sentence, even when that overruns the
    budget — a truncated fragment scores worse than a slightly long answer, and
    an empty answer fails submission validation outright.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    sentences = split_sentences(text)
    if not sentences:
        return text[:max_chars].rstrip()

    kept = sentences[0]
    for sentence in sentences[1:]:
        candidate = f"{kept} {sentence}"
        if len(candidate) > max_chars:
            break
        kept = candidate
    return kept.strip()
