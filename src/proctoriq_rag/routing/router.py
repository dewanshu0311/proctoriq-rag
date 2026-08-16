"""Smart Query Router — LLM intent classification, before retrieval.

Why this runs before retrieval rather than after
------------------------------------------------
Phase 2 established that reranking cannot fix the adversarial gap. Reranking
lifted adversarial doc-F1 from 0.450 to 0.643, but the gap versus lookup
questions held, because **the ranking is not wrong**. When a student asks how to
disable face verification, the genuinely most relevant passage *is* the
face-verification guide — not the policy boundary that answers them. No amount of
re-ordering fixes a ranking that is already correct by its own measure. Only
knowing what the question is *for* fixes it.

Four axes, each chosen because there is evidence it matters:

``intent``       lookup / policy_boundary / status_outcome / compound
``platform``     windows / mac / unspecified — 9 questions are platform-specific
                 and docs 01 and 02 are near-identical text
``phase``        pre / during / post — lighting appears in doc 03 for the mock
                 test and doc 06 for the live exam, same symptom, different
                 correct citation
``cardinality``  1 or 2 — Phase 2 proved no score-gap rule recovers this
                 (D-022), so it has to come from intent

Two properties this module is built around
------------------------------------------
**It biases, it does not filter.** A hard document filter on a misclassification
is unrecoverable: the correct section becomes unreachable and the question is
lost outright. A score boost degrades gracefully — a wrong platform call costs a
few rank positions, not the answer. Filtering is implemented too, so the choice
is measured rather than asserted, but bias is the default.

**It always has a keyless fallback.** Classification is an LLM call, and the
extractive pipeline currently needs no API key at all. A deterministic keyword
classifier backs every axis, so a Groq outage on Kaggle degrades the routing
rather than failing the run.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Sequence

Intent = Literal["lookup", "policy_boundary", "status_outcome", "compound"]
Platform = Literal["windows", "mac", "unspecified"]
Phase = Literal["pre", "during", "post"]

INTENTS: tuple[str, ...] = ("lookup", "policy_boundary", "status_outcome", "compound")
PLATFORMS: tuple[str, ...] = ("windows", "mac", "unspecified")
PHASES: tuple[str, ...] = ("pre", "during", "post")

#: Document prefixes belonging to each assessment phase. Derived from the corpus
#: ordering, which is itself chronological: install -> during-exam -> aftermath.
PHASE_DOCS: dict[str, tuple[str, ...]] = {
    "pre": ("01", "02", "03"),
    "during": ("04", "05", "06", "07"),
    "post": ("08", "09", "10"),
}

#: Documents that define policy boundaries. A policy_boundary question is asking
#: what the system will not do, and the answer lives here regardless of which
#: subsystem the question names.
POLICY_DOCS: tuple[str, ...] = ("07", "10")

PLATFORM_DOCS: dict[str, str] = {"windows": "01", "mac": "02"}


@dataclass(frozen=True)
class RouteDecision:
    """One question's classification, plus how it was obtained."""

    question_id: str
    intent: str = "lookup"
    platform: str = "unspecified"
    phase: str = "during"
    cardinality: int = 1
    source: str = "llm"          # llm | fallback | cache
    raw: str = ""

    @property
    def is_policy_boundary(self) -> bool:
        return self.intent in ("policy_boundary", "compound")

    def as_dict(self) -> dict:
        return asdict(self)


def _clamp(value: object, allowed: Sequence[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


CLASSIFY_PROMPT = """\
Classify this student support question for a proctored-exam platform.

Question: {question}

Answer with a single JSON object and nothing else:
{{"intent": ..., "platform": ..., "phase": ..., "cardinality": ...}}

intent — one of:
  "lookup"          asking how to do or fix something; the docs answer it directly
  "policy_boundary" asking to disable, bypass, be exempted from, or be guaranteed
                    an outcome from a proctoring or integrity control; also any
                    request for special treatment or an on-the-spot decision
  "status_outcome"  asking what a status means, where to check it, or how long a
                    review takes
  "compound"        genuinely two things at once: something the documentation
                    answers normally AND a request policy forbids. This includes
                    asking what a status or outcome MEANS and then asking for it
                    to be changed, overridden, guaranteed, or fast-tracked; and
                    describing a technical problem and then asking whether some
                    prohibited shortcut is acceptable.

platform — "windows", "mac", or "unspecified" if the question names neither

phase — one of:
  "pre"     before the exam: installing, logging in, mock test, readiness
  "during"  while the exam is running: disconnects, freezes, camera warnings,
            what is and is not permitted mid-exam
  "post"    after the exam: reporting an issue, re-attempt requests, statuses,
            contacting support

cardinality — 1 normally. Use 2 when answering well genuinely needs TWO different
documents, which happens in two recognisable shapes:
  - an explicit comparison between platforms or situations
  - a factual answer PLUS a policy boundary: the student asks what something
    means or what to do, and also asks for something to be overridden, approved
    on the spot, guaranteed, or exempted. One document explains the process or
    the status; a different one states what support or the system will never do.
Judge cardinality on its own merits — a question can be "compound" and still be
answerable from a single document.

JSON only."""


class QueryRouter:
    """Classifies questions before retrieval. LLM first, keyword fallback always."""

    def __init__(
        self,
        client: object | None = None,
        cache_path: Path | str | None = None,
        use_llm: bool = True,
    ) -> None:
        self.client = client
        self.cache_path = Path(cache_path) if cache_path else None
        self.use_llm = use_llm
        self._cache: dict[str, dict] = {}
        self.stats = {"llm": 0, "fallback": 0, "cache": 0}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._cache = {}

    # ── the keyless fallback ───────────────────────────────────────────────
    @staticmethod
    def fallback_classify(question_id: str, question: str) -> RouteDecision:
        """Deterministic keyword classifier. No API key, no network, no variance.

        Deliberately conservative: it defaults to ``lookup`` and cardinality 1,
        which is the majority class on both axes (29 of 50 and 36 of 50). A
        fallback that guesses aggressively would do more damage than one that
        declines to.
        """
        text = question.lower()

        boundary_markers = (
            "turn off", "disable", "bypass", "without my camera", "camera turned off",
            "won't apply", "wont apply", "exempt", "override", "guarantee",
            "basically guaranteed", "confirm that now", "fast-track", "fast track",
            "personally approve", "approve my re-attempt right now", "just tell me",
            "so it stops", "pause my assessment", "watch my screen",
        )
        status_markers = (
            "status shows", "check the status", "what does that mean", "pending review",
            "under verification", "flag cleared", "flag upheld", "how do i check",
            "hasn't changed", "re-attempt was approved",
        )
        compound_markers = ("also,", "— also", "and right after", "as well as", "also is it")

        if any(marker in text for marker in compound_markers) and any(
            marker in text for marker in boundary_markers
        ):
            intent = "compound"
        elif any(marker in text for marker in boundary_markers):
            intent = "policy_boundary"
        elif any(marker in text for marker in status_markers):
            intent = "status_outcome"
        else:
            intent = "lookup"

        if "mac" in text or "m-series" in text or "macbook" in text:
            platform = "mac"
        elif "windows" in text:
            platform = "windows"
        else:
            platform = "unspecified"

        if any(m in text for m in ("mock test", "install", "log in", "login", "otp",
                                   "pin", "before my", "day ahead")):
            phase = "pre"
        elif any(m in text for m in ("status", "re-attempt", "reattempt", "report",
                                     "support team", "review")):
            phase = "post"
        else:
            phase = "during"

        cardinality = 2 if ("versus" in text or " vs " in text
                            or "difference between" in text) else 1

        return widen_compound(RouteDecision(
            question_id=question_id, intent=intent, platform=platform,
            phase=phase, cardinality=cardinality, source="fallback",
        ), question)

    # ── LLM classification ─────────────────────────────────────────────────
    def _parse(self, question_id: str, raw: str) -> RouteDecision | None:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None

        try:
            cardinality = int(data.get("cardinality", 1))
        except (TypeError, ValueError):
            cardinality = 1

        return RouteDecision(
            question_id=question_id,
            intent=_clamp(data.get("intent"), INTENTS, "lookup"),
            platform=_clamp(data.get("platform"), PLATFORMS, "unspecified"),
            phase=_clamp(data.get("phase"), PHASES, "during"),
            cardinality=2 if cardinality >= 2 else 1,
            source="llm",
            raw=raw[:400],
        )

    def classify(self, question_id: str, question: str) -> RouteDecision:
        if question in self._cache:
            self.stats["cache"] += 1
            cached = dict(self._cache[question])
            cached["question_id"] = question_id
            cached.setdefault("source", "cache")
            return RouteDecision(**{
                k: v for k, v in cached.items()
                if k in RouteDecision.__dataclass_fields__
            })

        if self.use_llm and self.client is not None:
            try:
                raw = self.client.complete(CLASSIFY_PROMPT.format(question=question))
                decision = self._parse(question_id, raw)
                if decision is not None:
                    decision = widen_compound(decision, question)
                    self.stats["llm"] += 1
                    self._cache[question] = decision.as_dict()
                    return decision
            except Exception:  # noqa: BLE001 - any API failure degrades, never crashes
                pass

        self.stats["fallback"] += 1
        return self.fallback_classify(question_id, question)

    def classify_all(
        self, question_ids: Sequence[str], questions: Sequence[str]
    ) -> list[RouteDecision]:
        decisions = [
            self.classify(qid, q) for qid, q in zip(question_ids, questions)
        ]
        self.save()
        return decisions

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._cache, indent=2), encoding="utf-8"
            )


# ── compound widening ──────────────────────────────────────────────────────
#: Asking for a decision, override, exemption, or guarantee.
_BOUNDARY_REQUEST = (
    "can you", "can support", "can a support", "can they", "can my", "can i",
    "could you", "will that", "won't apply", "wont apply", "override", "exempt",
    "fast-track", "fast track", "approve", "confirm that", "guaranteed",
    "turn off", "disable", "bypass", "pause my", "okay to", "ok to",
)

#: Asking what something is, means, or what to do — the factual half.
#: Deliberately excludes "how do i" and "why can" — those are the phrasing of
#: most boundary REQUESTS too ("how do I turn off face verification"), so
#: including them made every pure-boundary question look compound and sent it to
#: a prompt that asks the model to answer a legitimate half that does not exist.
_FACTUAL_QUESTION = (
    "what does", "what do the", "what should", "how long",
    "where can i", "what happens", "what information", "status shows",
    "what's the difference", "what is the difference",
)

#: A described incident: the student is reporting a real problem, not only asking.
_INCIDENT = (
    "froze", "frozen", "disconnected", "dropped", "broke", "broken", "died",
    "unresponsive", "restart", "warning", "flagged", "interrupted", "failed",
    "not detected", "isn't", "won't let me",
)


def widen_compound(decision: "RouteDecision", question: str) -> "RouteDecision":
    """Reclassify as compound when a factual half and a boundary request co-occur.

    Router intent accuracy is 84%, so trusting a ``lookup`` call on a genuinely
    adversarial question loses integrity-refusal outright on those misses.
    Refusing-and-answering a compound question costs far less than missing a
    refusal entirely, so the asymmetry favours firing when in doubt.

    Deliberately a *general* signal rather than a rule tuned question-by-question
    against the key — fitting it to the holdout is exactly the overfitting this
    project has guarded against throughout. Whatever recall it achieves is
    reported as measured, not iterated until it matches.
    """
    text = question.lower()
    has_boundary = any(marker in text for marker in _BOUNDARY_REQUEST)
    has_factual = any(marker in text for marker in _FACTUAL_QUESTION)
    has_incident = any(marker in text for marker in _INCIDENT)

    if has_boundary and (has_factual or has_incident):
        # Intent only — cardinality is deliberately NOT raised here.
        #
        # The two uses of "compound" have opposite risk profiles and measurement
        # forced them apart. For REFUSAL firing the asymmetry favours recall:
        # missing a refusal loses integrity-refusal (15%) outright, while
        # refusing-and-answering a compound question costs little. For CARDINALITY
        # the asymmetry runs the other way: probe 5 showed a blanket second
        # citation costs 2.37 points, and a false positive always adds a wrong
        # citation while a true positive only pays when the SECOND ranked section
        # is also right.
        #
        # Measured: widening both together took recall 1/14 -> 7/14 but added 5
        # false positives, and the citation half fell from 27.17 to 26.50 — below
        # router-off at 26.93. Widening intent alone keeps the refusal recall and
        # gives the citation cost back.
        return RouteDecision(
            question_id=decision.question_id,
            intent="compound",
            platform=decision.platform,
            phase=decision.phase,
            cardinality=decision.cardinality,
            source=decision.source,
            raw=decision.raw,
        )

    return decision
