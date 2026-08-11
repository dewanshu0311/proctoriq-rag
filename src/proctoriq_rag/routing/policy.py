"""Turning a :class:`RouteDecision` into retrieval behaviour.

**Bias, do not filter.** This is the one design decision in the routing layer
that matters most. A hard document filter on a misclassification is
unrecoverable — the correct section is removed from consideration and the
question is lost outright, scoring zero on both citation dimensions. A score
boost degrades gracefully: a wrong platform call costs a few rank positions on a
question that might still land correctly.

The router is the first component in this project that can make retrieval
*worse*. Lookup document-F1 is already 1.000 locally, so on that class there is
nothing to gain and everything to lose. Filtering is implemented so the
comparison can be measured rather than asserted, but ``bias`` is the default and
the burden of proof is on ``filter`` to beat it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from proctoriq_rag.retrieval.citation import RankedSection
from proctoriq_rag.routing.router import PHASE_DOCS, PLATFORM_DOCS, POLICY_DOCS, RouteDecision


@dataclass(frozen=True)
class RoutingWeights:
    """Multiplicative score boosts. 1.0 means "no opinion".

    Values are deliberately mild. These multiply cross-encoder scores that are
    already well separated at the top, so a large boost would override the
    model's judgement wholesale rather than break ties — which is how routing
    turns a 1.000 lookup score into a regression.
    """

    platform_match: float = 1.25
    platform_mismatch: float = 0.80
    phase_match: float = 1.10
    policy_doc: float = 1.30
    mode: str = "bias"           # bias | filter | off


def _doc_prefix(doc_id: str) -> str:
    return doc_id[:2]


def apply_routing(
    sections: Sequence[RankedSection],
    decision: RouteDecision,
    weights: RoutingWeights = RoutingWeights(),
) -> list[RankedSection]:
    """Re-rank sections according to the routing decision.

    Returns a new list, descending by adjusted score. ``mode="off"`` returns the
    input unchanged, which is what the router-off arm of the A/B measures.
    """
    if weights.mode == "off" or not sections:
        return list(sections)

    adjusted: list[RankedSection] = []
    for section in sections:
        prefix = _doc_prefix(section.doc_id)
        factor = 1.0

        # Platform. Only ever applied to the two install guides — every other
        # document is platform-neutral and boosting it would be noise.
        if decision.platform in PLATFORM_DOCS and prefix in PLATFORM_DOCS.values():
            wanted = PLATFORM_DOCS[decision.platform]
            factor *= (
                weights.platform_match if prefix == wanted else weights.platform_mismatch
            )

        # Phase. This is the Q15/Q12 disambiguator: docs 01 §4 and 02 §5 restate
        # the 24-hour mock-test rule as closing cross-references, while doc 03 §2
        # is its canonical home and the only place carrying the "not a hard
        # technical requirement" qualifier. Both are "pre", so phase alone does
        # not separate them — but combined with platform=unspecified it stops the
        # platform-specific guides from winning a question that names no platform.
        if prefix in PHASE_DOCS.get(decision.phase, ()):
            factor *= weights.phase_match

        # Policy boundary. When the question is asking what the system will NOT
        # do, the answer lives in docs 07 and 10 regardless of which subsystem
        # the question names — which is exactly the failure reranking could not
        # fix, because the subsystem doc genuinely IS more similar.
        if decision.is_policy_boundary and prefix in POLICY_DOCS:
            factor *= weights.policy_doc

        if weights.mode == "filter":
            if decision.platform in PLATFORM_DOCS and prefix in PLATFORM_DOCS.values():
                if prefix != PLATFORM_DOCS[decision.platform]:
                    continue

        adjusted.append(
            RankedSection(
                doc_id=section.doc_id,
                section_title=section.section_title,
                score=section.score * factor,
                chunk_count=section.chunk_count,
            )
        )

    adjusted.sort(key=lambda s: -s.score)
    return adjusted


def cardinality_strategy(decision: RouteDecision):
    """Cardinality comes from intent, not from a score threshold.

    Phase 2 measured this directly: with genuinely discriminative cross-encoder
    scores, every score-gap rule still scored *below* fixed ``topk-1``,
    monotonically worse the more it cited. Whether a question needs two sources
    is a property of what it asks — a comparison wants two documents because it
    is a comparison, not because two sections happen to score similarly.

    Probe 5 bounds the prize: blanket ``topk-2`` cost 2.37 points, so citing two
    only where genuinely needed should recover that and add a little.
    """
    from proctoriq_rag.retrieval.citation import TopK

    return TopK(2 if decision.cardinality >= 2 else 1)
