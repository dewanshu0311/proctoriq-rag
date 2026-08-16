"""Router tests.

The load-bearing property is that routing cannot silently make retrieval worse:
biasing must be recoverable, filtering must be opt-in, and the keyless fallback
must always produce a valid decision.
"""

from __future__ import annotations

import json

import pytest

from proctoriq_rag.retrieval.citation import RankedSection
from proctoriq_rag.routing.policy import (
    RoutingWeights,
    apply_routing,
    cardinality_strategy,
)
from proctoriq_rag.routing.router import (
    INTENTS,
    PHASES,
    PLATFORMS,
    QueryRouter,
    RouteDecision,
)
from tests.conftest import requires_competition_data


class StubClient:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        return self.reply


def sec(doc: str, title: str, score: float) -> RankedSection:
    return RankedSection(doc_id=doc, section_title=title, score=score, chunk_count=1)


VALID = json.dumps({"intent": "policy_boundary", "platform": "mac",
                    "phase": "during", "cardinality": 2})


# ── schema validation ──────────────────────────────────────────────────────
def test_parses_a_valid_response():
    d = QueryRouter(client=StubClient(VALID)).classify("Qx", "can I disable it?")
    assert (d.intent, d.platform, d.phase, d.cardinality) == (
        "policy_boundary", "mac", "during", 2)
    assert d.source == "llm"


def test_tolerates_prose_around_the_json():
    reply = f"Sure, here you go:\n```json\n{VALID}\n```\nHope that helps!"
    assert QueryRouter(client=StubClient(reply)).classify("Qx", "q").intent == "policy_boundary"


def test_unknown_values_clamp_to_defaults():
    reply = json.dumps({"intent": "banana", "platform": "linux",
                        "phase": "eventually", "cardinality": "many"})
    d = QueryRouter(client=StubClient(reply)).classify("Qx", "q")
    assert d.intent == "lookup" and d.platform == "unspecified" and d.phase == "during"
    assert d.cardinality == 1


def test_cardinality_is_capped_at_two():
    reply = json.dumps({"intent": "lookup", "platform": "unspecified",
                        "phase": "pre", "cardinality": 7})
    assert QueryRouter(client=StubClient(reply)).classify("Qx", "q").cardinality == 2


def test_unparseable_reply_falls_back():
    d = QueryRouter(client=StubClient("no json here at all")).classify("Qx", "install psb")
    assert d.source == "fallback"


def test_api_failure_falls_back_rather_than_raising():
    class Boom:
        def complete(self, prompt, **kwargs):
            raise RuntimeError("rate limited")

    assert QueryRouter(client=Boom()).classify("Qx", "install psb").source == "fallback"


# ── the keyless fallback ───────────────────────────────────────────────────
def test_fallback_needs_no_client_at_all():
    d = QueryRouter(client=None, use_llm=False).classify("Qx", "how do I install PSB?")
    assert d.source == "fallback"
    assert d.intent in INTENTS and d.platform in PLATFORMS and d.phase in PHASES


@pytest.mark.parametrize("question,expected", [
    ("How do I turn off face verification for my exam?", "policy_boundary"),
    ("Can you personally approve my re-attempt right now?", "policy_boundary"),
    ("My status shows 'Pending Review', what does that mean?", "status_outcome"),
    ("PSB installation failed with 'Element not found', what should I do?", "lookup"),
])
def test_fallback_intent_on_clear_cases(question, expected):
    assert QueryRouter.fallback_classify("Qx", question).intent == expected


@pytest.mark.parametrize("question,expected", [
    ("On my Mac, PSB installation gives an error", "mac"),
    ("I'm on Windows and PSB installation failed", "windows"),
    ("My webcam isn't being detected", "unspecified"),
])
def test_fallback_platform(question, expected):
    assert QueryRouter.fallback_classify("Qx", question).platform == expected


def test_fallback_defaults_to_the_majority_class():
    """29 of 50 are lookup and 36 of 50 are single-source; guessing costs more."""
    d = QueryRouter.fallback_classify("Qx", "something entirely unclassifiable zzz")
    assert d.intent == "lookup"
    assert d.cardinality == 1


# ── caching and determinism ────────────────────────────────────────────────
def test_second_call_hits_the_cache_and_does_not_re_query(tmp_path):
    client = StubClient(VALID)
    router = QueryRouter(client=client, cache_path=tmp_path / "r.json")
    router.classify("Qx", "same question")
    router.classify("Qx", "same question")
    assert client.calls == 1
    assert router.stats["cache"] == 1


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "r.json"
    client = StubClient(VALID)
    QueryRouter(client=client, cache_path=path).classify_all(["Qx"], ["q"])
    QueryRouter(client=client, cache_path=path).classify_all(["Qx"], ["q"])
    assert client.calls == 1


def test_corrupt_cache_is_ignored(tmp_path):
    path = tmp_path / "r.json"
    path.write_text("not json", encoding="utf-8")
    assert QueryRouter(client=StubClient(VALID), cache_path=path).classify("Qx", "q")


# ── routing must not silently destroy retrieval ────────────────────────────
def test_mode_off_is_the_identity():
    sections = [sec("01_a", "S1", 0.9), sec("02_b", "S2", 0.8)]
    assert apply_routing(sections, RouteDecision("Qx"), RoutingWeights(mode="off")) == sections


def test_bias_reorders_but_never_removes():
    """A misclassification must cost rank positions, never reachability."""
    sections = [sec("02_mac_guide", "S1", 0.90), sec("01_win_guide", "S2", 0.89)]
    routed = apply_routing(
        sections, RouteDecision("Qx", platform="windows"), RoutingWeights(mode="bias")
    )
    assert len(routed) == len(sections)
    assert {s.citation for s in routed} == {s.citation for s in sections}
    assert routed[0].doc_id == "01_win_guide"


def test_filter_does_remove_and_is_therefore_not_the_default():
    sections = [sec("02_mac_guide", "S1", 0.9), sec("01_win_guide", "S2", 0.8)]
    routed = apply_routing(
        sections, RouteDecision("Qx", platform="windows"), RoutingWeights(mode="filter")
    )
    assert all(s.doc_id != "02_mac_guide" for s in routed)


def test_score_biasing_is_off_by_default():
    """Measured negative: bias cost -1.40 on the citation half and gained 0.0000
    on adversarial while regressing lookup. The code stays as evidence; it does
    not ship enabled."""
    assert RoutingWeights().mode == "off"
    sections = [sec("02_mac_guide", "S1", 0.90), sec("01_win_guide", "S2", 0.89)]
    assert apply_routing(
        sections, RouteDecision("Qx", platform="windows"), RoutingWeights()
    ) == sections


def test_policy_boundary_lifts_the_policy_documents():
    sections = [sec("06_camera_guide", "S1", 0.90), sec("07_policy", "S4", 0.80)]
    routed = apply_routing(
        sections, RouteDecision("Qx", intent="policy_boundary"), RoutingWeights(mode="bias")
    )
    assert routed[0].doc_id == "07_policy"


def test_platform_bias_only_touches_the_install_guides():
    """Every other document is platform-neutral; boosting it would be noise."""
    sections = [sec("06_camera_guide", "S1", 0.90), sec("09_status", "S2", 0.80)]
    routed = apply_routing(
        sections, RouteDecision("Qx", platform="mac"), RoutingWeights(mode="bias")
    )
    assert [s.doc_id for s in routed] == ["06_camera_guide", "09_status"]


def test_routing_output_stays_sorted():
    sections = [sec("01_a", "S1", 0.5), sec("07_p", "S2", 0.4), sec("03_c", "S3", 0.3)]
    routed = apply_routing(
        sections, RouteDecision("Qx", intent="policy_boundary"), RoutingWeights()
    )
    assert [s.score for s in routed] == sorted([s.score for s in routed], reverse=True)


def test_empty_input_is_safe():
    assert apply_routing([], RouteDecision("Qx"), RoutingWeights()) == []


# ── cardinality comes from intent ──────────────────────────────────────────
def test_cardinality_strategy_follows_the_decision():
    assert cardinality_strategy(RouteDecision("Qx", cardinality=1)).k == 1
    assert cardinality_strategy(RouteDecision("Qx", cardinality=2)).k == 2


@requires_competition_data
def test_platform_routing_separates_the_install_guides(corpus):
    """Docs 01 and 02 are near-identical text; platform is the only separator."""
    windows = sec("01_windows_installation_login_guide",
                  "Section 2: Common Installation Errors", 0.90)
    mac = sec("02_mac_installation_login_guide",
              "Section 2: Common Installation Errors", 0.91)

    # Explicit bias mode — the capability still works, it simply is not shipped.
    for platform, expected in (("windows", "01_windows_installation_login_guide"),
                               ("mac", "02_mac_installation_login_guide")):
        routed = apply_routing(
            [windows, mac], RouteDecision("Qx", platform=platform),
            RoutingWeights(mode="bias"),
        )
        assert routed[0].doc_id == expected
