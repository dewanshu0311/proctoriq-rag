"""Refusal template tests.

The load-bearing one is :func:`test_no_answer_contradicts_its_own_source`. A
confidently wrong answer against the grader's own source text is worse than a
weak one, and that is exactly the failure the first version of this module
shipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from proctoriq_rag.generation.answerer import ExtractiveAnswerer
from proctoriq_rag.generation.refusal import (
    VARIANTS,
    RefusalAnswerer,
    addresses_student,
    contradicts_source,
    looks_like_refusal,
    render,
)
from tests.conftest import REPO_ROOT, requires_competition_data

ANSWERS = REPO_ROOT / "outputs" / "refusal_answers.json"

#: Doc 05 §1 instructs a restart. Any answer calling that prohibited contradicts
#: the section it cites. This is the Q33 failure, preserved as a fixture.
FREEZE_SOURCE = (
    "If the assessment page becomes unresponsive: Wait a few seconds first and check "
    "whether the page resumes automatically. If the issue is due to your laptop itself "
    "becoming unresponsive, press and hold the power button to force a restart. After "
    "restarting, reopen the assessment using your original assessment link and continue."
)

Q33_BAD = (
    "Restarting your laptop is not permitted when the assessment page freezes. "
    "According to our proctoring policy, you should wait a few seconds first."
)

Q33_GOOD = (
    "If the assessment page froze, you should wait a few seconds first and check whether "
    "the page resumes automatically. If the issue is due to your laptop itself becoming "
    "unresponsive, press and hold the power button to force a restart. Quickly alt-tabbing "
    "to check your notes is not permitted."
)


class StubClient:
    def __init__(self, reply="A grounded refusal."):
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.reply


# ── the contradiction guard ────────────────────────────────────────────────
def test_detects_the_q33_contradiction():
    """The exact answer the first template produced must be caught."""
    assert "restart" in contradicts_source(Q33_BAD, FREEZE_SOURCE)


def test_accepts_the_corrected_q33_answer():
    """Instructing the restart AND refusing alt-tab is not a contradiction."""
    assert contradicts_source(Q33_GOOD, FREEZE_SOURCE) == []


def test_refusing_something_the_source_never_instructs_is_fine():
    source = "There is no setting or support-assisted method to disable face verification."
    text = "Disabling face verification is not permitted. There is no setting for it."
    assert contradicts_source(text, source) == []


def test_plain_extraction_never_contradicts():
    assert contradicts_source(FREEZE_SOURCE, FREEZE_SOURCE) == []


@requires_competition_data
@pytest.mark.skipif(not ANSWERS.exists(), reason="run scripts/measure_refusals.py first")
def test_no_answer_contradicts_its_own_source(corpus, answer_key):
    """No generated answer may call prohibited an action its passage instructs.

    Runs over every generated answer against the full text of its key sections.
    """
    data = json.loads(ANSWERS.read_text(encoding="utf-8"))
    qids = data["question_ids"]

    problems: list[str] = []
    for arm, texts in data["answers"].items():
        for qid, text in zip(qids, texts):
            entry = answer_key.get(qid)
            if entry is None:
                continue
            source = corpus.section_text(list(entry.pairs))
            offenders = contradicts_source(text, source)
            if offenders:
                problems.append(f"{arm}/{qid}: calls {offenders} prohibited")

    assert not problems, "answers contradict their cited source:\n" + "\n".join(problems)


# ── the three variants ─────────────────────────────────────────────────────
def test_three_variants_exist():
    assert set(VARIANTS) == {"pure_boundary", "compound", "plain"}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_no_variant_mandates_a_prohibition(variant):
    """Cause 2 of the Q33 failure: a prompt must never force a conclusion."""
    rendered = render(variant, "q?", "some passage")
    assert "Never describe an instructed action as prohibited" in rendered
    assert "Only call something forbidden if a passage actually says" in rendered


def test_pure_boundary_allows_declining_to_refuse():
    rendered = render("pure_boundary", "q?", "ctx")
    assert "ONLY if" in rendered and "If they do not, answer the question" in rendered


def test_compound_requires_both_halves():
    rendered = render("compound", "q?", "ctx")
    assert "Do not skip either part" in rendered
    assert "FIRST answer the legitimate part" in rendered


def test_plain_variant_forbids_refusal_language():
    assert "Do not refuse" in render("plain", "q?", "ctx")


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_suppresses_padding(variant):
    rendered = render(variant, "q?", "ctx")
    assert "no sign-off" in rendered and "no emoji" in rendered


# ── passage assembly ───────────────────────────────────────────────────────
@requires_competition_data
def test_refusal_receives_every_cited_passage(corpus):
    """Q33 invented a prohibition because the policy passage was never in context."""
    refuser = RefusalAnswerer(corpus=corpus, base=ExtractiveAnswerer(corpus),
                              client=StubClient())
    citations = [
        ("05_device_system_issue_playbook", "Section 1: Assessment Page Freezes or Gets Stuck"),
        ("07_permitted_prohibited_actions_policy", "Section 1: Prohibited Actions"),
    ]
    passages = refuser.passages(citations)
    assert "power button" in passages          # the legitimate half
    assert "three or four fingers" in passages  # the policy half


@requires_competition_data
def test_variant_selection_reaches_the_prompt(corpus):
    stub = StubClient()
    refuser = RefusalAnswerer(corpus=corpus, base=ExtractiveAnswerer(corpus), client=stub)
    refuser.answer_with_variant("q?", [], "compound")
    assert "Do not skip either part" in stub.prompts[0]


@requires_competition_data
def test_falls_back_rather_than_emitting_empty(corpus):
    class Boom:
        def complete(self, prompt, **kwargs):
            raise RuntimeError("rate limited")

    refuser = RefusalAnswerer(corpus=corpus, base=ExtractiveAnswerer(corpus), client=Boom())
    citations = [("07_permitted_prohibited_actions_policy",
                  "Section 4: This Policy Cannot Be Configured or Bypassed")]
    assert refuser.answer_with_variant("q?", citations, "pure_boundary").strip()


@requires_competition_data
def test_no_client_degrades_to_extractive(corpus):
    refuser = RefusalAnswerer(corpus=corpus, base=ExtractiveAnswerer(corpus), client=None)
    citations = [("07_permitted_prohibited_actions_policy",
                  "Section 4: This Policy Cannot Be Configured or Bypassed")]
    assert refuser.answer_with_variant("q?", citations, "pure_boundary").strip()


# ── structural checks ──────────────────────────────────────────────────────
def test_looks_like_refusal_distinguishes_declining_from_reciting():
    assert looks_like_refusal("Disabling face verification is not permitted.")
    assert not looks_like_refusal(
        "Face verification runs continuously throughout your session."
    )


def test_addresses_student():
    assert addresses_student("You cannot disable this.")
    assert not addresses_student("The policy defines a fixed integrity boundary.")
