"""Generation tests.

The load-bearing one is determinism: the entire probe design rests on extractive
``answer_text`` being byte-identical across runs and invariant to the citation
cardinality flag.
"""

from __future__ import annotations

import pytest

from proctoriq_rag.generation.answerer import (
    ExtractiveAnswerer,
    GroqAnswerer,
    SubsectionFocuser,
    build_answerer,
    select_source_citations,
)
from proctoriq_rag.generation.cleaning import (
    clean,
    normalize_whitespace,
    split_sentences,
    strip_markdown,
    trim_to_budget,
)
from proctoriq_rag.generation.groq_client import (
    GroqChatClient,
    GroqError,
    RetryPolicy,
    _is_retryable,
    discover_keys,
)
from proctoriq_rag.generation.prompts import (
    DEFAULT_TEMPLATE,
    TEMPLATES,
    TEMPLATES_BY_NAME,
    count_padding,
)
from tests.conftest import requires_competition_data

DOC01 = "01_windows_installation_login_guide"
INSTALL_ERRORS = "Section 2: Common Installation Errors"


class StubClient:
    """Records prompts, returns canned replies."""

    def __init__(self, reply="A grounded answer from the passage.", fail_times=0):
        self.reply = reply
        self.prompts: list[str] = []
        self.fail_times = fail_times
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls <= self.fail_times:
            raise RuntimeError("rate limit exceeded (429)")
        return self.reply


# ── cleaning ───────────────────────────────────────────────────────────────
def test_strip_markdown_removes_every_construct_in_the_corpus():
    text = (
        "### \"Element not found\"\n"
        "- **Uninstall** PSB via `Control Panel`.\n"
        "1. Restart your laptop.\n"
        "> a quote\n"
        "See [the guide](http://example.com).\n"
    )
    out = strip_markdown(text)
    for marker in ("###", "**", "`", "- ", "1. ", "> ", "](", "http"):
        assert marker not in out
    assert "Uninstall" in out and "Control Panel" in out and "the guide" in out


def test_clean_is_idempotent():
    text = "## Heading\n- **bold** item\n- second item\n"
    assert clean(clean(text)) == clean(text)


def test_normalize_whitespace_joins_bullets_into_sentences():
    out = normalize_whitespace("Uninstall PSB\nRestart your laptop\nInstall again")
    assert "\n" not in out
    assert out.count(".") >= 2


def test_clean_produces_no_double_spaces():
    assert "  " not in clean("a    b\n\n\nc")


def test_split_sentences_respects_abbreviations():
    text = "Automatic Termination vs. Logged Flag is the topic. The next sentence follows."
    assert len(split_sentences(text)) == 2


def test_trim_to_budget_cuts_on_sentence_boundary():
    text = "First sentence here. Second sentence here. Third sentence here."
    out = trim_to_budget(text, 45)
    assert out.endswith(".")
    assert "Third" not in out


def test_trim_keeps_first_sentence_even_when_over_budget():
    """An empty or fragmentary answer fails validation; a long one merely dilutes."""
    text = "A single very long sentence that exceeds the budget on its own by some margin."
    assert trim_to_budget(text, 10) == text


def test_trim_is_a_noop_under_budget():
    assert trim_to_budget("Short.", 500) == "Short."


# ── extractive: the determinism the probes depend on ───────────────────────
@requires_competition_data
def test_extractive_answers_every_question(corpus, answer_key):
    answerer = ExtractiveAnswerer(corpus)
    for entry in answer_key:
        text = answerer.answer("some question", list(entry.pairs))
        assert text.strip(), f"{entry.id} produced an empty answer"


@requires_competition_data
def test_extractive_is_byte_identical_across_runs(corpus):
    """The property the whole probe design rests on."""
    citations = [(DOC01, INSTALL_ERRORS)]
    question = "PSB shows Session Start Error, how do I resolve it?"
    first = ExtractiveAnswerer(corpus).answer(question, citations)
    for _ in range(5):
        assert ExtractiveAnswerer(corpus).answer(question, citations) == first


@requires_competition_data
def test_extractive_is_invariant_to_citation_cardinality(corpus):
    """topk-1 vs topk-2 must not change answer_text — this is what makes probe 5 clean."""
    question = "What's different about fixing this on Windows versus Mac?"
    one = [(DOC01, INSTALL_ERRORS)]
    two = one + [("02_mac_installation_login_guide", INSTALL_ERRORS)]
    answerer = ExtractiveAnswerer(corpus, answer_from="top1")
    assert answerer.answer(question, one) == answerer.answer(question, two)


@requires_competition_data
def test_answer_from_cited_does_use_both_sections(corpus):
    """The alternative mode exists and genuinely differs."""
    question = "Windows versus Mac?"
    one = [(DOC01, INSTALL_ERRORS)]
    two = one + [("02_mac_installation_login_guide", INSTALL_ERRORS)]
    answerer = ExtractiveAnswerer(corpus, answer_from="cited", max_chars=4000)
    assert answerer.answer(question, two) != answerer.answer(question, one)


@requires_competition_data
def test_extractive_contains_no_markdown(corpus, answer_key):
    answerer = ExtractiveAnswerer(corpus)
    for entry in answer_key:
        text = answerer.answer("q", list(entry.pairs))
        assert "###" not in text and "**" not in text and "\n" not in text


@requires_competition_data
def test_extractive_respects_the_char_budget(corpus, answer_key):
    answerer = ExtractiveAnswerer(corpus, max_chars=300)
    lengths = [len(answerer.answer("q", list(e.pairs))) for e in answer_key]
    assert max(lengths) < 900  # first-sentence overrun allowed, runaway is not


@requires_competition_data
def test_extractive_content_comes_from_the_source(corpus):
    """No invented content: every token must appear in the cited section."""
    citations = [(DOC01, INSTALL_ERRORS)]
    text = ExtractiveAnswerer(corpus).answer("session start error", citations)
    source = corpus.get_section(*citations[0]).body.lower()
    answer_tokens = {t for t in text.lower().replace(".", " ").split() if len(t) > 3}
    source_tokens = {t for t in source.replace(".", " ").split() if len(t) > 3}
    missing = answer_tokens - source_tokens
    assert not missing, f"tokens absent from source: {sorted(missing)[:10]}"


@requires_competition_data
def test_extractive_returns_empty_for_no_citations(corpus):
    assert ExtractiveAnswerer(corpus).answer("q", []) == ""


# ── subsection focusing ────────────────────────────────────────────────────
@requires_competition_data
def test_focuser_selects_the_matching_subsection(corpus):
    """Doc 01 §2 covers three unrelated errors; the answer should be about one."""
    focuser = SubsectionFocuser(corpus)
    session = focuser.focus("Session Start Error firewall", DOC01, INSTALL_ERRORS)
    element = focuser.focus("Element not found driver", DOC01, INSTALL_ERRORS)
    assert session != element
    assert "antivirus" in session.lower()
    assert "Control Panel" in element


@requires_competition_data
def test_focuser_returns_whole_section_when_no_subsections(corpus):
    focuser = SubsectionFocuser(corpus)
    title = "Section 1: Prohibited Actions"
    body = focuser.focus("q", "07_permitted_prohibited_actions_policy", title)
    assert "three or four fingers" in body


@requires_competition_data
def test_focus_never_changes_the_citation(corpus):
    """Focusing picks text from a ### block; the citation stays the parent ##."""
    answerer = ExtractiveAnswerer(corpus, focus_subsections=True)
    citations = [(DOC01, INSTALL_ERRORS)]
    answerer.answer("session start error", citations)
    assert citations == [(DOC01, INSTALL_ERRORS)]


def test_select_source_citations_modes():
    pairs = [("a", "S1"), ("b", "S2")]
    assert select_source_citations(pairs, "top1") == [("a", "S1")]
    assert select_source_citations(pairs, "cited") == pairs
    assert select_source_citations([], "top1") == []


# ── prompts ────────────────────────────────────────────────────────────────
def test_three_templates_with_distinct_names():
    assert len(TEMPLATES) == 3
    assert len({t.name for t in TEMPLATES}) == 3
    assert DEFAULT_TEMPLATE in TEMPLATES_BY_NAME


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.name)
def test_every_template_carries_the_style_contract(template):
    rendered = template.render("q?", [("Doc", "Section 1: X", "body text")])
    for requirement in ("No sign-off", "not stated in the passages", "recommendation into a rule"):
        assert requirement in rendered


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.name)
def test_render_includes_question_and_passages(template):
    rendered = template.render("Why is my camera flagged?", [("D", "S", "passage body")])
    assert "Why is my camera flagged?" in rendered
    assert "passage body" in rendered


def test_count_padding_detects_conversational_filler():
    assert count_padding("I'm sorry to hear that! Hope this helps!") >= 2
    assert count_padding("Restart your laptop and reopen PSB.") == 0


# ── groq client ────────────────────────────────────────────────────────────
def _isolate_keys(monkeypatch):
    """Clear every GROQ_API_KEY_* so a real local .env cannot leak into the test."""
    import os
    for name in list(os.environ):
        if name.startswith("GROQ_API_KEY"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "proctoriq_rag.generation.groq_client.load_dotenv", lambda *a, **k: {}
    )


def test_discover_keys_defaults_to_single_key(monkeypatch):
    """Single key is the default; rotation is opt-in complexity."""
    _isolate_keys(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "primary")
    monkeypatch.setenv("GROQ_API_KEY_1", "one")
    monkeypatch.setenv("GROQ_API_KEY_2", "two")
    monkeypatch.setenv("GROQ_ROTATE_KEYS", "0")
    assert discover_keys() == ["primary"]


def test_discover_keys_rotation_opt_in(monkeypatch):
    _isolate_keys(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "primary")
    monkeypatch.setenv("GROQ_API_KEY_1", "one")
    monkeypatch.setenv("GROQ_API_KEY_2", "two")
    assert discover_keys(rotate=True) == ["primary", "one", "two"]


def test_retry_policy_backs_off_and_is_bounded():
    policy = RetryPolicy(base_delay=1.0, max_delay=8.0, jitter=0.0)
    assert policy.delay_for(0) == 1.0
    assert policy.delay_for(1) == 2.0
    assert policy.delay_for(10) == 8.0


def test_rate_limit_errors_are_retryable():
    assert _is_retryable(RuntimeError("rate limit exceeded"))
    assert _is_retryable(RuntimeError("429 Too Many Requests"))
    assert not _is_retryable(ValueError("invalid model name"))


def test_client_raises_without_a_key(monkeypatch):
    _isolate_keys(monkeypatch)
    monkeypatch.setattr(
        "proctoriq_rag.generation.groq_client.discover_keys", lambda *a, **k: []
    )
    client = GroqChatClient(api_keys=[])
    with pytest.raises(GroqError, match="No Groq API key"):
        _ = client.clients


# ── generative mode ────────────────────────────────────────────────────────
@requires_competition_data
def test_groq_answerer_uses_the_template(corpus):
    stub = StubClient()
    answerer = GroqAnswerer(corpus, client=stub, template_name="terse-extractive")
    text = answerer.answer("Why?", [(DOC01, INSTALL_ERRORS)])
    assert text == "A grounded answer from the passage."
    assert "Why?" in stub.prompts[0]
    assert "No sign-off" in stub.prompts[0]


@requires_competition_data
def test_groq_answerer_falls_back_rather_than_emitting_empty(corpus):
    """An empty answer_text fails validation and scores zero on four dimensions."""
    class AlwaysFails:
        def complete(self, prompt, **kwargs):
            raise RuntimeError("boom")

    answerer = GroqAnswerer(corpus, client=AlwaysFails())
    text = answerer.answer("Session Start Error?", [(DOC01, INSTALL_ERRORS)])
    assert text.strip()


@requires_competition_data
def test_groq_answerer_falls_back_on_blank_reply(corpus):
    answerer = GroqAnswerer(corpus, client=StubClient(reply="   "))
    assert answerer.answer("q", [(DOC01, INSTALL_ERRORS)]).strip()


@requires_competition_data
def test_groq_prompt_only_contains_top1_passage(corpus):
    stub = StubClient()
    answerer = GroqAnswerer(corpus, client=stub, answer_from="top1")
    answerer.answer("q", [(DOC01, INSTALL_ERRORS),
                          ("02_mac_installation_login_guide", INSTALL_ERRORS)])
    assert "Mac Installation" not in stub.prompts[0]


@requires_competition_data
def test_build_answerer_dispatches(corpus):
    assert build_answerer(corpus, "extractive").name == "extractive"
    assert build_answerer(corpus, "generative", client=StubClient()).name.startswith("groq:")
    with pytest.raises(ValueError, match="unknown generation mode"):
        build_answerer(corpus, "telepathy")
