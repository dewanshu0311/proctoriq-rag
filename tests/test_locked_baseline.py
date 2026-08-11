"""The locked 79.27 configuration must stay one commit away, always.

Phase 4 introduces an LLM classifier, HyDE, fusion and generation — each a live
dependency with failure modes the current pipeline does not have: rate limits,
API outages, nondeterminism across Kaggle runs. The extractive configuration that
scored **79.27** has none of those, and two submissions get selected at the end.
One of them should be the low-variance option.

So this file pins that configuration. If any of these fail, the low-variance
fallback has drifted and the tag no longer reproduces what it claims to.

Tagged in git as ``v1.0-locked-79.27``.
"""

from __future__ import annotations

import pytest

from proctoriq_rag.config import SectionFormat, load_config
from proctoriq_rag.submission.writer import format_doc, format_section
from tests.conftest import requires_competition_data

#: Measured locally under the locked config. Public leaderboard: 79.27.
#: Local and public differ because the public split is ~20 of 50 questions.
LOCKED_LOCAL_METRICS = {
    "cite_exact": 0.5600,
    "cite_f1": 0.6667,
    "doc_f1": 0.8467,
    "doc_exact": 0.7000,
}

#: Derived from the probe deltas — see docs/SUBMISSION_LOG.md.
PUBLIC_DECOMPOSITION = {
    "retrieval_f1": 0.783,
    "citation_f1": 0.750,
    "answer_half": 52.36,
    "total": 79.27,
}


# ── the resolved format decisions ──────────────────────────────────────────
def test_doc_extension_stays_off():
    """D-004, probe 2: adding .md cost 15.66 points."""
    assert load_config().submission.doc_extension is False


def test_section_format_is_number_only():
    """D-005, probe 3: switching to number_only gained 11.25 points."""
    assert load_config().submission.section_format is SectionFormat.NUMBER_ONLY


@requires_competition_data
def test_locked_format_produces_only_section_n_strings(corpus):
    """Every emitted section string must be `Section N` or a bare header."""
    import re

    for section in corpus.sections():
        rendered = format_section(section.section_title, SectionFormat.NUMBER_ONLY)
        assert re.fullmatch(r"Section \d+|[A-Z].*", rendered)
        assert ":" not in rendered or not rendered.startswith("Section ")


@requires_competition_data
def test_locked_format_emits_no_extension(corpus):
    for doc_id in corpus.doc_ids():
        assert not format_doc(doc_id, extension=False).endswith(".md")


# ── the public decomposition arithmetic ────────────────────────────────────
def test_public_decomposition_reconstructs_the_observed_score():
    """Probe 2 zeroes both citation dimensions, so its 52.36 IS the answer half."""
    d = PUBLIC_DECOMPOSITION
    total = d["answer_half"] + 20 * d["retrieval_f1"] + 15 * d["citation_f1"]
    assert total == pytest.approx(d["total"], abs=0.01)


def test_headroom_is_where_we_think_it_is():
    """Citation half has ~8 points left, answer half ~13. Not 3 and 18."""
    d = PUBLIC_DECOMPOSITION
    citation_half = 20 * d["retrieval_f1"] + 15 * d["citation_f1"]
    assert citation_half == pytest.approx(26.91, abs=0.01)
    assert 35 - citation_half == pytest.approx(8.09, abs=0.01)
    assert 65 - d["answer_half"] == pytest.approx(12.64, abs=0.01)


# ── the pipeline itself still reproduces the locked output ─────────────────
@pytest.mark.slow
@requires_competition_data
def test_locked_pipeline_reproduces_its_metrics(corpus, answer_key):
    """The low-variance fallback must still score what the tag claims.

    Slow-marked: loads the cross-encoder. Run with ``pytest -m slow``.
    """
    import numpy as np
    import pandas as pd

    from proctoriq_rag.evaluation.scorer import Scorer
    from proctoriq_rag.pipeline import Pipeline, PipelineConfig

    config = load_config()
    questions = pd.read_csv(config.paths.test_csv)
    predictions = Pipeline(corpus=corpus, config=PipelineConfig()).run(
        questions["question_id"].astype(str).tolist(),
        questions["question"].astype(str).tolist(),
    )

    assert len(predictions) == 50
    assert all(p.answer_text.strip() for p in predictions)

    report = Scorer(corpus, answer_key).score(predictions)
    observed = {
        "cite_exact": float(np.mean([q.citation.exact_match for q in report.per_question])),
        "cite_f1": report.dimension_means["citation"],
        "doc_f1": report.dimension_means["retrieval"],
        "doc_exact": float(np.mean([q.retrieval.exact_match for q in report.per_question])),
    }
    for name, expected in LOCKED_LOCAL_METRICS.items():
        assert observed[name] == pytest.approx(expected, abs=1e-3), (
            f"{name}: locked baseline drifted, {observed[name]:.4f} != {expected:.4f}. "
            "The low-variance fallback no longer reproduces v1.0-locked-79.27."
        )
