"""Configuration for the ProctorIQ RAG pipeline.

Two facts about the competition's expected submission format are genuinely
unknown to us and will be resolved by leaderboard probe, not by reasoning:

1. whether ``cited_docs`` carries the ``.md`` extension, and
2. what form ``cited_sections`` takes.

Both are represented here as flags with every variant implemented and tested, so
resolving them is a one-line config change rather than a code change.

Deliberately absent: any path to the answer key. The key is a holdout test set;
nothing under ``src/`` may reference it. See ``tests/test_no_key_leakage.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml

# Repository root, derived from this file's location: src/proctoriq_rag/config.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class SectionFormat(str, Enum):
    """How a section header is rendered into the ``cited_sections`` column.

    Deriving from ``(str, Enum)`` rather than ``enum.StrEnum`` — StrEnum is
    Python 3.11+, and this package targets 3.10. Members compare equal to their
    string values either way, which is all we rely on.

    Given the header ``## Section 2: Common Installation Errors``:

    ``FULL_HEADER``  -> ``"Section 2: Common Installation Errors"``
    ``NUMBER_ONLY``  -> ``"Section 2"``
    ``TITLE_ONLY``   -> ``"Common Installation Errors"``
    """

    FULL_HEADER = "full_header"
    NUMBER_ONLY = "number_only"
    TITLE_ONLY = "title_only"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations, resolved absolute against the repo root."""

    kb_dir: Path = REPO_ROOT / "data" / "raw" / "kb"
    test_csv: Path = REPO_ROOT / "data" / "raw" / "test.csv"
    sample_submission: Path = REPO_ROOT / "data" / "raw" / "sample_submission.csv"
    submission_out: Path = REPO_ROOT / "outputs" / "submission.csv"


@dataclass(frozen=True)
class SubmissionConfig:
    """The submission-format decision surface. Applied only in ``submission.writer``."""

    doc_extension: bool = False
    section_format: SectionFormat = SectionFormat.FULL_HEADER
    separator: str = "|"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Local embedding model. Offline, no API key, no network at call time once cached."""

    diagnostic_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class Config:
    paths: PathsConfig = PathsConfig()
    submission: SubmissionConfig = SubmissionConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()


class ConfigError(ValueError):
    """Raised when a config file contains a value we cannot honour."""


def _resolve(value: Any, root: Path) -> Path:
    """Resolve a possibly-relative path string against the repo root."""
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def _parse_section_format(value: Any) -> SectionFormat:
    try:
        return SectionFormat(value)
    except ValueError:
        valid = ", ".join(m.value for m in SectionFormat)
        raise ConfigError(
            f"Unknown section_format {value!r}. Valid values: {valid}."
        ) from None


def load_config(
    path: Path | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    root: Path | None = None,
) -> Config:
    """Load configuration from YAML, then apply ``overrides``.

    ``overrides`` is a nested mapping mirroring the YAML structure, e.g.
    ``{"submission": {"section_format": "number_only"}}``. It takes precedence
    over the file, which takes precedence over the dataclass defaults. This is
    what probe scripts use to sweep format variants without editing the file.
    """
    root = root or REPO_ROOT
    path = DEFAULT_CONFIG_PATH if path is None else Path(path)

    raw: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw = loaded or {}

    for section, values in (overrides or {}).items():
        raw.setdefault(section, {})
        raw[section] = {**(raw[section] or {}), **dict(values)}

    paths_raw = raw.get("paths") or {}
    defaults = PathsConfig()
    paths = PathsConfig(
        kb_dir=_resolve(paths_raw.get("kb_dir", defaults.kb_dir), root),
        test_csv=_resolve(paths_raw.get("test_csv", defaults.test_csv), root),
        sample_submission=_resolve(
            paths_raw.get("sample_submission", defaults.sample_submission), root
        ),
        submission_out=_resolve(
            paths_raw.get("submission_out", defaults.submission_out), root
        ),
    )

    sub_raw = raw.get("submission") or {}
    sub_defaults = SubmissionConfig()
    submission = SubmissionConfig(
        doc_extension=bool(sub_raw.get("doc_extension", sub_defaults.doc_extension)),
        section_format=_parse_section_format(
            sub_raw.get("section_format", sub_defaults.section_format)
        ),
        separator=str(sub_raw.get("separator", sub_defaults.separator)),
    )

    emb_raw = raw.get("embedding") or {}
    embedding = EmbeddingConfig(
        diagnostic_model=str(
            emb_raw.get("diagnostic_model", EmbeddingConfig().diagnostic_model)
        )
    )

    return Config(paths=paths, submission=submission, embedding=embedding)


def with_submission(config: Config, **changes: Any) -> Config:
    """Return a copy of ``config`` with ``submission`` fields replaced.

    Convenience for format probes: ``with_submission(cfg, doc_extension=True)``.
    """
    if "section_format" in changes:
        changes["section_format"] = _parse_section_format(changes["section_format"])
    return replace(config, submission=replace(config.submission, **changes))
