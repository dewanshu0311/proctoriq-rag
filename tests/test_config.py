"""Config tests, with particular attention to the two unresolved format flags."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from proctoriq_rag.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigError,
    SectionFormat,
    load_config,
    with_submission,
)


def test_defaults_when_no_file_exists(tmp_path):
    config = load_config(tmp_path / "absent.yaml", root=tmp_path)
    assert config.submission.doc_extension is False
    assert config.submission.section_format is SectionFormat.FULL_HEADER
    assert config.submission.separator == "|"


def test_shipped_default_config_loads():
    assert DEFAULT_CONFIG_PATH.exists()
    config = load_config()
    assert isinstance(config, Config)
    assert config.paths.kb_dir.is_absolute()


def test_yaml_values_are_honoured(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        textwrap.dedent(
            """
            submission:
              doc_extension: true
              section_format: title_only
            """
        ),
        encoding="utf-8",
    )
    config = load_config(path, root=tmp_path)
    assert config.submission.doc_extension is True
    assert config.submission.section_format is SectionFormat.TITLE_ONLY


def test_overrides_beat_the_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("submission:\n  section_format: title_only\n", encoding="utf-8")
    config = load_config(
        path,
        overrides={"submission": {"section_format": "number_only"}},
        root=tmp_path,
    )
    assert config.submission.section_format is SectionFormat.NUMBER_ONLY


def test_invalid_section_format_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("submission:\n  section_format: sideways\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown section_format"):
        load_config(path, root=tmp_path)


def test_relative_paths_resolve_against_root(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("paths:\n  kb_dir: some/kb\n", encoding="utf-8")
    config = load_config(path, root=tmp_path)
    assert config.paths.kb_dir == tmp_path / "some" / "kb"


def test_absolute_paths_are_left_alone(tmp_path):
    absolute = (tmp_path / "elsewhere").resolve()
    path = tmp_path / "c.yaml"
    path.write_text(f"paths:\n  kb_dir: {absolute.as_posix()}\n", encoding="utf-8")
    assert load_config(path, root=tmp_path).paths.kb_dir == Path(absolute)


def test_with_submission_returns_a_copy(tmp_path):
    config = load_config(tmp_path / "absent.yaml", root=tmp_path)
    probed = with_submission(config, doc_extension=True, section_format="number_only")
    assert probed.submission.doc_extension is True
    assert probed.submission.section_format is SectionFormat.NUMBER_ONLY
    assert config.submission.doc_extension is False  # original untouched


def test_with_submission_rejects_a_bad_format(tmp_path):
    config = load_config(tmp_path / "absent.yaml", root=tmp_path)
    with pytest.raises(ConfigError):
        with_submission(config, section_format="sideways")


def test_section_format_members_compare_as_strings():
    """We target 3.10, so this is (str, Enum), not enum.StrEnum."""
    assert SectionFormat.FULL_HEADER == "full_header"
    assert SectionFormat("number_only") is SectionFormat.NUMBER_ONLY
    assert {m.value for m in SectionFormat} == {
        "full_header",
        "number_only",
        "title_only",
    }


def test_config_does_not_expose_an_answer_key_path(tmp_path):
    """src/ must not know the key exists — including through config."""
    config = load_config(tmp_path / "absent.yaml", root=tmp_path)
    assert not any("key" in field.lower() for field in vars(config.paths))
