#!/usr/bin/env python
"""Validate the answer key against the corpus and print an inventory.

    python scripts/validate_key.py

Exits 0 if the key is clean, 1 otherwise. This script is one of only three
places permitted to touch the answer key (here, ``diagnose_sections.py``, and
``tests/``); nothing under ``src/`` may.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from proctoriq_rag.config import load_config  # noqa: E402
from proctoriq_rag.corpus.loader import load_corpus  # noqa: E402
from proctoriq_rag.evaluation.answer_key import load_answer_key  # noqa: E402

DEFAULT_KEY_PATH = REPO_ROOT / "data" / "validation" / "answer_key.yaml"
EXPECTED_SECTION_COUNT = 53


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--kb-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config()
    kb_dir = args.kb_dir or config.paths.kb_dir

    rule = "=" * 78
    print(rule)
    print("CORPUS INVENTORY")
    print(rule)

    corpus = load_corpus(kb_dir)
    total_sections = len(corpus.sections())
    print(f"  knowledge base : {kb_dir}")
    print(f"  documents      : {len(corpus)}")
    print(f"  ## sections    : {total_sections}")
    for doc_id in corpus.doc_ids():
        titles = corpus.section_titles(doc_id)
        print(f"    {doc_id:<48} {len(titles):>2} sections")

    if total_sections != EXPECTED_SECTION_COUNT:
        print(
            f"\n  !! expected {EXPECTED_SECTION_COUNT} sections, found "
            f"{total_sections} — the parser or the corpus changed."
        )

    print()
    print(rule)
    print("ANSWER KEY")
    print(rule)
    print(f"  key file : {args.key}")

    key = load_answer_key(args.key, corpus, strict=False)
    problems = getattr(key, "problems", [])

    print(f"  entries  : {len(key)}")
    print(f"  version  : {key.version}")
    print(f"  kinds       : {key.counts('kind')}")
    print(f"  confidence  : {key.counts('confidence')}")

    multi = [e.id for e in key if e.is_multi_source]
    print(f"  multi-source: {len(multi)} -> {', '.join(multi)}")
    low = [e.id for e in key.by_confidence("low")]
    medium = [e.id for e in key.by_confidence("medium")]
    print(f"  low confidence   : {', '.join(low) or 'none'}")
    print(f"  medium confidence: {', '.join(medium) or 'none'}")

    print()
    print(rule)
    if problems:
        print(f"VALIDATION FAILED — {len(problems)} problem(s)")
        print(rule)
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("VALIDATION PASSED — key agrees with the corpus on every entry.")
    print(rule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
