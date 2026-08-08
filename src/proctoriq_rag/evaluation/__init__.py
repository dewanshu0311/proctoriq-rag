"""Local measurement harness.

**This subpackage is a one-way dependency.** It may import from the pipeline
(``corpus``, ``config``, ``submission``); the pipeline may never import from it.

It is the only code permitted to touch ``data/validation/answer_key.yaml``,
which is a hand-built holdout test set. If the key ever influenced pipeline
behaviour it would stop being an independent measurement — and the competition
rules explicitly forbid hardcoded answers. That boundary is enforced
structurally by ``tests/test_no_key_leakage.py``, not by convention.
"""
