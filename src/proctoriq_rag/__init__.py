"""ProctorIQ RAG — retrieval pipeline and local measurement harness.

Package layout
--------------
``config``      configuration, including the two unresolved submission-format flags
``corpus``      markdown knowledge-base loader
``submission``  submission.csv serialization (the ONLY place format flags apply)
``evaluation``  answer-key loader and local scorer — measurement only, never a pipeline input
``retrieval`` / ``routing`` / ``generation``
                empty in Phase 0; populated in Phase 1+

The ``evaluation`` subpackage is load-bearing in one direction only: it may read
the pipeline, and the pipeline may never read it. Enforced by
``tests/test_no_key_leakage.py``.
"""

import os as _os

# ── Keep HuggingFace on the PyTorch backend ────────────────────────────────
# `transformers` probes for TensorFlow and Flax at import time and raises if it
# finds Keras 3 without the `tf-keras` shim. Nothing here uses either backend —
# the cross-encoder and the sentence-transformers embedder are both PyTorch — but
# the probe fires anyway the moment anything pulls `transformers` in, including
# `langchain_text_splitters` inside the recursive chunker.
#
# That makes an unrelated TensorFlow install anywhere in the environment enough to
# break the test suite, which is exactly what happened once here: four chunking
# tests started failing with no change to this repository. Kaggle images ship
# TensorFlow as standard, so the notebook was one dependency resolution away from
# the same failure.
#
# Set before any submodule import, because `transformers` reads these once and
# caches the result.
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_FLAX", "0")

__version__ = "0.1.0"
