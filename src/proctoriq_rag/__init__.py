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

__version__ = "0.1.0"
