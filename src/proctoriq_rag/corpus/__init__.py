"""Knowledge-base loading."""

from proctoriq_rag.corpus.loader import (
    Corpus,
    Document,
    Section,
    load_corpus,
    split_sections,
)

__all__ = ["Corpus", "Document", "Section", "load_corpus", "split_sections"]
