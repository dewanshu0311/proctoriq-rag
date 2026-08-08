"""Primitive scoring functions. Pure, dependency-light, hand-checkable.

Everything here is deliberately simple enough that a test can assert an exact
expected number computed on paper. The scorer composes these; it does not
reimplement them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence, TypeVar

import numpy as np

H = TypeVar("H", bound=Hashable)


@dataclass(frozen=True)
class SetScore:
    """Precision / recall / F1 / exact-set-match for one prediction.

    We report all four because we do not know which the competition grader uses
    for "Retrieval quality" and "Citation accuracy". F1 is the most likely and is
    what the composite uses, but a config that wins on F1 and loses on
    exact-match is worth seeing before we commit to it.
    """

    precision: float
    recall: float
    f1: float
    exact_match: bool

    @property
    def as_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "exact_match": float(self.exact_match),
        }


PERFECT = SetScore(precision=1.0, recall=1.0, f1=1.0, exact_match=True)
ZERO = SetScore(precision=0.0, recall=0.0, f1=0.0, exact_match=False)


def set_scores(predicted: Iterable[H], expected: Iterable[H]) -> SetScore:
    """Set-based precision/recall/F1 between a prediction and the ground truth.

    Duplicates are collapsed — this is set semantics, deliberately. Where
    multiplicity matters (Q44 cites one document twice), the caller scores
    ``(doc, section)`` pairs instead, which are distinct.

    Empty-set convention, chosen so that "predicted nothing, expected nothing"
    is not punished and "predicted nothing, expected something" is:

    - both empty         -> all 1.0, ``exact_match=True``
    - exactly one empty  -> all 0.0, ``exact_match=False``
    """
    pred_set = set(predicted)
    true_set = set(expected)

    if not pred_set and not true_set:
        return PERFECT
    if not pred_set or not true_set:
        return ZERO

    overlap = len(pred_set & true_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(true_set)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return SetScore(
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match=pred_set == true_set,
    )


def multiset_scores(predicted: Sequence[H], expected: Sequence[H]) -> SetScore:
    """Multiset variant of :func:`set_scores`, honouring repeated elements.

    Reported alongside the set variant for the document dimension only, because
    Q44 cites the same document twice and we do not know whether the grader
    deduplicates. See ``docs/DECISIONS.md``.
    """
    from collections import Counter

    pred_counts = Counter(predicted)
    true_counts = Counter(expected)

    if not pred_counts and not true_counts:
        return PERFECT
    if not pred_counts or not true_counts:
        return ZERO

    overlap = sum((pred_counts & true_counts).values())
    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(true_counts.values())
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return SetScore(
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match=pred_counts == true_counts,
    )


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, clamped to [-1, 1].

    Returns 0.0 if either vector is all zeros — an undefined direction scores as
    "unrelated" rather than raising, because the caller is scoring a batch.
    """
    a = np.asarray(a, dtype="float64").ravel()
    b = np.asarray(b, dtype="float64").ravel()

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


def mean_or_none(values: Iterable[float | None]) -> float | None:
    """Mean of the non-``None`` values, or ``None`` if there are none.

    Used so an unmeasurable dimension propagates as ``None`` all the way to the
    report instead of silently becoming 0.0 and dragging the composite down.
    """
    present = [v for v in values if v is not None]
    if not present:
        return None
    return float(sum(present) / len(present))
