"""Score aggregation utilities."""

from __future__ import annotations

from math import sqrt
from typing import Iterable, List

from .models import ScoreSummary


def summarize_scores(round_id: int, proof_id: str, scores: Iterable[float]) -> ScoreSummary:
    values = list(scores)
    if not values:
        avg = 0.0
        std_dev = 0.0
        pass_ratio = 0.0
    else:
        avg = sum(values) / len(values)
        variance = sum((value - avg) ** 2 for value in values) / len(values)
        std_dev = sqrt(variance)
        pass_ratio = sum(1 for value in values if value >= 1.0) / len(values)
    return ScoreSummary(
        round_id=round_id,
        proof_id=proof_id,
        average_score=avg,
        std_deviation=std_dev,
        total_verifications=len(values),
        pass_ratio=pass_ratio,
        scores=values,
    )

