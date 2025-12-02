"""Output parsing helpers for solutions, self-evaluations, and verifier scores."""

from __future__ import annotations

import re
from typing import Tuple

SOLUTION_RE = re.compile(r"## Solution(?P<body>.*?)(## Self Evaluation|$)", re.IGNORECASE | re.DOTALL)
SELF_EVAL_RE = re.compile(r"## Self Evaluation(?P<body>.*)$", re.IGNORECASE | re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{\s*(?P<score>0(?:\.5)?|1(?:\.0)?)\s*\}")


def extract_solution_sections(text: str) -> Tuple[str, str]:
    solution = ""
    evaluation = ""

    solution_match = SOLUTION_RE.search(text)
    if solution_match:
        solution = solution_match.group("body").strip()

    evaluation_match = SELF_EVAL_RE.search(text)
    if evaluation_match:
        evaluation = evaluation_match.group("body").strip()

    return solution, evaluation


def extract_boxed_score(text: str) -> float:
    match = BOXED_RE.search(text)
    if not match:
        raise ValueError("Unable to locate boxed score in verifier output.")
    raw = match.group("score")
    if raw in {"1", "1.0"}:
        return 1.0
    if raw in {"0.5", "0.50"}:
        return 0.5
    return 0.0
