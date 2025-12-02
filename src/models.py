"""Data model definitions shared across the DeepSeekMath pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid


def make_proof_id(round_id: int, index: int) -> str:
    return f"r{round_id}_p{index:02d}_{uuid.uuid4().hex[:6]}"


def make_verification_id(proof_id: str, index: int) -> str:
    return f"v_{proof_id}_{index:02d}"


@dataclass(slots=True)
class Problem:
    problem_id: str
    question: str

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "Problem":
        return Problem(problem_id=data["problem_id"], question=data["question"])


@dataclass(slots=True)
class ProofRecord:
    round_id: int
    problem_id: str
    proof_id: str
    parent_proof_id: Optional[str]
    prompt_used: str
    raw_response: str
    parsed_solution: str
    parsed_self_eval: str
    is_final: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "problem_id": self.problem_id,
            "proof_id": self.proof_id,
            "parent_proof_id": self.parent_proof_id,
            "prompt_used": self.prompt_used,
            "raw_response": self.raw_response,
            "parsed_solution": self.parsed_solution,
            "parsed_self_eval": self.parsed_self_eval,
            "is_final": self.is_final,
        }


@dataclass(slots=True)
class VerificationRecord:
    round_id: int
    proof_id: str
    verification_id: str
    prompt_used: str
    raw_response: str
    parsed_score: float
    critique_content: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "proof_id": self.proof_id,
            "verification_id": self.verification_id,
            "prompt_used": self.prompt_used,
            "raw_response": self.raw_response,
            "parsed_score": self.parsed_score,
            "critique_content": self.critique_content,
        }


@dataclass(slots=True)
class CritiqueEntry:
    verification_id: str
    content: str
    score: float


@dataclass(slots=True)
class RefinementInput:
    round_id: int
    target_proof_id: str
    selected_critiques: List[CritiqueEntry]

    def to_json(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "target_proof_id": self.target_proof_id,
            "selected_critiques": [
                {
                    "verification_id": entry.verification_id,
                    "content": entry.content,
                    "score": entry.score,
                }
                for entry in self.selected_critiques
            ],
        }


@dataclass(slots=True)
class ScoreSummary:
    round_id: int
    proof_id: str
    average_score: float
    std_deviation: float
    total_verifications: int
    pass_ratio: float
    scores: List[float] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "proof_id": self.proof_id,
            "average_score": self.average_score,
            "std_deviation": self.std_deviation,
            "total_verifications": self.total_verifications,
            "pass_ratio": self.pass_ratio,
            "scores": self.scores,
        }
