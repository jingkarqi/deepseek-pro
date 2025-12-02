"""Prompt assembly helpers leveraging the shared math_templates module."""

from __future__ import annotations

from typing import Sequence

from .math_templates import math_templates
from .models import CritiqueEntry


GEN_TEMPLATE = math_templates["proof_generation"]
VERIFY_TEMPLATE = math_templates["proof_verification"]
REFINE_TEMPLATE = math_templates["proof_refinement"]


class PromptBuilder:
    """Centralizes prompt assembly so multiple components share identical text."""

    def __init__(self, problem_statement: str):
        self.problem_statement = problem_statement.strip()

    def build_generation_prompt(self) -> str:
        return GEN_TEMPLATE.format(question=self.problem_statement)

    def build_verification_prompt(self, proof_text: str) -> str:
        return VERIFY_TEMPLATE.format(statement=self.problem_statement, proof=proof_text.strip())

    def build_refinement_prompt(self, original_solution: str, critiques: Sequence[CritiqueEntry]) -> str:
        context_block = self._format_refinement_block(original_solution, critiques)
        generation_instruction = GEN_TEMPLATE.format(question=self.problem_statement)
        return REFINE_TEMPLATE.format(instruction=generation_instruction, proofs_to_refine=context_block)

    @staticmethod
    def _format_refinement_block(original_solution: str, critiques: Sequence[CritiqueEntry]) -> str:
        critique_lines = []
        for idx, critique in enumerate(critiques, start=1):
            critique_lines.append(f"{idx}. [{critique.score}] {critique.content.strip()}")
        critiques_block = "\n".join(critique_lines) if critique_lines else "1. No critiques collected."
        return (
            "### Solution Sample\n"
            f"{original_solution.strip()}\n\n"
            "### Integrity Evaluation\n"
            f"{critiques_block}"
        )

