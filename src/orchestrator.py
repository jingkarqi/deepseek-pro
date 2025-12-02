"""Main orchestration loop for the DeepSeekMath heavy compute pipeline."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import RunConfig, load_config_from_env
from .llm_client import DeepSeekTransport, LLMClient, LLMRequest
from .models import (
    CritiqueEntry,
    ProofRecord,
    VerificationRecord,
    make_proof_id,
    make_verification_id,
)
from .parsers import extract_boxed_score, extract_solution_sections
from .persistence import JsonlWriter, atomic_write_json, load_problem
from .prompt_builder import PromptBuilder
from .scoring import summarize_scores

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CandidateState:
    proof: ProofRecord
    critiques: List[CritiqueEntry]
    average_score: float
    pass_ratio: float


class HeavyComputeOrchestrator:
    """Coordinates generation, verification, scoring, and selection loops."""

    def __init__(
        self,
        config: RunConfig,
        generator_client: Optional[LLMClient] = None,
        verifier_client: Optional[LLMClient] = None,
    ) -> None:
        self.config = config
        self.run_dir = self.config.run_output_dir()
        generator_transport = self._maybe_create_deepseek_transport(role="generator")
        verifier_transport = self._maybe_create_deepseek_transport(role="verifier")

        self.generator_client = generator_client or LLMClient(
            max_concurrency=config.generator_concurrency,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
            transport=generator_transport,
        )
        self.verifier_client = verifier_client or LLMClient(
            max_concurrency=config.verifier_concurrency,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
            transport=verifier_transport,
        )

    async def run(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        problem = load_problem(self.config.input_problem_path)
        prompt_builder = PromptBuilder(problem.question)
        self._write_task_metadata(problem.problem_id)

        candidate_pool: List[CandidateState] = []
        population_target = self.config.effective_population()
        verification_target = self.config.effective_verifications()
        separator = "=" * 60

        for round_id in range(self.config.max_rounds + 1):
            logger.info("\n%s", separator)
            logger.info(
                "Starting round %s | population=%s verifications=%s candidate_pool=%s",
                round_id,
                population_target,
                verification_target,
                len(candidate_pool),
            )
            round_dir = self.config.round_dir(round_id)
            round_dir.mkdir(parents=True, exist_ok=True)

            logger.info("  [Generation]")
            gen_progress = StageProgress(
                label=f"    Round {round_id} generation",
                logger=logger,
                min_delta=0.10,
            )
            proofs, prompt_records = await self._generate_round(
                round_id,
                problem.problem_id,
                prompt_builder,
                candidate_pool,
                progress=gen_progress,
            )
            self._write_prompt_log(round_dir, round_id, prompt_records)
            self._write_proofs(round_dir, proofs)
            logger.info(
                "    finished | produced=%s prompts=%s progress %s (%s/%s proofs)",
                len(proofs),
                len(prompt_records),
                gen_progress.render_final(len(proofs)),
                len(proofs),
                population_target,
            )
            logger.info("")

            expected_verifications = len(proofs) * verification_target
            logger.info("  [Verification]")
            ver_progress = StageProgress(
                label=f"    Round {round_id} verification",
                logger=logger,
                min_delta=0.05,
            )
            verifications, verification_prompts = await self._verify_round(
                round_id,
                prompt_builder,
                proofs,
                progress=ver_progress,
            )
            self._write_verification_prompts(round_dir, verification_prompts)
            self._write_verifications(round_dir, verifications)
            logger.info(
                "    finished | proofs=%s verification_records=%s progress %s (%s/%s checks)",
                len(proofs),
                len(verifications),
                ver_progress.render_final(len(verifications)),
                len(verifications),
                expected_verifications,
            )
            logger.info("")

            candidate_pool = self._score_and_select(round_dir, round_id, proofs, verifications)
            self._write_run_state(round_id, candidate_pool)
            if candidate_pool:
                best = candidate_pool[0]
                logger.info(
                    "Round %s scoring finished | best_proof=%s avg=%.3f pass_ratio=%.3f",
                    round_id,
                    best.proof.proof_id,
                    best.average_score,
                    best.pass_ratio,
                )
            else:
                logger.info("Round %s scoring finished | no surviving candidates", round_id)
            logger.info("")

            if not candidate_pool:
                logger.info("No candidates found after round %s; stopping.", round_id)
                break

            solved = next(
                (candidate for candidate in candidate_pool if candidate.pass_ratio >= self.config.stop_threshold),
                None,
            )
            if solved:
                logger.info("Solved in round %s via proof %s", round_id, solved.proof.proof_id)
                self._write_final_solution(solved)
                break

    async def _generate_round(
        self,
        round_id: int,
        problem_id: str,
        prompt_builder: PromptBuilder,
        candidate_pool: Sequence[CandidateState],
        *,
        progress: Optional["StageProgress"] = None,
    ) -> Tuple[List[ProofRecord], List[Dict]]:
        population_target = self.config.effective_population()
        requests: List[LLMRequest] = []
        prompt_records: List[Dict] = []
        contexts = candidate_pool[:population_target] if round_id > 0 else []

        prompt_lookup: Dict[str, str] = {}

        if round_id == 0:
            for index in range(population_target):
                proof_id = make_proof_id(round_id, index)
                prompt = prompt_builder.build_generation_prompt()
                metadata = {"round_id": round_id, "proof_id": proof_id, "parent_proof_id": None, "mode": "generation"}
                requests.append(LLMRequest(prompt=prompt, reference_id=proof_id, role="generator", metadata=metadata))
                prompt_lookup[proof_id] = prompt
                prompt_records.append(
                    {"round_id": round_id, "proof_id": proof_id, "prompt": prompt, "mode": "generation"}
                )
        else:
            for index, candidate in enumerate(contexts):
                proof_id = make_proof_id(round_id, index)
                prompt = prompt_builder.build_refinement_prompt(candidate.proof.parsed_solution, candidate.critiques)
                metadata = {
                    "round_id": round_id,
                    "proof_id": proof_id,
                    "parent_proof_id": candidate.proof.proof_id,
                    "mode": "refinement",
                }
                requests.append(LLMRequest(prompt=prompt, reference_id=proof_id, role="generator", metadata=metadata))
                prompt_lookup[proof_id] = prompt
                prompt_records.append(
                    {
                        "round_id": round_id,
                        "proof_id": proof_id,
                        "parent_proof_id": candidate.proof.proof_id,
                        "prompt": prompt,
                        "mode": "refinement",
                        "critiques": [self._critique_to_json(entry) for entry in candidate.critiques],
                    }
                )

        if not requests:
            return [], prompt_records

        if progress:
            progress.start(len(requests))
        responses = await self.generator_client.run_batch(
            requests,
            progress_callback=progress.tick if progress else None,
        )
        if progress:
            progress.finish()
        proofs: List[ProofRecord] = []
        for response in responses:
            solution, evaluation = extract_solution_sections(response.text)
            parent_id = response.metadata.get("parent_proof_id")
            proofs.append(
                ProofRecord(
                    round_id=response.metadata["round_id"],
                    problem_id=problem_id,
                    proof_id=response.reference_id,
                    parent_proof_id=parent_id,
                    prompt_used=prompt_lookup.get(response.reference_id, ""),
                    raw_response=response.text,
                    parsed_solution=solution,
                    parsed_self_eval=evaluation,
                )
            )
        return proofs, prompt_records

    async def _verify_round(
        self,
        round_id: int,
        prompt_builder: PromptBuilder,
        proofs: Sequence[ProofRecord],
        *,
        progress: Optional["StageProgress"] = None,
    ) -> Tuple[List[VerificationRecord], List[Dict]]:
        verification_target = self.config.effective_verifications()
        requests: List[LLMRequest] = []
        prompt_records: List[Dict] = []

        prompt_lookup: Dict[str, str] = {}

        if not proofs:
            return [], []

        for proof in proofs:
            prompt = prompt_builder.build_verification_prompt(proof.parsed_solution or proof.raw_response)
            for index in range(verification_target):
                verification_id = make_verification_id(proof.proof_id, index)
                metadata = {
                    "round_id": round_id,
                    "proof_id": proof.proof_id,
                    "verification_id": verification_id,
                }
                requests.append(LLMRequest(prompt=prompt, reference_id=verification_id, role="verifier", metadata=metadata))
                prompt_lookup[verification_id] = prompt
                prompt_records.append(
                    {
                        "round_id": round_id,
                        "proof_id": proof.proof_id,
                        "verification_id": verification_id,
                        "prompt": prompt,
                    }
                )

        if progress:
            progress.start(len(requests))
        responses = await self.verifier_client.run_batch(
            requests,
            progress_callback=progress.tick if progress else None,
        )
        if progress:
            progress.finish()
        records: List[VerificationRecord] = []

        for response in responses:
            metadata = response.metadata
            try:
                score = extract_boxed_score(response.text)
            except ValueError as exc:
                logger.warning(
                    "Failed to parse score for verification %s: %s", response.reference_id, exc
                )
                score = 0.0
            records.append(
                VerificationRecord(
                    round_id=metadata["round_id"],
                    proof_id=metadata["proof_id"],
                    verification_id=response.reference_id,
                    prompt_used=prompt_lookup.get(response.reference_id, ""),
                    raw_response=response.text,
                    parsed_score=score,
                    critique_content=response.text,
                )
            )
        return records, prompt_records

    def _score_and_select(
        self,
        round_dir: Path,
        round_id: int,
        proofs: Sequence[ProofRecord],
        verifications: Sequence[VerificationRecord],
    ) -> List[CandidateState]:
        grouped: Dict[str, List[VerificationRecord]] = defaultdict(list)
        for verification in verifications:
            grouped[verification.proof_id].append(verification)

        summaries = []
        for proof in proofs:
            scores = [verification.parsed_score for verification in grouped.get(proof.proof_id, [])]
            summary = summarize_scores(round_id, proof.proof_id, scores)
            summaries.append(summary)

        summary_lookup = {summary.proof_id: summary for summary in summaries}

        summaries_path = round_dir / "05_parsed_scores.jsonl"
        with JsonlWriter(summaries_path) as writer:
            for summary in summaries:
                writer.write(summary.to_json())

        candidates = []
        for proof in proofs:
            summary = summary_lookup.get(proof.proof_id)
            if summary is None:
                continue
            candidates.append(
                CandidateState(
                    proof=proof,
                    critiques=self._select_critiques(grouped.get(proof.proof_id, [])),
                    average_score=summary.average_score,
                    pass_ratio=summary.pass_ratio,
                )
            )
        candidates.sort(key=lambda candidate: candidate.average_score, reverse=True)
        selected = candidates[: self.config.effective_population()]

        selected_path = round_dir / "06_selected_candidates.jsonl"
        with JsonlWriter(selected_path) as writer:
            for candidate in selected:
                writer.write(
                    {
                        "round_id": round_id,
                        "proof_id": candidate.proof.proof_id,
                        "avg_score": candidate.average_score,
                        "pass_ratio": candidate.pass_ratio,
                        "refinement_context": {
                            "original_solution": candidate.proof.parsed_solution,
                            "critiques": [self._critique_to_json(entry) for entry in candidate.critiques],
                        },
                    }
                )

        return selected

    def _select_critiques(self, verifications: Sequence[VerificationRecord]) -> List[CritiqueEntry]:
        critiques = [
            CritiqueEntry(verification_id=verification.verification_id, content=verification.critique_content, score=verification.parsed_score)
            for verification in verifications
        ]
        critiques.sort(key=lambda entry: entry.score)
        return critiques[: self.config.critiques_per_proof]

    def _write_prompt_log(self, round_dir: Path, round_id: int, prompt_records: List[Dict]) -> None:
        file_name = "01_generation_prompts.jsonl" if round_id == 0 else "01_refinement_prompts.jsonl"
        path = round_dir / file_name
        with JsonlWriter(path) as writer:
            for record in prompt_records:
                writer.write(record)

    def _write_proofs(self, round_dir: Path, proofs: Iterable[ProofRecord]) -> None:
        path = round_dir / "02_generated_proofs.jsonl"
        with JsonlWriter(path) as writer:
            for proof in proofs:
                writer.write(proof.to_json())

    def _write_verification_prompts(self, round_dir: Path, records: List[Dict]) -> None:
        path = round_dir / "03_verification_prompts.jsonl"
        with JsonlWriter(path) as writer:
            for record in records:
                writer.write(record)

    def _write_verifications(self, round_dir: Path, verifications: Iterable[VerificationRecord]) -> None:
        path = round_dir / "04_verification_results.jsonl"
        with JsonlWriter(path) as writer:
            for record in verifications:
                writer.write(record.to_json())

    def _write_task_metadata(self, problem_id: str) -> None:
        metadata = {
            "run_name": self.config.resolved_run_name(),
            "problem_id": problem_id,
            "config": self.config.to_dict(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(self.run_dir / "task_metadata.json", metadata)

    def _write_run_state(self, round_id: int, candidates: Sequence[CandidateState]) -> None:
        best = candidates[0] if candidates else None
        data = {
            "last_round": round_id,
            "run_name": self.config.resolved_run_name(),
            "best_proof_id": best.proof.proof_id if best else None,
            "best_score": best.average_score if best else None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(self.run_dir / "run_state.json", data)

    def _write_final_solution(self, candidate: CandidateState) -> None:
        payload = {
            "proof_id": candidate.proof.proof_id,
            "round_id": candidate.proof.round_id,
            "average_score": candidate.average_score,
            "solution": candidate.proof.parsed_solution,
            "self_evaluation": candidate.proof.parsed_self_eval,
            "critiques": [self._critique_to_json(entry) for entry in candidate.critiques],
        }
        atomic_write_json(self.run_dir / "final_solution.json", payload)

    @staticmethod
    def _critique_to_json(entry: CritiqueEntry) -> Dict[str, Any]:
        return {
            "verification_id": entry.verification_id,
            "content": entry.content,
            "score": entry.score,
        }


    def _maybe_create_deepseek_transport(self, role: str):
        api_key = self.config.deepseek_api_key
        if not api_key:
            return None

        if role == "generator":
            model = self.config.deepseek_generator_model or self.config.deepseek_default_model
            temperature = (
                self.config.deepseek_generator_temperature
                if self.config.deepseek_generator_temperature is not None
                else self.config.deepseek_default_temperature
            )
        else:
            model = self.config.deepseek_verifier_model or self.config.deepseek_default_model
            temperature = (
                self.config.deepseek_verifier_temperature
                if self.config.deepseek_verifier_temperature is not None
                else self.config.deepseek_default_temperature
            )

        return DeepSeekTransport(
            api_key=api_key,
            base_url=self.config.deepseek_base_url,
            model=model,
            system_prompt=self.config.deepseek_system_prompt,
            temperature=temperature,
            stream=self.config.deepseek_stream,
        )


class StageProgress:
    """Logger-based progress indicator with rate limiting."""

    def __init__(self, label: str, logger: logging.Logger, width: int = 30, min_delta: float = 0.05) -> None:
        self.label = label
        self.logger = logger
        self.width = width
        self.min_delta = min_delta
        self.total = 1
        self.current = 0
        self._active = False
        self._last_ratio = -1.0

    def start(self, total: int) -> None:
        self.total = max(total, 1)
        self.current = 0
        self._active = True
        self._last_ratio = -1.0
        self._log(force=True)

    def tick(self, current: int, total: int) -> None:
        if not self._active:
            self.start(total)
            return
        self.total = max(total, 1)
        self.current = current
        ratio = min(max(self.current / self.total, 0.0), 1.0)
        if ratio - self._last_ratio >= self.min_delta or ratio >= 1.0:
            self._log()

    def finish(self) -> None:
        if not self._active:
            return
        self.current = self.total
        if self._last_ratio < 1.0:
            self._log(force=True)
        self._active = False

    def render_final(self, produced: int) -> str:
        ratio = min(max(produced / max(self.total, 1), 0.0), 1.0)
        filled = int(round(ratio * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        return f"[{bar}] {ratio * 100:5.1f}%"

    def _log(self, force: bool = False) -> None:
        ratio = min(max(self.current / max(self.total, 1), 0.0), 1.0)
        if not force and ratio - self._last_ratio < self.min_delta:
            return
        self._last_ratio = ratio
        filled = int(round(ratio * self.width))
        filled = min(filled, self.width)
        bar = "#" * filled + "-" * (self.width - filled)
        self.logger.info("%s [%s] %5.1f%% (%s/%s)", self.label, bar, ratio * 100, self.current, self.total)

async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config_from_env()
    orchestrator = HeavyComputeOrchestrator(config=config)
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
