"""Async LLM client abstraction with pluggable transports."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
import random
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

try:  # Optional dependency for real DeepSeek calls
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMRequest:
    prompt: str
    reference_id: str
    role: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMResponse:
    reference_id: str
    role: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


TransportCallable = Callable[[LLMRequest], Awaitable[LLMResponse]]


class LLMClient:
    """Concurrency-safe batch executor for generator/verifier style prompts."""

    def __init__(
        self,
        *,
        max_concurrency: int = 32,
        timeout: int = 120,
        max_retries: int = 6,
        transport: Optional[TransportCallable] = None,
    ) -> None:
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.max_retries = max_retries
        self._transport = transport or DummyTransport().__call__

    async def run_batch(
        self,
        requests: Sequence[LLMRequest],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[LLMResponse]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        total = len(requests)
        completed = 0
        progress_lock = asyncio.Lock()

        async def _run_with_progress(req: LLMRequest) -> LLMResponse:
            nonlocal completed
            result = await self._run_single(req, semaphore)
            if progress_callback:
                async with progress_lock:
                    completed += 1
                    progress_callback(completed, total)
            return result

        tasks = [asyncio.create_task(_run_with_progress(req)) for req in requests]
        return await asyncio.gather(*tasks)

    async def _run_single(self, request: LLMRequest, semaphore: asyncio.Semaphore) -> LLMResponse:
        async with semaphore:
            for attempt in range(1, self.max_retries + 1):
                try:
                    result = await asyncio.wait_for(self._transport(request), timeout=self.timeout)
                    result.metadata = {**request.metadata, **result.metadata}
                    return result
                except Exception as exc:  # pragma: no cover - best-effort logging
                    log_func = logger.warning if attempt == self.max_retries else logger.info
                    log_func("LLM request failed (attempt %s/%s): %s", attempt, self.max_retries, exc)
                    if attempt == self.max_retries:
                        raise
                    await asyncio.sleep(min(2 ** attempt, 30))


class DummyTransport:
    """Simple transport used for local dry-run testing without hitting a real API."""

    def __init__(
        self,
        *,
        max_concurrency: int = 8,
        min_delay: float = 0.05,
        max_delay: float = 0.3,
        failure_rate: float = 0.002,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.failure_rate = failure_rate

    async def __call__(self, request: LLMRequest) -> LLMResponse:
        async with self._semaphore:
            await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))
            if random.random() < self.failure_rate:
                raise RuntimeError("Simulated dummy transport failure")

            if request.role == "verifier":
                score_text = random.choice(["0", "0.5", "1", "1.0"])
                synthesized = (
                    "Here is my evaluation of the solution:\n"
                    "This is a placeholder critique for dry runs.\n\n"
                    "Based on my evaluation, the final overal score should be:\n"
                    f"\\boxed{{{score_text}}}"
                )
            else:
                synthesized = (
                    "## Solution\n"
                    "This is a placeholder solution emitted by the dummy transport.\n\n"
                    "## Self Evaluation\n"
                    "Here is my evaluation of the solution: The placeholder is trivial.\n"
                    "\\boxed{0.5}"
                )
            return LLMResponse(reference_id=request.reference_id, role=request.role, text=synthesized)


class DeepSeekTransport:
    """Transport that forwards requests to the official DeepSeek Chat Completions API."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.2,
        stream: bool = False,
    ) -> None:
        if OpenAI is None:  # pragma: no cover - import guard
            raise RuntimeError("The openai package is required to use DeepSeekTransport.")
        if api_key is None:
            raise ValueError("api_key must be provided to use DeepSeekTransport.")
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.stream = stream
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    async def __call__(self, request: LLMRequest) -> LLMResponse:
        def _invoke() -> LLMResponse:
            messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": request.prompt}]
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=self.stream,
                temperature=self.temperature,
            )
            text = completion.choices[0].message.content or ""
            metadata = {"model": self.model}
            usage = getattr(completion, "usage", None)
            if usage:
                metadata["usage"] = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            return LLMResponse(reference_id=request.reference_id, role=request.role, text=text, metadata=metadata)

        return await asyncio.to_thread(_invoke)
