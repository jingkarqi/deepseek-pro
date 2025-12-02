"""Runtime configuration helpers for the DeepSeekMath high-compute pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
import uuid
from typing import Any, Dict, Optional


def _default_run_name() -> str:
    """Return a human-readable timestamp plus short suffix for run folders."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"run_{timestamp}_{suffix}"


@dataclass(slots=True)
class RunConfig:
    """
    全局运行参数，所有模块都从这里读取，便于统一调度。

    主要字段与可用的环境变量说明：
    - population_size (`POPULATION_SIZE`): 每轮保留的候选证明数量 N，默认 64。
    - verification_samples (`VERIFICATION_SAMPLES`): 单个证明的验证次数 M，默认 64。
    - critiques_per_proof (`CRITIQUES_PER_PROOF`): 进入下一轮时挑选的批评条数 K，默认 8。
    - max_rounds (`MAX_ROUNDS`): 最大迭代轮数 T，默认 16。
    - stop_threshold (`STOP_THRESHOLD`): 满足该置信度后提前终止，默认 1.0（即 64/64 全通过）。
    - input_problem_path (`PROBLEM_PATH`): 输入题目 JSON 路径。
    - output_root (`OUTPUT_ROOT`): 输出根目录，内部继续创建 run_xxx 结构。
    - log_dir (`LOG_DIR`): 日志目录。
    - run_name (`RUN_NAME`): 运行 ID，不提供时自动生成 run_时间戳_随机后缀。
    - dry_run_population / dry_run_verifications (`DRY_POPULATION` / `DRY_VERIFICATIONS`): 缩小 N、M 以便本地或 CI 快速测试。
    - generator_concurrency / verifier_concurrency (`GENERATOR_CONCURRENCY` / `VERIFIER_CONCURRENCY`): LLM 并发上限。
    - request_timeout (`REQUEST_TIMEOUT`): 单次推理的超时秒数。
    - max_retries (`MAX_RETRIES`): 请求失败后的最大重试次数。
    - deepseek_api_key (`DEEPSEEK_API_KEY`): 访问 DeepSeek API 所需的密钥（可留空以使用本地 DummyTransport）。
    - deepseek_base_url (`DEEPSEEK_BASE_URL`): DeepSeek API base URL，默认 https://api.deepseek.com。
    - deepseek_system_prompt (`DEEPSEEK_SYSTEM_PROMPT`): 自定义系统提示词，默认空串交由项目 Prompt 控制。
    - deepseek_stream (`DEEPSEEK_STREAM`): 是否启用流式输出，默认 False。
    - deepseek_default_model (`DEEPSEEK_MODEL`): Generator/Verifier 通用的模型名称，默认 deepseek-reasoner，可被角色专属模型覆盖。
    - deepseek_generator_model (`DEEPSEEK_GENERATOR_MODEL`): 仅生成端使用的模型（可选）。
    - deepseek_verifier_model (`DEEPSEEK_VERIFIER_MODEL`): 仅验证端使用的模型（可选）。
    - deepseek_default_temperature (`DEEPSEEK_TEMPERATURE`): 通用采样温度，默认 0.2。
    - deepseek_generator_temperature (`DEEPSEEK_GENERATOR_TEMP`): 生成端专属温度（可选）。
    - deepseek_verifier_temperature (`DEEPSEEK_VERIFIER_TEMP`): 验证端专属温度（可选）。
    - metadata: 预留字段，写入 task_metadata.json 便于审计。
    """

    population_size: int = 64
    verification_samples: int = 64
    critiques_per_proof: int = 8
    max_rounds: int = 16
    stop_threshold: float = 1.0
    input_problem_path: Path = Path("input/problem1.json")
    output_root: Path = Path("output")
    log_dir: Path = Path("log")
    run_name: Optional[str] = None
    dry_run_population: Optional[int] = None
    dry_run_verifications: Optional[int] = None
    generator_concurrency: int = 32
    verifier_concurrency: int = 128
    request_timeout: int = 4500  # seconds
    max_retries: int = 8
    metadata: Dict[str, Any] = field(default_factory=dict)
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_system_prompt: str = ""
    deepseek_stream: bool = False
    deepseek_default_model: str = "deepseek-reasoner"
    deepseek_generator_model: Optional[str] = None
    deepseek_verifier_model: Optional[str] = None
    deepseek_default_temperature: float = 0.2
    deepseek_generator_temperature: Optional[float] = None
    deepseek_verifier_temperature: Optional[float] = None

    def resolved_run_name(self) -> str:
        if not self.run_name:
            self.run_name = _default_run_name()
        return self.run_name

    def run_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def round_dir(self, round_id: int) -> Path:
        return self.run_output_dir() / f"round_{round_id}"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["input_problem_path"] = str(self.input_problem_path)
        data["output_root"] = str(self.output_root)
        data["log_dir"] = str(self.log_dir)
        # 避免将敏感信息直接写入元数据文件
        data.pop("deepseek_api_key", None)
        return data

    def effective_population(self) -> int:
        return self.dry_run_population or self.population_size

    def effective_verifications(self) -> int:
        return self.dry_run_verifications or self.verification_samples


def load_config_from_env() -> RunConfig:
    """
    读取环境变量覆盖默认参数，便于通过 `export KEY=value` 的方式在不同算力环境运行。
    仅在变量存在时才覆盖 RunConfig 字段，未设置的键则继续沿用默认值。
    """

    def _maybe_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        return int(raw) if raw is not None else default

    def _maybe_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        return float(raw) if raw is not None else default

    def _maybe_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    cfg = RunConfig(
        population_size=_maybe_int("POPULATION_SIZE", 64),
        verification_samples=_maybe_int("VERIFICATION_SAMPLES", 64),
        critiques_per_proof=_maybe_int("CRITIQUES_PER_PROOF", 8),
        max_rounds=_maybe_int("MAX_ROUNDS", 16),
        generator_concurrency=_maybe_int("GENERATOR_CONCURRENCY", 32),
        verifier_concurrency=_maybe_int("VERIFIER_CONCURRENCY", 128),
        request_timeout=_maybe_int("REQUEST_TIMEOUT", 120),
        max_retries=_maybe_int("MAX_RETRIES", 8),
    )

    dry_pop = os.getenv("DRY_POPULATION")
    dry_ver = os.getenv("DRY_VERIFICATIONS")
    run_name = os.getenv("RUN_NAME")

    if dry_pop:
        cfg.dry_run_population = int(dry_pop)
    if dry_ver:
        cfg.dry_run_verifications = int(dry_ver)
    if run_name:
        cfg.run_name = run_name

    problem_path = os.getenv("PROBLEM_PATH")
    output_root = os.getenv("OUTPUT_ROOT")
    log_dir = os.getenv("LOG_DIR")

    if problem_path:
        cfg.input_problem_path = Path(problem_path)
    if output_root:
        cfg.output_root = Path(output_root)
    if log_dir:
        cfg.log_dir = Path(log_dir)

    # DeepSeek 请求参数
    cfg.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    cfg.deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", cfg.deepseek_base_url)
    cfg.deepseek_system_prompt = os.getenv("DEEPSEEK_SYSTEM_PROMPT", cfg.deepseek_system_prompt)
    cfg.deepseek_stream = _maybe_bool("DEEPSEEK_STREAM", cfg.deepseek_stream)
    cfg.deepseek_default_model = os.getenv("DEEPSEEK_MODEL", cfg.deepseek_default_model)
    cfg.deepseek_default_temperature = _maybe_float("DEEPSEEK_TEMPERATURE", cfg.deepseek_default_temperature)

    generator_model = os.getenv("DEEPSEEK_GENERATOR_MODEL")
    verifier_model = os.getenv("DEEPSEEK_VERIFIER_MODEL")
    generator_temp = os.getenv("DEEPSEEK_GENERATOR_TEMP")
    verifier_temp = os.getenv("DEEPSEEK_VERIFIER_TEMP")

    if generator_model:
        cfg.deepseek_generator_model = generator_model
    if verifier_model:
        cfg.deepseek_verifier_model = verifier_model
    if generator_temp:
        cfg.deepseek_generator_temperature = float(generator_temp)
    if verifier_temp:
        cfg.deepseek_verifier_temperature = float(verifier_temp)

    return cfg
