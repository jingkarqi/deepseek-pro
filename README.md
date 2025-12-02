# DeepSeekMath-V2 Heavy Compute Pipeline

本项目复现了 DeepSeekMath-V2 论文中“High-Compute Search”流程：每一轮并发生成 64 份证明、对每份证明并发做 64 次验证、基于评分筛选后再迭代修正。为便于上手，这里提供一份完整的运行指南。

---

## 1. 目录结构

```
├── input/
│   └── problem1.json           # 题目输入 (problem_id + question)
├── output/
│   └── run_*/...               # 每轮生成/验证/评分的 JSONL 输出
├── src/
│   ├── orchestrator.py         # 主控逻辑
│   ├── config.py               # 全局参数定义
│   ├── llm_client.py           # LLM 并发客户端 & DummyTransport
│   ├── prompt_builder.py       # Prompt 组装
│   ├── parsers.py / scoring.py # 解析与算分工具
│   └── math_templates.py       # 题目/验证模板
└── plan.md / design.md ...     # 设计文档
```

---

## 2. 题目输入

请在 `input/problem1.json` 中提供题目，格式例如：

```json
{
  "problem_id": "hypergeo_001",
  "question": "化简: ${}_2F_1\\!\\left(-\\frac34,\\;\\frac34;\\;\\frac14;\\;\\frac{\\sqrt3}{2}\\right)$"
}
```

运行时 orchestrator 会自动读取该题目构建 Prompt。

---

## 3. 环境变量与全局参数

所有运行参数集中在 `RunConfig` 中，可通过环境变量覆盖。常用选项如下：

| 变量名 | 默认值 | 说明 |
|---|---|---|
| `POPULATION_SIZE` | 64 | 每轮候选证明数量 N |
| `VERIFICATION_SAMPLES` | 64 | 每个 proof 的验证次数 M |
| `CRITIQUES_PER_PROOF` | 8 | 每轮挑选的 critique 条数 K |
| `MAX_ROUNDS` | 16 | 最大迭代轮数 |
| `STOP_THRESHOLD` | 1.0 | 满足平均得分=1 时提前终止 |
| `DRY_POPULATION` / `DRY_VERIFICATIONS` | - | 缩小 N/M，便于 CI 快速演练 |
| `PROBLEM_PATH` | `input/problem1.json` | 题目路径 |
| `OUTPUT_ROOT` | `output` | 输出根目录 |
| `RUN_NAME` | 自动生成 | 自定义 run ID |
| `GENERATOR_CONCURRENCY` / `VERIFIER_CONCURRENCY` | 32 / 128 | LLM 并发上限 |
| `REQUEST_TIMEOUT` | 4500s | 单次请求超时 |
| `MAX_RETRIES` | 8 | LLM 请求最大重试次数 |

### DeepSeek API 相关

若要调用真实的 DeepSeek API，需要设置以下变量：

| 变量名 | 作用 |
|---|---|
| `DEEPSEEK_API_KEY` *(必填)* | 官方 API Key |
| `DEEPSEEK_BASE_URL` | API Base URL，默认 `https://api.deepseek.com` |
| `DEEPSEEK_SYSTEM_PROMPT` | 全局 system prompt（默认空串，由模板负责提示词） |
| `DEEPSEEK_MODEL` | 通用模型名称，默认 `deepseek-reasoner` |
| `DEEPSEEK_GENERATOR_MODEL` / `DEEPSEEK_VERIFIER_MODEL` | 若生成/验证使用不同模型（可选） |
| `DEEPSEEK_TEMPERATURE` | 默认温度 0.2 |
| `DEEPSEEK_GENERATOR_TEMP` / `DEEPSEEK_VERIFIER_TEMP` | 角色专用温度（可选） |
| `DEEPSEEK_STREAM` | 是否启用流式输出 (`true`/`false`) |

设置了 `DEEPSEEK_API_KEY` 后，orchestrator 会自动用真实 transport 替换 DummyTransport。

---

## 4. 本地 Dry Run（使用 DummyTransport）

默认情况下（未设置 `DEEPSEEK_API_KEY`），系统会启用 `DummyTransport`，具备：

- 并发限流（默认 8）和随机延迟（50–300 ms），模拟真实等待时间。
- 0.2% 的随机失败，用于测试自动重试。
- 生成/验证输出包含必要的 `## Solution` / `\boxed{score}` 片段，便于解析。

若想更快演练，可缩小规模：

```powershell
$env:DRY_POPULATION=2
$env:DRY_VERIFICATIONS=4
$env:MAX_ROUNDS=1
python -m src.orchestrator
```

运行结束后检查 `output/run_*/round_*/` 中的 JSONL 文件即可验证流程。

---

## 5. 真实运行步骤

1. **准备 Python 环境**：推荐 3.10+，并安装 `openai` SDK（用于调用 DeepSeek API）：
   ```bash
   pip install openai
   ```
2. **设置题目**：更新 `input/problem1.json`。
3. **配置环境变量**（示例 PowerShell）：
   ```powershell
   $env:DEEPSEEK_API_KEY="sk-xxx"
   $env:DEEPSEEK_MODEL="deepseek-reasoner"
   $env:POPULATION_SIZE=64
   $env:VERIFICATION_SAMPLES=64
   ```
4. **运行 orchestrator**：
   ```powershell
   python -m src.orchestrator
   ```
5. **查看输出**：在 `output/run_*` 中可获得：
   - `round_*/01_refinement_prompts.jsonl`、`02_generated_proofs.jsonl`、`03_verification_prompts.jsonl`、`04_verification_results.jsonl`、`05_parsed_scores.jsonl`、`06_selected_candidates.jsonl`。
   - `task_metadata.json`、`run_state.json`、`final_solution.json`（若找到满分证据）。

---

## 6. 日志与进度

- 每轮迭代都会打印分隔线与阶段信息（Generation/Verification）。
- 内置实时进度条（通过 logger 输出），示例：`Round 0 verification [##########--------------------] 34.0% (1394/4096)`.
- 如果某次 LLM 请求失败，`src.llm_client` 的日志会提示重试次数。

---

## 7. 常见问题

1. **找不到题目文件**：请确认 `input/problem1.json` 存在，或设置 `PROBLEM_PATH` 指向正确路径。
2. **解析失败**：若日志出现 “Unable to locate boxed score”，说明验证输出没有 `\boxed{score}`，可调整模板或确保模型按要求输出。
3. **API Key 未设置**：会自动回退到 DummyTransport；若想真机调用，必须导出 `DEEPSEEK_API_KEY`。
4. **输出过大**：可以临时调整 `POPULATION_SIZE` / `VERIFICATION_SAMPLES` 以降低算力需求。

---

祝调试顺利！如需进一步定制（如替换题库/模型、扩展评分策略等），可继续阅读 `design.md` 与 `plan.md`，该文档列出了模块分层与数据 schema 设计。Happy hacking 🚀

