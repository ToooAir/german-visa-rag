"""
Model benchmark: compare gpt-4o-mini / gpt-4.1-mini / gpt-5-mini (full reasoning)
/ gpt-5-mini (reasoning_effort=none) on the same two test turns.

Measures:
  - Time to first token (TTFT) via streaming
  - Total latency (wall-clock, stream-to-completion)
  - Response length (chars)
  - Tag count emitted
  - Tag quality: TP / FP / FN vs expected_tags

Usage:
  source .venv/bin/activate
  python -m eval.model_benchmark
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Optional, TypedDict

from src.config import settings
from src.llm.openai_client import OpenAIClient
from src.rag.prompt_builder import PromptRequest, get_prompt_builder

# ── Type Definitions ──────────────────────────────────────────────────────────


class TestCase(TypedDict):
    name: str
    question: str
    visa_type: str
    language: str
    requirements: Optional[dict]
    expected_tags: list[dict[str, str]]


class ModelConfig(TypedDict):
    label: str
    deployment: str
    reasoning_effort: Optional[str]


# ── Test cases ────────────────────────────────────────────────────────────────

TEST_CASES: list[TestCase] = [
    {
        "name": "CK T0 — first contact",
        "question": "我想申請 Chancenkarte，我有德文 B1 的 Goethe 證書，學歷是台灣大學的學士學位。",
        "visa_type": "chancenkarte",
        "language": "zh-TW",
        "requirements": None,
        "expected_tags": [
            {"type": "MILESTONE", "id": "1", "status": "current"},
            {"type": "REQ", "id": "1-1", "value": "TBC", "status": "warning"},
            {"type": "REQ", "id": "1-2", "value": "B1", "status": "required"},
            {"type": "REQ", "id": "1-3", "value": "TBC", "status": "warning"},
            {"type": "REQ", "id": "2-1", "value": "B1|2", "status": "required"},
        ],
    },
    {
        "name": "CK Path1 T0 — direct anabin confirm",
        "question": "我的台灣大學學位已在 anabin 查詢確認為 H+ 等級且評級是 'gleichwertig'，我想申請 Chancenkarte，年薪要求我沒有，但我有 €14,000 存款。",
        "visa_type": "chancenkarte",
        "language": "zh-TW",
        "requirements": None,
        "expected_tags": [
            {"type": "MILESTONE", "id": "1", "status": "current"},
            {"type": "REQ", "id": "1-1", "value": "MET", "status": "required"},
            {"type": "REQ", "id": "1-3", "value": "MET", "status": "required"},
        ],
    },
]

# ── Tag parsing ────────────────────────────────────────────────────────────────

_MILESTONE_RE = re.compile(r"\[MILESTONE:(\w+):(\w+)\]")
_REQ_RE = re.compile(r"\[REQ:([\w\-]+):([\w|]+):(\w+)\]")


def parse_tags(text: str) -> list[dict]:
    tags: list[dict] = []
    for m_id, m_status in _MILESTONE_RE.findall(text or ""):
        tags.append({"type": "MILESTONE", "id": m_id, "status": m_status})
    for r_id, r_val, r_status in _REQ_RE.findall(text or ""):
        tags.append({"type": "REQ", "id": r_id, "value": r_val, "status": r_status})
    return tags


def score_tags(predicted: list[dict], expected: list[dict]) -> tuple[int, int, int]:
    """Compute relaxed TP/FP/FN (id + type match; value/status exact)."""

    def key(t: dict) -> tuple:
        return (t["type"], t["id"], t.get("value", ""), t.get("status", ""))

    pred_set = {key(t) for t in predicted}
    exp_set = {key(t) for t in expected}
    tp = len(pred_set & exp_set)
    fp = len(pred_set - exp_set)
    fn = len(exp_set - pred_set)
    return tp, fp, fn


# ── Benchmark runner ───────────────────────────────────────────────────────────


@dataclass
class BenchResult:
    model_label: str
    case_name: str
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    response_chars: int = 0
    tag_count: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    error: Optional[str] = None
    response_tail: str = ""


async def run_one(
    client: OpenAIClient,
    case: TestCase,
    model_label: str,
    reasoning_effort: Optional[str] = None,
) -> BenchResult:
    pb = get_prompt_builder()
    req = PromptRequest(
        context="Germany visa information.",
        question=case["question"],
        language=case["language"],
        visa_type=case["visa_type"],
        requirements=case.get("requirements"),
    )
    sys_prompt = pb.build_system_prompt(req)
    user_msg = pb.build_user_message(case["question"])
    messages = [{"role": "system", "content": sys_prompt}, user_msg]

    result = BenchResult(model_label=model_label, case_name=case["name"])
    text_buf: list[str] = []
    ttft_recorded = False

    try:
        t_start = time.perf_counter()
        stream = client.call_streaming(
            messages=messages,
            temperature=0.1,
            reasoning_effort=reasoning_effort,
        )
        async for chunk in stream:
            if not ttft_recorded:
                result.ttft_ms = (time.perf_counter() - t_start) * 1000
                ttft_recorded = True
            text_buf.append(chunk)
        result.total_ms = (time.perf_counter() - t_start) * 1000

        full_text = "".join(text_buf)
        result.response_chars = len(full_text)
        result.response_tail = full_text[-300:]

        predicted = parse_tags(full_text)
        result.tag_count = len(predicted)
        result.tp, result.fp, result.fn = score_tags(predicted, case["expected_tags"])

    except Exception as exc:
        result.error = str(exc)[:120]
        result.total_ms = (time.perf_counter() - t_start) * 1000

    return result


# ── Model configs ──────────────────────────────────────────────────────────────

MODEL_CONFIGS: list[ModelConfig] = [
    {"label": "gpt-4o-mini", "deployment": "gpt-4o-mini", "reasoning_effort": None},
    {"label": "gpt-4.1-mini", "deployment": "gpt-4.1-mini", "reasoning_effort": None},
    {"label": "gpt-5-mini (full)", "deployment": "gpt-5-mini", "reasoning_effort": None},
    {"label": "gpt-5-mini (none)", "deployment": "gpt-5-mini", "reasoning_effort": "none"},
]


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    all_results: list[BenchResult] = []

    for cfg in MODEL_CONFIGS:
        client = OpenAIClient(
            api_key=settings.azure_openai_api_key or settings.openai_api_key,
            deployment=cfg["deployment"],
        )
        print(f"\n▶ Testing [{cfg['label']}] ...")

        for case in TEST_CASES:
            r = await run_one(client, case, cfg["label"], cfg["reasoning_effort"])
            all_results.append(r)
            status = "✅" if not r.error else "❌"
            f1 = (2 * r.tp / (2 * r.tp + r.fp + r.fn)) if (r.tp + r.fp + r.fn) > 0 else 0.0
            print(
                f"  {status} {case['name'][:35]:<35} "
                f"TTFT={r.ttft_ms:6.0f}ms  Total={r.total_ms:6.0f}ms  "
                f"chars={r.response_chars:4}  tags={r.tag_count}  "
                f"TP/FP/FN={r.tp}/{r.fp}/{r.fn}  F1={f1:.3f}"
            )
            if r.error:
                print(f"    ERROR: {r.error}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("  BENCHMARK SUMMARY")
    print("=" * 110)
    print(f"  {'Model':<28} {'Case':<38} {'TTFT(ms)':>9} {'Total(ms)':>10} {'Chars':>6} {'Tags':>5} {'F1':>6}")
    print("-" * 110)

    for r in all_results:
        f1 = (2 * r.tp / (2 * r.tp + r.fp + r.fn)) if (r.tp + r.fp + r.fn) > 0 else 0.0
        err = " ❌" if r.error else ""
        print(
            f"  {r.model_label:<28} {r.case_name:<38} "
            f"{r.ttft_ms:>9.0f} {r.total_ms:>10.0f} {r.response_chars:>6} {r.tag_count:>5} {f1:>6.3f}{err}"
        )

    print("=" * 110)

    # ── Per-model aggregate ───────────────────────────────────────────────────
    print("\n  Per-model aggregate (avg across test cases):")
    print(f"  {'Model':<28} {'Avg TTFT':>10} {'Avg Total':>10} {'Macro F1':>10} {'Errors':>7}")
    print("-" * 75)

    models = list(dict.fromkeys(r.model_label for r in all_results))
    for m in models:
        rs = [r for r in all_results if r.model_label == m and not r.error]
        errs = sum(1 for r in all_results if r.model_label == m and r.error)
        if not rs:
            print(f"  {m:<28} {'N/A':>10} {'N/A':>10} {'N/A':>10} {errs:>7}")
            continue
        avg_ttft = sum(r.ttft_ms for r in rs) / len(rs)
        avg_total = sum(r.total_ms for r in rs) / len(rs)
        f1s = [(2 * r.tp / (2 * r.tp + r.fp + r.fn)) if (r.tp + r.fp + r.fn) > 0 else 0.0 for r in rs]
        macro_f1 = sum(f1s) / len(f1s)
        print(f"  {m:<28} {avg_ttft:>9.0f}ms {avg_total:>9.0f}ms {macro_f1:>10.3f} {errs:>7}")

    print("=" * 110)


if __name__ == "__main__":
    import logging

    logging.getLogger("visa_rag").setLevel(logging.WARNING)
    asyncio.run(main())
