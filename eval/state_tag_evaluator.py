"""
State Tag F1 Evaluator.

Measures the accuracy of REQ and MILESTONE tag generation in multi-turn
visa consultations. Tags encode structured user state (requirements met/unmet,
conversation phase) and are the most distinctive feature of this RAG system.

Design decisions:
- LLM-direct: calls the model with a stub context, bypassing retrieval.
  Tags derive from DOMAIN_KNOWLEDGE in the system prompt + user-stated facts,
  not from retrieved documents. This isolates tag-generation accuracy from
  retrieval variability.
- Multi-turn state accumulation: REQ tags from each turn are merged and
  injected as CURRENT_UI_STATE in subsequent turns, exactly mirroring the
  real frontend behaviour.
- Two F1 variants:
    relaxed  — type + id + status must match (value ignored)
    strict   — type + id + value + status must all match
- Forbidden tags: (type, id, status) patterns that must NOT appear
  (e.g. [REQ:1-1:*:required] when user never confirmed funds).
  Each violation counts as an extra False Positive.

Usage:
    python -m eval.state_tag_evaluator                          # default dataset
    python -m eval.state_tag_evaluator eval/state_tag_dataset.json
"""

import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import settings
from src.llm import get_llm_client
from src.logger import logger
from src.observability.mlflow_tracker import get_mlflow_tracker
from src.rag.prompt_builder import PromptRequest, get_prompt_builder

# ─── Tag parsing (mirrors answer_generator.py) ────────────────────────────────

_MILESTONE_RE = re.compile(r"\[MILESTONE:([\d-]+):(\w+)\]")
_REQ_RE = re.compile(r"\[REQ:([\d-]+):([^:\]]+):(\w+)\]")

# Minimal stub context injected into every evaluation prompt.
# Keeps the prompt structurally valid without influencing DOMAIN_KNOWLEDGE-
# driven tag generation. Retrieved documents are intentionally omitted.
_EVAL_CONTEXT = (
    "General information about German visa and immigration requirements "
    "is available from official sources including make-it-in-germany.com "
    "and auswaertiges-amt.de. Refer to the DOMAIN_KNOWLEDGE section for "
    "structured eligibility rules and tag generation."
)


# ─── Data structures ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ParsedTag:
    """One extracted tag from raw LLM output."""

    tag_type: str  # "MILESTONE" or "REQ"
    id: str
    value: str  # For MILESTONE: mirrors status; for REQ: the VALUE field
    status: str  # "current" | "completed" | "required" | "warning"

    def __str__(self) -> str:
        if self.tag_type == "MILESTONE":
            return f"[MILESTONE:{self.id}:{self.status}]"
        return f"[REQ:{self.id}:{self.value}:{self.status}]"


@dataclass
class TurnResult:
    """Evaluation result for a single conversation turn."""

    conversation_id: str
    turn_index: int
    user_message: str
    predicted_tags: list[ParsedTag] = field(default_factory=list)
    expected_tags: list[dict] = field(default_factory=list)
    forbidden_tags: list[dict] = field(default_factory=list)
    raw_response: str = ""
    # Relaxed matching (id + status)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    # Strict matching (id + value + status)
    strict_tp: int = 0
    strict_f1: float = 0.0
    # Forbidden tags that incorrectly appeared
    forbidden_violations: int = 0


# ─── Tag parsing ──────────────────────────────────────────────────────────────


def _parse_tags(text: str) -> list[ParsedTag]:
    """Extract all MILESTONE and REQ tags from a raw LLM response."""
    tags: list[ParsedTag] = []
    for m_id, m_status in _MILESTONE_RE.findall(text):
        tags.append(ParsedTag(tag_type="MILESTONE", id=m_id, value=m_status, status=m_status))
    for r_id, r_val, r_status in _REQ_RE.findall(text):
        tags.append(ParsedTag(tag_type="REQ", id=r_id, value=r_val, status=r_status))
    return tags


# ─── Matching helpers ─────────────────────────────────────────────────────────


def _relaxed_match(predicted: ParsedTag, expected: dict) -> bool:
    """Relaxed: type + id + status must match. Value is ignored."""
    return (
        predicted.tag_type == expected["type"]
        and predicted.id == expected["id"]
        and predicted.status == expected["status"]
    )


def _strict_match(predicted: ParsedTag, expected: dict) -> bool:
    """Strict: type + id + status must match AND value must match.
    MILESTONE has no separate value field, so status-only match suffices."""
    if not _relaxed_match(predicted, expected):
        return False
    if predicted.tag_type == "MILESTONE":
        return True
    return predicted.value == expected.get("value", predicted.value)


def _is_forbidden_hit(predicted: ParsedTag, forbidden: dict) -> bool:
    """True if predicted tag matches a forbidden (type, id, status) pattern."""
    return (
        predicted.tag_type == forbidden["type"]
        and predicted.id == forbidden["id"]
        and predicted.status == forbidden["status"]
    )


# ─── F1 computation ───────────────────────────────────────────────────────────


def _compute_f1(
    predicted: list[ParsedTag],
    expected_tags: list[dict],
    forbidden_tags: list[dict],
    strict: bool = False,
) -> tuple[int, int, int, float, float, float, int]:
    """
    Compute TP, FP, FN, Precision, Recall, F1, and forbidden violation count.

    Forbidden tag hits are counted as additional False Positives so that
    No-Assumption Rule violations penalise precision directly.
    """
    match_fn = _strict_match if strict else _relaxed_match

    matched_expected: set[int] = set()
    tp = 0
    fp_from_unmatched = 0

    for pred in predicted:
        matched = False
        for i, exp in enumerate(expected_tags):
            if i not in matched_expected and match_fn(pred, exp):
                tp += 1
                matched_expected.add(i)
                matched = True
                break
        if not matched:
            fp_from_unmatched += 1

    # Forbidden violations count as extra FP
    forbidden_hits = sum(1 for pred in predicted for forb in forbidden_tags if _is_forbidden_hit(pred, forb))
    fp = fp_from_unmatched + forbidden_hits
    fn = len(expected_tags) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return tp, fp, fn, precision, recall, f1, forbidden_hits


# ─── State accumulation ───────────────────────────────────────────────────────


def _tags_to_requirements(tags: list[ParsedTag]) -> list[dict]:
    """
    Convert parsed REQ tags into the `requirements` format expected by
    PromptBuilder.build_system_prompt().

    Later tags with the same ID override earlier ones, mirroring the real
    frontend's state-update semantics (last-write-wins per requirement ID).
    """
    state: dict[str, dict] = {}
    for tag in tags:
        if tag.tag_type == "REQ":
            state[tag.id] = {"id": tag.id, "value": tag.value, "status": tag.status}
    return list(state.values())


def _merge_requirements(
    accumulated: list[dict],
    new_tags: list[ParsedTag],
) -> list[dict]:
    """Merge new REQ tags into accumulated state. New values override old."""
    merged: dict[str, dict] = {r["id"]: r for r in accumulated}
    for r in _tags_to_requirements(new_tags):
        merged[r["id"]] = r
    return list(merged.values())


# ─── Evaluator ────────────────────────────────────────────────────────────────


class StateTagEvaluator:
    """
    Evaluates State Tag (REQ + MILESTONE) generation accuracy using F1.

    Each conversation is evaluated turn-by-turn. REQ tags produced in turn N
    are accumulated and injected as CURRENT_UI_STATE into turn N+1, exactly
    mirroring the real multi-turn frontend flow.
    """

    def __init__(self):
        self.llm = get_llm_client()
        self.prompt_builder = get_prompt_builder()
        self.mlflow = get_mlflow_tracker()

    async def _call_llm(self, messages: list[dict]) -> str:
        """Call LLM with low temperature for deterministic tag generation."""
        return await self.llm.call_non_streaming(
            messages=messages,
            temperature=0.1,
            max_tokens=settings.max_response_tokens,
        )

    async def evaluate_conversation(self, conversation: dict) -> list[TurnResult]:
        """
        Evaluate one multi-turn conversation, accumulating REQ state across turns.

        Args:
            conversation: One entry from the dataset `conversations` array.

        Returns:
            List of TurnResult, one per turn.
        """
        conv_id: str = conversation["id"]
        visa_type: Optional[str] = conversation.get("visa_type")
        language: str = conversation.get("language", "zh-TW")
        turns: list[dict] = conversation["turns"]

        accumulated_requirements: list[dict] = []
        results: list[TurnResult] = []

        for turn in turns:
            turn_index: int = turn["turn_index"]
            user_message: str = turn["user"]
            expected_tags: list[dict] = turn.get("expected_tags", [])
            forbidden_tags: list[dict] = turn.get("forbidden_tags", [])

            logger.info(
                "Evaluating [%s] turn %d/%d: %s",
                conv_id,
                turn_index + 1,
                len(turns),
                user_message[:60],
            )

            # Build prompt — inject accumulated state as CURRENT_UI_STATE
            req = PromptRequest(
                context=_EVAL_CONTEXT,
                question=user_message,
                language=language,
                visa_type=visa_type,
                requirements=accumulated_requirements if accumulated_requirements else None,
            )
            system_prompt = self.prompt_builder.build_system_prompt(req)
            user_msg = self.prompt_builder.build_user_message(user_message)
            messages = [
                {"role": "system", "content": system_prompt},
                user_msg,
            ]

            try:
                raw_response = await self._call_llm(messages)
            except Exception as exc:
                logger.error("LLM call failed [%s] turn %d: %s", conv_id, turn_index, exc)
                raw_response = ""

            predicted = _parse_tags(raw_response)

            # ── Relaxed F1 ──────────────────────────────────────────────────
            tp, fp, fn, precision, recall, f1, forbidden_hits = _compute_f1(
                predicted, expected_tags, forbidden_tags, strict=False
            )
            # ── Strict F1 ───────────────────────────────────────────────────
            s_tp, _, _, _, _, s_f1, _ = _compute_f1(predicted, expected_tags, forbidden_tags, strict=True)

            result = TurnResult(
                conversation_id=conv_id,
                turn_index=turn_index,
                user_message=user_message,
                predicted_tags=predicted,
                expected_tags=expected_tags,
                forbidden_tags=forbidden_tags,
                raw_response=raw_response,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                strict_tp=s_tp,
                strict_f1=s_f1,
                forbidden_violations=forbidden_hits,
            )
            results.append(result)

            # Accumulate state for next turn
            accumulated_requirements = _merge_requirements(accumulated_requirements, predicted)

        return results

    # ─── Aggregate + report ───────────────────────────────────────────────────

    async def evaluate_all(
        self,
        dataset_path: str,
        output_dir: str = "eval/results",
    ) -> dict:
        """
        Evaluate all conversations in the dataset and produce a JSON report.

        Args:
            dataset_path: Path to state_tag_dataset.json.
            output_dir: Directory to write the report JSON.

        Returns:
            Full report dict (also written to disk and logged to MLflow).
        """
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        all_results: list[TurnResult] = []

        for conversation in dataset["conversations"]:
            turn_results = await self.evaluate_conversation(conversation)
            all_results.extend(turn_results)

        aggregate = self._aggregate(all_results, len(dataset["conversations"]))
        turn_details = [self._serialise_turn(r) for r in all_results]

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/state_tag_report_{timestamp}.json"

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "dataset": dataset_path,
            "aggregate": aggregate,
            "turns": turn_details,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("State Tag F1 report saved to %s", output_path)

        if self.mlflow:
            self._log_to_mlflow(aggregate)

        return report

    @staticmethod
    def _aggregate(results: list[TurnResult], conv_count: int) -> dict:
        """Compute macro and micro F1 across all turns."""
        n = len(results)
        if n == 0:
            return {}

        # Macro averages — mean of per-turn metrics
        macro_precision = sum(r.precision for r in results) / n
        macro_recall = sum(r.recall for r in results) / n
        macro_f1 = sum(r.f1 for r in results) / n
        macro_strict_f1 = sum(r.strict_f1 for r in results) / n

        # Micro averages — pool TP/FP/FN across all turns
        total_tp = sum(r.tp for r in results)
        total_fp = sum(r.fp for r in results)
        total_fn = sum(r.fn for r in results)
        total_forbidden = sum(r.forbidden_violations for r in results)

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )

        return {
            "macro_f1_relaxed": round(macro_f1, 4),
            "macro_f1_strict": round(macro_strict_f1, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "micro_f1": round(micro_f1, 4),
            "micro_precision": round(micro_precision, 4),
            "micro_recall": round(micro_recall, 4),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "total_forbidden_violations": total_forbidden,
            "turn_count": n,
            "conversation_count": conv_count,
        }

    @staticmethod
    def _serialise_turn(r: TurnResult) -> dict:
        return {
            "conversation_id": r.conversation_id,
            "turn_index": r.turn_index,
            "user_message": r.user_message[:100],
            "predicted_tags": [
                {"type": t.tag_type, "id": t.id, "value": t.value, "status": t.status} for t in r.predicted_tags
            ],
            "expected_tags": r.expected_tags,
            "forbidden_tags": r.forbidden_tags,
            "tp": r.tp,
            "fp": r.fp,
            "fn": r.fn,
            "precision": round(r.precision, 4),
            "recall": round(r.recall, 4),
            "f1_relaxed": round(r.f1, 4),
            "f1_strict": round(r.strict_f1, 4),
            "forbidden_violations": r.forbidden_violations,
            "raw_response_tail": r.raw_response[-400:] if r.raw_response else "",
        }

    def _log_to_mlflow(self, aggregate: dict) -> None:
        try:
            import mlflow

            with mlflow.start_run(run_name="state_tag_f1_evaluation"):
                for key, value in aggregate.items():
                    if isinstance(value, float):
                        mlflow.log_metric(key, value)
                    elif isinstance(value, int):
                        mlflow.log_metric(key, float(value))
                logger.info("State Tag F1 metrics logged to MLflow")
        except Exception as exc:
            logger.warning("MLflow logging failed: %s", exc)


# ─── CLI entry point ──────────────────────────────────────────────────────────


def _print_report(report: dict) -> None:
    agg = report["aggregate"]
    turns = report["turns"]

    print("\n" + "=" * 64)
    print("  STATE TAG F1 EVALUATION REPORT")
    print("=" * 64)
    print(f"  Conversations : {agg['conversation_count']}")
    print(f"  Turns         : {agg['turn_count']}")
    print()
    print(f"  Macro F1  (relaxed) : {agg['macro_f1_relaxed']:.3f}")
    print(f"  Macro F1  (strict)  : {agg['macro_f1_strict']:.3f}")
    print(f"  Macro Precision     : {agg['macro_precision']:.3f}")
    print(f"  Macro Recall        : {agg['macro_recall']:.3f}")
    print(f"  Micro F1            : {agg['micro_f1']:.3f}")
    print()
    print(f"  TP / FP / FN        : {agg['total_tp']} / {agg['total_fp']} / {agg['total_fn']}")
    print(f"  Forbidden violations: {agg['total_forbidden_violations']}")
    print()
    print("  Per-turn breakdown:")
    print(f"  {'Conv':20s} {'T':>2}  {'F1-R':>5}  {'F1-S':>5}  {'P':>5}  {'R':>5}  {'Forb':>4}  Message")
    print("  " + "-" * 90)
    for t in turns:
        cid = t["conversation_id"][:18]
        ti = t["turn_index"]
        f1r = t["f1_relaxed"]
        f1s = t["f1_strict"]
        p = t["precision"]
        r = t["recall"]
        fv = t["forbidden_violations"]
        msg = t["user_message"][:38]
        fv_mark = f"⚠{fv}" if fv else "  -"
        print(f"  {cid:20s}  {ti}  {f1r:.3f}  {f1s:.3f}  {p:.3f}  {r:.3f}  {fv_mark:>4}  {msg}")
    print("=" * 64)


async def main() -> None:
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "eval/state_tag_dataset.json"

    evaluator = StateTagEvaluator()
    report = await evaluator.evaluate_all(dataset_path)
    _print_report(report)


if __name__ == "__main__":
    asyncio.run(main())
