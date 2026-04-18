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
from src.rag.tag_filter import apply_milestone2_filter, apply_path1_filter

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
    # Production F1 (after post-processing filter applied)
    prod_f1: float = 0.0
    prod_strict_f1: float = 0.0
    prod_tp: int = 0
    prod_fp: int = 0
    prod_fn: int = 0
    prod_forbidden_violations: int = 0


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


def _build_confirmed_state(
    accumulated_reqs: list[dict],
    accumulated_milestones: dict[str, str],
) -> set[ParsedTag]:
    """
    Build the set of tags already confirmed in prior turns.
    Used by _filter_idempotent_reemissions to identify re-emitted tags.
    """
    confirmed: set[ParsedTag] = set()
    for r in accumulated_reqs:
        confirmed.add(ParsedTag(tag_type="REQ", id=r["id"], value=r["value"], status=r["status"]))
    for m_id, m_status in accumulated_milestones.items():
        confirmed.add(ParsedTag(tag_type="MILESTONE", id=m_id, value=m_status, status=m_status))
    return confirmed


def _dedup_tags(tags: list[ParsedTag]) -> list[ParsedTag]:
    """
    Remove exact within-turn duplicates (type + id + status + value all identical).

    The LLM sometimes outputs the same tag in both the conversational prose and
    the formal tag block at the end of a response. Each instance is captured by
    the regex parser, inflating FP counts. Only true duplicates (all four fields
    identical) are removed — tags with the same (type, id) but different status
    or value are kept, so a legitimate same-turn state update (e.g. TBC → C1)
    is not lost.
    """
    seen: set[tuple[str, str, str, str]] = set()
    result: list[ParsedTag] = []
    for tag in tags:
        key = (tag.tag_type, tag.id, tag.status, tag.value)
        if key not in seen:
            seen.add(key)
            result.append(tag)
    return result


def _filter_idempotent_reemissions(
    predicted: list[ParsedTag],
    confirmed_state: set[ParsedTag],
) -> list[ParsedTag]:
    """
    Remove predicted tags that are identical to already-confirmed state
    (type + id + status + value all match).

    LLM defensive re-emission of already-confirmed tags is correct behaviour
    for state reconstructibility and must not be penalised as False Positive.

    Phase-transition failures are still penalised: e.g. predicting
    [MILESTONE:1:current] when [MILESTONE:2:current] is expected is NOT
    filtered because the id differs.
    """
    return [tag for tag in predicted if tag not in confirmed_state]


# ─── F1 computation ───────────────────────────────────────────────────────────


def _compute_f1(
    predicted: list[ParsedTag],
    expected_tags: list[dict],
    forbidden_tags: list[dict],
    strict: bool = False,
    unfiltered_predicted: Optional[list[ParsedTag]] = None,
) -> tuple[int, int, int, float, float, float, int]:
    """
    Compute TP, FP, FN, Precision, Recall, F1, and forbidden violation count.

    Forbidden tag hits are counted as additional False Positives so that
    No-Assumption Rule violations penalise precision directly.

    Args:
        predicted: Filtered predictions (re-emissions removed) — used for TP and FP.
        expected_tags: Ground-truth tags for this turn.
        forbidden_tags: Tag patterns that must not appear.
        strict: If True use strict matching (id+value+status); else relaxed (id+status).
        unfiltered_predicted: Raw predictions before the idempotent re-emission filter.
            When provided, an expected tag that was not matched by *filtered* predictions
            is still not counted as FN if it appears in *unfiltered* predictions — i.e.
            the LLM re-emitted it correctly and should not be penalised for doing so.
            If None, FN is computed solely from filtered predictions (legacy behaviour).
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

    # FN: expected tags not matched by filtered predictions.
    # If unfiltered_predicted is provided, a re-emitted tag (present in unfiltered but
    # removed by the filter) is NOT a false negative — the LLM produced it correctly.
    unmatched_expected = [exp for i, exp in enumerate(expected_tags) if i not in matched_expected]
    if unfiltered_predicted is not None:
        fn = sum(1 for exp in unmatched_expected if not any(match_fn(pred, exp) for pred in unfiltered_predicted))
    else:
        fn = len(unmatched_expected)

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
        accumulated_milestones: dict[str, str] = {}  # milestone_id → status
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

            predicted = _dedup_tags(_parse_tags(raw_response))

            # Filter idempotent re-emissions before scoring.
            # Tags identical to already-confirmed prior-turn state are correct
            # LLM behaviour (state reconstructibility) and must not count as FP.
            confirmed_state = _build_confirmed_state(accumulated_requirements, accumulated_milestones)
            scored_predicted = _filter_idempotent_reemissions(predicted, confirmed_state)

            # ── Relaxed F1 ──────────────────────────────────────────────────
            # FP: scored_predicted (filtered); FN: checks unfiltered to avoid
            # penalising the LLM for re-emitting already-confirmed TBC tags.
            tp, fp, fn, precision, recall, f1, forbidden_hits = _compute_f1(
                scored_predicted,
                expected_tags,
                forbidden_tags,
                strict=False,
                unfiltered_predicted=predicted,
            )
            # ── Strict F1 ───────────────────────────────────────────────────
            s_tp, _, _, _, _, s_f1, _ = _compute_f1(
                scored_predicted,
                expected_tags,
                forbidden_tags,
                strict=True,
                unfiltered_predicted=predicted,
            )

            # ── Production F1 (post-processing filters applied) ─────────────
            state_reqs = [{"id": r["id"], "value": r["value"], "status": r["status"]} for r in accumulated_requirements]
            state_milestones_list = [{"id": k, "status": v} for k, v in accumulated_milestones.items()]
            pred_req_dicts = [
                {"id": t.id, "value": t.value, "status": t.status} for t in predicted if t.tag_type == "REQ"
            ]
            pred_milestone_dicts = [{"id": t.id, "status": t.status} for t in predicted if t.tag_type == "MILESTONE"]
            filtered_reqs = apply_path1_filter(visa_type, state_reqs, pred_req_dicts)
            filtered_milestones = apply_milestone2_filter(
                visa_type, state_reqs, state_milestones_list, filtered_reqs, pred_milestone_dicts
            )
            prod_predicted = _dedup_tags(
                [ParsedTag(tag_type="REQ", id=r["id"], value=r["value"], status=r["status"]) for r in filtered_reqs]
                + [
                    ParsedTag(tag_type="MILESTONE", id=m["id"], value=m["status"], status=m["status"])
                    for m in filtered_milestones
                ]
            )
            prod_scored = _filter_idempotent_reemissions(prod_predicted, confirmed_state)
            p_tp, p_fp, p_fn, _, _, p_f1, p_forb = _compute_f1(
                prod_scored, expected_tags, forbidden_tags, strict=False, unfiltered_predicted=prod_predicted
            )
            _, _, _, _, _, ps_f1, _ = _compute_f1(
                prod_scored, expected_tags, forbidden_tags, strict=True, unfiltered_predicted=prod_predicted
            )

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
                prod_f1=p_f1,
                prod_strict_f1=ps_f1,
                prod_tp=p_tp,
                prod_fp=p_fp,
                prod_fn=p_fn,
                prod_forbidden_violations=p_forb,
            )
            results.append(result)

            # Accumulate state for next turn (use raw predicted, not filtered,
            # so the full LLM state is carried forward regardless of scoring).
            accumulated_requirements = _merge_requirements(accumulated_requirements, predicted)
            for tag in predicted:
                if tag.tag_type == "MILESTONE":
                    accumulated_milestones[tag.id] = tag.status

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

        # Production (post-filter) aggregate
        macro_prod_f1 = sum(r.prod_f1 for r in results) / n
        macro_prod_strict_f1 = sum(r.prod_strict_f1 for r in results) / n
        prod_tp = sum(r.prod_tp for r in results)
        prod_fp = sum(r.prod_fp for r in results)
        prod_fn = sum(r.prod_fn for r in results)
        prod_forbidden = sum(r.prod_forbidden_violations for r in results)
        prod_micro_p = prod_tp / (prod_tp + prod_fp) if (prod_tp + prod_fp) > 0 else 0.0
        prod_micro_r = prod_tp / (prod_tp + prod_fn) if (prod_tp + prod_fn) > 0 else 0.0
        prod_micro_f1 = (
            2 * prod_micro_p * prod_micro_r / (prod_micro_p + prod_micro_r)
            if (prod_micro_p + prod_micro_r) > 0
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
            # Production metrics (after post-processing filters)
            "prod_macro_f1_relaxed": round(macro_prod_f1, 4),
            "prod_macro_f1_strict": round(macro_prod_strict_f1, 4),
            "prod_micro_f1": round(prod_micro_f1, 4),
            "prod_total_tp": prod_tp,
            "prod_total_fp": prod_fp,
            "prod_total_fn": prod_fn,
            "prod_total_forbidden_violations": prod_forbidden,
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
            "prod_f1_relaxed": round(r.prod_f1, 4),
            "prod_f1_strict": round(r.prod_strict_f1, 4),
            "prod_tp": r.prod_tp,
            "prod_fp": r.prod_fp,
            "prod_fn": r.prod_fn,
            "prod_forbidden_violations": r.prod_forbidden_violations,
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


# ─── Retroactive rescore ──────────────────────────────────────────────────────


def rescore_report(report_path: str, output_dir: str = "eval/results") -> dict:
    """
    Re-score an existing report JSON using the current evaluator logic
    (including the idempotent re-emission filter) without making new LLM calls.

    Reads predicted_tags from each turn, simulates multi-turn state accumulation,
    applies _filter_idempotent_reemissions, and recomputes F1 metrics.

    Args:
        report_path: Path to an existing state_tag_report_*.json file.
        output_dir:  Directory for the revised report (suffix: _revised).

    Returns:
        Revised report dict (also written to disk).
    """
    with open(report_path, "r", encoding="utf-8") as f:
        old_report = json.load(f)

    # Group turns by conversation_id, preserving turn order
    conv_turns: dict[str, list[dict]] = {}
    for turn in old_report["turns"]:
        cid = turn["conversation_id"]
        conv_turns.setdefault(cid, []).append(turn)

    revised_turns: list[TurnResult] = []
    conv_count = len(conv_turns)

    for cid, turns in conv_turns.items():
        accumulated_requirements: list[dict] = []
        accumulated_milestones: dict[str, str] = {}

        for turn in sorted(turns, key=lambda t: t["turn_index"]):
            # Reconstruct ParsedTag list from stored dict representation
            # Apply exact-match dedup (same key as evaluate_conversation)
            predicted: list[ParsedTag] = _dedup_tags(
                [
                    ParsedTag(
                        tag_type=t["type"],
                        id=t["id"],
                        value=t["value"],
                        status=t["status"],
                    )
                    for t in turn.get("predicted_tags", [])
                ]
            )
            expected_tags: list[dict] = turn.get("expected_tags", [])
            forbidden_tags: list[dict] = turn.get("forbidden_tags", [])

            # Apply idempotent re-emission filter
            confirmed_state = _build_confirmed_state(accumulated_requirements, accumulated_milestones)
            scored_predicted = _filter_idempotent_reemissions(predicted, confirmed_state)

            # Recompute F1: FP from filtered, FN from unfiltered
            tp, fp, fn, precision, recall, f1, forbidden_hits = _compute_f1(
                scored_predicted,
                expected_tags,
                forbidden_tags,
                strict=False,
                unfiltered_predicted=predicted,
            )
            s_tp, _, _, _, _, s_f1, _ = _compute_f1(
                scored_predicted,
                expected_tags,
                forbidden_tags,
                strict=True,
                unfiltered_predicted=predicted,
            )

            revised_turns.append(
                TurnResult(
                    conversation_id=cid,
                    turn_index=turn["turn_index"],
                    user_message=turn.get("user_message", ""),
                    predicted_tags=scored_predicted,
                    expected_tags=expected_tags,
                    forbidden_tags=forbidden_tags,
                    raw_response="",  # not re-stored; original report has raw_response_tail
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
            )

            # Advance accumulated state (use raw predicted, not filtered)
            accumulated_requirements = _merge_requirements(accumulated_requirements, predicted)
            for tag in predicted:
                if tag.tag_type == "MILESTONE":
                    accumulated_milestones[tag.id] = tag.status

    aggregate = StateTagEvaluator._aggregate(revised_turns, conv_count)
    turn_details = [StateTagEvaluator._serialise_turn(r) for r in revised_turns]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    source_stem = Path(report_path).stem
    output_path = f"{output_dir}/{source_stem}_revised.json"

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "source_report": report_path,
        "rescore_note": "Retroactive rescore: idempotent re-emission filter applied. No new LLM calls.",
        "aggregate": aggregate,
        "turns": turn_details,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Revised report saved to %s", output_path)
    return report


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
    print("  [Raw LLM output]")
    print(f"  Macro F1  (relaxed) : {agg['macro_f1_relaxed']:.3f}")
    print(f"  Macro F1  (strict)  : {agg['macro_f1_strict']:.3f}")
    print(f"  Macro Precision     : {agg['macro_precision']:.3f}")
    print(f"  Macro Recall        : {agg['macro_recall']:.3f}")
    print(f"  Micro F1            : {agg['micro_f1']:.3f}")
    print(f"  TP / FP / FN        : {agg['total_tp']} / {agg['total_fp']} / {agg['total_fn']}")
    print(f"  Forbidden violations: {agg['total_forbidden_violations']}")
    print()
    prod = agg.get("prod_macro_f1_relaxed") is not None
    if prod:
        print("  [Production (post-filter)]")
        print(f"  Prod Macro F1 (relaxed) : {agg['prod_macro_f1_relaxed']:.3f}")
        print(f"  Prod Macro F1 (strict)  : {agg['prod_macro_f1_strict']:.3f}")
        print(f"  Prod Micro F1           : {agg['prod_micro_f1']:.3f}")
        print(f"  Prod TP / FP / FN       : {agg['prod_total_tp']} / {agg['prod_total_fp']} / {agg['prod_total_fn']}")
        print(f"  Prod Forbidden          : {agg['prod_total_forbidden_violations']}")
        print()
    print("  Per-turn breakdown:")
    hdr = f"  {'Conv':20s} {'T':>2}  {'F1-R':>5}  {'pF1-R':>6}  {'F1-S':>5}  {'pF1-S':>6}  {'P':>5}  {'R':>5}  {'Forb':>4}  Message"
    print(hdr)
    print("  " + "-" * 105)
    for t in turns:
        cid = t["conversation_id"][:18]
        ti = t["turn_index"]
        f1r = t["f1_relaxed"]
        pf1r = t.get("prod_f1_relaxed", f1r)
        f1s = t["f1_strict"]
        pf1s = t.get("prod_f1_strict", f1s)
        p = t["precision"]
        r = t["recall"]
        fv = t["forbidden_violations"]
        pfv = t.get("prod_forbidden_violations", fv)
        msg = t["user_message"][:30]
        fv_mark = f"⚠{pfv}" if pfv else "  -"
        print(f"  {cid:20s}  {ti}  {f1r:.3f}  {pf1r:.3f}  {f1s:.3f}  {pf1s:.3f}  {p:.3f}  {r:.3f}  {fv_mark:>4}  {msg}")
    print("=" * 64)


async def main() -> None:
    args = sys.argv[1:]

    # --rescore <report.json> [<report2.json> ...] — retroactive rescore mode
    if args and args[0] == "--rescore":
        report_paths = args[1:] if len(args) > 1 else []
        if not report_paths:
            print("Usage: python -m eval.state_tag_evaluator --rescore <report.json> [...]")
            sys.exit(1)
        for path in report_paths:
            print(f"\nRescoring: {path}")
            report = rescore_report(path)
            _print_report(report)
        return

    dataset_path = args[0] if args else "eval/state_tag_dataset.json"
    evaluator = StateTagEvaluator()
    report = await evaluator.evaluate_all(dataset_path)
    _print_report(report)


if __name__ == "__main__":
    asyncio.run(main())
