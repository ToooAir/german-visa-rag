"""
State tag post-processing filters.

Implements deterministic rules that are structurally fragile when expressed
solely as LLM prompt instructions — due to attention dilution and inter-rule
conflicts that emerge as prompt complexity grows.

Two filters are provided:

  1. apply_path1_filter
     Suppress REQ:1-2 (language threshold) when Chancenkarte Path 1 is active.
     Trigger: REQ:1-3:MET:required present in state OR this turn's output.
     Guarantee: REQ:1-2 is NEVER emitted in Path 1 context.

  2. apply_milestone2_filter
     Inject MILESTONE:2:current when all required criteria for the visa type
     are confirmed (across CURRENT_UI_STATE + this turn).
     Also removes MILESTONE:1 from the emitted set on injection (supersession).

Both filters accept and return plain dicts so they can be used in both the
production streaming path (answer_generator.py) and the offline evaluator
(eval/state_tag_evaluator.py) without shared data-class coupling.

Dict formats expected:
  REQ requirement:  {"id": str, "value": str, "status": str}
  Milestone:        {"id": str, "status": str}
"""

from __future__ import annotations

from typing import Optional

# Salary values that satisfy the Blue Card salary criterion.
_SALARY_MET_VALUES: frozenset[str] = frozenset({"MET", "SALARY_MET", "SHORTAGE_SALARY_MET", "GRADUATE_SALARY_MET"})


def apply_path1_filter(
    visa_type: Optional[str],
    state_requirements: list[dict],
    new_requirements: list[dict],
) -> list[dict]:
    """Remove REQ:1-2 from new_requirements when Chancenkarte Path 1 is active.

    Path 1 is active when REQ:1-3:MET:required appears in either
    state_requirements (CURRENT_UI_STATE from prior turns) or new_requirements
    (this turn's LLM output).  Both Case A(a) and Case A(b) are covered.

    Returns new_requirements unchanged for non-Chancenkarte visa types or when
    Path 1 is not active.
    """
    if visa_type != "chancenkarte":
        return new_requirements

    path1_active = any(
        r.get("id") == "1-3" and r.get("value") == "MET" and r.get("status") == "required"
        for r in state_requirements + new_requirements
    )
    if not path1_active:
        return new_requirements

    _LANG_IDS = {"1-2", "2-1", "2-7"}
    return [r for r in new_requirements if r.get("id") not in _LANG_IDS]


def apply_milestone2_filter(
    visa_type: Optional[str],
    state_requirements: list[dict],
    state_milestones: list[dict],
    new_requirements: list[dict],
    new_milestones: list[dict],
) -> list[dict]:
    """Inject MILESTONE:2:current when all visa REQ criteria are confirmed.

    Merges state and new requirements (last-write-wins by ID) to evaluate
    whether the MILESTONE:2 trigger condition is satisfied.

    On injection:
    - Appends {"id": "2", "status": "current"} to the result.
    - Removes MILESTONE:1 from new_milestones (supersession rule).

    Returns new_milestones unchanged when:
    - visa_type is None or unrecognised,
    - MILESTONE:2 is already present in state or new milestones,
    - trigger conditions are not met.
    """
    if not visa_type:
        return new_milestones

    # MILESTONE:2 already fired — nothing to inject.
    if any(m.get("id") == "2" for m in state_milestones + new_milestones):
        return new_milestones

    # Merge requirements: state first, new overwrites by ID.
    merged: dict[str, dict] = {r["id"]: r for r in state_requirements}
    merged.update({r["id"]: r for r in new_requirements})

    def is_met(req_id: str) -> bool:
        r = merged.get(req_id)
        return r is not None and r.get("value") == "MET" and r.get("status") == "required"

    def is_confirmed(req_id: str) -> bool:
        r = merged.get(req_id)
        return r is not None and r.get("status") == "required"

    fire = False
    if visa_type == "chancenkarte":
        if is_met("1-3"):  # Path 1 — language threshold waived
            fire = is_met("1-1")
        else:  # Path 2 — language must be confirmed (not TBC)
            lang_ok = is_confirmed("1-2")
            fire = is_met("1-1") and lang_ok and is_met("1-3")
    elif visa_type == "blue_card":
        salary_req = merged.get("2")
        salary_ok = (
            salary_req is not None
            and salary_req.get("status") == "required"
            and salary_req.get("value") in _SALARY_MET_VALUES
        )
        fire = is_met("1") and salary_ok
    elif visa_type == "student":
        fire = all(is_met(i) for i in ["1", "2", "3", "4"])
    elif visa_type == "skilled_worker":
        fire = is_confirmed("1") and is_confirmed("2")

    if not fire:
        return new_milestones

    # Inject MILESTONE:2, suppress MILESTONE:1 (supersession).
    result = [m for m in new_milestones if m.get("id") != "1"]
    result.append({"id": "2", "status": "current"})
    return result
