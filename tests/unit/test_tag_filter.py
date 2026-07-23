"""Unit tests for src/rag/tag_filter.py."""

from src.rag.tag_filter import (
    apply_english_c1_split_filter,
    apply_feg_path_a_filter,
    apply_milestone2_filter,
    apply_path1_filter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def req(id_, value, status="required"):
    return {"id": id_, "value": value, "status": status}


def milestone(id_, status="current"):
    return {"id": id_, "status": status}


# ---------------------------------------------------------------------------
# apply_english_c1_split_filter
# ---------------------------------------------------------------------------


class TestApplyEnglishC1SplitFilter:
    def test_c1_1_in_2_1_is_rerouted_to_2_7(self):
        reqs = [req("2-1", "C1|1")]
        result = apply_english_c1_split_filter(reqs)
        assert result == [{"id": "2-7", "value": "EN_C1|1", "status": "required"}]

    def test_en_c1_1_in_2_1_is_rerouted_to_2_7(self):
        reqs = [req("2-1", "EN_C1|1")]
        result = apply_english_c1_split_filter(reqs)
        assert result == [{"id": "2-7", "value": "EN_C1|1", "status": "required"}]

    def test_german_a2_1pt_not_affected(self):
        """German A2 = A2|1 must NOT be rerouted — only C1|1 / EN_C1|1 are English."""
        reqs = [req("2-1", "A2|1")]
        result = apply_english_c1_split_filter(reqs)
        assert result[0]["id"] == "2-1"
        assert result[0]["value"] == "A2|1"

    def test_german_c1_4pt_not_affected(self):
        reqs = [req("2-1", "C1|4")]
        result = apply_english_c1_split_filter(reqs)
        assert result[0]["id"] == "2-1"
        assert result[0]["value"] == "C1|4"

    def test_stacking_case_both_tags(self):
        """German C1 (4pt) + English C1 (1pt): only the English one is rerouted."""
        reqs = [req("2-1", "C1|4"), req("2-1", "C1|1")]
        result = apply_english_c1_split_filter(reqs)
        ids = [(r["id"], r["value"]) for r in result]
        assert ("2-1", "C1|4") in ids
        assert ("2-7", "EN_C1|1") in ids

    def test_non_language_tags_unchanged(self):
        reqs = [req("2-2", "5_YEARS_EXP|3"), req("2-3", "UNDER_35|2")]
        result = apply_english_c1_split_filter(reqs)
        assert result == reqs

    def test_empty_list(self):
        assert apply_english_c1_split_filter([]) == []

    def test_status_preserved_on_reroute(self):
        reqs = [{"id": "2-1", "value": "C1|1", "status": "warning"}]
        result = apply_english_c1_split_filter(reqs)
        assert result[0]["status"] == "warning"


# ---------------------------------------------------------------------------
# apply_path1_filter
# ---------------------------------------------------------------------------


class TestApplyPath1Filter:
    def test_non_chancenkarte_passthrough(self):
        new = [req("1-2", "B2")]
        assert apply_path1_filter("blue_card", [], new) == new

    def test_path1_not_active_passthrough(self):
        new = [req("1-2", "C1"), req("2-1", "C1|4")]
        result = apply_path1_filter("chancenkarte", [], new)
        assert result == new

    def test_path1_active_via_state_suppresses_1_2(self):
        state = [req("1-3", "MET")]
        new = [req("1-2", "C1"), req("1-1", "MET")]
        result = apply_path1_filter("chancenkarte", state, new)
        ids = [r["id"] for r in result]
        assert "1-2" not in ids
        assert "1-1" in ids

    def test_path1_active_via_new_req_suppresses_1_2(self):
        new = [req("1-3", "MET"), req("1-2", "B1"), req("1-1", "MET")]
        result = apply_path1_filter("chancenkarte", [], new)
        ids = [r["id"] for r in result]
        assert "1-2" not in ids

    def test_path1_suppresses_req_2_1(self):
        """Path 1 must also strip REQ:2-1 (German language points)."""
        state = [req("1-3", "MET")]
        new = [req("2-1", "C1|4"), req("1-1", "MET")]
        result = apply_path1_filter("chancenkarte", state, new)
        ids = [r["id"] for r in result]
        assert "2-1" not in ids
        assert "1-1" in ids

    def test_path1_suppresses_req_2_7(self):
        """Path 1 must also strip REQ:2-7 (English C1 bonus)."""
        state = [req("1-3", "MET")]
        new = [req("2-7", "EN_C1|1"), req("1-1", "MET")]
        result = apply_path1_filter("chancenkarte", state, new)
        ids = [r["id"] for r in result]
        assert "2-7" not in ids
        assert "1-1" in ids

    def test_path1_suppresses_all_language_ids_together(self):
        """All three language-related IDs are stripped in one pass."""
        state = [req("1-3", "MET")]
        new = [req("1-2", "C1"), req("2-1", "C1|4"), req("2-7", "EN_C1|1"), req("1-1", "MET")]
        result = apply_path1_filter("chancenkarte", state, new)
        ids = [r["id"] for r in result]
        assert "1-2" not in ids
        assert "2-1" not in ids
        assert "2-7" not in ids
        assert "1-1" in ids

    def test_path1_requires_met_value(self):
        """REQ:1-3 with value != MET does not activate Path 1."""
        state = [req("1-3", "PARTIAL", status="warning")]
        new = [req("1-2", "B1"), req("2-1", "B1|2")]
        result = apply_path1_filter("chancenkarte", state, new)
        ids = [r["id"] for r in result]
        assert "1-2" in ids
        assert "2-1" in ids


# ---------------------------------------------------------------------------
# apply_feg_path_a_filter
# ---------------------------------------------------------------------------


class TestApplyFegPathAFilter:
    def test_non_skilled_worker_passthrough(self):
        """Filter only applies to skilled_worker (FEG)."""
        new = [req("1", "MET"), req("4", "TBC", status="warning")]
        assert apply_feg_path_a_filter("blue_card", [], new) == new

    def test_path_a_not_active_passthrough(self):
        """Without REQ:1:MET, Path A is not active — REQ:4 kept."""
        new = [req("1", "TBC", status="warning"), req("4", "TBC", status="warning")]
        result = apply_feg_path_a_filter("skilled_worker", [], new)
        assert result == new

    def test_path_a_active_via_new_strips_pending_language(self):
        """REQ:1:MET this turn → drop pending REQ:4:warning (Path A is language-exempt)."""
        new = [req("1", "MET"), req("4", "TBC", status="warning")]
        result = apply_feg_path_a_filter("skilled_worker", [], new)
        ids = [r["id"] for r in result]
        assert "4" not in ids
        assert "1" in ids

    def test_path_a_active_via_state_strips_pending_language(self):
        """REQ:1:MET from CURRENT_UI_STATE also activates the filter."""
        state = [req("1", "MET")]
        new = [req("4", "TBC", status="warning"), req("2", "TBC", status="warning")]
        result = apply_feg_path_a_filter("skilled_worker", state, new)
        ids = [r["id"] for r in result]
        assert "4" not in ids
        assert "2" in ids

    def test_confirmed_a2_preserved(self):
        """A completed Path B (REQ:4:A2:required) must NOT be stripped even if REQ:1:MET."""
        state = [req("1", "MET")]
        new = [req("4", "A2", status="required")]
        result = apply_feg_path_a_filter("skilled_worker", state, new)
        assert new[0] in result

    def test_path_b_in_progress_keeps_language(self):
        """Path B in progress (REQ:1:TBC) keeps its pending REQ:4:TBC."""
        new = [req("1", "TBC", status="warning"), req("4", "TBC", status="warning")]
        result = apply_feg_path_a_filter("skilled_worker", [], new)
        ids = [r["id"] for r in result]
        assert "4" in ids


# ---------------------------------------------------------------------------
# apply_milestone2_filter
# ---------------------------------------------------------------------------


class TestApplyMilestone2Filter:
    def test_non_visa_type_passthrough(self):
        new_m = [milestone("1")]
        assert apply_milestone2_filter(None, [], [], [], new_m) == new_m

    def test_milestone2_already_in_state(self):
        state_m = [milestone("2")]
        new_m = [milestone("1")]
        result = apply_milestone2_filter("chancenkarte", [], state_m, [], new_m)
        assert result == new_m  # no injection

    def test_chancenkarte_path1_fires_when_fin_and_qual_met(self):
        state_r = [req("1-1", "MET"), req("1-3", "MET")]
        result = apply_milestone2_filter("chancenkarte", state_r, [], [], [])
        ids = [m["id"] for m in result]
        assert "2" in ids

    def test_chancenkarte_path1_no_fire_when_fin_missing(self):
        state_r = [req("1-3", "MET"), req("1-1", "TBC", status="warning")]
        result = apply_milestone2_filter("chancenkarte", state_r, [], [], [])
        ids = [m["id"] for m in result]
        assert "2" not in ids

    def test_chancenkarte_path2_fires_when_all_three_met(self):
        state_r = [req("1-1", "MET"), req("1-2", "B1"), req("1-3", "MET")]
        result = apply_milestone2_filter("chancenkarte", state_r, [], [], [])
        ids = [m["id"] for m in result]
        assert "2" in ids

    def test_chancenkarte_path2_no_fire_without_language(self):
        # Path 2: 1-3 not MET (partial recognition) → language REQ:1-2 required but missing
        state_r = [req("1-1", "MET"), req("1-3", "PARTIAL", status="warning")]
        result = apply_milestone2_filter("chancenkarte", state_r, [], [], [])
        ids = [m["id"] for m in result]
        assert "2" not in ids

    def test_milestone1_suppressed_on_milestone2_injection(self):
        state_r = [req("1-1", "MET"), req("1-3", "MET")]
        new_m = [milestone("1")]
        result = apply_milestone2_filter("chancenkarte", state_r, [], [], new_m)
        ids = [m["id"] for m in result]
        assert "1" not in ids
        assert "2" in ids

    def test_blue_card_fires_when_qual_and_salary_met(self):
        state_r = [req("1", "MET"), req("2", "SALARY_MET")]
        result = apply_milestone2_filter("blue_card", state_r, [], [], [])
        assert any(m["id"] == "2" for m in result)

    def test_blue_card_no_fire_without_salary(self):
        state_r = [req("1", "MET")]
        result = apply_milestone2_filter("blue_card", state_r, [], [], [])
        assert not any(m["id"] == "2" for m in result)
