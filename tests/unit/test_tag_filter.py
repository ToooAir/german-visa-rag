"""Unit tests for src/rag/tag_filter.py."""

from src.rag.tag_filter import apply_milestone2_filter, apply_path1_filter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def req(id_, value, status="required"):
    return {"id": id_, "value": value, "status": status}


def milestone(id_, status="current"):
    return {"id": id_, "status": status}


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
