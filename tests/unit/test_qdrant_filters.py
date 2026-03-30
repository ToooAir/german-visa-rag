"""Unit tests for QdrantWrapper.build_filter_authority_and_visa"""

import pytest
from qdrant_client.http.models import FieldCondition, Filter, MatchAny

from src.models.chunk import AuthorityLevel
from src.vector_db.qdrant_client_wrapper import QdrantWrapper


@pytest.fixture
def wrapper(monkeypatch):
    # Prevent actual Qdrant connections during construction
    monkeypatch.setattr(
        "src.vector_db.qdrant_client_wrapper.AsyncQdrantClient",
        lambda **kw: None,
    )
    monkeypatch.setattr(
        "src.vector_db.qdrant_client_wrapper.QdrantClient",
        lambda **kw: None,
    )
    return QdrantWrapper()


class TestBuildFilterAuthorityAndVisa:
    def test_official_only_includes_official(self, wrapper):
        f = wrapper.build_filter_authority_and_visa(AuthorityLevel.OFFICIAL)
        # Single condition returns a FieldCondition directly
        assert isinstance(f, FieldCondition)
        assert f.match.any == ["official"]

    def test_semi_official_includes_official_and_semi(self, wrapper):
        f = wrapper.build_filter_authority_and_visa(AuthorityLevel.SEMI_OFFICIAL)
        assert isinstance(f, FieldCondition)
        assert set(f.match.any) == {"official", "semi_official"}

    def test_third_party_includes_all_levels(self, wrapper):
        f = wrapper.build_filter_authority_and_visa(AuthorityLevel.THIRD_PARTY)
        assert isinstance(f, FieldCondition)
        assert set(f.match.any) == {"official", "semi_official", "third_party"}

    def test_visa_types_filter_added(self, wrapper):
        f = wrapper.build_filter_authority_and_visa(
            AuthorityLevel.OFFICIAL,
            visa_types=["chancenkarte", "work_visa"],
        )
        # With visa_types, result is a Filter with must conditions
        assert isinstance(f, Filter)
        assert f.must is not None
        assert len(f.must) == 2

        keys = {c.key for c in f.must}
        assert "authority_level" in keys
        assert "visa_types" in keys

    def test_visa_types_uses_match_any(self, wrapper):
        f = wrapper.build_filter_authority_and_visa(
            AuthorityLevel.SEMI_OFFICIAL,
            visa_types=["chancenkarte"],
        )
        visa_cond = next(c for c in f.must if c.key == "visa_types")
        assert isinstance(visa_cond.match, MatchAny)
        assert "chancenkarte" in visa_cond.match.any

    def test_no_visa_types_returns_single_condition(self, wrapper):
        f = wrapper.build_filter_authority_and_visa(AuthorityLevel.OFFICIAL, visa_types=None)
        # No visa filter → single FieldCondition, not a Filter wrapper
        assert isinstance(f, FieldCondition)

    def test_empty_visa_types_no_visa_condition(self, wrapper):
        f = wrapper.build_filter_authority_and_visa(AuthorityLevel.OFFICIAL, visa_types=[])
        assert isinstance(f, FieldCondition)
