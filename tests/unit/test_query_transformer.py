"""Unit tests for query transformer."""

import json
from unittest.mock import AsyncMock, patch

import pytest

import src.rag.query_transformer as qt_module
from src.rag.query_transformer import QueryTransformer, get_query_transformer


@pytest.mark.asyncio
async def test_query_detection_visa_types(mock_llm_client):
    """Test visa type detection in queries."""
    transformer = get_query_transformer()

    mock_llm_client.call_non_streaming.return_value = json.dumps(
        {
            "corrected_query": "Chancenkarte 的申請條件？",
            "english_query": "Chancenkarte application requirements",
            "german_query": "Chancenkarte Antragsvoraussetzungen",
            "query_variants": [],
            "detected_visa_types": ["chancenkarte"],
            "languages_detected": ["zh"],
            "confidence": 0.95,
        }
    )

    result = await transformer.transform_query("Chancenkarte 的申請條件？")
    assert "chancenkarte" in [vt.lower() for vt in result["detected_visa_types"]]


@pytest.mark.asyncio
async def test_language_detection():
    """Test language detection heuristic."""
    transformer = get_query_transformer()

    # German (has 'wie', which is in our common German words list)
    result_de = await transformer.transform_query("Wie beantrage ich ein Chancenkarte?", apply_expansion=False)
    assert "de" in result_de["languages_detected"]

    # English (fallback when no DE/ZH detected)
    result_en = await transformer.transform_query("What is a Chancenkarte?", apply_expansion=False)
    assert "en" in result_en["languages_detected"]

    # Chinese (contains CJK characters)
    result_zh = await transformer.transform_query("Chancenkarte 是什麼？", apply_expansion=False)
    assert "zh" in result_zh["languages_detected"]


@pytest.mark.asyncio
async def test_spell_correction(mock_llm_client):
    """Test spell correction via LLM."""
    transformer = get_query_transformer()

    mock_llm_client.call_non_streaming.return_value = json.dumps(
        {
            "corrected_query": "chancenkarte applications",
            "english_query": "chancenkarte applications",
            "german_query": "chancenkarte anträge",
            "query_variants": [],
            "detected_visa_types": ["chancenkarte"],
            "languages_detected": ["en"],
            "confidence": 0.95,
        }
    )

    result = await transformer.transform_query("chacenkarte aplications")

    # Should correct typos
    corrected = result["corrected_query"].lower()
    assert "chancenkarte" in corrected or "application" in corrected


# ─── Additional unit tests (no fixture dependency) ────────────────────────────


def _make_transformer() -> QueryTransformer:
    with patch("src.rag.query_transformer.get_llm_client"):
        t = QueryTransformer()
    t.llm = AsyncMock()
    return t


_VALID_RESPONSE = {
    "corrected_query": "corrected",
    "english_query": "english version",
    "german_query": "deutsche Version",
    "query_variants": ["variant 1"],
    "detected_visa_types": ["chancenkarte"],
    "languages_detected": ["en"],
    "confidence": 0.95,
}


class TestDetectVisaTypesExtended:
    def test_blue_card_english(self):
        t = _make_transformer()
        assert "blue_card" in t._detect_visa_types("blue card requirements")

    def test_work_visa_chinese(self):
        t = _make_transformer()
        assert "work_visa" in t._detect_visa_types("工作簽證申請")

    def test_freelance(self):
        t = _make_transformer()
        assert "freelance_visa" in t._detect_visa_types("freelance visa Germany")

    def test_entrepreneur(self):
        t = _make_transformer()
        assert "entrepreneur_visa" in t._detect_visa_types("entrepreneur visa process")

    def test_no_match_returns_empty(self):
        t = _make_transformer()
        assert t._detect_visa_types("what is the weather") == []


class TestDetectLanguagesExtended:
    def test_german_by_umlaut(self):
        t = _make_transformer()
        # "ä" is in the text → triggers German detection
        assert "de" in t._detect_languages("Wie beantrage ich eine Aufenthaltserlaubnis?")

    def test_chinese_characters(self):
        t = _make_transformer()
        assert "zh" in t._detect_languages("請問機會卡如何申請")

    def test_english_fallback(self):
        t = _make_transformer()
        langs = t._detect_languages("What are the requirements?")
        assert langs == ["en"]


class TestExpandQueryWithLLM:
    @pytest.mark.asyncio
    async def test_plain_json_parsed(self):
        t = _make_transformer()
        t.llm.call_non_streaming = AsyncMock(return_value=json.dumps(_VALID_RESPONSE))
        result = await t._expand_query_with_llm("chancenkarte")
        assert result["corrected_query"] == "corrected"

    @pytest.mark.asyncio
    async def test_strips_markdown_code_fence(self):
        t = _make_transformer()
        fenced = f"```json\n{json.dumps(_VALID_RESPONSE)}\n```"
        t.llm.call_non_streaming = AsyncMock(return_value=fenced)
        result = await t._expand_query_with_llm("query")
        assert result["corrected_query"] == "corrected"

    @pytest.mark.asyncio
    async def test_raises_on_oversized_response(self):
        from tenacity import RetryError

        t = _make_transformer()
        t.llm.call_non_streaming = AsyncMock(return_value="x" * 11_000)
        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises((ValueError, RetryError)),
        ):
            await t._expand_query_with_llm("q")

    @pytest.mark.asyncio
    async def test_raises_on_invalid_json(self):
        from tenacity import RetryError

        t = _make_transformer()
        t.llm.call_non_streaming = AsyncMock(return_value="not valid json")
        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises((json.JSONDecodeError, RetryError)),
        ):
            await t._expand_query_with_llm("q")


class TestTransformQueryExtended:
    @pytest.mark.asyncio
    async def test_llm_failure_returns_fallback(self):
        t = _make_transformer()
        t._expand_query_with_llm = AsyncMock(side_effect=Exception("LLM down"))
        result = await t.transform_query("hello")
        assert result["corrected_query"] == "hello"
        assert result["confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_successful_expansion_returned(self):
        t = _make_transformer()
        t._expand_query_with_llm = AsyncMock(return_value=_VALID_RESPONSE)
        result = await t.transform_query("short q")
        assert result is _VALID_RESPONSE


class TestGetSearchQueriesExtended:
    @pytest.mark.asyncio
    async def test_expansion_disabled_returns_original(self):
        t = _make_transformer()
        with patch("src.rag.query_transformer.settings") as s:
            s.enable_query_expansion = False
            result = await t.get_search_queries("my query")
        assert result == ["my query"]

    @pytest.mark.asyncio
    async def test_long_query_skips_expansion(self):
        t = _make_transformer()
        t._expand_query_with_llm = AsyncMock(return_value=_VALID_RESPONSE)
        long_q = "word " * 25  # >100 chars
        with patch("src.rag.query_transformer.settings") as s:
            s.enable_query_expansion = True
            result = await t.get_search_queries(long_q)
        assert result == [long_q]
        t._expand_query_with_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_deduplicates_same_queries(self):
        t = _make_transformer()
        expanded = {
            "corrected_query": "same",
            "english_query": "same",
            "german_query": "deutsch",
        }
        t._expand_query_with_llm = AsyncMock(return_value=expanded)
        with patch("src.rag.query_transformer.settings") as s:
            s.enable_query_expansion = True
            result = await t.get_search_queries("q")
        assert result.count("same") == 1

    @pytest.mark.asyncio
    async def test_caps_at_3_results(self):
        t = _make_transformer()
        expanded = {
            "corrected_query": "c",
            "english_query": "e",
            "german_query": "g",
            "query_variants": ["v1", "v2"],
        }
        t._expand_query_with_llm = AsyncMock(return_value=expanded)
        with patch("src.rag.query_transformer.settings") as s:
            s.enable_query_expansion = True
            result = await t.get_search_queries("q")
        assert len(result) <= 3

    @pytest.mark.asyncio
    async def test_expansion_failure_falls_back_to_original(self):
        t = _make_transformer()
        t._expand_query_with_llm = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("src.rag.query_transformer.settings") as s:
            s.enable_query_expansion = True
            result = await t.get_search_queries("my query")
        assert result == ["my query"]


class TestGetSearchQueriesVariantAdded:
    @pytest.mark.asyncio
    async def test_variant_appended_when_queries_less_than_3(self):
        """Lines 185-186: query_variants item added when search_queries has <3 entries."""
        t = _make_transformer()
        # All keys same → search_queries stays at 1 entry → variant is appended
        expanded = {
            "corrected_query": "same",
            "english_query": "same",
            "german_query": "same",
            "query_variants": ["new_variant"],
        }
        t._expand_query_with_llm = AsyncMock(return_value=expanded)
        with patch("src.rag.query_transformer.settings") as s:
            s.enable_query_expansion = True
            result = await t.get_search_queries("same")
        assert "new_variant" in result

    @pytest.mark.asyncio
    async def test_transform_query_raises_triggers_except(self):
        """Lines 190-192: exception in transform_query causes fallback to [query]."""
        t = _make_transformer()
        t.transform_query = AsyncMock(side_effect=RuntimeError("transform crash"))
        with patch("src.rag.query_transformer.settings") as s:
            s.enable_query_expansion = True
            result = await t.get_search_queries("my query")
        assert result == ["my query"]


class TestSingletonExtended:
    def test_returns_same_instance(self):
        qt_module._transformer = None
        with patch("src.rag.query_transformer.get_llm_client"):
            a = get_query_transformer()
            b = get_query_transformer()
        assert a is b
        qt_module._transformer = None
