"""Unit tests for query transformer."""

import json

import pytest

from src.rag.query_transformer import get_query_transformer


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
