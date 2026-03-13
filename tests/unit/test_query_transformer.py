"""Unit tests for query transformer."""

import pytest
from src.rag.query_transformer import get_query_transformer


@pytest.mark.asyncio
async def test_query_detection_visa_types():
    """Test visa type detection in queries."""
    transformer = get_query_transformer()
    
    result = await transformer.transform_query("Chancenkarte 的申請條件？")
    
    assert "chancenkarte" in [vt.lower() for vt in result["detected_visa_types"]]


@pytest.mark.asyncio
async def test_language_detection():
    """Test language detection."""
    transformer = get_query_transformer()
    
    # German
    result_de = await transformer.transform_query("Wie beantrage ich ein Chancenkarte?")
    assert "de" in result_de["languages_detected"]
    
    # English
    result_en = await transformer.transform_query("What is a Chancenkarte?")
    assert "en" in result_en["languages_detected"]
    
    # Chinese
    result_zh = await transformer.transform_query("Chancenkarte 是什麼？")
    assert "zh" in result_zh["languages_detected"]


@pytest.mark.asyncio
async def test_spell_correction():
    """Test spell correction."""
    transformer = get_query_transformer()
    
    result = await transformer.transform_query("chacenkarte aplications")
    
    # Should correct typos
    corrected = result["corrected_query"].lower()
    assert "chancenkarte" in corrected or "application" in corrected
