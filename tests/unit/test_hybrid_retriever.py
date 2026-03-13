"""Unit tests for Hybrid Retriever weighting logic."""

import pytest
from unittest.mock import AsyncMock, patch
from src.rag.hybrid_retriever import HybridRetriever
from datetime import datetime, timedelta

@pytest.mark.asyncio
async def test_retriever_recency_and_authority_weighting():
    """Test if official and recent documents get higher scores."""
    
    # Mock Qdrant results
    mock_qdrant_results = [
        {
            "id": 1,
            "score": 0.8,
            "payload": {
                "authority_level": "official",
                "fetched_at": datetime.utcnow().isoformat(), # Very recent
                "text": "Doc 1"
            }
        },
        {
            "id": 2,
            "score": 0.8, # Same base score
            "payload": {
                "authority_level": "third_party",
                "fetched_at": (datetime.utcnow() - timedelta(days=200)).isoformat(), # Old
                "text": "Doc 2"
            }
        }
    ]
    
    # Create mock qdrant client
    mock_qdrant = AsyncMock()
    mock_qdrant.hybrid_search.return_value = mock_qdrant_results
    
    # Patch embedder to avoid real API calls
    with patch("src.rag.hybrid_retriever.embedder.embed_single", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.1] * 1536
        
        retriever = HybridRetriever(qdrant_client=mock_qdrant)
        results = await retriever.retrieve("test query")
        
        assert len(results) == 2
        # Doc 1 should be ranked higher due to authority boost (x1.2) and recency
        assert results[0]["id"] == 1
        assert results[1]["id"] == 2
        assert results[0]["adjusted_score"] > results[1]["adjusted_score"]
