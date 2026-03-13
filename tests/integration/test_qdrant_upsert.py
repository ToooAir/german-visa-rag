"""Integration logic test for Qdrant payload formatting."""

import pytest
from src.models.chunk import Chunk, ChunkMetadata, AuthorityLevel, VisaType
from src.models.chunk import QdrantPayload

def test_qdrant_payload_conversion():
    """Ensure internal Chunk model correctly converts to Qdrant Payload."""
    chunk = Chunk(
        metadata=ChunkMetadata(
            chunk_id="test_123",
            parent_doc_id="doc_1",
            source_url="http://test.com",
            authority_level=AuthorityLevel.OFFICIAL,
            visa_types=[VisaType.CHANCENKARTE],
            text_hash="abc123hash",
            is_parent=False
        ),
        text="Sample text for Qdrant."
    )
    
    payload = QdrantPayload.from_chunk(chunk)
    payload_dict = payload.to_dict()
    
    assert payload_dict["chunk_id"] == "test_123"
    assert payload_dict["authority_level"] == "official"
    assert payload_dict["visa_types"] == ["chancenkarte"]
    assert "Sample text" in payload_dict["text"]
    assert payload_dict["is_parent"] == False
