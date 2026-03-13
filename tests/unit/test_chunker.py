"""Unit tests for the Parent-Child Chunker."""

import pytest
from src.ingestion.chunker import ParentChildChunker
from src.models.chunk import AuthorityLevel

def test_split_by_headers():
    """Test if markdown is correctly split by H2/H3 headers."""
    chunker = ParentChildChunker()
    markdown = "Intro text\n## Section 1\nContent 1\n### Section 2\nContent 2"
    
    sections = chunker.split_by_headers(markdown)
    
    assert len(sections) == 3
    assert sections[0][0] == "Introduction"  # Default header
    assert "Intro text" in sections[0][1]
    assert sections[1][0] == "Section 1"
    assert "Content 1" in sections[1][1]

def test_chunk_document_parent_child_relationship():
    """Test if parent and child chunks are correctly created."""
    chunker = ParentChildChunker(child_chunk_size=50)
    markdown = "## Visa Rules\nThis is sentence one. This is sentence two. This is sentence three."
    
    chunks = chunker.chunk_document(
        markdown_text=markdown,
        source_url="http://test.com",
        doc_id="doc_123",
        title="Test Doc"
    )
    
    # Check parent chunk
    parent_chunks = [c for c in chunks if c.metadata.is_parent]
    assert len(parent_chunks) == 1
    assert "Visa Rules" in parent_chunks[0].text
    
    # Check child chunks
    child_chunks = [c for c in chunks if not c.metadata.is_parent]
    assert len(child_chunks) > 0
    
    # Verify linking
    for child in child_chunks:
        assert child.metadata.parent_doc_id == "doc_123"
        assert child.metadata.section_header == "Visa Rules"
