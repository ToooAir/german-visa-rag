"""Unit tests for the Parent-Child Chunker."""

from src.ingestion.chunker import ParentChildChunker


def test_split_by_headers():
    """Test if markdown is correctly split by H2/H3 headers."""
    chunker = ParentChildChunker()
    # Intro must be >= 300 chars to be kept
    intro = "This is a long introduction to ensure it passes the 300-character threshold filter. " * 5
    markdown = f"{intro}\n## Section 1\nContent 1\n### Section 2\nContent 2"

    sections = chunker.split_by_headers(markdown)

    assert len(sections) == 3
    assert sections[0][0] == "Introduction"
    assert "introduction" in sections[0][1].lower()
    assert sections[1][0] == "Section 1"
    assert "Content 1" in sections[1][1]


def test_chunk_document_parent_child_relationship():
    """Test if parent and child chunks are correctly created."""
    chunker = ParentChildChunker(child_chunk_size=50, min_child_length=10)
    # Parent text (header + content) must be >= 100
    # Child must be >= 150 (since we didn't override min_child_length in this test specifically to lower values)
    # Wait, the test uses min_child_length=10... but the DEFAULT is 150.
    # ParentChildChunker constructor: min_child_length=min_child_length (passed as 10)
    # So if the test passed 10, it should be fine.
    # Let me check the failure again: "assert 0 == 1".
    # Ah, ParentChildChunker.__init__: min_child_length=min_child_length (10)
    # BUT, parent_text length check in chunk_document uses self.min_parent_length (default 100).
    # markdown = "## Visa Rules\nThis is sentence one. This is sentence two. This is sentence three."
    # "Visa Rules\n\nThis is sentence one. This is sentence two. This is sentence three." -> 75 chars.
    # 75 < 100 -> dropped!
    content = (
        "This is a much longer sentence to ensure that our total parent chunk length exceeds the 100 character threshold. "
        * 2
    )
    markdown = f"## Visa Rules\n{content}"

    chunks = chunker.chunk_document(
        markdown_text=markdown, source_url="http://test.com", doc_id="doc_123", title="Test Doc"
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
