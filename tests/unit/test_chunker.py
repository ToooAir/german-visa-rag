"""Unit tests for the Parent-Child Chunker."""

import src.ingestion.chunker as chunker_module
from src.ingestion.chunker import MIN_INTRO_LENGTH, ParentChildChunker, get_chunker


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


# ─── split_by_headers ────────────────────────────────────────────────────────


class TestSplitByHeaders:
    def test_short_intro_is_dropped(self):
        chunker = ParentChildChunker()
        # intro is too short (< MIN_INTRO_LENGTH = 300 chars)
        markdown = "Short intro.\n## Section\nContent here."
        sections = chunker.split_by_headers(markdown)
        headers = [s[0] for s in sections]
        assert "Introduction" not in headers
        assert "Section" in headers

    def test_no_headers_short_content_is_dropped(self):
        """Entire doc with no headers and short content → nothing kept."""
        chunker = ParentChildChunker()
        markdown = "Just a short line."
        sections = chunker.split_by_headers(markdown)
        assert sections == []

    def test_no_headers_long_content_kept_as_introduction(self):
        """Entire doc with no headers but long content → kept as Introduction."""
        chunker = ParentChildChunker()
        long_text = "x " * (MIN_INTRO_LENGTH // 2 + 10)  # > 300 chars
        sections = chunker.split_by_headers(long_text)
        assert len(sections) == 1
        assert sections[0][0] == "Introduction"

    def test_h3_header_creates_section(self):
        chunker = ParentChildChunker()
        markdown = "### Requirements\nYou need a passport."
        sections = chunker.split_by_headers(markdown)
        assert any(s[0] == "Requirements" for s in sections)

    def test_empty_section_content_not_appended(self):
        """A header with only whitespace content is not added as a section."""
        chunker = ParentChildChunker()
        markdown = "## Empty Section\n   \n## Real Section\nActual content here."
        sections = chunker.split_by_headers(markdown)
        assert not any(s[1].strip() == "" for s in sections)


# ─── _derive_context_label ───────────────────────────────────────────────────


class TestDeriveContextLabel:
    def test_generic_header_uses_title(self):
        chunker = ParentChildChunker()
        label = chunker._derive_context_label("Introduction", "Visa Requirements Guide")
        assert label == "Visa Requirements Guide"

    def test_non_generic_header_returned_as_is(self):
        chunker = ParentChildChunker()
        label = chunker._derive_context_label("Eligibility Criteria", "Visa Guide")
        assert label == "Eligibility Criteria"

    def test_overview_is_generic(self):
        chunker = ParentChildChunker()
        label = chunker._derive_context_label("Overview", "My Title")
        assert label == "My Title"

    def test_generic_header_no_title_returns_default(self):
        chunker = ParentChildChunker()
        label = chunker._derive_context_label("Introduction", "")
        assert label == "Overview"


# ─── split_into_sentences ────────────────────────────────────────────────────


class TestSplitIntoSentences:
    def test_empty_paragraphs_skipped(self):
        chunker = ParentChildChunker()
        text = "Para one.\n\n\n\nPara two."
        result = chunker.split_into_sentences(text, max_size=500)
        assert len(result) == 1  # Both fit in one chunk

    def test_small_paragraphs_combined(self):
        chunker = ParentChildChunker()
        text = "Para one.\n\nPara two.\n\nPara three."
        result = chunker.split_into_sentences(text, max_size=500)
        assert len(result) == 1
        combined = result[0]
        assert "Para one" in combined and "Para two" in combined

    def test_large_paragraph_split_by_sentence(self):
        chunker = ParentChildChunker()
        # Each sentence ~30 chars, max_size=60 → two sentences per chunk
        text = "First sentence here. Second sentence here. Third sentence here."
        result = chunker.split_into_sentences(text, max_size=40)
        assert len(result) >= 2

    def test_paragraph_exceeding_max_size_hard_split(self):
        """A paragraph with no sentence punctuation gets hard-split by character."""
        chunker = ParentChildChunker()
        # 200 chars with no sentence boundaries
        text = "a" * 200
        result = chunker.split_into_sentences(text, max_size=50)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 50

    def test_current_chunk_flushed_at_end(self):
        chunker = ParentChildChunker()
        text = "Short para."
        result = chunker.split_into_sentences(text, max_size=500)
        assert len(result) == 1
        assert "Short para" in result[0]


# ─── chunk_document branches ────────────────────────────────────────────────


class TestChunkDocumentBranches:
    def test_oversized_parent_chunk_is_trimmed(self):
        from src.ingestion.chunker import MAX_CHUNK_LENGTH

        chunker = ParentChildChunker(min_child_length=10)
        # Create a section whose parent_text exceeds MAX_CHUNK_LENGTH
        huge_content = "word " * (MAX_CHUNK_LENGTH // 4)
        markdown = f"## Big Section\n{huge_content}"
        chunks = chunker.chunk_document(
            markdown_text=markdown,
            source_url="http://test.com",
            doc_id="doc_big",
            title="Big Doc",
        )
        parents = [c for c in chunks if c.metadata.is_parent]
        assert len(parents) == 1
        assert "[Truncated]" in parents[0].text

    def test_short_parent_section_skipped(self):
        chunker = ParentChildChunker(min_parent_length=200)
        # Section content that produces a parent_text < 200 chars
        markdown = "## Tiny\nHi."
        chunks = chunker.chunk_document(
            markdown_text=markdown,
            source_url="http://test.com",
            doc_id="doc_tiny",
            title="Tiny Doc",
        )
        assert chunks == []

    def test_short_child_chunk_skipped(self):
        chunker = ParentChildChunker(min_child_length=1000, min_parent_length=10)
        # Make sure parent passes but child is shorter than 1000
        content = "This is a short child." * 3  # ~66 chars — under 1000
        markdown = f"## Section\n{content}"
        chunks = chunker.chunk_document(
            markdown_text=markdown,
            source_url="http://test.com",
            doc_id="doc_x",
            title="X",
        )
        # Parent exists but no children
        assert any(c.metadata.is_parent for c in chunks)
        assert not any(not c.metadata.is_parent for c in chunks)

    def test_extract_urls_from_child(self):
        chunker = ParentChildChunker(min_child_length=10)
        url = "https://example.com/visa"
        content = ("word " * 30) + f" See {url} for details."
        markdown = f"## Links\n{content}"
        chunks = chunker.chunk_document(
            markdown_text=markdown,
            source_url="http://test.com",
            doc_id="doc_url",
            title="URL Doc",
        )
        children = [c for c in chunks if not c.metadata.is_parent]
        all_urls = [u for c in children for u in c.metadata.referenced_urls]
        assert url in all_urls

    def test_returns_empty_for_blank_markdown(self):
        chunker = ParentChildChunker()
        chunks = chunker.chunk_document(
            markdown_text="   ",
            source_url="http://test.com",
            doc_id="doc_blank",
            title="Blank",
        )
        assert chunks == []


# ─── clean_markdown ───────────────────────────────────────────────────────────


class TestCleanMarkdown:
    def test_removes_images(self):
        chunker = ParentChildChunker()
        text = "Before ![alt](http://img.com/a.png) After"
        result = chunker.clean_markdown(text)
        assert "![" not in result
        assert "Before" in result

    def test_removes_social_share_links(self):
        chunker = ParentChildChunker()
        text = "Content. [Teilen auf LinkedIn](https://linkedin.com/share) More."
        result = chunker.clean_markdown(text)
        assert "Teilen auf" not in result

    def test_removes_download_links(self):
        chunker = ParentChildChunker()
        text = "See [Download the form](https://example.com/form.pdf) here."
        result = chunker.clean_markdown(text)
        assert "Download" not in result


# ─── get_chunker singleton ────────────────────────────────────────────────────


class TestGetChunker:
    def setup_method(self):
        chunker_module._chunker = None

    def teardown_method(self):
        chunker_module._chunker = None

    def test_returns_same_instance(self):
        a = get_chunker()
        b = get_chunker()
        assert a is b

    def test_returns_parent_child_chunker(self):
        assert isinstance(get_chunker(), ParentChildChunker)
