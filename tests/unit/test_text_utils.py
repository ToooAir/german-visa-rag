"""Unit tests for src/utils/text_utils.py"""

from src.utils.text_utils import (
    clean_markdown,
    extract_section_title,
    normalize_whitespace,
    truncate_text,
)


class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("hello   world") == "hello world"

    def test_collapses_tabs_to_space(self):
        assert normalize_whitespace("hello\t\tworld") == "hello world"

    def test_preserves_single_newline(self):
        assert normalize_whitespace("line1\nline2") == "line1\nline2"

    def test_collapses_excess_newlines_to_two(self):
        assert normalize_whitespace("a\n\n\n\nb") == "a\n\nb"

    def test_strips_leading_trailing_whitespace(self):
        assert normalize_whitespace("  hello  ") == "hello"

    def test_empty_string(self):
        assert normalize_whitespace("") == ""

    def test_only_whitespace(self):
        assert normalize_whitespace("   \n\n  ") == ""


class TestCleanMarkdown:
    def test_removes_html_comments(self):
        assert clean_markdown("hello <!-- comment --> world") == "hello world"

    def test_removes_multiline_html_comments(self):
        result = clean_markdown("before\n<!-- multi\nline -->\nafter")
        assert "<!--" not in result
        assert "after" in result

    def test_removes_inline_html_tags(self):
        assert clean_markdown("hello <b>world</b>") == "hello world"

    def test_normalizes_headers(self):
        result = clean_markdown("##  Too Many Spaces")
        assert result == "## Too Many Spaces"

    def test_passes_through_clean_text(self):
        text = "## Title\n\nSome text here."
        assert clean_markdown(text) == text

    def test_removes_markdown_images(self):
        assert clean_markdown("![logo](../img/lay/BMJV.svg) Real content") == "Real content"

    def test_flattens_links_to_anchor_text(self):
        assert clean_markdown("See [§ 18g AufenthG](__18g.html) now") == "See § 18g AufenthG now"

    def test_strips_gesetze_navigation_boilerplate(self):
        """gesetze-im-internet.de wraps a nav icon inside a link — keep only the label."""
        nav = '[weiter![next](../img/button/p_right.gif "next")](__18h.html "to next")'
        assert clean_markdown(nav) == "weiter"

    def test_keeps_link_label_when_url_dropped(self):
        result = clean_markdown("[Startseite](../index.html) | body")
        assert "Startseite | body" == result
        assert "index.html" not in result


class TestExtractSectionTitle:
    def test_finds_nearest_header(self):
        md = "# H1\n\nsome text\n\n## H2\n\nmore text"
        # start_pos points into "more text" — should find H2
        pos = md.index("more text")
        assert extract_section_title(md, pos) == "## H2"

    def test_returns_general_when_no_header(self):
        md = "just plain text without headers"
        assert extract_section_title(md, len(md)) == "General"

    def test_finds_h1_when_only_header(self):
        md = "# Title\n\nsome content"
        pos = md.index("some content")
        assert extract_section_title(md, pos) == "# Title"


class TestTruncateText:
    def test_no_truncation_when_short_enough(self):
        assert truncate_text("hello", 10) == "hello"

    def test_truncates_to_max_length(self):
        result = truncate_text("hello world", 8)
        assert len(result) == 8
        assert result.endswith("...")

    def test_exact_length_not_truncated(self):
        assert truncate_text("hello", 5) == "hello"

    def test_custom_suffix(self):
        result = truncate_text("hello world", 7, suffix="…")
        assert result.endswith("…")
        assert len(result) == 7
