"""Text processing utilities for normalization, deduplication, and extraction."""

import re


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces but preserve newlines."""
    # Collapse 3 or more newlines to 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace multiple spaces with a single space (excluding newlines)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_markdown(text: str) -> str:
    """Clean markdown text for processing."""
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Remove markdown images — decorative logos/nav-button icons carry no textual
    # content, yet survive HTML-tag stripping (e.g. gesetze-im-internet.de pads
    # legal text with ![](../img/...) logos and prev/next navigation icons).
    # Must run before link flattening because images share the [..](..) syntax.
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    # Flatten markdown links to their anchor text, dropping the URL. Citations are
    # attached from chunk metadata (source_url), so inline link targets are noise.
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Remove inline HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Normalize markdown headers
    text = re.sub(r"^(\#{2,})\s+", r"\1 ", text, flags=re.MULTILINE)
    return normalize_whitespace(text)


def extract_section_title(markdown_text: str, start_pos: int) -> str:
    """Extract the current section title from markdown."""
    lines = markdown_text[:start_pos].split("\n")
    for line in reversed(lines):
        if re.match(r"^#+\s+", line):
            return line.strip()
    return "General"


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix
