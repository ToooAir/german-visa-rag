"""Text processing utilities for normalization, deduplication, and extraction."""

import re
from typing import List


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces/newlines."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Preserve paragraph breaks
    text = re.sub(r' \n ', '\n\n', text)
    return text.strip()


def clean_markdown(text: str) -> str:
    """Clean markdown text for processing."""
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Remove inline HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize markdown headers
    text = re.sub(r'^(\#{2,})\s+', r'\1 ', text, flags=re.MULTILINE)
    return normalize_whitespace(text)


def extract_section_title(markdown_text: str, start_pos: int) -> str:
    """Extract the current section title from markdown."""
    lines = markdown_text[:start_pos].split('\n')
    for line in reversed(lines):
        if re.match(r'^#+\s+', line):
            return line.strip()
    return "General"


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
