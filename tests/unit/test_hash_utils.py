"""Unit tests for src/utils/hash_utils.py"""

from src.utils.hash_utils import compute_canonical_hash, compute_content_hash


class TestComputeCanonicalHash:
    def test_is_deterministic(self):
        assert compute_canonical_hash("hello world") == compute_canonical_hash("hello world")

    def test_case_insensitive(self):
        assert compute_canonical_hash("Hello World") == compute_canonical_hash("hello world")

    def test_whitespace_normalized(self):
        assert compute_canonical_hash("hello   world") == compute_canonical_hash("hello world")

    def test_different_text_differs(self):
        assert compute_canonical_hash("foo") != compute_canonical_hash("bar")

    def test_returns_hex_string(self):
        result = compute_canonical_hash("test")
        assert len(result) == 64  # SHA-256 hex = 64 chars
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_string(self):
        result = compute_canonical_hash("")
        assert len(result) == 64


class TestComputeContentHash:
    def test_is_deterministic(self):
        assert compute_content_hash("hello") == compute_content_hash("hello")

    def test_case_sensitive(self):
        # Unlike canonical hash, content hash should be case-sensitive
        assert compute_content_hash("Hello") != compute_content_hash("hello")

    def test_whitespace_sensitive(self):
        # Content hash preserves exact spacing
        assert compute_content_hash("a b") != compute_content_hash("a  b")

    def test_returns_hex_string(self):
        result = compute_content_hash("test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)
