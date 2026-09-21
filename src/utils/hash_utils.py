"""Hashing utilities for canonical deduplication."""

import hashlib


def compute_canonical_hash(text: str) -> str:
    """
    Compute canonical hash for deduplication.

    Normalized to handle minor formatting differences:
    - Case-insensitive
    - Whitespace normalized
    - Special chars standardized
    """
    # Normalize whitespace
    normalized = " ".join(text.lower().split())

    # Compute SHA-256
    hash_obj = hashlib.sha256(normalized.encode("utf-8"))
    return hash_obj.hexdigest()


def compute_content_hash(text: str) -> str:
    """Compute simple SHA256 hash for content versioning."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_query_fingerprint(text: str, length: int = 12) -> str:
    """Short, stable id for a user query — safe to log and to track.

    User queries carry personal data (degree, salary, nationality, age), so
    observability records this fingerprint instead of the text. The same
    question always yields the same fingerprint, so repeats still group across
    log lines and MLflow runs without persisting what anyone actually asked.
    """
    return compute_canonical_hash(text)[:length]
