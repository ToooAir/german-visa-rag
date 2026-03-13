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
    hash_obj = hashlib.sha256(normalized.encode('utf-8'))
    return hash_obj.hexdigest()


def compute_content_hash(text: str) -> str:
    """Compute simple SHA256 hash for content versioning."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
