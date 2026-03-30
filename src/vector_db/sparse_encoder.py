"""
Lightweight BM25-style sparse vector encoder.

Generates sparse token-frequency vectors for hybrid search with Qdrant.
Uses a hash-based vocabulary (no training corpus needed) to map tokens
to integer indices. Designed for German/English/Chinese legal text.
"""

import re
from math import log

from qdrant_client.http.models import SparseVector

from src.logger import logger


class SparseEncoder:
    r"""
    Lightweight sparse vector encoder using TF-IDF-style term weighting.

    Converts text to a SparseVector (indices, values) for Qdrant hybrid search.
    Uses a hash-based vocabulary with configurable size to avoid collisions.

    Key design:
    - No external models or training data required
    - Deterministic: same text always produces same vector
    - Multilingual: handles German, English (regex \w+ covers unicode)
    - Sub-word n-grams: adds character 2-grams for terms < 4 chars (handles
      legal codes like "§18c") and prefix tokens for prefixes of long words
    """

    def __init__(self, vocab_size: int = 30_000):
        """
        Args:
            vocab_size: Hash table size. Higher = fewer collisions but more
                        memory. 30k is a good tradeoff for < 10k documents.
        """
        self.vocab_size = vocab_size

    def _tokenize(self, text: str) -> list[str]:
        r"""
        Extract tokens from text.

        Applies:
        - Lowercasing
        - Unicode word tokenization (handles German umlauts: ä, ö, ü, ß)
        - Filters short stop-like tokens (< 2 chars)
        - Adds sub-word tokens for short/special terms like §18c, ICT-Karte
        """
        text = text.lower()
        # Primary tokens: unicode word chars (handles ä, ö, ü, ß, numbers, §)
        tokens = re.findall(r"[\w§]+", text, re.UNICODE)

        extra = []
        for tok in tokens:
            if len(tok) >= 4:
                # Add word prefix for partial matching (e.g. "niederlassung" catches "niederlassungserlaubnis")
                extra.append(tok[:4] + "__")
            elif len(tok) >= 2:
                # 2-char bigrams for short legal codes (§18, 18c, etc.)
                for i in range(len(tok) - 1):
                    extra.append(tok[i : i + 2] + "_bg")

        all_tokens = [t for t in tokens if len(t) >= 2] + extra
        return all_tokens

    def _token_to_index(self, token: str) -> int:
        """Map a token string to a stable integer index via hash."""
        return abs(hash(token)) % self.vocab_size

    def encode(self, text: str) -> SparseVector:
        """
        Encode text to a Qdrant SparseVector.

        Returns:
            SparseVector with indices (token hashes) and values (TF scores).
            Returns empty SparseVector if text is empty.
        """
        if not text or not text.strip():
            return SparseVector(indices=[], values=[])

        tokens = self._tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        # Count term frequencies
        tf: dict[int, float] = {}
        for token in tokens:
            idx = self._token_to_index(token)
            tf[idx] = tf.get(idx, 0.0) + 1.0

        # Normalize TF by document length (augmented TF)
        max_tf = max(tf.values())
        for idx in tf:
            tf[idx] = 0.5 + 0.5 * tf[idx] / max_tf

        # Apply log(1 + tf) smoothing for better score distribution
        indices = sorted(tf.keys())
        values = [round(0.5 + 0.5 * log(1.0 + tf[idx]), 6) for idx in indices]

        return SparseVector(indices=indices, values=values)

    def encode_batch(self, texts: list[str]) -> list[SparseVector]:
        """
        Encode a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            List of SparseVectors in the same order as input
        """
        results = []
        for text in texts:
            try:
                results.append(self.encode(text))
            except Exception as e:
                logger.warning("Sparse encoding failed for text, using empty vector: %s", e)
                results.append(SparseVector(indices=[], values=[]))
        return results


# Module-level singleton
_encoder: SparseEncoder | None = None


def get_sparse_encoder(vocab_size: int = 30_000) -> SparseEncoder:
    """Get or create the global sparse encoder singleton."""
    global _encoder
    if _encoder is None:
        _encoder = SparseEncoder(vocab_size=vocab_size)
        logger.info("Sparse encoder initialized (vocab_size=%d)", vocab_size)
    return _encoder
