"""Unit tests for the sparse BM25 encoder."""

from unittest.mock import patch

import pytest
from qdrant_client.http.models import SparseVector

from src.vector_db.sparse_encoder import SparseEncoder, get_sparse_encoder


@pytest.fixture
def encoder():
    return SparseEncoder(vocab_size=30_000)


class TestSparseEncoderBasics:
    def test_empty_string_returns_empty_vector(self, encoder):
        result = encoder.encode("")
        assert isinstance(result, SparseVector)
        assert result.indices == []
        assert result.values == []

    def test_whitespace_only_returns_empty_vector(self, encoder):
        result = encoder.encode("   ")
        assert result.indices == []
        assert result.values == []

    def test_normal_text_returns_nonempty_vector(self, encoder):
        result = encoder.encode("Chancenkarte requirements Germany")
        assert len(result.indices) > 0
        assert len(result.values) == len(result.indices)

    def test_indices_and_values_same_length(self, encoder):
        for text in ["short", "a longer German legal text", "§18c AufenthG"]:
            result = encoder.encode(text)
            assert len(result.indices) == len(result.values), f"Mismatch for: {text}"

    def test_values_are_positive(self, encoder):
        result = encoder.encode("test document with multiple words")
        assert all(v > 0 for v in result.values)

    def test_indices_are_non_negative_integers(self, encoder):
        result = encoder.encode("test phrase")
        assert all(isinstance(i, int) and i >= 0 for i in result.indices)

    def test_indices_within_vocab_size(self, encoder):
        result = encoder.encode("a longer text with many unique tokens here")
        assert all(i < encoder.vocab_size for i in result.indices)


class TestSparseEncoderDeterminism:
    def test_same_text_produces_same_indices(self, encoder):
        text = "Niederlassungserlaubnis Chancenkarte §18c"
        r1 = encoder.encode(text)
        r2 = encoder.encode(text)
        assert r1.indices == r2.indices
        assert r1.values == r2.values

    def test_indices_are_sorted(self, encoder):
        result = encoder.encode("some random text for testing purposes")
        assert result.indices == sorted(result.indices)


class TestSparseEncoderGermanTerms:
    def test_handles_german_umlauts(self, encoder):
        """German characters ä, ö, ü, ß must be handled without error."""
        result = encoder.encode("Aufenthaltserlaubnis für qualifizierte Beschäftigung")
        assert len(result.indices) > 0

    def test_handles_legal_code(self, encoder):
        """Legal codes like §18c should produce tokens and not crash."""
        result = encoder.encode("§18c AufenthG Niederlassungserlaubnis")
        assert len(result.indices) > 0

    def test_different_texts_produce_different_vectors(self, encoder):
        r1 = encoder.encode("Chancenkarte requirements")
        r2 = encoder.encode("Niederlassungserlaubnis conditions")
        # The sets of indices should differ
        assert set(r1.indices) != set(r2.indices)


class TestSparseEncoderEmptyTokens:
    def test_all_single_char_tokens_returns_empty_vector(self, encoder):
        """Line 84: text that tokenizes to nothing (all tokens < 2 chars) → empty SparseVector."""
        # "a b c" produces tokens ['a','b','c'], all filtered (len < 2), extra also empty
        result = encoder.encode("a b c")
        assert result.indices == []
        assert result.values == []


class TestSparseEncoderBatch:
    def test_batch_encoding_matches_single(self, encoder):
        texts = ["first text", "second text", "§18c AufenthG"]
        batch_results = encoder.encode_batch(texts)
        for text, batch_result in zip(texts, batch_results):
            single_result = encoder.encode(text)
            assert batch_result.indices == single_result.indices
            assert batch_result.values == single_result.values

    def test_batch_empty_list(self, encoder):
        result = encoder.encode_batch([])
        assert result == []

    def test_batch_preserves_order(self, encoder):
        texts = ["alpha", "beta", "gamma"]
        results = encoder.encode_batch(texts)
        assert len(results) == 3
        for i, (text, result) in enumerate(zip(texts, results)):
            expected = encoder.encode(text)
            assert result.indices == expected.indices


class TestSparseEncoderBatchError:
    def test_encode_exception_returns_empty_vector(self, encoder):
        """Lines 117-119: encode raises → batch catches and returns empty SparseVector."""
        with patch.object(encoder, "encode", side_effect=RuntimeError("encode failed")):
            results = encoder.encode_batch(["some text"])
        assert len(results) == 1
        assert results[0].indices == []
        assert results[0].values == []


class TestSparseEncoderSingleton:
    def test_get_sparse_encoder_returns_instance(self):
        enc = get_sparse_encoder()
        assert isinstance(enc, SparseEncoder)

    def test_get_sparse_encoder_is_singleton(self):
        enc1 = get_sparse_encoder()
        enc2 = get_sparse_encoder()
        assert enc1 is enc2
