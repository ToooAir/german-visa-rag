"""Unit tests for src/storage/sqlite_state_store.py"""

import tempfile
from pathlib import Path

import pytest

import src.storage.sqlite_state_store as store_module
from src.storage.sqlite_state_store import SQLiteStateStore, get_state_store

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_store() -> SQLiteStateStore:
    """Create an in-memory (tmp) SQLiteStateStore for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = SQLiteStateStore(db_path=db_path)
    # db_path now points inside a deleted tempdir — but SQLite creates it again
    # Actually we need the tempdir to persist. Better: use a new tmp for each:
    return store


@pytest.fixture
def store(tmp_path) -> SQLiteStateStore:
    return SQLiteStateStore(db_path=tmp_path / "test.db")


# ─── register_source_document ─────────────────────────────────────────────────


class TestRegisterSourceDocument:
    def test_registers_new_document(self, store):
        doc_id = store.register_source_document(
            source_url="https://example.com/page",
            title="Page Title",
            authority_level="official",
            visa_types=["chancenkarte"],
        )
        assert doc_id is not None
        assert int(doc_id) > 0

    def test_duplicate_returns_existing_id(self, store):
        id1 = store.register_source_document("https://dup.com/", "Dup")
        id2 = store.register_source_document("https://dup.com/", "Dup")
        assert id1 == id2

    def test_different_urls_return_different_ids(self, store):
        id1 = store.register_source_document("https://a.com/", "A")
        id2 = store.register_source_document("https://b.com/", "B")
        assert id1 != id2


# ─── mark_document_* ──────────────────────────────────────────────────────────


class TestMarkDocumentStatus:
    def test_mark_processing(self, store):
        doc_id = store.register_source_document("https://proc.com/", "Proc")
        store.mark_document_processing(doc_id)
        meta = store.get_document_metadata("https://proc.com/")
        assert meta["status"] == "processing"

    def test_mark_ingested(self, store):
        doc_id = store.register_source_document("https://ing.com/", "Ing")
        store.mark_document_ingested(doc_id, content_hash="abc123")
        meta = store.get_document_metadata("https://ing.com/")
        assert meta["status"] == "ingested"

    def test_mark_failed(self, store):
        doc_id = store.register_source_document("https://fail.com/", "Fail")
        store.mark_document_failed(doc_id, error_message="timeout")
        meta = store.get_document_metadata("https://fail.com/")
        assert meta["status"] == "failed"


# ─── get_document_metadata ────────────────────────────────────────────────────


class TestGetDocumentMetadata:
    def test_returns_none_for_unknown_url(self, store):
        assert store.get_document_metadata("https://unknown.example.com/") is None

    def test_returns_dict_for_known_url(self, store):
        store.register_source_document("https://known.com/", "Known")
        meta = store.get_document_metadata("https://known.com/")
        assert isinstance(meta, dict)
        assert "status" in meta


# ─── chunks ───────────────────────────────────────────────────────────────────


class TestChunks:
    def test_register_chunk(self, store):
        doc_id = store.register_source_document("https://c.com/", "C")
        row_id = store.register_chunk(
            chunk_id="chunk-1",
            parent_doc_id=doc_id,
            source_url="https://c.com/",
            text="Sample chunk text",
            text_hash="hash-abc",
            is_parent=False,
        )
        assert row_id is not None

    def test_check_chunk_duplicate_false_when_not_exists(self, store):
        assert store.check_chunk_duplicate("nonexistent-hash") is False

    def test_check_chunk_duplicate_true_when_exists(self, store):
        doc_id = store.register_source_document("https://d.com/", "D")
        store.register_chunk("c2", doc_id, "https://d.com/", "text", "unique-hash")
        assert store.check_chunk_duplicate("unique-hash") is True

    def test_update_chunk_qdrant_id(self, store):
        doc_id = store.register_source_document("https://e.com/", "E")
        store.register_chunk("c3", doc_id, "https://e.com/", "text", "hash-xyz")
        # Should not raise
        store.update_chunk_qdrant_id("c3", qdrant_point_id=42)

    def test_delete_document_chunks(self, store):
        doc_id = store.register_source_document("https://del.com/", "Del")
        store.register_chunk("c4", doc_id, "https://del.com/", "text", "hash-del")
        assert store.check_chunk_duplicate("hash-del") is True
        store.delete_document_chunks(doc_id)
        assert store.check_chunk_duplicate("hash-del") is False


# ─── ingestion_runs ───────────────────────────────────────────────────────────


class TestIngestionRuns:
    def test_create_and_finalize_run(self, store):
        run_id = "run-001"
        store.create_ingestion_run(run_id, triggered_by="test")
        store.finalize_ingestion_run(
            run_id,
            documents_processed=5,
            chunks_ingested=20,
            chunks_skipped=2,
            error_count=0,
            total_tokens=1000,
        )
        stats = store.get_stats()
        assert stats["total_ingestion_runs"] >= 1

    def test_get_stats_returns_expected_keys(self, store):
        stats = store.get_stats()
        assert "ingested_documents" in stats
        assert "active_chunks" in stats
        assert "total_ingestion_runs" in stats


# ─── get_pending_documents ────────────────────────────────────────────────────


class TestGetPendingDocuments:
    def test_returns_pending_docs(self, store):
        store.register_source_document("https://pend.com/", "Pend")
        pending = store.get_pending_documents(limit=10)
        assert any(d["source_url"] == "https://pend.com/" for d in pending)

    def test_ingested_docs_not_returned(self, store):
        doc_id = store.register_source_document("https://done.com/", "Done")
        store.mark_document_ingested(doc_id, content_hash="xyz")
        pending = store.get_pending_documents(limit=10)
        assert not any(d["source_url"] == "https://done.com/" for d in pending)

    def test_limit_respected(self, store):
        for i in range(5):
            store.register_source_document(f"https://p{i}.com/", f"P{i}")
        pending = store.get_pending_documents(limit=3)
        assert len(pending) <= 3


# ─── discovery cache ──────────────────────────────────────────────────────────


class TestDiscoveryCache:
    def test_save_and_retrieve_urls(self, store):
        urls = [
            {
                "url": "https://example.com/page1",
                "authority_level": "official",
                "visa_types": ["chancenkarte"],
                "relevance_score": 0.9,
                "from_sitemap": True,
                "from_crawling": False,
            }
        ]
        store.save_discovered_urls("example.com", urls)
        cached = store.get_cached_discovery("example.com", max_age_hours=24)
        assert cached is not None
        assert len(cached) == 1
        assert cached[0]["url"] == "https://example.com/page1"

    def test_returns_none_for_unknown_domain(self, store):
        assert store.get_cached_discovery("unknown.domain", max_age_hours=24) is None

    def test_replaces_old_entries_on_save(self, store):
        urls_v1 = [
            {
                "url": "https://x.com/old",
                "authority_level": "official",
                "visa_types": ["general"],
                "relevance_score": 0.5,
                "from_sitemap": False,
                "from_crawling": True,
            }
        ]
        urls_v2 = [
            {
                "url": "https://x.com/new",
                "authority_level": "official",
                "visa_types": ["general"],
                "relevance_score": 0.8,
                "from_sitemap": True,
                "from_crawling": False,
            }
        ]
        store.save_discovered_urls("x.com", urls_v1)
        store.save_discovered_urls("x.com", urls_v2)
        cached = store.get_cached_discovery("x.com", max_age_hours=24)
        urls_found = [c["url"] for c in cached]
        assert "https://x.com/new" in urls_found
        assert "https://x.com/old" not in urls_found

    def test_clear_discovery_cache_specific_domain(self, store):
        urls = [
            {
                "url": "https://y.com/p",
                "authority_level": "official",
                "visa_types": ["general"],
                "relevance_score": 0.5,
                "from_sitemap": False,
                "from_crawling": False,
            }
        ]
        store.save_discovered_urls("y.com", urls)
        store.clear_discovery_cache(domain="y.com")
        assert store.get_cached_discovery("y.com", max_age_hours=24) is None

    def test_clear_discovery_cache_all(self, store):
        urls = [
            {
                "url": "https://z.com/p",
                "authority_level": "official",
                "visa_types": ["general"],
                "relevance_score": 0.5,
                "from_sitemap": False,
                "from_crawling": False,
            }
        ]
        store.save_discovered_urls("z.com", urls)
        store.clear_discovery_cache()
        assert store.get_cached_discovery("z.com", max_age_hours=24) is None


# ─── Singleton ────────────────────────────────────────────────────────────────


class TestGetStateStore:
    def test_returns_same_instance(self, tmp_path):
        store_module._state_store = None
        from unittest.mock import patch

        with patch("src.storage.sqlite_state_store.settings") as s:
            s.sqlite_db_path = tmp_path / "singleton.db"
            a = get_state_store()
            b = get_state_store()
        assert a is b
        store_module._state_store = None
