"""
SQLite-based state store for tracking ingestion progress, deduplication,
document versioning, and URL discovery caching.
Ensures no duplicate chunks are ingested.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from src.config import settings
from src.logger import logger


class SQLiteStateStore:
    """SQLite state store for ingestion tracking and deduplication."""

    def __init__(self, db_path: Path = None):
        """Initialize state store with database path."""
        self.db_path = db_path or settings.sqlite_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Table: tracked_documents (source documents)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tracked_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_url TEXT UNIQUE NOT NULL,
                    source_hash TEXT,
                    document_title TEXT,
                    authority_level TEXT DEFAULT 'third_party',
                    visa_types TEXT,  -- JSON array
                    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_fetched_at TIMESTAMP,
                    last_modified_at TIMESTAMP,
                    content_hash TEXT,
                    status TEXT DEFAULT 'pending',  -- pending, processing, ingested, failed
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Table: chunks (deduplicated chunks)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT UNIQUE NOT NULL,
                    parent_doc_id TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    text_hash TEXT UNIQUE NOT NULL,
                    text_length INTEGER,
                    is_parent BOOLEAN DEFAULT 0,
                    section_header TEXT,
                    language TEXT DEFAULT 'de',
                    qdrant_point_id INTEGER,
                    status TEXT DEFAULT 'active',  -- active, archived, deleted
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_doc_id) REFERENCES tracked_documents(id)
                )
            """)

            # Table: ingestion_runs (audit trail)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT DEFAULT 'running',  -- running, completed, failed
                    documents_processed INTEGER DEFAULT 0,
                    chunks_ingested INTEGER DEFAULT 0,
                    chunks_skipped INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    error_details TEXT,  -- JSON
                    triggered_by TEXT DEFAULT 'manual'  -- manual, scheduler
                )
            """)

            # Table: discovered_urls (URL discovery cache)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS discovered_urls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    url TEXT NOT NULL,
                    authority_level TEXT DEFAULT 'third_party',
                    visa_types TEXT,
                    relevance_score REAL DEFAULT 0.5,
                    from_sitemap INTEGER DEFAULT 0,
                    from_crawling INTEGER DEFAULT 0,
                    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(domain, url)
                )
            """)

            # Indices for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_text_hash ON chunks(text_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_url ON tracked_documents(source_url)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_status ON chunks(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_status ON tracked_documents(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_discovered_domain ON discovered_urls(domain)")

            conn.commit()
            logger.info("SQLite state store initialized at %s", self.db_path)

    @contextmanager
    def _get_connection(self):
        """Get database connection context."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def register_source_document(
        self,
        source_url: str,
        title: str,
        authority_level: str = "third_party",
        visa_types: Optional[list[str]] = None,
    ) -> str:
        """
        Register a source document for tracking.

        Args:
            source_url: URL of the document
            title: Document title
            authority_level: Authority classification
            visa_types: List of visa categories

        Returns:
            Document ID for tracking
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    INSERT INTO tracked_documents
                    (source_url, document_title, authority_level, visa_types, status)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        source_url,
                        title,
                        authority_level,
                        json.dumps(visa_types or []),
                        "pending",
                    ),
                )
                conn.commit()
                return str(cursor.lastrowid)

            except sqlite3.IntegrityError:
                # Document already tracked, return existing ID
                cursor.execute("SELECT id FROM tracked_documents WHERE source_url = ?", (source_url,))
                result = cursor.fetchone()
                return str(result[0]) if result else None

    def mark_document_processing(self, doc_id: str):
        """Mark document as currently being processed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE tracked_documents
                SET status = 'processing', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (doc_id,),
            )
            conn.commit()

    def mark_document_ingested(self, doc_id: str, content_hash: str):
        """Mark document as successfully ingested."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE tracked_documents
                SET status = 'ingested',
                    content_hash = ?,
                    last_fetched_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (content_hash, doc_id),
            )
            conn.commit()

    def mark_document_failed(self, doc_id: str, error_message: str):
        """Mark document as failed ingestion."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE tracked_documents
                SET status = 'failed',
                    error_message = ?,
                    retry_count = retry_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (error_message, doc_id),
            )
            conn.commit()

    def get_document_metadata(self, source_url: str) -> Optional[dict[str, Any]]:
        """Get document status and content hash."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, status, content_hash FROM tracked_documents WHERE source_url = ?", (source_url,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def check_chunk_duplicate(self, text_hash: str) -> bool:
        """Check if chunk (by hash) already exists."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM chunks WHERE text_hash = ? AND status = 'active'", (text_hash,))
            return cursor.fetchone() is not None

    def register_chunk(
        self,
        chunk_id: str,
        parent_doc_id: str,
        source_url: str,
        text: str,
        text_hash: str,
        is_parent: bool = False,
        section_header: Optional[str] = None,
        language: str = "de",
    ) -> int:
        """
        Register a chunk for tracking.

        Returns:
            Chunk row ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO chunks
                    (chunk_id, parent_doc_id, source_url, text_hash, text_length,
                     is_parent, section_header, language, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        chunk_id,
                        parent_doc_id,
                        source_url,
                        text_hash,
                        len(text),
                        int(is_parent),
                        section_header,
                        language,
                        "active",
                    ),
                )
                conn.commit()
                return cursor.lastrowid
            except Exception as e:
                logger.error("Error registering chunk %s: %s", chunk_id, e)
                return None

    def update_chunk_qdrant_id(self, chunk_id: str, qdrant_point_id: int):
        """Link chunk to its Qdrant point ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE chunks
                SET qdrant_point_id = ?
                WHERE chunk_id = ?
            """,
                (qdrant_point_id, chunk_id),
            )
            conn.commit()

    def delete_document_chunks(self, doc_id: str):
        """Delete all chunks associated with a document."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE parent_doc_id = ?", (doc_id,))
            conn.commit()
            logger.debug("Deleted existing chunks for doc_id %s from SQLite", doc_id)

    def create_ingestion_run(self, run_id: str, triggered_by: str = "manual") -> str:
        """Create new ingestion run record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO ingestion_runs (run_id, triggered_by)
                VALUES (?, ?)
            """,
                (run_id, triggered_by),
            )
            conn.commit()
            return run_id

    def finalize_ingestion_run(
        self,
        run_id: str,
        documents_processed: int,
        chunks_ingested: int,
        chunks_skipped: int,
        error_count: int = 0,
        total_tokens: int = 0,
        error_details: Optional[str] = None,
    ):
        """Finalize ingestion run with summary."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE ingestion_runs
                SET status = 'completed',
                    end_time = CURRENT_TIMESTAMP,
                    documents_processed = ?,
                    chunks_ingested = ?,
                    chunks_skipped = ?,
                    error_count = ?,
                    total_tokens = ?,
                    error_details = ?
                WHERE run_id = ?
            """,
                (
                    documents_processed,
                    chunks_ingested,
                    chunks_skipped,
                    error_count,
                    total_tokens,
                    error_details,
                    run_id,
                ),
            )
            conn.commit()

    def get_pending_documents(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get documents pending ingestion."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM tracked_documents
                WHERE status IN ('pending', 'failed')
                AND retry_count < 3
                ORDER BY updated_at ASC
                LIMIT ?
            """,
                (limit,),
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_stats(self) -> dict[str, Any]:
        """Get ingestion statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM tracked_documents WHERE status = 'ingested'")
            ingested_docs = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM chunks WHERE status = 'active'")
            active_chunks = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM ingestion_runs")
            total_runs = cursor.fetchone()[0]

            cursor.execute("""
                SELECT SUM(chunks_ingested), SUM(total_tokens)
                FROM ingestion_runs WHERE status = 'completed'
            """)
            row = cursor.fetchone()
            total_chunks_ingested = row[0] or 0
            total_tokens = row[1] or 0

            cursor.execute("SELECT COUNT(DISTINCT domain) FROM discovered_urls")
            cached_domains = cursor.fetchone()[0]

            return {
                "ingested_documents": ingested_docs,
                "active_chunks": active_chunks,
                "cached_discovery_domains": cached_domains,
                "total_ingestion_runs": total_runs,
                "total_chunks_ingested": total_chunks_ingested,
                "total_tokens_used": total_tokens,
            }

    # ============================================
    # Discovery Cache Methods
    # ============================================

    def save_discovered_urls(
        self,
        domain: str,
        urls_with_metadata: list[dict[str, Any]],
    ):
        """
        Save discovered URLs to cache, replacing any previous entries for the domain.

        Args:
            domain: The domain these URLs belong to
            urls_with_metadata: List of dicts with keys:
                url, authority_level, visa_types, relevance_score,
                from_sitemap (bool), from_crawling (bool)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Delete old entries for this domain
            cursor.execute("DELETE FROM discovered_urls WHERE domain = ?", (domain,))

            for entry in urls_with_metadata:
                cursor.execute(
                    """
                    INSERT INTO discovered_urls
                    (domain, url, authority_level, visa_types, relevance_score,
                     from_sitemap, from_crawling)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        domain,
                        entry["url"],
                        entry.get("authority_level", "third_party"),
                        json.dumps(entry.get("visa_types", ["general"])),
                        entry.get("relevance_score", 0.5),
                        int(entry.get("from_sitemap", False)),
                        int(entry.get("from_crawling", False)),
                    ),
                )
            conn.commit()
            logger.info("Cached %d discovered URLs for %s", len(urls_with_metadata), domain)

    def get_cached_discovery(
        self,
        domain: str,
        max_age_hours: int = 24,
    ) -> Optional[list[dict[str, Any]]]:
        """
        Return cached discovery URLs if they exist and are fresh enough.

        Args:
            domain: Domain to look up
            max_age_hours: Maximum cache age in hours

        Returns:
            List of URL dicts if cache is fresh, None if stale or missing
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT url, authority_level, visa_types, relevance_score,
                       from_sitemap, from_crawling, discovered_at
                FROM discovered_urls
                WHERE domain = ?
                  AND discovered_at > datetime('now', ?)
                ORDER BY relevance_score DESC
            """,
                (domain, f"-{max_age_hours} hours"),
            )

            rows = cursor.fetchall()
            if not rows:
                return None

            results = []
            for row in rows:
                results.append(
                    {
                        "url": row["url"],
                        "authority_level": row["authority_level"],
                        "visa_types": json.loads(row["visa_types"]) if row["visa_types"] else ["general"],
                        "relevance_score": row["relevance_score"],
                        "from_sitemap": bool(row["from_sitemap"]),
                        "from_crawling": bool(row["from_crawling"]),
                        "discovered_at": row["discovered_at"],
                    }
                )
            return results

    def clear_discovery_cache(self, domain: Optional[str] = None):
        """
        Clear the discovery cache.

        Args:
            domain: If provided, clear only this domain. Otherwise clear all.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if domain:
                cursor.execute("DELETE FROM discovered_urls WHERE domain = ?", (domain,))
                logger.info("Cleared discovery cache for %s", domain)
            else:
                cursor.execute("DELETE FROM discovered_urls")
                logger.info("Cleared all discovery cache")
            conn.commit()


# Singleton instance
_state_store = None


def get_state_store() -> SQLiteStateStore:
    """Get or create state store singleton."""
    global _state_store
    if _state_store is None:
        _state_store = SQLiteStateStore()
    return _state_store
