import argparse
import asyncio

from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from src.ingestion.cli import load_seed_urls
from src.ingestion.crawl_strategy import get_strategy_registry
from src.logger import logger
from src.storage.sqlite_state_store import get_state_store
from src.vector_db.qdrant_client_wrapper import get_qdrant_client


def _pinned_urls() -> set[str]:
    """URLs pinned in seed_urls.yml, which must survive a purge.

    A pin exists precisely because discovery cannot reach the page: anabin is a
    dynamic JSF app with only static entry points, and several official pages sit
    outside their domain's allowed path patterns. Judging those by the patterns
    alone deletes them, and the next ingest puts them back, so the purge and the
    seed list would fight each other on every run.
    """
    return {doc["url"] for doc in load_seed_urls() if doc.get("url")}


async def purge_orphans(dry_run: bool = False):
    registry = get_strategy_registry()
    pinned = _pinned_urls()
    state_store = get_state_store()
    qdrant = get_qdrant_client()

    logger.info("🧹 Starting orphan purge based on latest crawl strategies...")

    # 1. Get all tracked documents
    # Using row_factory=sqlite3.Row as configured in SQLiteStateStore
    with state_store._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, source_url FROM tracked_documents")
        docs = cursor.fetchall()

    purged_count = 0
    total_docs = len(docs)

    for doc in docs:
        doc_id = str(doc["id"])
        url = doc["source_url"]

        if url in pinned:
            continue

        # Check if URL is still allowed under current strategies
        strategy = registry.get_strategy(url)
        if not strategy.is_url_allowed(url):
            if dry_run:
                logger.info(f"[dry-run] would purge: {url}")
                purged_count += 1
                continue

            logger.info(f"🚫 Purging disallowed URL: {url}")

            # Delete from Qdrant
            q_filter = Filter(must=[FieldCondition(key="parent_doc_id", match=MatchValue(value=doc_id))])
            try:
                await qdrant.delete_by_filter(q_filter)
            except Exception as e:
                logger.error(f"Error deleting from Qdrant for {url}: {e}")

            # Delete chunks from SQLite
            state_store.delete_document_chunks(doc_id)

            # Delete document itself from SQLite
            with state_store._get_connection() as conn:
                conn.execute("DELETE FROM tracked_documents WHERE id = ?", (doc_id,))
                conn.commit()

            purged_count += 1

    verb = "Would delete" if dry_run else "Deleted"
    logger.info(
        f"✨ Purge {'dry-run' if dry_run else 'complete'}. "
        f"{verb} {purged_count}/{total_docs} orphaned/disallowed documents "
        f"({len(pinned)} pinned URLs skipped)."
    )
    return purged_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="List what would be deleted and exit")
    asyncio.run(purge_orphans(dry_run=parser.parse_args().dry_run))
