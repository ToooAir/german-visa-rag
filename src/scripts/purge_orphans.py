import asyncio
import json
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from src.ingestion.crawl_strategy import get_strategy_registry
from src.storage.sqlite_state_store import get_state_store
from src.vector_db.qdrant_client_wrapper import get_qdrant_client
from src.logger import logger


async def purge_orphans():
    registry = get_strategy_registry()
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

        # Check if URL is still allowed under current strategies
        strategy = registry.get_strategy(url)
        if not strategy.is_url_allowed(url):
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

    logger.info(f"✨ Purge complete. Deleted {purged_count}/{total_docs} orphaned/disallowed documents.")


if __name__ == "__main__":
    asyncio.run(purge_orphans())
