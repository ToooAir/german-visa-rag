"""Utility script to clean up local SQLite state and Qdrant collection."""

import asyncio
import os
from pathlib import Path
from src.config import settings
from src.vector_db.qdrant_client_wrapper import get_qdrant_client
from src.logger import logger

async def cleanup():
    logger.info("Starting cleanup process...")
    
    # 1. Clean SQLite
    db_path = settings.sqlite_db_path
    if db_path.exists():
        os.remove(db_path)
        logger.info(f"Deleted SQLite database at {db_path}")
    
    # 2. Clean Qdrant
    try:
        qdrant = get_qdrant_client()
        await qdrant.client.delete_collection(settings.qdrant_collection_name)
        logger.info(f"Deleted Qdrant collection: {settings.qdrant_collection_name}")
    except Exception as e:
        logger.warning(f"Could not delete Qdrant collection (might not exist): {e}")

    logger.info("Cleanup completed successfully!")

if __name__ == "__main__":
    asyncio.run(cleanup())
