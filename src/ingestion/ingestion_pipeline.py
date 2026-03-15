"""
Complete ingestion pipeline orchestrating crawling, chunking, embedding,
deduplication, and Qdrant upsert.
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio

from src.config import settings
from src.logger import logger
from src.models.chunk import QdrantPayload, VisaType, AuthorityLevel
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
from src.ingestion.crawler import get_crawler
from src.ingestion.chunker import get_chunker
from src.storage.sqlite_state_store import get_state_store
from src.vector_db.qdrant_client_wrapper import get_qdrant_client
from src.vector_db.embedder import embedder, QuotaExhaustedError
from src.observability.mlflow_tracker import get_mlflow_tracker

from qdrant_client.http.models import PointStruct


class IngestionPipeline:
    """
    End-to-end ingestion pipeline.
    
    Flow:
    1. Fetch source URLs
    2. Crawl HTML → Markdown
    3. Parse and chunk (Parent-Child)
    4. Deduplicate by text hash
    5. Generate embeddings
    6. Upsert to Qdrant + SQLite tracking
    """

    def __init__(self):
        self.crawler = get_crawler()
        self.chunker = get_chunker()
        self.state_store = get_state_store()
        self.qdrant = get_qdrant_client()
        self.mlflow = get_mlflow_tracker()

    async def run_full_ingestion(
        self,
        source_documents: List[Dict[str, Any]],
        triggered_by: str = "manual",
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute complete ingestion pipeline.
        
        Args:
            source_documents: List of docs with url, title, authority_level, visa_types
            triggered_by: 'manual' or 'scheduler'
            
        Returns:
            Ingestion summary statistics
        """
        run_id = str(uuid.uuid4())
        self.state_store.create_ingestion_run(run_id, triggered_by)
        
        logger.info(
            "Starting ingestion pipeline",
            extra={
                "run_id": run_id,
                "documents_count": len(source_documents),
                "triggered_by": triggered_by,
            }
        )
        
        # Statistics
        documents_processed = 0
        chunks_ingested = 0
        chunks_skipped = 0
        errors = []
        total_tokens = 0
        quota_exhausted = False
        documents_skipped_quota = 0
        
        # ── Pre-flight: check embedding API quota ──
        try:
            api_ok = await embedder.preflight_check()
            if not api_ok:
                return {
                    "run_id": run_id,
                    "success": False,
                    "documents_processed": 0,
                    "chunks_ingested": 0,
                    "chunks_skipped": 0,
                    "errors": ["Embedding API preflight check failed (non-quota). Check logs."],
                    "total_tokens": 0,
                    "quota_exhausted": False,
                }
        except QuotaExhaustedError as qe:
            logger.error(f"⛔ Embedding quota exhausted before crawling: {qe}")
            self.state_store.finalize_ingestion_run(
                run_id, 0, 0, 0, error_count=1, total_tokens=0,
            )
            return {
                "run_id": run_id,
                "success": False,
                "documents_processed": 0,
                "chunks_ingested": 0,
                "chunks_skipped": 0,
                "errors": [str(qe)],
                "total_tokens": 0,
                "quota_exhausted": True,
                "wait_seconds": qe.wait_seconds,
            }
        
        # Ensure Qdrant collection exists
        await self.qdrant.ensure_collection_exists()
        
        # ── Concurrent execution with Semaphore ──
        # Use an explicit small integer default for concurrency if no setting exists
        max_concurrent = int(getattr(settings, "crawler_max_concurrent_requests", 5))
        sem = asyncio.Semaphore(max_concurrent)
        quota_flag = [False] # Use list for shared mutable state
        
        async def sem_process(source_doc):
            async with sem:
                if quota_flag[0]: # Skip if another task already hit quota
                    return {"success": False, "skipped_quota": True}
                try:
                    return await self._process_single_document(source_doc, force=force)
                except QuotaExhaustedError:
                    quota_flag[0] = True
                    raise

        # Process each source document
        tasks = [sem_process(doc) for doc in source_documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, QuotaExhaustedError) or quota_flag[0]:
                quota_exhausted = True
                if isinstance(result, QuotaExhaustedError):
                    errors.append(str(result))
            elif isinstance(result, Exception):
                logger.error(f"Unexpected error in pipeline task: {result}")
                errors.append(str(result))
            elif isinstance(result, dict):
                if result.get("skipped_quota"):
                    documents_skipped_quota += 1
                elif result["success"]:
                    documents_processed += 1
                    chunks_ingested += result["chunks_ingested"]
                    chunks_skipped += result["chunks_skipped"]
                    total_tokens += result.get("tokens_used", 0)
                else:
                    errors.append(result["error"])
        
        # Finalize run
        self.state_store.finalize_ingestion_run(
            run_id,
            documents_processed=documents_processed,
            chunks_ingested=chunks_ingested,
            chunks_skipped=chunks_skipped,
            error_count=len(errors),
            total_tokens=total_tokens,
        )
        
        summary = {
            "run_id": run_id,
            "success": len(errors) == 0,
            "documents_processed": documents_processed,
            "chunks_ingested": chunks_ingested,
            "chunks_skipped": chunks_skipped,
            "errors": errors,
            "total_tokens": total_tokens,
            "quota_exhausted": quota_exhausted,
            "documents_skipped_quota": documents_skipped_quota,
        }
        
        logger.info("Ingestion pipeline completed", extra=summary)
        
        # Log to MLflow
        if self.mlflow:
            self.mlflow.log_ingestion_run(summary)
        
        return summary

    async def _process_single_document(
        self,
        source_doc: Dict[str, Any],
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a single source document through full pipeline.
        
        Returns:
            Result dict with success flag and statistics
        """
        url = source_doc.get("url")
        title = source_doc.get("title", url)
        authority_level = AuthorityLevel(source_doc.get("authority_level", "third_party"))
        visa_types = [VisaType(vt) for vt in source_doc.get("visa_types", [])]
        
        logger.debug(f"Processing document: {url}")
        
        try:
            # Register in state store
            doc_id = self.state_store.register_source_document(
                url, title, authority_level.value, [vt.value for vt in visa_types]
            )
            
            self.state_store.mark_document_processing(doc_id)
            
            # Step 1: Crawl
            crawled = await self.crawler.crawl_document(url)
            if not crawled:
                self.state_store.mark_document_failed(doc_id, "Crawling failed")
                return {"success": False, "error": f"Failed to crawl {url}"}
            
            markdown_text = crawled["markdown"]
            fetched_at = datetime.fromisoformat(crawled["fetched_at"])
            content_hash = self._compute_document_hash(markdown_text)
            
            # --- Optimization: Skip if content hasn't changed ---
            existing = self.state_store.get_document_metadata(url)
            if (
                not force 
                and existing 
                and existing["status"] == "ingested" 
                and existing["content_hash"] == content_hash
            ):
                logger.info(f"⏭️  Document content hasn't changed, skipping: {url}")
                return {
                    "success": True,
                    "chunks_ingested": 0,
                    "chunks_skipped": 0,
                    "tokens_used": 0,
                }
            
            # --- Replacement Logic: Delete old chunks if re-processing ---
            if existing:
                logger.info(f"Replacing existing chunks for: {url}")
                # 1. Delete from Qdrant
                from_qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="parent_doc_id",
                            match=MatchValue(value=str(doc_id))
                        )
                    ]
                )
                await self.qdrant.delete_by_filter(from_qdrant_filter)
                
                # 2. Delete from SQLite
                self.state_store.delete_document_chunks(doc_id)
            
            # Step 2: Chunk (Parent-Child)
            chunks = self.chunker.chunk_document(
                markdown_text=markdown_text,
                source_url=url,
                doc_id=doc_id,
                title=title,
                authority_level=authority_level,
                visa_types=visa_types,
                language="de",
                published_at=None,
            )
            
            if not chunks:
                self.state_store.mark_document_failed(doc_id, "Chunking produced no chunks")
                return {"success": False, "error": f"No chunks created for {url}"}
            
            logger.debug(f"Generated {len(chunks)} chunks for {url}")
            
            # Step 3: Deduplicate
            chunks_to_ingest = []
            seen_hashes = set()
            skipped_count = 0
            
            for chunk in chunks:
                text_hash = chunk.metadata.text_hash
                
                # Check DB for duplicates
                if self.state_store.check_chunk_duplicate(text_hash):
                    logger.debug(f"Skipping duplicate chunk (DB): {chunk.metadata.chunk_id}")
                    skipped_count += 1
                    continue
                
                # Check current batch for duplicates (internal within document)
                if text_hash in seen_hashes:
                    logger.debug(f"Skipping duplicate chunk (Batch): {chunk.metadata.chunk_id}")
                    skipped_count += 1
                    continue
                
                chunks_to_ingest.append(chunk)
                seen_hashes.add(text_hash)
            
            logger.debug(
                f"Deduplication: {len(chunks_to_ingest)} to ingest, {skipped_count} skipped"
            )
            
            # Step 4: Embed
            logger.debug(f"Embedding {len(chunks_to_ingest)} chunks")
            
            texts_to_embed = [c.text for c in chunks_to_ingest]
            embeddings = await embedder.embed_texts(texts_to_embed)
            
            if len(embeddings) != len(chunks_to_ingest):
                raise ValueError("Embedding count mismatch")
            
            # Step 5: Prepare Qdrant points
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks_to_ingest, embeddings)):
                payload = QdrantPayload.from_chunk(chunk)
                
                # Use hash-based ID for deterministic point IDs
                point_id = int(hash(chunk.metadata.text_hash) & 0x7FFFFFFF)
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload.to_dict(),
                )
                points.append(point)
            
            # Step 6: Upsert to Qdrant
            logger.debug(f"Upserting {len(points)} points to Qdrant")
            await self.qdrant.upsert_points(points, wait=True)
            
            # Step 7: Update state store with tracking
            for chunk, point in zip(chunks_to_ingest, points):
                chunk_row_id = self.state_store.register_chunk(
                    chunk_id=chunk.metadata.chunk_id,
                    parent_doc_id=doc_id,
                    source_url=url,
                    text=chunk.text,
                    text_hash=chunk.metadata.text_hash,
                    is_parent=chunk.metadata.is_parent,
                    section_header=chunk.metadata.section_header,
                    language=chunk.metadata.language,
                )
                self.state_store.update_chunk_qdrant_id(
                    chunk.metadata.chunk_id,
                    point.id,
                )
            
            # Mark document as ingested
            content_hash = self._compute_document_hash(markdown_text)
            self.state_store.mark_document_ingested(doc_id, content_hash)
            
            logger.info(
                f"Successfully ingested document",
                extra={
                    "url": url,
                    "chunks_ingested": len(chunks_to_ingest),
                    "chunks_skipped": skipped_count,
                }
            )
            
            return {
                "success": True,
                "chunks_ingested": len(chunks_to_ingest),
                "chunks_skipped": skipped_count,
                "tokens_used": 0,  # TODO: track embedding tokens
            }
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}", exc_info=True)
            if doc_id:
                self.state_store.mark_document_failed(doc_id, str(e))
            return {"success": False, "error": str(e)}

    def _compute_document_hash(self, text: str) -> str:
        """Compute hash of entire document for versioning."""
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()


# Singleton instance
_pipeline = None


def get_ingestion_pipeline() -> IngestionPipeline:
    """Get or create ingestion pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline
