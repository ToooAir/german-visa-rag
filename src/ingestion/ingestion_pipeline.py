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
from src.ingestion.crawler import get_crawler
from src.ingestion.chunker import get_chunker
from src.storage.sqlite_state_store import get_state_store
from src.vector_db.qdrant_client_wrapper import get_qdrant_client
from src.vector_db.embedder import embedder
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
        
        # Ensure Qdrant collection exists
        await self.qdrant.ensure_collection_exists()
        
        # Process each source document
        for source_doc in source_documents:
            try:
                result = await self._process_single_document(source_doc)
                
                if result["success"]:
                    documents_processed += 1
                    chunks_ingested += result["chunks_ingested"]
                    chunks_skipped += result["chunks_skipped"]
                    total_tokens += result.get("tokens_used", 0)
                else:
                    errors.append(result["error"])
                    
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                errors.append(str(e))
        
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
        }
        
        logger.info("Ingestion pipeline completed", extra=summary)
        
        # Log to MLflow
        if self.mlflow:
            self.mlflow.log_ingestion_run(summary)
        
        return summary

    async def _process_single_document(
        self,
        source_doc: Dict[str, Any],
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
            skipped_count = 0
            
            for chunk in chunks:
                text_hash = chunk.metadata.text_hash
                
                if self.state_store.check_chunk_duplicate(text_hash):
                    logger.debug(f"Skipping duplicate chunk: {chunk.metadata.chunk_id}")
                    skipped_count += 1
                else:
                    chunks_to_ingest.append(chunk)
            
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
