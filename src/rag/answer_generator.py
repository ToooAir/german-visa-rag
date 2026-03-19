"""
Answer generation with streaming support, caching, and source attribution.
Handles LLM calls and formats responses for OpenAI-compatible endpoints.
"""

from typing import Dict, Any, List, AsyncIterator, Optional
import time
import uuid
import asyncio

from src.config import settings
from src.logger import logger
from src.exceptions import LLMGenerationError
from src.storage.redis_cache import query_cache
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.query_transformer import get_query_transformer
from src.rag.reranker import get_reranker
from src.rag.prompt_builder import get_prompt_builder
from src.llm import get_llm_client  
from src.llm.token_counter import get_token_counter
from src.observability.mlflow_tracker import get_mlflow_tracker


class AnswerGenerator:
    """
    Generate answers with full RAG pipeline:
    0. Cache Check
    1. Query Transformation
    2. Hybrid Retrieval
    3. Reranking
    4. Prompt Building
    5. LLM Generation (streaming or non-streaming)
    6. Observability & Cache Update
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.query_transformer = get_query_transformer()
        self.reranker = get_reranker()
        self.prompt_builder = get_prompt_builder()
        self.llm = get_llm_client()
        self.token_counter = get_token_counter()
        self.mlflow = get_mlflow_tracker()

    async def generate_answer(self, query: str) -> Dict[str, Any]:
        """Generate answer without streaming."""
        start_time = time.time()
        
        # 0. Check Redis Cache
        cached_result = await query_cache.get(query)
        if cached_result:
            cached_result["metadata"]["latency_seconds"] = time.time() - start_time
            cached_result["metadata"]["cache_hit"] = True
            return cached_result

        try:
            # 1. Query transformation
            logger.info("Step 1: Transforming query")
            try:
                transformed = await self.query_transformer.transform_query(query)
                main_query = transformed["corrected_query"]
            except Exception as e:
                logger.error(f"Query expansion/correction failed: {e}")
                main_query = query # Fallback to original query
            
            # 2. Retrieval
            try:
                search_queries = await self.query_transformer.get_search_queries(query)
                logger.info(f"Step 2: Retrieving context for queries: {search_queries}")
                
                # For now, we'll retrieve for all and then deduplicate/rank
                # In a more advanced version, we could use reciprocal rank fusion
                all_results = await self.retriever.retrieve_batch(queries=search_queries)
                
                # Flatten and deduplicate by chunk_id
                retrieval_results = []
                seen_chunk_ids = set()
                for batch in all_results:
                    for result in batch:
                        chunk_id = result.get("metadata", {}).get("chunk_id")
                        if chunk_id not in seen_chunk_ids:
                            retrieval_results.append(result)
                            seen_chunk_ids.add(chunk_id)
                
                # Re-sort by adjusted_score (since they come from different queries)
                retrieval_results = sorted(
                    retrieval_results,
                    key=lambda x: x.get("adjusted_score", 0),
                    reverse=True,
                )
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
                raise
            
            if not retrieval_results:
                logger.warning("No context found during retrieval")
                return {
                    "answer": "我查閱的資料庫中暫時沒有相關信息。請查詢官方資源：https://www.make-it-in-germany.com",
                    "sources": [],
                    "metadata": {
                        "query": query,
                        "retrieval_count": 0,
                        "cache_hit": False,
                    },
                }
            
            # 3. Reranking
            reranked = await self.reranker.rerank(
                query=main_query,
                documents=retrieval_results,
                top_k=settings.retrieval_top_k_reranked,
            )
            
            # 4. Build context & prompt
            context = self.prompt_builder.build_context_from_retrieval(reranked)
            system_prompt = self.prompt_builder.build_system_prompt(context=context, question=query)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
            
            # 5. Call LLM
            logger.info(f"Step 5: Calling LLM ({settings.openai_model})")
            try:
                response_text = await self.llm.call_non_streaming(
                    messages=messages,
                    temperature=0.3,
                    max_tokens=settings.max_response_tokens,
                )
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                raise
            
            # Add disclaimer
            response_text = self.prompt_builder.add_disclaimer(response_text)
            
            # Extract sources
            sources = [
                {
                    "url": r.get("metadata", {}).get("source_url"),
                    "title": r.get("metadata", {}).get("source_title"),
                    "authority": r.get("metadata", {}).get("authority_level"),
                }
                for r in reranked
            ]
            
            latency = time.time() - start_time
            
            result = {
                "answer": response_text,
                "sources": sources,
                "metadata": {
                    "query": query,
                    "retrieval_count": len(reranked),
                    "latency_seconds": latency,
                    "cache_hit": False,
                },
            }

            # 6. Update Cache & Observability
            await query_cache.set(query, result)
            
            if self.mlflow:
                input_tokens = self.token_counter.count_messages(messages)
                output_tokens = self.token_counter.count_text(response_text)
                self.mlflow.log_query_result(
                    query,
                    {
                        "latency_seconds": latency,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": self.token_counter.estimate_cost(input_tokens, output_tokens),
                    },
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}", exc_info=True)
            raise LLMGenerationError(f"生成答案失敗: {str(e)}")

    async def generate_answer_streaming(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Generate answer with streaming response (SSE)."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Starting answer generation", extra={"request_id": request_id})

        # 0. Check Redis Cache
        yield self._format_status_chunk("analyzing")
        cached_result = await query_cache.get(query)
        if cached_result:
            logger.info("Streaming from cache", extra={"request_id": request_id})
            chunk_size = 20
            answer = cached_result["answer"]
            # Simulate streaming effect for cached response
            for i in range(0, len(answer), chunk_size):
                yield self._format_sse_chunk(answer[i:i+chunk_size])
                await asyncio.sleep(0.02)
            yield "data: [DONE]\n\n"
            return
        
        try:
            # 1. Query Transformation
            transformed = await self.query_transformer.transform_query(query)
            main_query = transformed["corrected_query"]
            
            # 2. Hybrid Retrieval
            yield self._format_status_chunk("retrieving")
            search_queries = await self.query_transformer.get_search_queries(query)
            logger.info(f"Streaming Retrieval for queries: {search_queries}", extra={"request_id": request_id})
            
            all_results = await self.retriever.retrieve_batch(
                queries=search_queries,
                top_k=top_k or settings.retrieval_top_k_hybrid,
            )
            
            # Flatten and deduplicate
            retrieval_results = []
            seen_chunk_ids = set()
            for batch in all_results:
                for result in batch:
                    chunk_id = result.get("metadata", {}).get("chunk_id")
                    if chunk_id not in seen_chunk_ids:
                        retrieval_results.append(result)
                        seen_chunk_ids.add(chunk_id)
            
            retrieval_results = sorted(
                retrieval_results,
                key=lambda x: x.get("adjusted_score", 0),
                reverse=True,
            )
            
            if not retrieval_results:
                yield self._format_sse_chunk(
                    "我查閱的資料庫中暫時沒有相關信息。請查詢官方資源：https://www.make-it-in-germany.com"
                )
                yield "data: [DONE]\n\n"
                return
            
            # 3. Reranking
            reranked = await self.reranker.rerank(
                query=main_query,
                documents=retrieval_results,
                top_k=settings.retrieval_top_k_reranked,
            )
            yield self._format_status_chunk("extracting")
            
            # 4. Build Context & Validation
            context = self.prompt_builder.build_context_from_retrieval(
                reranked,
                top_k=settings.retrieval_top_k_reranked,
            )
            
            if not self.prompt_builder.validate_context_for_injection(context):
                logger.warning("Suspicious context detected", extra={"request_id": request_id})
                yield self._format_sse_chunk("安全驗證失敗，無法處理此請求。")
                yield "data: [DONE]\n\n"
                return
            
            # Build Prompt
            system_prompt = self.prompt_builder.build_system_prompt(context=context, question=query)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
            
            # Pre-count tokens for metrics
            input_tokens = self.token_counter.count_messages(messages)
            
            # 5. Stream LLM Response
            yield self._format_status_chunk("synthesizing")
            full_response = ""
            milestone_pattern = re.compile(r"\[MILESTONE:(\d+):(\w+)\]")
            
            try:
                async for chunk in await self.llm.call_streaming(
                    messages=messages,
                    temperature=0.3, 
                    max_tokens=settings.max_response_tokens,
                ):
                    # Check for milestones in chunks (simplified buffer check)
                    # Note: In a real prod app, we'd use a text window buffer to handle split tags
                    if "[MILESTONE:" in chunk:
                        matches = milestone_pattern.findall(chunk)
                        for m_id, m_status in matches:
                            yield self._format_milestone_chunk(m_id, m_status)
                        
                        # Strip the tag from the chunk before sending to user
                        clean_chunk = milestone_pattern.sub("", chunk)
                        if clean_chunk:
                            yield self._format_sse_chunk(clean_chunk)
                    else:
                        yield self._format_sse_chunk(chunk)
                    
                    full_response += chunk
                
            except Exception as e:
                logger.error(f"LLM streaming failed: {e}", extra={"request_id": request_id})
                yield self._format_sse_chunk(f"\n\n[生成中斷：{str(e)}]")
            
            # Add disclaimer
            disclaimer = "\n\n" + self.prompt_builder.DISCLAIMER
            yield self._format_sse_chunk(disclaimer)
            full_response += disclaimer
            
            # 6. Observability & Write to Cache
            output_tokens = self.token_counter.count_text(full_response)
            latency = time.time() - start_time
            cost = self.token_counter.estimate_cost(input_tokens, output_tokens)
            
            logger.info(
                "Answer generation completed",
                extra={
                    "request_id": request_id,
                    "latency_seconds": latency,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "estimated_cost_usd": cost,
                    "retrieval_count": len(reranked),
                },
            )
            
            # Log to MLflow
            if self.mlflow:
                self.mlflow.log_query_result(
                    query,
                    {
                        "latency_seconds": latency,
                        "cost_usd": cost,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                )

            # Extract sources for cache
            sources = [
                {
                    "url": r.get("metadata", {}).get("source_url"),
                    "title": r.get("metadata", {}).get("source_title"),
                    "authority": r.get("metadata", {}).get("authority_level"),
                }
                for r in reranked
            ]
            
            # Save to Redis
            await query_cache.set(query, {
                "answer": full_response,
                "sources": sources,
                "metadata": {
                    "query": query,
                    "retrieval_count": len(reranked),
                    "cache_hit": False
                }
            })
            
            # Yield metadata chunk containing sources for the frontend
            import json
            metadata_chunk = {
                "choices": [],
                "metadata": {
                    "sources": sources
                }
            }
            yield f"data: {json.dumps(metadata_chunk, ensure_ascii=False)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(
                f"Answer generation failed: {e}",
                extra={"request_id": request_id},
                exc_info=True,
            )
            yield self._format_sse_chunk(f"[錯誤：{str(e)}]")
            yield "data: [DONE]\n\n"

    @staticmethod
    def _format_sse_chunk(content: str) -> str:
        """Format content as SSE JSON chunk (OpenAI-compatible)."""
        import json
        chunk = {
            "choices": [
                {
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ]
        }
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    @staticmethod
    def _format_milestone_chunk(milestone_id: str, status: str) -> str:
        """Format milestone update as SSE JSON chunk."""
        import json
        chunk = {
            "choices": [],
            "metadata": {
                "achieved_milestone": {
                    "id": milestone_id,
                    "status": status
                }
            }
        }
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    @staticmethod
    def _format_status_chunk(status: str) -> str:
        """Format status update as SSE JSON chunk."""
        import json
        chunk = {
            "choices": [],
            "metadata": {
                "status": status
            }
        }
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# Singleton instance removed in favor of Dependency Injection

