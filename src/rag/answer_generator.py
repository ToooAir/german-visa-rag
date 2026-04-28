"""
Answer generation with streaming support, caching, and source attribution.
Handles LLM calls and formats responses for OpenAI-compatible endpoints.
"""

import asyncio
import json
import re
import time
import uuid
from typing import Any, AsyncIterator, Optional

from src.config import settings
from src.exceptions import LLMGenerationError
from src.llm import get_llm_client
from src.llm.token_counter import get_token_counter
from src.logger import logger
from src.observability.mlflow_tracker import get_mlflow_tracker
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.prompt_builder import PromptRequest, get_prompt_builder
from src.rag.query_transformer import get_query_transformer
from src.rag.reranker import get_reranker
from src.rag.tag_filter import apply_english_c1_split_filter, apply_milestone2_filter, apply_path1_filter
from src.storage.redis_cache import query_cache

# Fallback messages when no retrieval results are found.
_NO_INFO_MSG: dict[str, str] = {
    "en": "I couldn't find relevant information in my database. "
    "Please check official resources: https://www.make-it-in-germany.com",
    "de": "Ich konnte keine relevanten Informationen in meiner Datenbank finden. "
    "Bitte prüfen Sie die offiziellen Ressourcen: https://www.make-it-in-germany.com",
    "zh-TW": "我查閱的資料庫中暫時沒有相關信息。" "請查詢官方資源：https://www.make-it-in-germany.com",
}


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

    # ─── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _flatten_and_deduplicate(
        batched_results: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Flatten batch results, deduplicate by chunk_id, sort by adjusted_score."""
        seen: set = set()
        flat: list[dict[str, Any]] = []
        for batch in batched_results:
            for result in batch:
                chunk_id = result.get("metadata", {}).get("chunk_id")
                if chunk_id not in seen:
                    flat.append(result)
                    seen.add(chunk_id)
        flat.sort(key=lambda x: x.get("adjusted_score", 0), reverse=True)
        return flat

    @staticmethod
    def _build_sources(reranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract source metadata from reranked documents."""
        return [
            {
                "url": r.get("metadata", {}).get("source_url"),
                "title": r.get("metadata", {}).get("source_title"),
                "authority": r.get("metadata", {}).get("authority_level"),
            }
            for r in reranked
        ]

    @staticmethod
    def _format_sse_chunk(content: str) -> str:
        """Format content as SSE JSON chunk (OpenAI-compatible)."""
        chunk = {"choices": [{"delta": {"content": content}, "finish_reason": None}]}
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    @staticmethod
    def _format_milestone_chunk(milestone_id: str, status: str) -> str:
        """Format milestone update as SSE JSON chunk."""
        chunk = {
            "choices": [],
            "metadata": {"achieved_milestone": {"id": milestone_id, "status": status}},
        }
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    @staticmethod
    def _format_req_chunk(req_id: str, value: str, status: str) -> str:
        """Format requirement update as SSE JSON chunk."""
        chunk = {"metadata": {"updated_requirement": {"id": req_id, "value": value, "status": status}}}
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    @staticmethod
    def _format_status_chunk(status: str) -> str:
        """Format status update as SSE JSON chunk."""
        chunk = {"choices": [], "metadata": {"status": status}}
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    @staticmethod
    def _format_search_queries_chunk(queries: list[str]) -> str:
        """Format search queries as SSE JSON chunk."""
        chunk = {"choices": [], "metadata": {"search_queries": queries}}
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    # ─── Public API ─────────────────────────────────────────────────────────

    async def generate_answer(
        self,
        query: str,
        language: str = "auto",
        visa_type: Optional[str] = None,
        requirements: Optional[list[dict[str, str]]] = None,
        top_k: Optional[int] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Generate answer without streaming."""
        start_time = time.time()

        # 0. Cache check
        cached_result = await query_cache.get(query)
        if cached_result:
            cached_result["metadata"]["latency_seconds"] = time.time() - start_time
            cached_result["metadata"]["cache_hit"] = True
            return cached_result

        try:
            # 1. Query transformation + build search queries (single LLM call)
            logger.info("Step 1: Transforming query")
            try:
                transformed = await self.query_transformer.transform_query(query)
                main_query = transformed["corrected_query"]
            except Exception as e:
                logger.error("Query expansion/correction failed: %s", e)
                main_query = query
                transformed = {}

            if not settings.enable_query_expansion or len(query) > 100:
                search_queries = [query]
            else:
                search_queries = [main_query]
                for key in ("english_query", "german_query"):
                    val = transformed.get(key)
                    if val and val not in search_queries:
                        search_queries.append(val)
                if len(search_queries) < 3:
                    for variant in transformed.get("query_variants", []):
                        if variant and variant not in search_queries:
                            search_queries.append(variant)
                            break
                search_queries = list(dict.fromkeys(filter(None, search_queries)))[:3]

            # 2. Retrieval
            logger.info("Step 2: Retrieving for queries: %s", search_queries)

            visa_types_filter = [visa_type] if visa_type else None
            all_results = await self.retriever.retrieve_batch(
                queries=search_queries,
                top_k=top_k or settings.retrieval_top_k_hybrid,
                visa_types=visa_types_filter,
            )
            retrieval_results = self._flatten_and_deduplicate(all_results)

            # Fallback: retry without visa filter if no results
            if not retrieval_results and visa_type:
                logger.info("No results with visa_type filter, retrying with broad search")
                all_results = await self.retriever.retrieve_batch(
                    queries=search_queries,
                    top_k=top_k or settings.retrieval_top_k_hybrid,
                )
                retrieval_results = self._flatten_and_deduplicate(all_results)

            if not retrieval_results:
                logger.warning("No context found after retrieval fallback")
                return {
                    "answer": _NO_INFO_MSG.get(language, _NO_INFO_MSG["en"]),
                    "sources": [],
                    "contexts": [],
                    "metadata": {"query": query, "retrieval_count": 0, "cache_hit": False},
                }

            # 3. Reranking
            reranked = await self.reranker.rerank(
                query=main_query,
                documents=retrieval_results,
                top_k=settings.retrieval_top_k_reranked,
            )

            # 4. Build prompt
            context = self.prompt_builder.build_context_from_retrieval(reranked, language=language or "en")
            request = PromptRequest(
                context=context,
                question=query,
                language=language,
                visa_type=visa_type,
                requirements=requirements,
            )
            system_prompt = self.prompt_builder.build_system_prompt(request)
            messages = [
                {"role": "system", "content": system_prompt},
                self.prompt_builder.build_user_message(query),
            ]

            # 5. LLM call
            logger.info("Step 5: Calling LLM (%s)", settings.openai_model)
            response_text = await self.llm.call_non_streaming(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or settings.max_response_tokens,
            )

            if len(response_text) > settings.max_response_chars:
                logger.warning(
                    "LLM response (%d chars) exceeds max_response_chars (%d); truncating",
                    len(response_text),
                    settings.max_response_chars,
                )
                response_text = response_text[: settings.max_response_chars]

            sources = self._build_sources(reranked)
            latency = time.time() - start_time

            result = {
                "answer": response_text,
                "sources": sources,
                "contexts": [(doc.get("text") or doc.get("content", "")) for doc in reranked],
                "metadata": {
                    "query": query,
                    "retrieval_count": len(reranked),
                    "latency_seconds": latency,
                    "cache_hit": False,
                },
            }

            # 6. Cache + observability
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
            logger.error("Answer generation failed: %s", e, exc_info=True)
            raise LLMGenerationError(f"Answer generation failed: {e}") from e

    async def generate_answer_streaming(
        self,
        query: str,
        language: str = "auto",
        visa_type: Optional[str] = None,
        requirements: Optional[list[dict[str, str]]] = None,
        top_k: Optional[int] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Generate answer with streaming response (SSE)."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        logger.info("Starting streaming answer generation (request_id=%s)", request_id)

        # 0. Cache check
        yield self._format_status_chunk("analyzing")
        cached_result = await query_cache.get(query)
        if cached_result:
            logger.info("Streaming from cache (request_id=%s)", request_id)
            answer = cached_result.get("answer", "")
            sources = cached_result.get("sources", [])

            yield f"data: {json.dumps({'choices': [], 'metadata': {'sources': sources}}, ensure_ascii=False)}\n\n"

            # Support legacy caches where tags were embedded in answer text
            milestone_pattern = re.compile(r"\[MILESTONE:([\d-]+):(\w+)\]")
            req_pattern = re.compile(r"\[REQ:([\d-]+):([^:]+):(\w+)\]")

            for m_id, m_status in milestone_pattern.findall(answer):
                yield self._format_milestone_chunk(m_id, m_status)
            for r_id, r_val, r_status in req_pattern.findall(answer):
                yield self._format_req_chunk(r_id, r_val, r_status)

            # New-format cache fields take precedence
            for m in cached_result.get("milestones", []):
                yield self._format_milestone_chunk(m["id"], m["status"])
            for r in cached_result.get("requirements", []):
                yield self._format_req_chunk(r["id"], r["value"], r["status"])

            clean_answer = milestone_pattern.sub("", req_pattern.sub("", answer))
            chunk_size = 30
            for i in range(0, len(clean_answer), chunk_size):
                yield self._format_sse_chunk(clean_answer[i : i + chunk_size])
                await asyncio.sleep(0.01)

            yield "data: [DONE]\n\n"
            return

        try:
            # 1. Query transformation + build search queries (single LLM call)
            transformed = await self.query_transformer.transform_query(query)
            main_query = transformed["corrected_query"]

            # 2. Retrieval
            yield self._format_status_chunk("retrieving")
            if not settings.enable_query_expansion or len(query) > 100:
                search_queries = [query]
            else:
                search_queries = [main_query]
                for key in ("english_query", "german_query"):
                    val = transformed.get(key)
                    if val and val not in search_queries:
                        search_queries.append(val)
                if len(search_queries) < 3:
                    for variant in transformed.get("query_variants", []):
                        if variant and variant not in search_queries:
                            search_queries.append(variant)
                            break
                search_queries = list(dict.fromkeys(filter(None, search_queries)))[:3]
            yield self._format_search_queries_chunk(search_queries)
            logger.info("Streaming retrieval for queries: %s (request_id=%s)", search_queries, request_id)

            visa_types_filter = [visa_type] if visa_type else None
            all_results = await self.retriever.retrieve_batch(
                queries=search_queries,
                top_k=top_k or settings.retrieval_top_k_hybrid,
                visa_types=visa_types_filter,
            )
            retrieval_results = self._flatten_and_deduplicate(all_results)

            if not retrieval_results and visa_type:
                logger.info("Streaming: no results with filter, retrying broad search")
                all_results = await self.retriever.retrieve_batch(queries=search_queries)
                retrieval_results = self._flatten_and_deduplicate(all_results)

            if not retrieval_results:
                yield self._format_sse_chunk(_NO_INFO_MSG.get(language, _NO_INFO_MSG["en"]))
                yield "data: [DONE]\n\n"
                return

            # 3. Reranking
            reranked = await self.reranker.rerank(
                query=main_query,
                documents=retrieval_results,
                top_k=settings.retrieval_top_k_reranked,
            )

            sources = self._build_sources(reranked)
            yield f"data: {json.dumps({'choices': [], 'metadata': {'sources': sources}}, ensure_ascii=False)}\n\n"
            yield self._format_status_chunk("extracting")

            # 4. Build prompt
            context = self.prompt_builder.build_context_from_retrieval(reranked, language=language or "en")
            request = PromptRequest(
                context=context,
                question=query,
                language=language,
                visa_type=visa_type,
                requirements=requirements,
            )
            system_prompt = self.prompt_builder.build_system_prompt(request)
            messages = [
                {"role": "system", "content": system_prompt},
                self.prompt_builder.build_user_message(query),
            ]
            input_tokens = self.token_counter.count_messages(messages)

            # 5. Stream LLM response
            yield self._format_status_chunk("synthesizing")
            full_response = ""
            # Raw (pre-filter) tag lists — collected during stream
            raw_milestones: list[dict] = []
            raw_requirements: list[dict] = []

            # Broad strip patterns catch malformed tags; strict patterns update UI state only
            strip_milestone = re.compile(r"\[MILESTONE:[^\]]+\]")
            strip_req = re.compile(r"\[REQ:[^\]]+\]")
            extract_milestone = re.compile(r"\[MILESTONE:([\d-]+):(\w+)\]")
            extract_req = re.compile(r"\[REQ:([\d-]+):([^:]+):(\w+)\]")

            tag_buffer = ""
            _size_limit_hit = False

            try:
                async for chunk in self.llm.call_streaming(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens or settings.max_response_tokens,
                ):
                    tag_buffer += chunk

                    if "[" in tag_buffer:
                        for m_id, m_status in extract_milestone.findall(tag_buffer):
                            raw_milestones.append({"id": m_id, "status": m_status})

                        for r_id, r_val, r_status in extract_req.findall(tag_buffer):
                            raw_requirements.append({"id": r_id, "value": r_val, "status": r_status})

                        tag_buffer = strip_milestone.sub("", tag_buffer)
                        tag_buffer = strip_req.sub("", tag_buffer)

                        if "[" in tag_buffer:
                            parts = tag_buffer.split("[")
                            to_send = "[".join(parts[:-1])
                            tag_buffer = "[" + parts[-1]
                            if to_send:
                                yield self._format_sse_chunk(to_send)
                                full_response += to_send
                        else:
                            if tag_buffer:
                                yield self._format_sse_chunk(tag_buffer)
                                full_response += tag_buffer
                            tag_buffer = ""
                    else:
                        if tag_buffer:
                            yield self._format_sse_chunk(tag_buffer)
                            full_response += tag_buffer
                        tag_buffer = ""

                    if len(full_response) > settings.max_response_chars:
                        logger.warning(
                            "Streaming response exceeded max_response_chars (%d) — aborting LLM stream (request_id=%s)",
                            settings.max_response_chars,
                            request_id,
                        )
                        _size_limit_hit = True
                        yield self._format_sse_chunk("\n\n[Response truncated: size limit reached]")
                        break

                # Flush remaining buffer
                if tag_buffer and not _size_limit_hit:
                    for m_id, m_status in extract_milestone.findall(tag_buffer):
                        raw_milestones.append({"id": m_id, "status": m_status})
                    for r_id, r_val, r_status in extract_req.findall(tag_buffer):
                        raw_requirements.append({"id": r_id, "value": r_val, "status": r_status})

                    clean_last = strip_milestone.sub("", strip_req.sub("", tag_buffer))
                    if clean_last:
                        yield self._format_sse_chunk(clean_last)
                        full_response += clean_last

                # Apply post-processing filters and emit tag events
                state_reqs: list[dict] = requirements or []
                corrected_reqs = apply_english_c1_split_filter(raw_requirements)
                filtered_reqs = apply_path1_filter(visa_type, state_reqs, corrected_reqs)
                filtered_milestones = apply_milestone2_filter(visa_type, state_reqs, [], filtered_reqs, raw_milestones)

                for r in filtered_reqs:
                    logger.info("Updating requirement: %s=%s (%s)", r["id"], r["value"], r["status"])
                    yield self._format_req_chunk(r["id"], r["value"], r["status"])
                for m in filtered_milestones:
                    logger.info("Triggering milestone: %s=%s", m["id"], m["status"])
                    yield self._format_milestone_chunk(m["id"], m["status"])

                # Expose filtered results for cache / observability
                achieved_milestones = filtered_milestones
                updated_requirements = filtered_reqs

            except Exception as e:
                logger.error("LLM streaming failed: %s (request_id=%s)", e, request_id)
                yield self._format_sse_chunk(f"\n\n[Generation interrupted: {e}]")
                achieved_milestones = []
                updated_requirements = []

            # 6. Observability + cache
            output_tokens = self.token_counter.count_text(full_response)
            latency = time.time() - start_time
            cost = self.token_counter.estimate_cost(input_tokens, output_tokens)

            logger.info(
                "Streaming completed (request_id=%s): latency=%.2fs tokens=%d/%d cost=$%.5f",
                request_id,
                latency,
                input_tokens,
                output_tokens,
                cost,
                extra={"retrieval_count": len(reranked)},
            )

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

            await query_cache.set(
                query,
                {
                    "answer": full_response,
                    "sources": sources,
                    "milestones": achieved_milestones,
                    "requirements": updated_requirements,
                    "metadata": {
                        "query": query,
                        "retrieval_count": len(reranked),
                        "cache_hit": False,
                    },
                },
            )

            yield f"data: {json.dumps({'choices': [], 'metadata': {'sources': sources, 'latency_seconds': latency}}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error("Answer generation failed: %s (request_id=%s)", e, request_id, exc_info=True)
            yield self._format_sse_chunk(f"[Error: {e}]")
            yield "data: [DONE]\n\n"


# Singleton instance removed in favour of Dependency Injection
