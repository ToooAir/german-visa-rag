import asyncio
import json
import os
import sys

# Ensure src is in path
sys.path.append(os.getcwd())

from src.rag.answer_generator import AnswerGenerator
from src.rag.hybrid_retriever import HybridRetriever
from src.vector_db.qdrant_client_wrapper import QdrantWrapper
from src.vector_db.embedder import OpenAIEmbedder

from dotenv import load_dotenv
load_dotenv()

async def test_rag_trace_stream():
    print("Testing RAG Trace Streaming Metadata...")
    
    # Initialize components
    api_key = os.getenv("OPENAI_API_KEY")
    qdrant = QdrantWrapper()
    # HybridRetriever only takes qdrant_client, embedder is imported at module level
    retriever = HybridRetriever(qdrant)
    generator = AnswerGenerator(retriever)
    
    query = "What are the points for Chancenkarte if I am 35?"
    
    print(f"Query: {query}")
    print("-" * 50)
    
    found_search_queries = False
    found_early_sources = False
    found_status = []
    
    async for chunk in generator.generate_answer_streaming(query, language="en"):
        if chunk.startswith("data: "):
            data_str = chunk[6:].strip()
            if data_str == "[DONE]":
                print("\n[DONE]")
                break
                
            try:
                data = json.loads(data_str)
                metadata = data.get("metadata", {})
                
                if "status" in metadata:
                    status = metadata["status"]
                    found_status.append(status)
                    print(f"Status Update: {status}")
                
                if "search_queries" in metadata:
                    queries = metadata["search_queries"]
                    found_search_queries = True
                    print(f"Search Queries Found: {queries}")
                
                if "sources" in metadata:
                    sources = metadata["sources"]
                    # If sources found while no content has been streamed yet, it's early visibility
                    if not data.get("choices") or not data["choices"][0].get("delta", {}).get("content"):
                        found_early_sources = True
                        print(f"Early Sources Found: {len(sources)} sources")
                
                if data.get("choices") and data["choices"][0].get("delta", {}).get("content"):
                    content = data["choices"][0]["delta"]["content"]
                    print(content, end="", flush=True)
                    
            except json.JSONDecodeError:
                pass

    print("\n" + "-" * 50)
    print("Verification Results:")
    print(f"✅ Status Flow: {' -> '.join(found_status)}")
    print(f"✅ Search Queries Emitted: {found_search_queries}")
    print(f"✅ Early Sources Emitted: {found_early_sources}")
    
    assert found_search_queries, "Should have emitted search queries"
    assert found_early_sources, "Should have emitted sources before content"
    assert "retrieving" in found_status, "Should have retrieving status"
    assert "synthesizing" in found_status, "Should have synthesizing status"
    
    print("\nALL TRACE VERIFICATIONS PASSED!")

if __name__ == "__main__":
    asyncio.run(test_rag_trace_stream())
