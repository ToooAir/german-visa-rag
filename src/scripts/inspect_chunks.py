import sys
import os
import argparse
import random
from typing import List

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.vector_db.qdrant_client_wrapper import get_qdrant_client
from src.config import settings
from src.logger import logger

def inspect_chunks(sample_size: int = 5, collection_name: str = None):
    """
    Randomly sample chunks from Qdrant and print a quality scorecard.
    """
    wrapper = get_qdrant_client()
    client = wrapper.sync_client
    collection_name = collection_name or settings.qdrant_collection_name
    
    print(f"\n{'='*60}")
    print(f"🔍 Vector DB Quality Inspector - Collection: {collection_name}")
    print(f"{'='*60}\n")
    
    try:
        # Get total count
        collection_info = client.get_collection(collection_name=collection_name)
        total_points = collection_info.points_count
        print(f"📊 Total Chunks in DB: {total_points}")
        
        if total_points == 0:
            print("❌ No data found in the collection.")
            return

        # Simple random sampling (scroll with offset if needed, but here we just take some)
        # For a true random sample in Qdrant, we might need more complex logic, 
        # but for diagnostics, taking the first N is often enough or using a random offset.
        offset = random.randint(0, max(0, total_points - sample_size))
        
        results, next_offset = client.scroll(
            collection_name=collection_name,
            limit=sample_size,
            with_payload=True,
            with_vectors=False,
        )
        
        for i, point in enumerate(results):
            payload = point.payload
            text = payload.get("text", "")
            title = payload.get("source_title", "Unknown Title")
            url = payload.get("source_url", "N/A")
            is_parent = payload.get("is_parent", False)
            header = payload.get("section_header", "N/A")
            
            print(f"--- [Sample {i+1}] {'(PARENT)' if is_parent else '(CHILD)'} ---")
            print(f"📌 Source: {title}")
            print(f"🔗 URL: {url}")
            print(f"📂 Header: {header}")
            print(f"📏 Length: {len(text)} chars")
            print("-" * 30)
            print(f"TEXT CONTENT:\n{text[:500]}..." if len(text) > 500 else f"TEXT CONTENT:\n{text}")
            print("-" * 30)
            
            # Simple Heuristics for Quality Scorecard
            issues = []
            if len(text) < 100: issues.append("⚠️ Too short (<100 chars)")
            if "cookie" in text.lower() or "privacy policy" in text.lower(): issues.append("⚠️ Potential web noise (cookies/legal)")
            if not header or header == "Introduction": issues.append("ℹ️ generic header context")
            
            if issues:
                print("🚩 QUALITY ALERTS:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("✅ QUALITY LOOKS GOOD (Semantic & Contextual)")
            
            print("\n")

    except Exception as e:
        print(f"❌ Error inspecting collection: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect chunks in Qdrant for diagnostic purposes.")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of chunks to sample")
    parser.add_argument("--collection", type=str, help="Qdrant collection name")
    
    args = parser.parse_args()
    inspect_chunks(sample_size=args.sample_size, collection_name=args.collection)
