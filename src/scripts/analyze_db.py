import sys
import os
import asyncio
from typing import Dict, Counter

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.vector_db.qdrant_client_wrapper import get_qdrant_client
from src.config import settings

async def analyze_distribution():
    wrapper = get_qdrant_client()
    client = wrapper.client
    collection_name = settings.qdrant_collection_name
    
    print(f"Analyzing collection: {collection_name}")
    
    try:
        counts = Counter()
        offset = None
        limit = 100
        total_scanned = 0
        
        while True:
            points, next_offset = await client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                break
                
            for point in points:
                auth = point.payload.get("authority_level", "unknown")
                counts[auth] += 1
                total_scanned += 1
            
            offset = next_offset
            if not offset or total_scanned >= 2000: # Scan up to 2000 for a quick snapshot
                break
        
        print(f"\nSource Authority Distribution (sampled first {total_scanned} points):")
        for auth, count in counts.items():
            print(f"  - {auth}: {count} ({count/total_scanned:.1%})")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await wrapper.close()

if __name__ == "__main__":
    asyncio.run(analyze_distribution())
