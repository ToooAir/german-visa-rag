import asyncio
import os
import sys

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import settings
from src.vector_db.qdrant_client_wrapper import get_qdrant_client


async def count_noise():
    wrapper = get_qdrant_client()
    client = wrapper.client
    collection_name = settings.qdrant_collection_name

    count = 0
    offset = None
    scanned = 0

    try:
        while True:
            points, next_offset = await client.scroll(
                collection_name=collection_name, limit=100, offset=offset, with_payload=True, with_vectors=False
            )

            if not points:
                break

            for p in points:
                title = p.payload.get("source_title", "")
                if "Reise- und Sicherheitshinweise" in title:
                    count += 1

            scanned += len(points)
            offset = next_offset
            if not offset or scanned >= 3000:  # limit scan for speed
                break

        print(f"Sampled {scanned} chunks.")
        print(f"Found {count} travel advice (noise) chunks.")
        print(f"Noise ratio: {count/scanned:.1%}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await wrapper.close()


if __name__ == "__main__":
    asyncio.run(count_noise())
