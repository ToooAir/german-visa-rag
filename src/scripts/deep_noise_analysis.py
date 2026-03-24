import sys
import os
import asyncio
import re
from typing import List, Dict, Counter

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.vector_db.qdrant_client_wrapper import get_qdrant_client
from src.config import settings

async def deep_noise_analysis(sample_size: int = 200):
    wrapper = get_qdrant_client()
    client = wrapper.client
    collection_name = settings.qdrant_collection_name
    
    print(f"Deep Noise Analysis (Sample Size: {sample_size}) on {collection_name}")
    
    noise_patterns = {
        "travel_advice": re.compile(r"Reise- und Sicherheitshinweise", re.I),
        "legal_footer": re.compile(r"Impressum|Datenschutz|Cookie|Privacy Policy|All rights reserved", re.I),
        "navigational": re.compile(r"Navigation|Menu|Breadcrumb|Search|Suche", re.I),
        "broken_content": re.compile(r"\[Download\]|\[PDF\]|fileadmin|/downloads/", re.I),
        "press_news": re.compile(r"Pressemitteilung|Current issues|Latest News|Veranstaltungen", re.I),
        "short_fragment": re.compile(r"^.{1,100}$", re.S),
        "social_media": re.compile(r"Facebook|Twitter|LinkedIn|Instagram|Share this", re.I)
    }
    
    results = Counter()
    noise_examples = {k: [] for k in noise_patterns}
    total_len = 0
    
    try:
        offset = None
        scanned = 0
        
        while scanned < sample_size:
            points, next_offset = await client.scroll(
                collection_name=collection_name,
                limit=min(100, sample_size - scanned),
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                break
                
            for p in points:
                payload = p.payload
                text = payload.get("text", "")
                title = payload.get("source_title", "")
                url = payload.get("source_url", "")
                full_content = f"{title} {text}"
                
                is_noisy = False
                for name, pattern in noise_patterns.items():
                    if pattern.search(full_content):
                        results[name] += 1
                        if len(noise_examples[name]) < 3:
                            noise_examples[name].append({"url": url, "title": title, "snippet": text[:100]})
                        is_noisy = True
                
                if not is_noisy:
                    results["clean"] += 1
                
                total_len += len(text)
                scanned += 1
            
            offset = next_offset
            if not offset:
                break
        
        print(f"\nSummary of {scanned} sampled chunks:")
        print(f"{'Pattern':<20} | {'Count':<6} | {'Percentage':<10}")
        print("-" * 40)
        for name in list(noise_patterns.keys()) + ["clean"]:
            count = results[name]
            print(f"{name:<20} | {count:<6} | {count/scanned:.1%}")
            
        print("\n--- Examples of Potential Noise ---")
        for name, examples in noise_examples.items():
            if examples:
                print(f"\n[{name.upper()}]")
                for ex in examples:
                    print(f"  - Title: {ex['title']}")
                    print(f"    URL: {ex['url']}")
                    print(f"    Snippet: {ex['snippet']}...")
                    
    except Exception as e:
        print(f"Error during analysis: {e}")
    finally:
        await wrapper.close()

if __name__ == "__main__":
    asyncio.run(deep_noise_analysis(200))
