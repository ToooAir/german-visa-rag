import asyncio
import os

import openai
from dotenv import load_dotenv

from src.llm.openai_client import OpenAIClient
from src.vector_db.embedder import OpenAIEmbedder, QuotaExhaustedError


async def test_provider(provider_name, use_azure):
    """
    Test connectivity for a specific provider (OpenAI or Azure).
    Verifies both Embedding and Chat Completion.
    """
    print(f"\n--- Testing {provider_name} ---")
    load_dotenv(override=True)

    # Configure for specific test
    os.environ["USE_AZURE_OPENAI"] = str(use_azure)

    try:
        if use_azure:
            # Azure configuration
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            llm_deployment = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-41-mini")
            emb_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

            print(f"Azure Config: Endpoint={endpoint}, LLM={llm_deployment}, EMB={emb_deployment}")

            # Explicitly set settings for this test run to override singleton behavior in src.config
            from src.config import settings

            settings.use_azure_openai = True
            settings.azure_openai_api_key = api_key
            settings.azure_openai_endpoint = endpoint
            settings.azure_llm_deployment = llm_deployment

            # Initialize clients
            llm = OpenAIClient(api_key=api_key)
            embedder = OpenAIEmbedder(
                api_key=api_key,
                model=emb_deployment,
                is_azure=True,
                azure_endpoint=endpoint,
                azure_api_version=api_version,
                azure_deployment=emb_deployment,
            )
        else:
            # OpenAI / GitHub Models configuration
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            emb_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

            print(f"OpenAI Config: Base={base_url}, LLM={model}, EMB={emb_model}")

            # Reset settings for standard OpenAI
            from src.config import settings

            settings.use_azure_openai = False

            llm = OpenAIClient(api_key=api_key, model=model, base_url=base_url)
            embedder = OpenAIEmbedder(api_key=api_key, model=emb_model, base_url=base_url)

        # 1. Test Embedding
        print("Testing Embedding...")
        emb = await embedder.embed_single("Hello world")
        print(f"✅ Embedding Success (Size: {len(emb)})")

        # 2. Test Chat
        print("Testing Chat Completion...")
        res = await llm.call_non_streaming([{"role": "user", "content": "Say 'Provider OK'"}])
        print(f"✅ Chat Success: {res.strip()}")

    except QuotaExhaustedError as e:
        print(f"❌ {provider_name} Quota/Rate Limit Exhausted!")
        print("   Reason: API key credits exhausted or hard rate limit hit.")

        # Display new detailed recovery info
        if e.reset_requests:
            print(f"   Recovery (Requests): Resets in {e.reset_requests}")
        if e.reset_tokens:
            print(f"   Recovery (Tokens): Resets in {e.reset_tokens}")
        if e.wait_seconds > 0:
            print(f"   Recovery (Wait): Try again in {e.wait_seconds} seconds.")

        if not (e.reset_requests or e.reset_tokens or e.wait_seconds > 0):
            print("   Recovery: Check your billing dashboard at https://platform.openai.com/account/billing")
        print(f"   Detail: {str(e)}")

    except openai.RateLimitError as e:
        print(f"❌ {provider_name} Rate Limit Reached!")
        headers = getattr(e, "response", None).headers if hasattr(e, "response") else {}
        reset_requests = headers.get("x-ratelimit-reset-requests")
        reset_tokens = headers.get("x-ratelimit-reset-tokens")

        print("   Reason: Too many requests (RPM/TPM limit).")
        if reset_requests:
            print(f"   Recovery (Requests): Resets in {reset_requests}")
        if reset_tokens:
            print(f"   Recovery (Tokens): Resets in {reset_tokens}")
        if not reset_requests and not reset_tokens:
            # Fallback to parsing message
            wait_seconds = OpenAIEmbedder._parse_wait_seconds(str(e))
            if wait_seconds > 0:
                print(f"   Recovery: Try again in {wait_seconds} seconds.")
            else:
                print(f"   Detail: {str(e)}")

    except openai.AuthenticationError:
        print(f"❌ {provider_name} Authentication Failed: Invalid API Key.")

    except Exception as e:
        print(f"❌ {provider_name} Failed: {type(e).__name__}: {str(e)}")


async def main():
    print("Starting Connectivity Tests for German Visa RAG Providers...")

    # Test OpenAI / GitHub
    await test_provider("OpenAI (GitHub Models)", use_azure=False)

    # Test Azure
    await test_provider("Azure OpenAI", use_azure=True)


if __name__ == "__main__":
    # Ensure PYTHONPATH is set so 'src' can be found
    # Run with: export PYTHONPATH=$PYTHONPATH:$(pwd) && python scripts/test_provider.py
    asyncio.run(main())
