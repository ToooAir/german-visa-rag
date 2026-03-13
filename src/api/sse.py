"""Server-Sent Events (SSE) streaming utilities."""

from typing import AsyncIterator
import json
from fastapi.responses import StreamingResponse


class SSEFormatter:
    """Format responses as Server-Sent Events."""

    @staticmethod
    def format_event(event: str, data: dict) -> str:
        """
        Format event as SSE.
        
        Args:
            event: Event type
            data: Event data dict
            
        Returns:
            SSE-formatted string
        """
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    @staticmethod
    def format_done() -> str:
        """Format end-of-stream marker."""
        return "data: [DONE]\n\n"

    @staticmethod
    async def stream_to_sse(
        async_generator: AsyncIterator[str],
    ) -> AsyncIterator[str]:
        """
        Convert async generator to SSE format.
        
        The generator should yield JSON strings or [DONE].
        """
        async for item in async_generator:
            yield item


def create_sse_response(
    async_generator: AsyncIterator[str],
) -> StreamingResponse:
    """Create SSE streaming response."""
    return StreamingResponse(
        SSEFormatter.stream_to_sse(async_generator),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
