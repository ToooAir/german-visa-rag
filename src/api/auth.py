"""API authentication and authorization using API keys."""

from typing import Optional
from fastapi import HTTPException, Header, status

from src.config import settings
from src.logger import logger


class APIKeyAuth:
    """API Key-based authentication."""

    @staticmethod
    async def verify_api_key(
        x_api_key: Optional[str] = Header(None),
    ) -> str:
        """
        Verify API key from header.
        
        Args:
            x_api_key: API key from X-API-Key header
            
        Returns:
            API key if valid
            
        Raises:
            HTTPException if invalid
        """
        if not x_api_key:
            logger.warning("Request missing API key")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key in X-API-Key header",
            )
        
        if x_api_key != settings.api_key:
            logger.warning(f"Invalid API key attempt: {x_api_key[:10]}...")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key",
            )
        
        return x_api_key


auth = APIKeyAuth()
