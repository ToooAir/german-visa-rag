"""API authentication and authorization using API keys."""

import secrets
from typing import Optional

from fastapi import Header, HTTPException, status

from src.config import settings
from src.logger import logger


class APIKeyAuth:
    """API Key-based authentication."""

    @staticmethod
    async def verify_admin_key(
        x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
    ) -> str:
        """Verify the privileged key guarding /admin/*.

        Production once ran with a global require_api_key flag unset, which turned
        verification off for every endpoint at once: /admin/ingest/stats answered
        with no key at all, and /admin/ingest/single would have accepted an
        arbitrary URL into the corpus. That flag is gone; this check has no off
        switch.

        Fails closed. With no admin_api_key configured the endpoints refuse to
        serve rather than falling open, because the failure mode of the old design
        was precisely that a missing setting meant "allow everyone".

        Read from its own header, so a key a client happens to send as X-API-Key
        cannot satisfy it by accident.
        """
        if not settings.admin_api_key:
            logger.error("Admin endpoint called but ADMIN_API_KEY is not configured — refusing")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Admin API is not configured on this deployment",
            )

        if not x_admin_key:
            logger.warning("Admin request missing key")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing admin key in X-Admin-Key header",
            )

        if not secrets.compare_digest(x_admin_key, settings.admin_api_key):
            logger.warning("Invalid admin key attempt")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid admin key",
            )

        return x_admin_key


auth = APIKeyAuth()
