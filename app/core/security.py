"""
Security module.
This microservice does NOT implement end-user authentication — that responsibility
belongs to the .NET backend. However, this module provides:

  1. Optional shared-secret validation so only the .NET API can call this service
     (defense-in-depth when deployed inside a private VNet is preferred, but the
     header approach works when a VNet is not available).
  2. Input sanitization helpers to prevent path traversal via blob IDs.
  3. A configurable CORS policy factory consumed by main.py.

Set INTERNAL_API_KEY to a non-empty value in the environment to enable key checking.
Leave it empty (default) to rely solely on network-level isolation (recommended for
Azure Container Apps with ingress restricted to internal traffic).
"""

import re
from typing import Optional

from fastapi import Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Blob IDs must be safe filenames: alphanumerics, hyphens, underscores, dots, slashes
_SAFE_BLOB_ID_RE = re.compile(r"^[\w\-./]+$")
_MAX_BLOB_ID_LENGTH = 512


def validate_blob_id(blob_id: str) -> str:
    """
    Validate a blob ID to prevent path-traversal or injection attacks.
    Raises ValueError on invalid input; returns the cleaned ID on success.
    """
    if not blob_id or len(blob_id) > _MAX_BLOB_ID_LENGTH:
        raise ValueError(f"Blob ID exceeds maximum length of {_MAX_BLOB_ID_LENGTH}.")
    if ".." in blob_id:
        raise ValueError("Blob ID must not contain '..' sequences.")
    if not _SAFE_BLOB_ID_RE.match(blob_id):
        raise ValueError(
            "Blob ID contains invalid characters. "
            "Only alphanumerics, hyphens, underscores, dots, and forward slashes are allowed."
        )
    return blob_id


async def verify_internal_api_key(
    x_internal_api_key: Optional[str] = Header(default=None),
) -> None:
    """
    FastAPI dependency that enforces an optional shared-secret header.
    If INTERNAL_API_KEY is not configured, this check is skipped entirely.
    """
    settings = get_settings()
    expected_key: Optional[str] = getattr(settings, "INTERNAL_API_KEY", None)

    if not expected_key:
        # Network-level isolation is the primary control; skip header check.
        return

    if x_internal_api_key != expected_key:
        logger.warning("Rejected request: invalid or missing X-Internal-Api-Key header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal API key.",
        )


def build_cors_middleware_kwargs() -> dict:
    """Return kwargs suitable for add_middleware(CORSMiddleware, **kwargs)."""
    settings = get_settings()
    return {
        "allow_origins": settings.CORS_ORIGINS,
        "allow_credentials": False,   # no cookies in a microservice
        "allow_methods": settings.CORS_METHODS,
        "allow_headers": settings.CORS_HEADERS,
    }
