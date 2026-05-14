"""
Exception handling middleware and utilities.
Catches unhandled exceptions at the ASGI boundary and converts them into
structured JSON error responses rather than leaking raw Python stack traces.
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log every request with its duration and outcome.
    Attaches a unique request_id to structured log context for traceability.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        # Bind request context so all log entries within this request carry it
        import structlog
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Request completed",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )
            response.headers["X-Request-Id"] = request_id
            return response

        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "Unhandled exception in request",
                duration_ms=round(duration_ms, 2),
                error=str(exc),
            )
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "An unexpected internal error occurred.",
                    "request_id": request_id,
                },
                headers={"X-Request-Id": request_id},
            )
