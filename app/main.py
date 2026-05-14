"""
FastAPI application factory and entry point.

Startup sequence
----------------
1. Logging configured first — every subsequent log is captured.
2. Key Vault secrets loaded via Managed Identity (if AZURE_KEYVAULT_URL is set).
3. Azure Blob Storage client initialised (validates AD token + account URL).
4. ML model loaded from disk into memory.
5. Application begins accepting requests only after all of the above succeed.

Shutdown sequence
-----------------
1. Azure Blob Storage client closed (flushes HTTP connection pool).
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.security import build_cors_middleware_kwargs
from app.ml.model_loader import load_model
from app.services.blob_service import blob_storage_service
from app.services.keyvault_service import load_secrets_from_keyvault
from app.utils.middleware import RequestLoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """
    ASGI lifespan handler — startup before yield, shutdown after yield.
    FastAPI guarantees this completes before the first request is served.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    configure_logging()
    logger = get_logger(__name__)

    settings = get_settings()
    logger.info(
        "Starting up",
        app=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )

    # 1. Pull secrets from Key Vault (no-op if AZURE_KEYVAULT_URL is unset)
    await load_secrets_from_keyvault()

    # 2. Initialise Azure Blob Storage client (Managed Identity auth)
    await blob_storage_service.initialise()

    # 3. Load ML model — blocking by design; no requests served until ready
    load_model()

    logger.info("Startup complete — service is ready to accept requests")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down")
    await blob_storage_service.close()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """
    Application factory.
    Separating construction from the module-level variable makes it easy to
    create isolated instances in integration tests.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "ML inference microservice for the Signature Verification System. "
            "Intended to be called exclusively by the .NET backend API."
        ),
        # Disable interactive docs in production to reduce attack surface
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        openapi_url="/openapi.json" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan,
    )

    # ── Middleware (outermost = first to execute on each request) ─────────────
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(CORSMiddleware, **build_cors_middleware_kwargs())

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(api_router)

    return app


# Module-level app instance consumed by Uvicorn: `uvicorn app.main:app`
app = create_app()
