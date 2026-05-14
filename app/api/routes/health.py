"""
Health check endpoint.
Used by Azure Container Apps' liveness and readiness probes.

GET /health  → returns {"status": "ok", "model_loaded": true, "version": "1.0.0"}
              HTTP 200 when healthy, HTTP 503 when the model is not yet loaded.

Configure in Azure Container Apps:
  Liveness probe:  GET /health  (period: 30s, failure threshold: 3)
  Readiness probe: GET /health  (period: 10s, failure threshold: 3)
"""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.ml.model_loader import is_model_loaded
from app.models.response_models import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    include_in_schema=True,
)
async def health_check() -> JSONResponse:
    settings = get_settings()
    model_loaded = is_model_loaded()

    response_body = HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=settings.APP_VERSION,
    )

    http_status = (
        status.HTTP_200_OK if model_loaded else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(
        content=response_body.model_dump(),
        status_code=http_status,
    )
