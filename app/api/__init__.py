"""
api — FastAPI routers.
Aggregates all sub-routers and re-exports api_router for consumption by main.py.

Routers
-------
routes.health  : GET  /health  — liveness / readiness probe for Azure Container Apps.
routes.process : POST /process — full signature verification pipeline.
"""

from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.process import router as process_router

api_router = APIRouter()

#Health check — no prefix so Azure probes hit /health directly
api_router.include_router(health_router, tags=["Health"])

# Core inference endpoint
api_router.include_router(process_router, tags=["Inference"])
