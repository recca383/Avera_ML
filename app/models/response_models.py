"""
Response models — Pydantic schemas for outgoing API responses.
All numeric fields are rounded at the serialization boundary so callers
receive consistent precision without surprises.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ProcessResponse(BaseModel):
    """
    Inference result returned to the .NET backend after the full pipeline completes.

    Fields
    ------
    case_name           : Echoed from the request for easy correlation.
    verdict             : GENUINE or FORGED classification.
    confidence_genuine  : Probability (0–100) that the signature is genuine.
    confidence_forged   : Probability (0–100) that the signature is forged.
    distance            : Raw embedding distance produced by the Siamese network.
    threshold           : Distance threshold used for this inference run.
    gradcam_blob_ids    : Blob IDs of the uploaded Grad-CAM visualization images.
    """

    case_name: str = Field(..., description="Case identifier echoed from the request.")
    verdict: Literal["GENUINE", "FORGED"] = Field(
        ..., description="Final verdict: GENUINE or FORGED."
    )
    confidence_genuine: float = Field(
        ..., ge=0.0, le=100.0, description="Confidence that the signature is genuine (%)."
    )
    confidence_forged: float = Field(
        ..., ge=0.0, le=100.0, description="Confidence that the signature is forged (%)."
    )
    distance: float = Field(
        ..., ge=0.0, description="Embedding distance between questioned and reference signatures."
    )
    threshold: float = Field(
        ..., ge=0.0, description="Decision threshold applied during this inference."
    )
    gradcam_blob_ids: List[str] = Field(
        ..., description="Blob IDs of the Grad-CAM visualization images uploaded to Azure Blob Storage."
    )

    model_config = {"json_schema_extra": {
        "example": {
            "case_name": "Case 001",
            "verdict": "GENUINE",
            "confidence_genuine": 94.25,
            "confidence_forged": 5.75,
            "distance": 0.214562,
            "threshold": 0.485123,
            "gradcam_blob_ids": ["gradcam-output/case-001/query_case-001_original.png"],
        }
    }}


class HealthResponse(BaseModel):
    """Lightweight health-check response consumed by Azure Container Apps probe."""

    status: Literal["ok", "degraded"] = "ok"
    model_loaded: bool = False
    version: Optional[str] = None
