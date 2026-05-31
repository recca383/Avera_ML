"""
Request models — Pydantic schemas for incoming API payloads.
Validation is strict: missing or malformed fields will return HTTP 422 automatically.
"""

from typing import List

from pydantic import BaseModel, Field, field_validator

from app.core.security import validate_blob_id


class ProcessRequest(BaseModel):
    """
    Payload sent by the .NET backend to trigger a signature verification pipeline.

    Fields
    ------
    case_name           : Human-readable case identifier (used in logging and response).
    reference_image_ids : 1–10 blob IDs for reference (genuine) signatures.
    questioned_image_id : Blob ID for the questioned signature to be verified.
    """

    case_name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        examples=["Case 001"],
        description="Unique case name or identifier.",
    )
    reference_image_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        examples=[["ref1.png", "ref2.png", "ref3.png", "ref4.png"]],
        description="List of blob IDs for reference signature images (1–10).",
    )
    questioned_image_id: str = Field(
        ...,
        examples=["questioned.png"],
        description="Blob ID for the questioned signature image.",
    )
    output_blob_name: str | None = Field(
        None,
        examples=["gradcam-output/case-001.png"],
        description="Optional blob ID where the Grad-CAM result image should be uploaded.",
    )

    @field_validator("reference_image_ids", mode="before")
    @classmethod
    def validate_reference_ids(cls, v: List[str]) -> List[str]:
        return [validate_blob_id(blob_id) for blob_id in v]

    @field_validator("questioned_image_id", mode="before")
    @classmethod
    def validate_questioned_id(cls, v: str) -> str:
        return validate_blob_id(v)

    @field_validator("output_blob_name", mode="before")
    @classmethod
    def validate_output_blob_name(cls, v: str | None) -> str | None:
        return validate_blob_id(v) if v is not None else None

    model_config = {"json_schema_extra": {
        "example": {
            "case_name": "Case-0001",
            "reference_image_ids": ["G1.png", "G2.png", "G3.png", "G4.png"],
            "questioned_image_id": "F1.png",
            "output_blob_name": "output.json",
        }
    }}
