"""
/process endpoint.
Orchestrates the complete signature verification pipeline:
  1. Download images from Azure Blob Storage
  2. Preprocess images into tensors
  3. Run Siamese network inference
  4. Generate Grad-CAM heatmap
  5. Upload Grad-CAM image to Azure Blob Storage
  6. Return structured verification result

Error handling philosophy
-------------------------
- HTTP 400 → invalid request data (bad blob IDs, malformed JSON)
- HTTP 404 → one or more blobs not found in storage
- HTTP 422 → Pydantic validation failure (handled automatically by FastAPI)
- HTTP 500 → unexpected internal error (logged with full traceback)
- HTTP 503 → model not loaded (startup dependency failure)
"""

import asyncio
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.logging import get_logger
from app.core.security import verify_internal_api_key
from app.models.request_models import ProcessRequest
from app.models.response_models import ProcessResponse
from app.services.blob_service import BlobStorageService, get_blob_storage_service
from app.services.gradcam_service import GradCAMService, get_gradcam_service
from app.services.inference_service import InferenceService, get_inference_service
from app.services.preprocessing_service import PreprocessingService, get_preprocessing_service

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/process",
    response_model=ProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Run signature verification pipeline",
    description=(
        "Downloads reference and questioned signature images from Azure Blob Storage, "
        "runs Siamese network inference, generates a Grad-CAM heatmap, uploads it back "
        "to Blob Storage, and returns the verification result."
    ),
    responses={
        400: {"description": "Invalid request — bad blob ID format or empty image data"},
        404: {"description": "One or more blobs not found in Azure Blob Storage"},
        503: {"description": "ML model not loaded (service starting up)"},
    },
    dependencies=[Depends(verify_internal_api_key)],
)
async def process(
    request: ProcessRequest,
    blob_svc: Annotated[BlobStorageService, Depends(get_blob_storage_service)],
    preprocessing_svc: Annotated[PreprocessingService, Depends(get_preprocessing_service)],
    inference_svc: Annotated[InferenceService, Depends(get_inference_service)],
    gradcam_svc: Annotated[GradCAMService, Depends(get_gradcam_service)],
) -> ProcessResponse:
    """
    Full signature verification pipeline endpoint.
    """
    from app.ml.model_loader import is_model_loaded

    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model is not yet loaded. Please retry in a moment.",
        )

    logger.info(
        "Processing verification request",
        case_name=request.case_name,
        num_references=len(request.reference_image_ids),
        questioned_id=request.questioned_image_id,
    )

    # ── Step 1: Download images concurrently ─────────────────────────────────
    try:
        download_tasks = [
            blob_svc.download_blob(f"{request.case_name}/{blob_id}")
            for blob_id in request.reference_image_ids
        ] + [blob_svc.download_blob(f"{request.case_name}/{request.questioned_image_id}")]

        all_bytes: List[bytes] = await asyncio.gather(*download_tasks)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.error("Blob download failed", case_name=request.case_name, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to retrieve images from storage: {exc}",
        ) from exc

    reference_bytes_list = all_bytes[:-1]
    questioned_bytes = all_bytes[-1]

    if not questioned_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Questioned image blob '{request.questioned_image_id}' is empty.",
        )

    # ── Step 2: Preprocess images ─────────────────────────────────────────────
    try:
        reference_tensors, questioned_tensor, questioned_pil = await asyncio.to_thread(
            _preprocess_all,
            preprocessing_svc,
            reference_bytes_list,
            questioned_bytes,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image preprocessing failed: {exc}",
        ) from exc

    # ── Steps 3 + 4: Inference & Grad-CAM (sequential to avoid model state race) ───────
    try:
        # Run inference first (inference_mode, no gradients needed)
        inference_result = await inference_svc.verify(reference_tensors, questioned_tensor)
        # Then run Grad-CAM (requires train mode and gradients enabled)
        gradcam_result = await gradcam_svc.generate(questioned_tensor, questioned_pil, request.case_name)
    except Exception as exc:
        logger.exception(
            "Inference or Grad-CAM generation failed",
            case_name=request.case_name,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during inference. Please try again.",
        ) from exc

    verdict, conf_genuine, conf_forged, distance, threshold = inference_result
    gradcam_png_bytes, gradcam_blob_id = gradcam_result
    if request.output_blob_name:
        gradcam_blob_id = request.output_blob_name

    # ── Step 5: Upload Grad-CAM image ─────────────────────────────────────────
    try:
        await blob_svc.upload_blob(
            blob_id=f"{request.case_name}/{gradcam_blob_id}",
            data=gradcam_png_bytes,
            content_type="image/png",
        )
    except RuntimeError as exc:
        logger.error(
            "Grad-CAM upload failed",
            case_name=request.case_name,
            gradcam_blob_id=gradcam_blob_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to upload Grad-CAM image: {exc}",
        ) from exc

    logger.info(
        "Verification pipeline completed",
        case_name=request.case_name,
        verdict=verdict,
        distance=round(distance, 6),
        gradcam_blob_id=gradcam_blob_id,
    )

    return ProcessResponse(
        case_name=request.case_name,
        verdict=verdict,
        confidence_genuine=conf_genuine,
        confidence_forged=conf_forged,
        distance=distance,
        threshold=threshold,
        gradcam_blob_id=gradcam_blob_id,
    )


# ── Helper (synchronous, runs in thread pool) ─────────────────────────────────

def _preprocess_all(
    svc: PreprocessingService,
    reference_bytes_list: List[bytes],
    questioned_bytes: bytes,
):
    """
    Preprocess all images in a single thread-pool task to reduce context-switch overhead.
    Returns (reference_tensors, questioned_tensor, questioned_pil).
    """
    import torch

    reference_tensors = svc.preprocess_batch(reference_bytes_list)
    questioned_tensor = svc.preprocess_image_bytes(questioned_bytes)
    questioned_pil = svc.bytes_to_pil(questioned_bytes)
    return reference_tensors, questioned_tensor, questioned_pil
