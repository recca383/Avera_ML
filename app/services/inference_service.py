"""
Inference service.
Runs the Siamese network forward pass and converts the raw embedding distance
into a human-readable verdict with confidence scores.

Architecture note
-----------------
The Siamese network produces an embedding vector for each input image.
Verification is performed by computing the L2 (Euclidean) distance between
the questioned embedding and the mean of the reference embeddings.
- distance < threshold  → GENUINE
- distance >= threshold → FORGED

If your model outputs similarity scores or logits directly (not embeddings),
replace the distance computation in _compute_distance() with the appropriate
post-processing for your architecture.
"""

import asyncio
from typing import List, Tuple

import torch
import torch.nn.functional as F

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.model_loader import get_model, get_model_device

logger = get_logger(__name__)


class InferenceService:
    """
    Runs Siamese network inference for signature verification.
    All heavy tensor operations are executed inside asyncio.to_thread()
    to avoid blocking the async event loop.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    # ── Public async API ─────────────────────────────────────────────────────

    async def verify(
        self,
        reference_tensors: torch.Tensor,   # (N, 1, H, W)
        questioned_tensor: torch.Tensor,   # (1, 1, H, W)
    ) -> Tuple[str, float, float, float, float]:
        """
        Run verification asynchronously.

        Returns
        -------
        tuple of (verdict, confidence_genuine, confidence_forged, distance, threshold)
        """
        return await asyncio.to_thread(
            self._run_inference, reference_tensors, questioned_tensor
        )

    # ── Synchronous core (runs in thread pool) ────────────────────────────────

    def _run_inference(
        self,
        reference_tensors: torch.Tensor,
        questioned_tensor: torch.Tensor,
    ) -> Tuple[str, float, float, float, float]:
        model = get_model()
        device = get_model_device()
        threshold = self._settings.INFERENCE_THRESHOLD

        # Debug: check actual device of model parameters
        first_param = next(model.parameters())
        actual_param_device = first_param.device
        model_dtype = first_param.dtype
        
        # Log device mismatch if detected
        if actual_param_device != device:
            logger.warning(
                "Model parameter device mismatch",
                expected_device=str(device),
                actual_param_device=str(actual_param_device),
                tensor_device=str(reference_tensors.device),
            )
        
        # Ensure tensors are on the correct device AND the actual model parameter device
        reference_tensors = reference_tensors.to(device=actual_param_device, dtype=model_dtype)
        questioned_tensor = questioned_tensor.to(device=actual_param_device, dtype=model_dtype)

        with torch.inference_mode():
            # ── Embed all reference images ───────────────────────────────────
            # Shape: (N, embedding_dim)
            reference_embeddings = model(reference_tensors)

            # Mean reference embedding aggregates across all provided references.
            mean_ref_embedding = reference_embeddings.mean(dim=0, keepdim=True)  # (1, D)

            # ── Embed questioned image ───────────────────────────────────────
            questioned_embedding = model(questioned_tensor)  # (1, D)

            # ── Compute L2 distance ──────────────────────────────────────────
            distance = self._compute_distance(mean_ref_embedding, questioned_embedding)

        verdict, conf_genuine, conf_forged = self._classify(distance, threshold)

        logger.info(
            "Inference complete",
            verdict=verdict,
            distance=round(distance, 6),
            threshold=threshold,
            confidence_genuine=round(conf_genuine, 4),
        )

        return verdict, conf_genuine, conf_forged, distance, threshold

    @staticmethod
    def _compute_distance(
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
    ) -> float:
        """
        Compute normalised L2 distance between two embedding vectors.
        L2-normalising before distance ensures the result is bounded in [0, 2].
        """
        a = F.normalize(embedding_a, p=2, dim=1)
        b = F.normalize(embedding_b, p=2, dim=1)
        dist = torch.norm(a - b, p=2, dim=1).item()
        return float(dist)

    @staticmethod
    def _classify(
        distance: float,
        threshold: float,
    ) -> Tuple[str, float, float]:
        """
        Convert distance to verdict and confidence percentages.

        Confidence is derived from a sigmoid-like mapping so that:
        - distance = 0       → 100 % genuine
        - distance = threshold → ~50 % genuine
        - distance >> threshold → ~0 % genuine

        This is an approximation. Replace with calibrated probabilities from
        a dedicated calibration head if your model provides them.
        """
        import math

        # Sigmoid centred on the threshold, scaled for sensitivity.
        scale = 6.0 / threshold if threshold > 0 else 6.0
        conf_genuine_raw = 1.0 / (1.0 + math.exp(scale * (distance - threshold)))
        conf_genuine = round(conf_genuine_raw * 100.0, 4)
        conf_forged = round(100.0 - conf_genuine, 4)

        verdict = "GENUINE" if distance < threshold else "FORGED"
        return verdict, conf_genuine, conf_forged


# ── Module-level singleton ────────────────────────────────────────────────────
inference_service = InferenceService()


def get_inference_service() -> InferenceService:
    """FastAPI dependency: returns the application-scoped InferenceService."""
    return inference_service
