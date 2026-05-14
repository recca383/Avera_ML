"""
Grad-CAM service.
Generates a Gradient-weighted Class Activation Mapping (Grad-CAM) heatmap for
the questioned signature image to provide visual explainability.

Architecture note
-----------------
Grad-CAM requires access to intermediate feature maps and their gradients.
Because the model is loaded as a TorchScript module, we use register_forward_hook
and register_full_backward_hook on the last convolutional layer.

If your model is NOT a TorchScript module (i.e. you have the Python class), you
can subclass it and override forward() to expose the target layer directly, which
gives cleaner gradient access.

The target layer name ("features.7" in the default) must match the actual layer
name in your exported model. Update TARGET_LAYER_NAME to match your architecture.
"""

import asyncio
import io
import uuid
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.model_loader import get_model, get_model_device

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
# Name of the target convolutional layer within the TorchScript module.
# Inspect your model with: [name for name, _ in model.named_modules()]
# Common examples:
#   - ResNet-based:  "backbone.layer4"
#   - VGG-based:     "features.28"
#   - Custom CNN:    "encoder.conv5"
TARGET_LAYER_NAME = "backbone.layer4"


class GradCAMService:
    """
    Produces Grad-CAM heatmap overlays for explainability.
    The heat map is blended with the original questioned image and returned
    as PNG bytes ready to be uploaded to Azure Blob Storage.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    # ── Public async API ─────────────────────────────────────────────────────

    async def generate(
        self,
        questioned_tensor: torch.Tensor,   # (1, 1, H, W)
        original_pil_image: Image.Image,   # RGB PIL image (H, W)
        case_name: str,
    ) -> Tuple[bytes, str]:
        """
        Generate Grad-CAM heatmap, overlay it on the original image, and return
        the result as PNG bytes together with a unique blob ID.

        Returns
        -------
        (png_bytes, blob_id)
        """
        return await asyncio.to_thread(
            self._run_gradcam, questioned_tensor, original_pil_image, case_name
        )

    # ── Synchronous core (runs in thread pool) ────────────────────────────────

    def _run_gradcam(
        self,
        questioned_tensor: torch.Tensor,
        original_pil_image: Image.Image,
        case_name: str,
    ) -> Tuple[bytes, str]:
        device = get_model_device()
        tensor = questioned_tensor.to(device)

        cam = self._compute_gradcam(tensor)

        if cam is None:
            # Fall back to a plain copy of the original if Grad-CAM fails.
            logger.warning(
                "Grad-CAM computation failed; returning original image as fallback",
                case_name=case_name,
            )
            cam = np.zeros(
                (self._settings.MODEL_INPUT_SIZE, self._settings.MODEL_INPUT_SIZE),
                dtype=np.float32,
            )

        overlay_bytes = self._overlay_heatmap(cam, original_pil_image)
        blob_id = self._build_blob_id(case_name)
        return overlay_bytes, blob_id

    def _compute_gradcam(self, tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Compute the Grad-CAM activation map for the questioned image tensor.
        Returns a float32 ndarray of shape (H, W) with values in [0, 1],
        or None if the target layer could not be found.
        """
        model = get_model()

        # Storage for hook data
        activations: list = []
        gradients: list = []

        def forward_hook(module, _input, output):  # noqa: ANN001
            activations.append(output.detach())

        def backward_hook(module, _grad_input, grad_output):  # noqa: ANN001
            gradients.append(grad_output[0].detach())

        # Register hooks on the target layer
        target_layer = self._find_layer(model, TARGET_LAYER_NAME)
        if target_layer is None:
            logger.warning(
                "Target layer not found; Grad-CAM unavailable",
                target_layer=TARGET_LAYER_NAME,
            )
            return None

        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            tensor.requires_grad_(True)

            # Forward pass — the Siamese network expects a single image here
            # We compute the embedding norm as the scalar to back-propagate through.
            embedding = model(tensor)
            score = embedding.norm(p=2, dim=1).sum()

            model.zero_grad()
            score.backward()

            if not activations or not gradients:
                return None

            acts = activations[0]     # (1, C, h, w)
            grads = gradients[0]      # (1, C, h, w)

            # Global average pool the gradients (channel weights)
            weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

            # Weighted combination of activation maps
            cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
            cam = F.relu(cam)

            # Upsample to model input size
            target_size = self._settings.MODEL_INPUT_SIZE
            cam = F.interpolate(
                cam,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )

            cam_np = cam.squeeze().cpu().numpy()

            # Normalise to [0, 1]
            cam_min, cam_max = cam_np.min(), cam_np.max()
            if cam_max - cam_min > 1e-8:
                cam_np = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                cam_np = np.zeros_like(cam_np)

            return cam_np.astype(np.float32)

        finally:
            fwd_handle.remove()
            bwd_handle.remove()
            tensor.requires_grad_(False)

    def _overlay_heatmap(
        self,
        cam: np.ndarray,
        original_pil: Image.Image,
    ) -> bytes:
        """
        Blend the Grad-CAM heatmap with the original image and encode as PNG bytes.
        """
        alpha = self._settings.GRADCAM_ALPHA
        target_size = self._settings.MODEL_INPUT_SIZE

        # Convert original PIL image to OpenCV BGR
        original_rgb = np.array(original_pil.convert("RGB").resize((target_size, target_size)))
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

        # Apply JET colormap to the activation map
        cam_uint8 = (cam * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

        # Blend
        overlay = cv2.addWeighted(heatmap_bgr, alpha, original_bgr, 1 - alpha, 0)

        # Encode to PNG bytes in-memory (no temp file)
        success, buffer = cv2.imencode(".png", overlay)
        if not success:
            raise RuntimeError("Failed to encode Grad-CAM overlay as PNG.")

        return buffer.tobytes()

    @staticmethod
    def _find_layer(model: torch.nn.Module, layer_name: str) -> Optional[torch.nn.Module]:
        """
        Traverse module hierarchy using dot-notation (e.g. 'backbone.layer4').
        Returns None if the layer path does not exist.
        """
        parts = layer_name.split(".")
        current = model
        for part in parts:
            current = getattr(current, part, None)
            if current is None:
                return None
        return current

    def _build_blob_id(self, case_name: str) -> str:
        """
        Construct a unique, URL-safe blob ID for the Grad-CAM output image.
        Format: gradcam-output/<sanitised_case>/<uuid>.png
        """
        prefix = self._settings.GRADCAM_OUTPUT_PREFIX
        # Sanitise case_name for use in a blob path
        safe_case = "".join(c if c.isalnum() or c in "-_" else "_" for c in case_name)
        unique = uuid.uuid4().hex[:12]
        return f"{prefix}/{safe_case}/{unique}.png"


# ── Module-level singleton ────────────────────────────────────────────────────
gradcam_service = GradCAMService()


def get_gradcam_service() -> GradCAMService:
    """FastAPI dependency: returns the application-scoped GradCAMService."""
    return gradcam_service
