"""
Grad-CAM service.
Generates Gradient-weighted Class Activation Mapping (Grad-CAM) coordinates
and bounding box information for mobile app visualization.

Output Format
-------------
Returns a JSON object containing:
  - cam_grid: 7×7 Grad-CAM heat intensity values [0.0–1.0]
  - ink_bbox: Normalized bounding box {x, y, w, h, cx, cy, ...}
  - stroke_markers: Top-5 hottest points [(id, cx_norm, cy_norm, score), ...]

All coordinates are normalized to [0.0–1.0] relative to 224×224 image size.
The mobile app multiplies by actual screen dimensions to draw overlays.
"""

import asyncio
import io
import json
import uuid
from typing import Dict, List, Optional, Tuple

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
TARGET_LAYER_NAME = "backbone.conv_layers.24"
CAM_GRID_SIZE = 7      # 7×7 grid for mobile app
TOP_MARKERS = 5        # Top 5 hottest points


class GradCAMService:
    """
    Produces Grad-CAM coordinates and bounding box data for explainability.
    Returns JSON serialization with:
      - 7×7 cam_grid heat map
      - ink_bbox (signature bounding box)
      - stroke_markers (top-5 critical points)
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
        Generate Grad-CAM coordinates and bounding box data as JSON.

        Returns
        -------
        (json_bytes, blob_id)
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
        """
        Compute Grad-CAM, extract coordinates, and return JSON.
        """
        model = get_model()
        first_param = next(model.parameters())
        actual_device = first_param.device
        model_dtype = first_param.dtype
        
        tensor = questioned_tensor.to(device=actual_device, dtype=model_dtype)

        # Compute full-resolution Grad-CAM
        cam_fullres = self._compute_gradcam(tensor)

        if cam_fullres is None:
            # Fallback: empty coordinates
            logger.warning(
                "Grad-CAM computation failed; returning empty coordinates",
                case_name=case_name,
            )
            cam_fullres = np.zeros(
                (self._settings.MODEL_INPUT_SIZE, self._settings.MODEL_INPUT_SIZE),
                dtype=np.float32,
            )

        # Extract coordinates from full-resolution CAM
        cam_grid = self._extract_cam_grid(cam_fullres)
        
        # Compute ink bounding box from original image
        orig_gray = np.array(original_pil_image.convert("L"))
        ink_bbox = self._compute_ink_bbox(orig_gray)
        
        # Extract stroke markers (top-5 hottest points)
        stroke_markers = self._extract_stroke_markers(cam_fullres)

        # Serialize to JSON
        payload = {
            "cam_grid": {
                "grid_size": CAM_GRID_SIZE,
                "values": cam_grid,  # List[List[float]] 7×7
            },
            "ink_bbox": ink_bbox,
            "stroke_markers": stroke_markers,
        }

        json_bytes = json.dumps(payload).encode("utf-8")
        blob_id = self._build_blob_id(case_name)

        logger.info(
            "Grad-CAM coordinates generated",
            case_name=case_name,
            json_size=len(json_bytes),
            blob_id=blob_id,
        )

        return json_bytes, blob_id

    def _compute_gradcam(self, tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Compute the Grad-CAM activation map (full resolution 224×224).
        Returns a float32 ndarray of shape (H, W) with values in [0, 1],
        or None if the target layer could not be found.
        """
        model = get_model()
        
        # Get the actual device from model parameters
        first_param = next(model.parameters())
        actual_device = first_param.device
        model_dtype = first_param.dtype
        
        tensor = tensor.to(device=actual_device, dtype=model_dtype)

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
            # Save original model state
            original_training = model.training
            model.train()  # Enable gradients

            tensor.requires_grad_(True)

            with torch.enable_grad():
                # Forward pass
                embedding = model(tensor)
                score = embedding.norm(p=2, dim=1).sum()

                model.zero_grad()
                score.backward()

            if not activations or not gradients:
                return None

            acts = activations[0]     # (1, C, h, w)
            grads = gradients[0]      # (1, C, h, w)

            # Global average pool the gradients
            weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

            # Weighted combination
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

            # Normalize to [0, 1]
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
            model.train(original_training)

    def _extract_cam_grid(self, cam_fullres: np.ndarray) -> List[List[float]]:
        """
        Downsample full-resolution CAM (224×224) to a 7×7 grid.
        Each cell contains the mean intensity of its region.
        
        Returns
        -------
        List[List[float]] of shape (7, 7) with values in [0.0–1.0]
        """
        H, W = cam_fullres.shape
        grid_size = CAM_GRID_SIZE
        cell_h = H // grid_size
        cell_w = W // grid_size

        cam_grid = []
        for r in range(grid_size):
            row = []
            for c in range(grid_size):
                y1 = r * cell_h
                y2 = (r + 1) * cell_h if r < grid_size - 1 else H
                x1 = c * cell_w
                x2 = (c + 1) * cell_w if c < grid_size - 1 else W
                
                cell_mean = float(cam_fullres[y1:y2, x1:x2].mean())
                row.append(round(cell_mean, 4))
            cam_grid.append(row)

        return cam_grid

    def _compute_ink_bbox(self, gray_image: np.ndarray) -> Dict[str, float]:
        """
        Compute the bounding box of ink (non-white pixels) in the image.
        Returns normalized coordinates [0.0–1.0] relative to image size.
        
        Parameters
        ----------
        gray_image : np.ndarray
            Greyscale image (H, W)
            
        Returns
        -------
        Dict with keys: x, y, w, h, x2, y2, cx, cy, ratio
        """
        # Find non-white pixels (signature ink)
        # Assuming white background (255), ink is darker (<200)
        binary = gray_image < 200
        coords = np.argwhere(binary)

        if len(coords) == 0:
            # Empty image — return default bbox
            return {
                "x": 0.0,
                "y": 0.0,
                "w": 1.0,
                "h": 1.0,
                "x2": 1.0,
                "y2": 1.0,
                "cx": 0.5,
                "cy": 0.5,
                "ratio": 1.0,
            }

        # coords is (rows, cols) = (y, x)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Add padding
        pad = 10
        x1 = max(0, x_min - pad)
        y1 = max(0, y_min - pad)
        x2 = min(gray_image.shape[1], x_max + pad)
        y2 = min(gray_image.shape[0], y_max + pad)

        w_px = x2 - x1
        h_px = y2 - y1

        # Normalize by image size
        S = float(self._settings.MODEL_INPUT_SIZE)

        bbox_norm = {
            "x": round(x1 / S, 4),
            "y": round(y1 / S, 4),
            "w": round(w_px / S, 4),
            "h": round(h_px / S, 4),
            "x2": round(x2 / S, 4),
            "y2": round(y2 / S, 4),
            "cx": round((x1 + x2) / 2 / S, 4),
            "cy": round((y1 + y2) / 2 / S, 4),
            "ratio": round(w_px / max(h_px, 1), 2),
        }

        return bbox_norm

    def _extract_stroke_markers(self, cam_fullres: np.ndarray) -> List[Dict]:
        """
        Find the top-N hottest points in the Grad-CAM map.
        Returns normalized coordinates [0.0–1.0].
        
        Returns
        -------
        List[Dict] with keys: id, cx_norm, cy_norm, cx_px, cy_px, score
        """
        # Flatten and find top-N indices
        flat_cam = cam_fullres.flatten()
        top_n_indices = np.argsort(flat_cam)[-TOP_MARKERS:][::-1]

        markers = []
        S = float(self._settings.MODEL_INPUT_SIZE)

        for rank, flat_idx in enumerate(top_n_indices):
            cy_px, cx_px = np.unravel_index(flat_idx, cam_fullres.shape)
            score = float(flat_cam[flat_idx])

            marker = {
                "id": rank + 1,
                "cx_norm": round(float(cx_px) / S, 4),
                "cy_norm": round(float(cy_px) / S, 4),
                "cx_px": int(cx_px),
                "cy_px": int(cy_px),
                "score": round(score, 4),
            }
            markers.append(marker)

        return markers

    @staticmethod
    def _find_layer(model: torch.nn.Module, layer_name: str) -> Optional[torch.nn.Module]:
        """
        Traverse module hierarchy using dot-notation.
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
        Construct a unique blob ID for the JSON output.
        Format: gradcam-output/<sanitised_case>/<uuid>.json
        """
        prefix = self._settings.GRADCAM_OUTPUT_PREFIX
        safe_case = "".join(c if c.isalnum() or c in "-_" else "_" for c in case_name)
        unique = uuid.uuid4().hex[:12]
        return f"{prefix}/{safe_case}/{unique}.json"


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
        
        # Get the actual device from model parameters, not just what we stored
        first_param = next(model.parameters())
        actual_device = first_param.device
        model_dtype = first_param.dtype
        
        # Ensure tensor is on the same device and dtype as the model
        tensor = tensor.to(device=actual_device, dtype=model_dtype)

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
            # Save original model state to restore after Grad-CAM computation
            original_training = model.training
            model.train()  # Set to training mode to enable gradient computation

            tensor.requires_grad_(True)

            # Explicitly enable gradients within the forward/backward pass
            with torch.enable_grad():
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
            model.train(original_training)  # Restore original model state

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
