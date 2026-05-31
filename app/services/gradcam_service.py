"""
Grad-CAM service.
Generates Gradient-weighted Class Activation Mapping (Grad-CAM) coordinates
and bounding box information for mobile app visualization.

Output Format
-------------
Returns a JSON object (as bytes) containing:
  - cam_grid:       7×7 Grad-CAM heat intensity values [0.0–1.0]
  - ink_bbox:       Normalized bounding box {x, y, w, h, cx, cy, …}
  - stroke_markers: Top-5 hottest points [{id, cx_norm, cy_norm, score}, …]

All coordinates are normalized to [0.0–1.0] relative to 224×224 image size.
The mobile app multiplies by actual screen dimensions to draw overlays.

Blob name written to Azure: V1.json
"""

import asyncio
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.model_loader import get_model

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_LAYER_NAME = "backbone.conv_layers.24"
CAM_GRID_SIZE = 7    # 7×7 grid for mobile app
TOP_MARKERS = 5      # top-N hottest points exported as stroke markers


class GradCAMService:
    """
    Produces Grad-CAM coordinates and bounding-box data for explainability.

    Calling ``generate()`` returns:
        (json_bytes, blob_id)

    where ``json_bytes`` is a UTF-8-encoded JSON payload ready to be uploaded
    to Azure Blob Storage as ``V1.json``, and ``blob_id`` is the full blob path.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    # ── Public async API ──────────────────────────────────────────────────────

    async def generate(
        self,
        questioned_tensor: torch.Tensor,  # (1, 1, H, W)
        original_pil_image: Image.Image,  # PIL image (any mode; converted internally)
        case_name: str,
    ) -> Tuple[bytes, str]:
        """
        Generate Grad-CAM JSON coordinate payload.

        Returns
        -------
        (json_bytes, blob_id)
            json_bytes : UTF-8 JSON ready to upload as application/json
            blob_id    : full blob path, e.g. "gradcam-output/case_xyz/V1.json"
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
        Compute Grad-CAM, extract coordinates, serialise to JSON, return bytes.
        """
        model = get_model()
        first_param = next(model.parameters())
        actual_device = first_param.device
        model_dtype = first_param.dtype

        tensor = questioned_tensor.to(device=actual_device, dtype=model_dtype)

        # ── Grad-CAM activation map (224×224 float32) ─────────────────────────
        cam_fullres = self._compute_gradcam(tensor)

        if cam_fullres is None:
            logger.warning(
                "Grad-CAM computation failed; returning zero map",
                case_name=case_name,
            )
            cam_fullres = np.zeros(
                (self._settings.MODEL_INPUT_SIZE, self._settings.MODEL_INPUT_SIZE),
                dtype=np.float32,
            )

        # ── Coordinate structures ─────────────────────────────────────────────
        cam_grid = self._extract_cam_grid(cam_fullres)

        orig_gray = np.array(original_pil_image.convert("L"))
        ink_bbox = self._compute_ink_bbox(orig_gray)

        stroke_markers = self._extract_stroke_markers(cam_fullres)

        # ── Serialise ─────────────────────────────────────────────────────────
        payload = {
            "cam_grid": {
                "grid_size": CAM_GRID_SIZE,
                "values": cam_grid,  # List[List[float]] 7×7
                "how_to_draw": (
                    "For each cell (r,c): draw rect at "
                    "x=c/7·screenW, y=r/7·screenH, "
                    "w=screenW/7, h=screenH/7, "
                    "color=jet_colormap(values[r][c])"
                ),
            },
            "ink_bbox": {
                **ink_bbox,
                "how_to_draw": (
                    "Draw rect at "
                    "x=x·screenW, y=y·screenH, "
                    "w=w·screenW, h=h·screenH"
                ),
            },
            "stroke_markers": {
                "count": len(stroke_markers),
                "markers": stroke_markers,
                "how_to_draw": (
                    "For each marker: draw circle at "
                    "(cx_norm·screenW, cy_norm·screenH), "
                    'label it marker["id"], '
                    "draw arrow from bubble in margin toward this point"
                ),
            },
        }

        json_bytes = json.dumps(payload, indent=2).encode("utf-8")
        blob_id = self._build_blob_id(case_name)

        logger.info(
            "Grad-CAM JSON coordinates generated",
            case_name=case_name,
            json_size_bytes=len(json_bytes),
            blob_id=blob_id,
        )

        return json_bytes, blob_id

    # ── Grad-CAM computation ──────────────────────────────────────────────────

    def _compute_gradcam(self, tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Compute the Grad-CAM activation map (full resolution 224×224).

        Returns a float32 ndarray of shape (H, W) with values in [0, 1],
        or None if the target layer could not be found.
        """
        model = get_model()
        first_param = next(model.parameters())
        actual_device = first_param.device
        model_dtype = first_param.dtype

        tensor = tensor.to(device=actual_device, dtype=model_dtype)

        activations: list = []
        gradients: list = []

        def forward_hook(module, _input, output):  # noqa: ANN001
            activations.append(output.detach())

        def backward_hook(module, _grad_input, grad_output):  # noqa: ANN001
            gradients.append(grad_output[0].detach())

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
            original_training = model.training
            model.train()  # enable gradient computation

            tensor.requires_grad_(True)

            with torch.enable_grad():
                embedding = model(tensor)
                score = embedding.sum()
                model.zero_grad()
                score.backward()

            if not activations or not gradients:
                return None

            acts = activations[0]   # (1, C, h, w)
            grads = gradients[0]    # (1, C, h, w)

            weights = grads.mean(dim=(2, 3), keepdim=True)        # (1, C, 1, 1)
            cam = (weights * acts).sum(dim=1, keepdim=True)       # (1, 1, h, w)
            cam = F.relu(cam)

            target_size = self._settings.MODEL_INPUT_SIZE
            cam = F.interpolate(
                cam,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )

            cam_np = cam.squeeze().cpu().numpy()

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

    # ── Coordinate extraction helpers ─────────────────────────────────────────

    def _extract_cam_grid(self, cam_fullres: np.ndarray) -> List[List[float]]:
        """
        Downsample full-resolution CAM (224×224) to a 7×7 grid.
        Each cell holds the mean intensity of its region.

        Returns
        -------
        List[List[float]] of shape (7, 7) with values in [0.0–1.0].
        """
        H, W = cam_fullres.shape
        cell_h = H // CAM_GRID_SIZE
        cell_w = W // CAM_GRID_SIZE

        cam_grid = []
        for r in range(CAM_GRID_SIZE):
            row = []
            for c in range(CAM_GRID_SIZE):
                y1 = r * cell_h
                y2 = (r + 1) * cell_h if r < CAM_GRID_SIZE - 1 else H
                x1 = c * cell_w
                x2 = (c + 1) * cell_w if c < CAM_GRID_SIZE - 1 else W
                row.append(round(float(cam_fullres[y1:y2, x1:x2].mean()), 4))
            cam_grid.append(row)

        return cam_grid

    def _compute_ink_bbox(self, gray_image: np.ndarray) -> Dict[str, float]:
        """
        Compute the bounding box of ink (non-white pixels) in the image.
        Returns normalized coordinates [0.0–1.0] relative to image size.

        Parameters
        ----------
        gray_image : np.ndarray
            Greyscale image (H, W).

        Returns
        -------
        Dict with keys: x, y, w, h, x2, y2, cx, cy, ratio.
        """
        binary = gray_image < 200          # ink = darker than near-white
        coords = np.argwhere(binary)

        if len(coords) == 0:
            return {
                "x": 0.0, "y": 0.0,
                "w": 1.0, "h": 1.0,
                "x2": 1.0, "y2": 1.0,
                "cx": 0.5, "cy": 0.5,
                "ratio": 1.0,
            }

        # coords layout: (rows=y, cols=x)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        pad = 10
        x1 = max(0, x_min - pad)
        y1 = max(0, y_min - pad)
        x2 = min(gray_image.shape[1], x_max + pad)
        y2 = min(gray_image.shape[0], y_max + pad)

        w_px = x2 - x1
        h_px = y2 - y1
        S = float(self._settings.MODEL_INPUT_SIZE)

        return {
            "x":     round(x1 / S, 4),
            "y":     round(y1 / S, 4),
            "w":     round(w_px / S, 4),
            "h":     round(h_px / S, 4),
            "x2":    round(x2 / S, 4),
            "y2":    round(y2 / S, 4),
            "cx":    round((x1 + x2) / 2 / S, 4),
            "cy":    round((y1 + y2) / 2 / S, 4),
            "ratio": round(w_px / max(h_px, 1), 2),
        }

    def _extract_stroke_markers(self, cam_fullres: np.ndarray) -> List[Dict]:
        """
        Find the top-N hottest points in the Grad-CAM map.
        Returns normalized coordinates [0.0–1.0].

        Returns
        -------
        List[Dict] with keys: id, cx_norm, cy_norm, cx_px, cy_px, score.
        """
        flat_cam = cam_fullres.flatten()
        top_indices = np.argsort(flat_cam)[-TOP_MARKERS:][::-1]

        S = float(self._settings.MODEL_INPUT_SIZE)
        markers = []

        for rank, flat_idx in enumerate(top_indices):
            cy_px, cx_px = np.unravel_index(flat_idx, cam_fullres.shape)
            score = float(flat_cam[flat_idx])
            markers.append({
                "id":      rank + 1,
                "cx_norm": round(float(cx_px) / S, 4),
                "cy_norm": round(float(cy_px) / S, 4),
                "cx_px":   int(cx_px),
                "cy_px":   int(cy_px),
                "score":   round(score, 4),
            })

        return markers

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _find_layer(
        model: torch.nn.Module, layer_name: str
    ) -> Optional[torch.nn.Module]:
        """Traverse module hierarchy using dot-notation."""
        current = model
        for part in layer_name.split("."):
            current = getattr(current, part, None)
            if current is None:
                return None
        return current

    def _build_blob_id(self, case_name: str) -> str:
        """
        Construct the blob path for the JSON output.

        Format: ``{prefix}/{sanitised_case}/V1.json``

        The filename is always ``V1.json`` so the mobile app and downstream
        services can locate it with a predictable path.
        """
        prefix = self._settings.GRADCAM_OUTPUT_PREFIX
        safe_case = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in case_name
        )
        return f"{prefix}/{safe_case}/V1.json"


# ── Module-level singleton ────────────────────────────────────────────────────
gradcam_service = GradCAMService()


def get_gradcam_service() -> GradCAMService:
    """FastAPI dependency: returns the application-scoped GradCAMService."""
    return gradcam_service