"""
Preprocessing service.
Converts raw image bytes (downloaded from Azure Blob Storage) into normalised
PyTorch tensors ready for Siamese network inference.

Design decisions
----------------
- All processing is done in-memory using NumPy + PIL; no temp files are written.
- Images are converted to greyscale (single channel) to match the expected
  training pipeline for handwritten signature models. Adjust _to_tensor() if
  your model uses RGB input.
- The preprocessing pipeline mirrors the transforms used during training to
  avoid train/inference distribution mismatch.
"""

import io
from typing import List

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class PreprocessingService:
    """
    Stateless image preprocessor.
    Methods are synchronous because they are CPU-bound and fast (<5 ms per image).
    Run them inside asyncio.to_thread() if profiling reveals contention.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._target_size = (
            self._settings.MODEL_INPUT_SIZE,
            self._settings.MODEL_INPUT_SIZE,
        )

        # Match the P2 training pipeline: grayscale images normalised to 0.5/0.5.
        self._mean = 0.5
        self._std = 0.5

    # ── Public API ────────────────────────────────────────────────────────────

    def preprocess_image_bytes(self, image_bytes: bytes) -> torch.Tensor:
        """
        Convert raw image bytes → (1, 1, H, W) float32 tensor (batched, single image).

        Parameters
        ----------
        image_bytes : Raw bytes from Azure Blob Storage.

        Returns
        -------
        torch.Tensor  shape (1, 1, H, W), dtype=float32, values in [0, 1] normalised.
        """
        pil_image = self._load_pil_image(image_bytes)
        pil_image = self._clean_border(pil_image)
        pil_image = self._resize_and_pad(pil_image)
        tensor = self._to_tensor(pil_image)               # (1, H, W)
        tensor = self._normalise(tensor)
        tensor = tensor.unsqueeze(0)                       # (1, 1, H, W)
        return tensor

    def preprocess_batch(self, images_bytes: List[bytes]) -> torch.Tensor:
        """
        Preprocess a list of raw image byte strings → (N, 1, H, W) tensor.
        Used for the reference image batch.
        """
        tensors = [self.preprocess_image_bytes(b).squeeze(0) for b in images_bytes]
        return torch.stack(tensors, dim=0)   # (N, 1, H, W)

    def bytes_to_pil(self, image_bytes: bytes) -> Image.Image:
        """Return a PIL Image (RGB) for Grad-CAM overlay generation."""
        img = self._load_pil_image(image_bytes)
        img = self._clean_border(img)
        img = self._resize_and_pad(img)
        return img.convert("RGB")

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_pil_image(image_bytes: bytes) -> Image.Image:
        try:
            return Image.open(io.BytesIO(image_bytes))
        except Exception as exc:
            raise ValueError(f"Cannot decode image bytes: {exc}") from exc

    def _clean_border(self, img: Image.Image, border: int = 4) -> Image.Image:
        """Remove scanner/edge artefacts by whitening the image border."""
        img = img.convert("L")
        img_np = np.array(img)
        img_np[:border, :] = 255
        img_np[-border:, :] = 255
        img_np[:, :border] = 255
        img_np[:, -border:] = 255
        return Image.fromarray(img_np)

    def _resize_and_pad(self, img: Image.Image) -> Image.Image:
        """
        Resize the image with aspect ratio preserved and add a white pad.

        This mirrors the notebook pipeline: the longest side is scaled to the
        target size, the image is centered on a white square canvas, and the
        paste boundary is softened with a small Gaussian blur mask so the CNN
        does not learn a hard white border edge.
        """
        img = img.convert("L")
        w, h = img.size
        scale = self._settings.MODEL_INPUT_SIZE / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("L", self._target_size, 255)
        paste_x = (self._target_size[0] - new_w) // 2
        paste_y = (self._target_size[1] - new_h) // 2
        canvas.paste(img, (paste_x, paste_y))

        mask = Image.new("L", self._target_size, 255)
        interior = Image.new("L", (new_w, new_h), 0)
        mask.paste(interior, (paste_x, paste_y))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=3))

        white = Image.new("L", self._target_size, 255)
        result = Image.composite(
            canvas,
            white,
            Image.fromarray(255 - np.array(mask)),
        )
        return result

    @staticmethod
    def _to_tensor(img: Image.Image) -> torch.Tensor:
        """PIL (H, W) greyscale → float32 tensor (1, H, W) in [0, 1]."""
        np_img = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np_img).unsqueeze(0)   # (1, H, W)
        return tensor

    def _normalise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply mean/std normalisation matching the training pipeline."""
        return (tensor - self._mean) / self._std


# ── Module-level singleton ────────────────────────────────────────────────────
preprocessing_service = PreprocessingService()


def get_preprocessing_service() -> PreprocessingService:
    """FastAPI dependency: returns the application-scoped PreprocessingService."""
    return preprocessing_service
