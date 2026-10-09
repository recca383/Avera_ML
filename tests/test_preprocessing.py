"""
Preprocessing parity tests (Pipeline 32).

Run from the project root:  python -m unittest discover -s tests
"""

import io
import unittest

import cv2
import numpy as np
from PIL import Image

from app.services.preprocessing_service import PreprocessingService, get_ink_bbox


def _signature(margin: int) -> np.ndarray:
    """Synthetic anti-aliased signature on white paper with `margin` px around the ink."""
    w, h = 320, 120
    sig = np.full((h, w), 255, dtype=np.uint8)
    pts = np.array([[10, 90], [50, 20], [90, 95], [140, 30], [190, 85], [240, 25], [305, 70]], np.int32)
    cv2.polylines(sig, [pts], False, 0, 3, cv2.LINE_AA)
    cv2.ellipse(sig, (120, 70), (25, 15), 0, 0, 360, 40, 2, cv2.LINE_AA)
    cv2.circle(sig, (280, 15), 3, 0, -1, cv2.LINE_AA)
    return cv2.copyMakeBorder(sig, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)


def _png_bytes(gray: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(gray).save(buf, format="PNG")
    return buf.getvalue()


class CropToInkParityTest(unittest.TestCase):
    """
    A signature with large white margins must preprocess to the same tensor as
    the same signature already cropped close to the ink. The pre-cropped copy
    keeps a 20 px margin so the 8 px border whitening doesn't erase real ink.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.service = PreprocessingService()
        cls.wide = _png_bytes(_signature(margin=250))
        cls.cropped = _png_bytes(_signature(margin=20))

    def test_tensor_matches_precropped(self) -> None:
        wide = self.service.preprocess_image_bytes(self.wide)
        cropped = self.service.preprocess_image_bytes(self.cropped)
        self.assertEqual(tuple(wide.shape), (1, 1, 224, 224))
        self.assertEqual(wide.shape, cropped.shape)
        self.assertLess(float((wide - cropped).abs().mean()), 1e-3)
        self.assertLess(float((wide - cropped).abs().max()), 0.05)

    def test_pil_matches_precropped(self) -> None:
        wide = np.asarray(self.service.bytes_to_pil(self.wide).convert("L"), dtype=np.float32)
        cropped = np.asarray(self.service.bytes_to_pil(self.cropped).convert("L"), dtype=np.float32)
        self.assertEqual(wide.shape, (224, 224))
        self.assertLess(float(np.abs(wide - cropped).mean()), 0.5)

    def test_margins_do_not_shrink_signature(self) -> None:
        # Without the crop, the 250 px margins would shrink the ink to a small
        # patch in the middle of the 224x224 canvas.
        tensor = self.service.preprocess_image_bytes(self.wide)[0, 0].numpy()
        ink_cols = np.where((tensor > 0).any(axis=0))[0]
        self.assertGreater(ink_cols.max() - ink_cols.min(), 200)

    def test_blank_image_is_not_cropped(self) -> None:
        blank = np.full((100, 150), 255, dtype=np.uint8)
        self.assertEqual(get_ink_bbox(blank), (0, 0, 150, 100))
        tensor = self.service.preprocess_image_bytes(_png_bytes(blank))
        self.assertEqual(tuple(tensor.shape), (1, 1, 224, 224))

    def test_ink_bbox_pads_and_clamps(self) -> None:
        img = np.full((50, 80), 255, dtype=np.uint8)
        img[10:20, 30:40] = 0
        self.assertEqual(get_ink_bbox(img, pad=6), (24, 4, 45, 25))
        img[0:5, 0:5] = 0
        self.assertEqual(get_ink_bbox(img, pad=6)[:2], (0, 0))


if __name__ == "__main__":
    unittest.main()
