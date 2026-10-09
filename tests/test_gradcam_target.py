"""
Grad-CAM explains the verification decision.

1. The score Grad-CAM backpropagates is the verdict distance from
   InferenceService for the same inputs.
2. Model-randomisation sanity check (Adebayo et al., 2018, "Sanity Checks for
   Saliency Maps"): with randomly initialised weights the heatmap must change
   substantially, i.e. correlate < 0.5 with the trained model's heatmap. A map
   that barely changes would reflect the input image, not what the model learned.

Inputs are synthetic signatures, so the tests do not depend on the (gitignored)
sample data. The randomisation test needs the trained weights at MODEL_PATH.

Run from the project root:  python -m unittest discover -s tests
"""

import io
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import app.ml.model_loader as model_loader
from app.core.config import get_settings
from app.ml.architecture import SiameseNineNet
from app.services.gradcam_service import GradCAMService
from app.services.inference_service import InferenceService
from app.services.preprocessing_service import PreprocessingService

MODEL_FILE = Path(get_settings().MODEL_PATH)
RANDOM_SEEDS = (0, 1, 2)


def _signature_png(seed: int, stretch: float = 1.0, slope: float = 0.0, loops: int = 1) -> bytes:
    """Synthetic signature: a wavy stroke with loops and a dot, on white paper."""
    rng = np.random.default_rng(seed)
    img = np.full((140, 360), 255, dtype=np.uint8)
    xs = np.linspace(20, 20 + 300 * stretch, 12)
    ys = 75 - slope * (xs - 180) + rng.normal(0, 14, len(xs))
    cv2.polylines(img, [np.stack([xs, ys], 1).astype(np.int32)], False, 0, 3, cv2.LINE_AA)
    for k in range(loops):
        cv2.ellipse(img, (70 + 70 * k, 70), (18, 12), 15 * k, 0, 360, 20, 2, cv2.LINE_AA)
    cv2.circle(img, (int(20 + 300 * stretch) - 10, 30), 4, 0, -1, cv2.LINE_AA)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _case(seed: int, forged: bool):
    pre = PreprocessingService()
    refs = pre.preprocess_batch([_signature_png(seed * 10 + i) for i in range(4)])
    q_png = _signature_png(seed * 10 + 7, stretch=0.7, slope=0.15, loops=3) if forged else _signature_png(seed * 10 + 5)
    return refs, pre.preprocess_image_bytes(q_png)


def _random_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return SiameseNineNet().eval()


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two heatmaps. A constant map (all zero after
    Grad-CAM's ReLU) shares no structure with the other map, so it scores 0."""
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


class _SwapModel(unittest.TestCase):
    """Saves and restores the model_loader singleton around each test."""

    def setUp(self) -> None:
        self._saved = (model_loader._model, model_loader._model_device)

    def tearDown(self) -> None:
        model_loader._model, model_loader._model_device = self._saved

    @staticmethod
    def use(model: torch.nn.Module) -> None:
        model_loader._model = model
        model_loader._model_device = torch.device("cpu")


class ScoreIsVerdictDistanceTest(_SwapModel):
    def _check(self, model: torch.nn.Module) -> None:
        self.use(model)
        gradcam, inference = GradCAMService(), InferenceService()
        for seed in (1, 2):
            for forged in (False, True):
                refs, q = _case(seed, forged)
                prototype = gradcam._reference_prototype(list(refs.unbind(0)))
                cam, score = gradcam._gradcam_with_score(q, prototype)
                distance = inference._run_inference(refs, q)[3]
                with self.subTest(seed=seed, forged=forged):
                    self.assertIsNotNone(score)
                    self.assertAlmostEqual(score, distance, places=5)
                    self.assertEqual(cam.shape, (224, 224))

    def test_random_weights(self) -> None:
        self._check(_random_model(0))

    @unittest.skipUnless(MODEL_FILE.exists(), "trained model weights not available")
    def test_trained_weights(self) -> None:
        model_loader._model = None
        model_loader.load_model()
        self._check(model_loader._model)


@unittest.skipUnless(MODEL_FILE.exists(), "trained model weights not available")
class ModelRandomisationSanityTest(_SwapModel):
    def test_heatmap_changes_with_random_weights(self) -> None:
        model_loader._model = None
        model_loader.load_model()
        trained = model_loader._model
        gradcam = GradCAMService()
        for seed in (1, 2):
            for forged in (False, True):
                refs, q = _case(seed, forged)
                self.use(trained)
                cam_trained = gradcam._compute_gradcam(q, gradcam._reference_prototype(list(refs.unbind(0))))
                self.assertIsNotNone(cam_trained)
                for rs in RANDOM_SEEDS:
                    self.use(_random_model(rs))
                    cam_random = gradcam._compute_gradcam(q, gradcam._reference_prototype(list(refs.unbind(0))))
                    if cam_random is None:          # degenerate random net: no usable gradient
                        cam_random = np.zeros_like(cam_trained)
                    corr = _correlation(cam_trained, cam_random)
                    with self.subTest(seed=seed, forged=forged, random_seed=rs):
                        self.assertLess(corr, 0.5)


if __name__ == "__main__":
    unittest.main()
