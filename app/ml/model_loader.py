"""
ML model loader.
Responsible for loading the exported Siamese network from disk exactly once
during application startup and keeping it resident in memory for the lifetime
of the process.

Design decisions
----------------
- torch.jit.load() is used to load a TorchScript-exported model, which is the
  recommended export format for production PyTorch deployments (no Python class
  definitions needed at inference time).
- If your Colab export used torch.save(model.state_dict(), ...) instead, replace
  the loader with the commented-out alternative that reconstructs the architecture
  first and then loads weights.
- The model is pinned to CPU by default. If the container has a GPU, change
  DEVICE to "cuda" and rebuild the image with the CUDA base.
- model.eval() and torch.inference_mode() are applied to disable dropout /
  batch-norm training behaviour and gradient computation, both of which are
  unnecessary and wasteful at inference time.
"""

import threading
from pathlib import Path
from typing import Optional

import torch

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Global model state — protected by a lock to be safe against parallel startup calls.
_model_lock = threading.Lock()
_model: Optional[torch.jit.ScriptModule] = None
_model_device: Optional[torch.device] = None


def load_model() -> None:
    """
    Load the TorchScript model from disk into the global singleton.
    Idempotent: safe to call multiple times (no-op after first successful load).
    Must be called from the FastAPI lifespan startup handler.
    """
    global _model, _model_device  # noqa: PLW0603

    with _model_lock:
        if _model is not None:
            logger.debug("Model already loaded; skipping.")
            return

        settings = get_settings()
        model_path = Path(settings.MODEL_PATH)

        if not model_path.exists():
            # In production, the model file must be present; fail fast.
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                "Ensure the exported model is included in the Docker image."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading model", path=str(model_path), device=str(device))

        try:
            from app.ml.architecture import SiameseNineNet

            state_dict = torch.load(str(model_path), map_location=device)
            if not isinstance(state_dict, dict):
                raise RuntimeError(
                    "Model file did not contain a state_dict for SiameseNineNet."
                )

            model = SiameseNineNet()
            model.load_state_dict(state_dict)
        except Exception as fallback_exc:
            logger.error("Failed to load state_dict model", error=str(fallback_exc))
            raise RuntimeError(
                "Model loading failed: unable to load TorchScript or state_dict model. "
                f"Details: {fallback_exc}"
            ) from fallback_exc
        

        model.eval()
        model = model.to(device)

        _model = model
        _model_device = device

        logger.info(
            "Model loaded successfully",
            device=str(device),
            path=str(model_path),
        )


def get_model() -> torch.jit.ScriptModule:
    """
    Return the loaded model.
    Raises RuntimeError if called before load_model() has completed.
    """
    if _model is None:
        raise RuntimeError(
            "Model has not been loaded. "
            "Ensure load_model() is called during application startup."
        )
    return _model


def get_model_device() -> torch.device:
    """Return the torch.device the model currently resides on."""
    if _model_device is None:
        raise RuntimeError("Model has not been loaded.")
    return _model_device


def is_model_loaded() -> bool:
    """Return True if the model singleton is ready for inference."""
    return _model is not None
