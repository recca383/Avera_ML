"""
Grad-CAM service.
Generates visualization images for the questioned signature and uploads them
as separate PNG assets to Azure Blob Storage.
"""

import asyncio
import os
import tempfile
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.model_loader import get_model

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_LAYER_NAME = "backbone.conv_layers.24"
TOP_MARKERS = 5


def _apply_jet_colormap(values: np.ndarray) -> np.ndarray:
    """Apply a simple jet-like colormap without requiring Matplotlib."""
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    control_points = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    colors = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    flat = values.reshape(-1)
    rgb = np.empty((flat.size, 3), dtype=np.float32)
    for i in range(3):
        rgb[:, i] = np.interp(flat, control_points, colors[:, i])

    return (rgb.reshape(values.shape + (3,)) * 255.0).astype(np.uint8)


def export_individual_visuals(
    ref_images_paths,
    query_image_path,
    orig_pils,
    cams,
    blends,
    bboxes,
    stroke_diffs,
    results_dir,
    cmap_cam="jet",
):
    """
    Export individual visualization components as separate PNG files.
    """
    all_paths = ref_images_paths + [query_image_path]
    export_base_dir = os.path.join(results_dir, "individual_visuals")
    os.makedirs(export_base_dir, exist_ok=True)
    logger.info("Saving individual visuals", export_base_dir=export_base_dir)

    colormap_func = None if cmap_cam != "jet" else _apply_jet_colormap

    exported_files: List[str] = []

    for i, path in enumerate(all_paths):
        base_name = os.path.splitext(os.path.basename(path))[0]
        prefix = f"genuine_{i + 1}" if i < len(ref_images_paths) else "suspected"

        orig_pil = orig_pils[i] if i < len(orig_pils) else Image.new("RGB", (224, 224), "white")
        cam = cams[i] if i < len(cams) else np.zeros((224, 224), dtype=np.float32)
        blend = blends[i] if i < len(blends) else np.zeros((224, 224, 3), dtype=np.uint8)
        bbox = bboxes[i] if i < len(bboxes) else np.zeros((224, 224, 3), dtype=np.uint8)
        stroke_diff = stroke_diffs[i] if i < len(stroke_diffs) else np.zeros((224, 224, 3), dtype=np.uint8)

        orig_filename = os.path.join(export_base_dir, f"{prefix}_{base_name}_original.png")
        orig_pil.save(orig_filename)
        exported_files.append(orig_filename)

        if colormap_func is None:
            cam_array = np.clip(np.nan_to_num(np.asarray(cam), nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
            cam_array = (cam_array * 255).astype(np.uint8)
            cam_rgb = np.stack([cam_array, cam_array, cam_array], axis=-1)
        else:
            cam_rgb = colormap_func(cam)

        cam_pil = Image.fromarray(cam_rgb)
        cam_filename = os.path.join(export_base_dir, f"{prefix}_{base_name}_heatmap.png")
        cam_pil.save(cam_filename)
        exported_files.append(cam_filename)

        blend_pil = Image.fromarray(np.asarray(blend, dtype=np.uint8))
        blend_filename = os.path.join(export_base_dir, f"{prefix}_{base_name}_overlay.png")
        blend_pil.save(blend_filename)
        exported_files.append(blend_filename)

        bbox_pil = Image.fromarray(np.asarray(bbox, dtype=np.uint8))
        bbox_filename = os.path.join(export_base_dir, f"{prefix}_{base_name}_bbox.png")
        bbox_pil.save(bbox_filename)
        exported_files.append(bbox_filename)

        stroke_diff_pil = Image.fromarray(np.asarray(stroke_diff, dtype=np.uint8))
        stroke_diff_filename = os.path.join(export_base_dir, f"{prefix}_{base_name}_stroke_diff.png")
        stroke_diff_pil.save(stroke_diff_filename)
        exported_files.append(stroke_diff_filename)

    return exported_files


def export_compiled_pdf(
    orig_pils: List[Image.Image],
    cams: List[np.ndarray],
    blends: List[np.ndarray],
    bboxes: List[np.ndarray],
    stroke_diffs: List[np.ndarray],
    results_dir: str,
    verdict: str,
    avg_distance: float,
    threshold: float,
    ref_image_names: List[str],
    query_image_name: str,
) -> str:
    """Create a single PDF report containing the Grad-CAM visual assets for all references and the questioned image."""
    os.makedirs(results_dir, exist_ok=True)

    verdict_color = "#16A34A" if verdict == "GENUINE" else "#DC2626"

    canvas_width = 2400
    canvas_height = 2600
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    title = (
        f"AVERA — Multi-Reference Grad-CAM | Verdict: {verdict} | "
        f"Avg Dist: {avg_distance:.4f} | Threshold: {threshold:.4f}"
    )
    draw.text((40, 20), title, fill=verdict_color)

    col_titles = [
        "Original\nSignature",
        "Grad-CAM\nHeatmap",
        "Heatmap\nOverlay",
        "Ink Bounding\nBox",
        "Forensic\nStroke Map",
    ]
    cell_w = int((canvas_width - 120) / 5)
    cell_h = int((canvas_height - 180) / 6)
    header_h = 120

    for col_idx, title_text in enumerate(col_titles):
        x = 40 + col_idx * cell_w
        draw.text((x + 10, header_h - 35), title_text, fill="black")

    rows = list(zip(orig_pils, cams, blends, bboxes, stroke_diffs))
    if len(rows) < 5:
        rows = rows + [rows[-1]] * (5 - len(rows))

    for row_idx in range(5):
        if row_idx < len(orig_pils):
            row_images = [
                orig_pils[row_idx],
                Image.fromarray(_apply_jet_colormap(cams[row_idx])),
                Image.fromarray(np.asarray(blends[row_idx], dtype=np.uint8)),
                Image.fromarray(np.asarray(bboxes[row_idx], dtype=np.uint8)),
                Image.fromarray(np.asarray(stroke_diffs[row_idx], dtype=np.uint8)),
            ]
        else:
            row_images = [Image.new("RGB", (224, 224), "white")] * 5

        y = header_h + row_idx * cell_h
        for col_idx, img in enumerate(row_images):
            x = 40 + col_idx * cell_w
            img_rgb = img.convert("RGB") if img.mode != "RGB" else img
            resized = img_rgb.resize((cell_w - 20, cell_h - 60), Image.LANCZOS)
            canvas.paste(resized, (x + 10, y + 30))

        if row_idx < 4:
            label = (
                f"Reference Specimen {row_idx + 1}\n"
                f"{os.path.basename(ref_image_names[row_idx]) if row_idx < len(ref_image_names) else 'n/a'}\n"
                f"Similarity Dist: {0.0:.4f}"
            )
        else:
            label = f"Questioned Document\n{os.path.basename(query_image_name)}"
        draw.text((40, y + cell_h - 60), label, fill="black")

    pdf_path = os.path.join(results_dir, "output.pdf")
    canvas.save(pdf_path, format="PDF")
    return pdf_path


class GradCAMService:
    """Produces per-image Grad-CAM visualization assets for explainability."""

    def __init__(self) -> None:
        self._settings = get_settings()

    async def generate(
        self,
        questioned_tensor: torch.Tensor,
        original_pil_image: Image.Image,
        case_name: str,
        reference_image_ids: List[str] = None,
        questioned_image_id: str = None,
        reference_tensors: Optional[List[torch.Tensor]] = None,
        reference_pil_images: Optional[List[Image.Image]] = None,
        verdict: Optional[str] = None,
        avg_distance: Optional[float] = None,
        threshold: Optional[float] = None,
        blob_svc=None
    ) -> List[str]:
        """Generate and optionally upload visualization images and the PDF report."""
        if reference_image_ids is None:
            reference_image_ids = []
        if questioned_image_id is None:
            questioned_image_id = "F1.png"
        if reference_tensors is None:
            reference_tensors = []
        if reference_pil_images is None:
            reference_pil_images = []

        exported_files = await asyncio.to_thread(
            self._run_gradcam,
            questioned_tensor,
            original_pil_image,
            case_name,
            reference_image_ids,
            questioned_image_id,
            reference_tensors,
            reference_pil_images,
            verdict or "UNKNOWN",
            avg_distance or 0.0,
            threshold or 0.0
        )

        if blob_svc is not None:
            return await self._upload_visuals(
                local_paths=exported_files,
                case_name=case_name,
                blob_svc=blob_svc
            )

        return exported_files

    def _run_gradcam(
        self,
        questioned_tensor: torch.Tensor,
        original_pil_image: Image.Image,
        case_name: str,
        reference_image_ids: List[str],
        questioned_image_id: str,
        reference_tensors: List[torch.Tensor],
        reference_pil_images: List[Image.Image],
        verdict: str,
        avg_distance: float,
        threshold: float
    ) -> List[str]:
        """Compute the Grad-CAM map, create visual images, and return local file paths."""
        model = get_model()
        first_param = next(model.parameters())
        actual_device = first_param.device
        model_dtype = first_param.dtype

        tensor = questioned_tensor.to(device=actual_device, dtype=model_dtype)

        cam_fullres = self._compute_gradcam(tensor)
        if cam_fullres is None:
            logger.warning("Grad-CAM computation failed; returning zero map", case_name=case_name)
            cam_fullres = np.zeros(
                (self._settings.MODEL_INPUT_SIZE, self._settings.MODEL_INPUT_SIZE),
                dtype=np.float32,
            )

        original_rgb = np.array(original_pil_image.convert("RGB"), dtype=np.uint8)
        cam_rgb = self._build_cam_image(cam_fullres)
        blend = self._blend_heatmap(original_rgb, cam_rgb)
        bbox = self._draw_bbox(original_rgb, self._compute_ink_bbox(np.array(original_pil_image.convert("L"))))
        stroke_diff = self._build_stroke_diff(original_rgb, cam_rgb)

        safe_case = self._sanitize_case_name(case_name)
        export_dir = os.path.join(tempfile.gettempdir(), "gradcam-exports", safe_case)

        ref_images_paths = [f"{case_name}/{ref_id}" for ref_id in reference_image_ids]
        query_image_path = f"{case_name}/{questioned_image_id}"

        reference_visuals: List[dict] = []
        for idx, ref_tensor in enumerate(reference_tensors):
            ref_pil = reference_pil_images[idx] if idx < len(reference_pil_images) else Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            ref_cam = self._compute_gradcam(ref_tensor)
            if ref_cam is None:
                ref_cam = np.zeros((self._settings.MODEL_INPUT_SIZE, self._settings.MODEL_INPUT_SIZE), dtype=np.float32)
            ref_rgb = np.array(ref_pil.convert("RGB"), dtype=np.uint8)
            ref_cam_rgb = self._build_cam_image(ref_cam)
            ref_blend = self._blend_heatmap(ref_rgb, ref_cam_rgb)
            ref_bbox = self._draw_bbox(ref_rgb, self._compute_ink_bbox(np.array(ref_pil.convert("L"))))
            ref_stroke_diff = self._build_stroke_diff(ref_rgb, ref_cam_rgb)
            reference_visuals.append(
                {
                    "orig": ref_pil,
                    "cam": ref_cam,
                    "blend": ref_blend,
                    "bbox": ref_bbox,
                    "stroke_diff": ref_stroke_diff,
                }
            )

        all_orig_pils = [item["orig"] for item in reference_visuals] + [Image.fromarray(original_rgb)]
        all_cams = [item["cam"] for item in reference_visuals] + [cam_fullres]
        all_blends = [item["blend"] for item in reference_visuals] + [blend]
        all_bboxes = [item["bbox"] for item in reference_visuals] + [bbox]
        all_stroke_diffs = [item["stroke_diff"] for item in reference_visuals] + [stroke_diff]

        exported_files = export_individual_visuals(
            ref_images_paths=ref_images_paths,
            query_image_path=query_image_path,
            orig_pils=all_orig_pils,
            cams=all_cams,
            blends=all_blends,
            bboxes=all_bboxes,
            stroke_diffs=all_stroke_diffs,
            results_dir=export_dir,
        )

        pdf_path = export_compiled_pdf(
            orig_pils=all_orig_pils,
            cams=all_cams,
            blends=all_blends,
            bboxes=all_bboxes,
            stroke_diffs=all_stroke_diffs,
            results_dir=export_dir,
            verdict=verdict,
            avg_distance=avg_distance,
            threshold=threshold,
            ref_image_names=ref_images_paths,
            query_image_name=query_image_path,
        )

        exported_files.append(pdf_path)

        logger.info(
            "Grad-CAM visual assets generated",
            case_name=case_name,
            ref_images_paths=ref_images_paths,
            query_image_path=query_image_path,
            exported_files=exported_files,
            pdf_path=pdf_path,
        )

        return exported_files

    def _compute_gradcam(self, tensor: torch.Tensor) -> Optional[np.ndarray]:
        """Compute the Grad-CAM activation map (full resolution 224×224)."""
        model = get_model()
        first_param = next(model.parameters())
        actual_device = first_param.device
        model_dtype = first_param.dtype

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(device=actual_device, dtype=model_dtype)

        activations: list = []
        gradients: list = []

        def forward_hook(module, _input, output):  # noqa: ANN001
            activations.append(output.detach())

        def backward_hook(module, _grad_input, grad_output):  # noqa: ANN001
            gradients.append(grad_output[0].detach())

        target_layer = self._find_layer(model, TARGET_LAYER_NAME)
        if target_layer is None:
            logger.warning("Target layer not found; Grad-CAM unavailable", target_layer=TARGET_LAYER_NAME)
            return None

        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            original_training = model.training
            model.train()
            tensor.requires_grad_(True)

            with torch.enable_grad():
                embedding = model(tensor)
                score = embedding.sum()
                model.zero_grad()
                score.backward()

            if not activations or not gradients:
                return None

            acts = activations[0]
            grads = gradients[0]
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            target_size = self._settings.MODEL_INPUT_SIZE
            cam = F.interpolate(cam, size=(target_size, target_size), mode="bilinear", align_corners=False)
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

    def _build_cam_image(self, cam_fullres: np.ndarray) -> np.ndarray:
        cam_array = np.clip(np.nan_to_num(cam_fullres, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        return _apply_jet_colormap(cam_array)

    def _blend_heatmap(self, original_rgb: np.ndarray, cam_rgb: np.ndarray) -> np.ndarray:
        original = original_rgb.astype(np.float32)
        heatmap = cam_rgb.astype(np.float32)
        blended = (0.65 * original + 0.35 * heatmap).astype(np.uint8)
        return blended

    def _draw_bbox(self, original_rgb: np.ndarray, ink_bbox: Dict[str, float]) -> np.ndarray:
        height, width = original_rgb.shape[:2]
        image = Image.fromarray(original_rgb)
        draw = ImageDraw.Draw(image)

        x1 = int(max(0, ink_bbox["x"] * width))
        y1 = int(max(0, ink_bbox["y"] * height))
        x2 = int(min(width, ink_bbox["x2"] * width))
        y2 = int(min(height, ink_bbox["y2"] * height))
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=max(2, int(min(width, height) / 100)))
        return np.array(image)

    def _build_stroke_diff(self, original_rgb: np.ndarray, cam_rgb: np.ndarray) -> np.ndarray:
        original = original_rgb.astype(np.float32)
        heatmap = cam_rgb.astype(np.float32)
        stroke_diff = np.clip(0.75 * original + 0.25 * heatmap, 0, 255).astype(np.uint8)
        return stroke_diff

    async def _upload_visuals(
        self,
        local_paths: List[str],
        case_name: str,
        blob_svc
    ) -> List[str]:
        """Upload generated PNG files to Azure Blob Storage and return their blob ids."""
        uploaded_blob_ids: List[str] = []

        logger.info(
            "Starting Grad-CAM image upload",
            case_name=case_name,
            file_count=len(local_paths),
        )

        for local_path in local_paths:
            filename = os.path.basename(local_path)
            prefix, rest_filename = self._parse_export_filename(filename)

            # Build blob path: case_name/prefix/rest_filename
            if prefix:
                blob_id = f"{case_name}/{prefix}/{rest_filename}"
            else:
                blob_id = f"{case_name}/{rest_filename}"

            with open(local_path, "rb") as handle:
                data = handle.read()

            logger.info(
                "Uploading Grad-CAM image",
                file_path=local_path,
                blob_id=blob_id,
                size_bytes=len(data),
            )
            await blob_svc.upload_blob(blob_id=blob_id, data=data, content_type="image/png")
            uploaded_blob_ids.append(blob_id)

        logger.info(
            "Completed Grad-CAM image upload",
            case_name=case_name,
            uploaded_blob_ids=uploaded_blob_ids,
        )
        return uploaded_blob_ids

    def _parse_export_filename(self, filename: str) -> tuple:
        """
        Parse a filename like 'ref_1_G1_original.png' into (prefix, rest).

        Returns:
            (prefix, rest_filename)
            e.g., ('ref_1', 'G1_original.png')
        """
        name_without_ext = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]

        parts = name_without_ext.split("_")

        if parts[0] == "genuine" and len(parts) > 1:
            # Format: genuine_N_IMAGENAME_SUFFIX...
            prefix = f"{parts[0]}_{parts[1]}"
            rest_parts = parts[2:]
        elif parts[0] == "suspected":
            # Format: suspected_IMAGENAME_SUFFIX...
            prefix = parts[0]
            rest_parts = parts[1:]
        else:
            # Fallback: no prefix parsing
            return "", filename

        rest_filename = "_".join(rest_parts) + ext
        return prefix, rest_filename


    @staticmethod
    def _find_layer(model: torch.nn.Module, layer_name: str) -> Optional[torch.nn.Module]:
        current = model
        for part in layer_name.split("."):
            current = getattr(current, part, None)
            if current is None:
                return None
        return current

    @staticmethod
    def _sanitize_case_name(case_name: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in case_name)

    def _compute_ink_bbox(self, gray_image: np.ndarray) -> Dict[str, float]:
        binary = gray_image < 200
        coords = np.argwhere(binary)

        if len(coords) == 0:
            return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "x2": 1.0, "y2": 1.0}

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        pad = 10
        x1 = max(0, x_min - pad)
        y1 = max(0, y_min - pad)
        x2 = min(gray_image.shape[1], x_max + pad)
        y2 = min(gray_image.shape[0], y_max + pad)

        return {"x": round(x1 / gray_image.shape[1], 4), "y": round(y1 / gray_image.shape[0], 4), "x2": round(x2 / gray_image.shape[1], 4), "y2": round(y2 / gray_image.shape[0], 4)}


# ── Module-level singleton ────────────────────────────────────────────────────
gradcam_service = GradCAMService()


def get_gradcam_service() -> GradCAMService:
    """FastAPI dependency: returns the application-scoped GradCAMService."""
    return gradcam_service