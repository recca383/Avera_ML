"""
Grad-CAM service.
Generates visualization images for the questioned signature and uploads them
as separate PNG assets to Azure Blob Storage.
"""

import asyncio
import math
import os
import tempfile
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.model_loader import get_model

logger = get_logger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_LAYER_NAME = "backbone.conv_layers.26"
TOP_MARKERS = 5


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if os.name == "nt":
        candidates.extend(
            [
                r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
                r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
            ]
        )

    candidates.append("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")

    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue

    return ImageFont.load_default()


def _get_ink_bbox(gray_np: np.ndarray, pad: int = 10) -> tuple[int, int, int, int]:
    gray_np = np.asarray(gray_np)
    if gray_np.dtype != np.uint8:
        gray_np = np.clip(gray_np, 0, 255).astype(np.uint8)

    _, binary = cv2.threshold(gray_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.argwhere(binary > 0)
    if len(coords) == 0:
        return 0, 0, gray_np.shape[1], gray_np.shape[0]

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    height, width = gray_np.shape[:2]

    return (
        max(0, int(x_min) - pad),
        max(0, int(y_min) - pad),
        min(width, int(x_max) + pad),
        min(height, int(y_max) + pad),
    )


def _get_skeleton(gray_np: np.ndarray) -> np.ndarray:
    gray_np = np.asarray(gray_np)
    if gray_np.dtype != np.uint8:
        gray_np = np.clip(gray_np, 0, 255).astype(np.uint8)

    _, binary = cv2.threshold(gray_np, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = binary.astype(np.uint8)

    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(binary, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary = eroded.copy()

        if cv2.countNonZero(binary) == 0:
            break

    return skeleton.astype(np.uint8)


def _compute_global_top_markers(
    reference_gray_images: List[np.ndarray],
    query_gray_image: np.ndarray,
    min_score: float = 0.15,
    max_markers: int = TOP_MARKERS,
    region: int = 15,
) -> List[tuple[float, int, int]]:
    skel_query = _get_skeleton(query_gray_image)
    skels_ref = [_get_skeleton(gray_image) for gray_image in reference_gray_images]

    height, width = skel_query.shape
    combined = np.zeros((height, width), dtype=np.float32)

    for skel_ref in skels_ref:
        ref_only = ((skel_ref == 1) & (skel_query == 0)).astype(np.float32)
        query_only = ((skel_query == 1) & (skel_ref == 0)).astype(np.float32)
        combined += ref_only + query_only

    kernel = np.ones((region, region), np.float32) / float(region * region)
    density = cv2.filter2D(combined, -1, kernel)

    peak_score = float(density.max())
    if peak_score <= 0:
        return []

    score_threshold = min_score * peak_score
    results: List[tuple[float, int, int]] = []
    temp = density.copy()

    for _ in range(max_markers):
        index = int(np.argmax(temp))
        cy, cx = divmod(index, width)
        score = float(temp[cy, cx])

        if score < score_threshold:
            break

        results.append((score, cx, cy))

        y1, y2 = max(0, cy - region), min(height, cy + region)
        x1, x2 = max(0, cx - region), min(width, cx + region)
        temp[y1:y2, x1:x2] = 0

    return results


def _generate_bounding_box_visualization(orig_gray_pil: Image.Image) -> np.ndarray:
    gray_np = np.asarray(orig_gray_pil.convert("L"), dtype=np.uint8)
    _, binary = cv2.threshold(gray_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 20]

    canvas = cv2.cvtColor(gray_np, cv2.COLOR_GRAY2BGR)

    if contours:
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 180, 0), 2, cv2.LINE_AA)
        label = f"W:{w} H:{h} R:{w / max(h, 1):.2f}"
        cv2.putText(
            canvas,
            label,
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 140, 0),
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _generate_stroke_difference_visualization(
    base_pil: Image.Image,
    global_top_markers: List[tuple[float, int, int]],
    image_size: int,
) -> np.ndarray:
    base_gray = np.asarray(base_pil.convert("L"), dtype=np.uint8)

    margin_px = int(image_size * 0.30)
    total_size = image_size + 2 * margin_px

    expanded = np.ones((total_size, total_size, 3), dtype=np.uint8) * 245
    signature_rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    expanded[
        margin_px : margin_px + image_size,
        margin_px : margin_px + image_size,
    ] = signature_rgb

    cv2.rectangle(
        expanded,
        (margin_px - 1, margin_px - 1),
        (margin_px + image_size, margin_px + image_size),
        (200, 200, 200),
        1,
    )

    dark_red = (30, 30, 180)
    white = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bubble_radius = 14
    bubble_dist = int(margin_px * 1.2)
    num_markers = len(global_top_markers)
    angle_step = 360 / num_markers if num_markers else 0

    for idx, (_score, cx, cy) in enumerate(global_top_markers):
        ex = cx + margin_px
        ey = cy + margin_px

        angle_rad = math.radians(idx * angle_step)
        bx = int(ex + bubble_dist * math.cos(angle_rad))
        by = int(ey + bubble_dist * math.sin(angle_rad))

        bx = max(bubble_radius + 4, min(total_size - bubble_radius - 4, bx))
        by = max(bubble_radius + 4, min(total_size - bubble_radius - 4, by))

        cv2.arrowedLine(
            expanded,
            (bx, by),
            (ex, ey),
            dark_red,
            thickness=1,
            tipLength=0.05,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(expanded, (bx, by), bubble_radius, dark_red, -1, cv2.LINE_AA)
        cv2.circle(expanded, (bx, by), bubble_radius + 1, white, 1, cv2.LINE_AA)

        label = str(idx + 1)
        (text_w, text_h), _ = cv2.getTextSize(label, font, 0.42, 1)
        cv2.putText(
            expanded,
            label,
            (bx - text_w // 2, by + text_h // 2),
            font,
            0.42,
            white,
            1,
            cv2.LINE_AA,
        )

    legend_y = total_size - 36
    cv2.rectangle(expanded, (0, legend_y), (total_size, total_size), (235, 235, 235), -1)
    cv2.putText(
        expanded,
        f"Markers 1-{len(global_top_markers)}: key discrepancy locations",
        (6, legend_y + 14),
        font,
        0.32,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        expanded,
        "Same number = same location across all rows",
        (6, legend_y + 30),
        font,
        0.32,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )

    return cv2.cvtColor(expanded, cv2.COLOR_BGR2RGB)


def _build_heatmap_overlay(original_pil: Image.Image, cam_fullres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cam_up = np.array(
        Image.fromarray((cam_fullres * 255).astype(np.uint8)).resize(
            (original_pil.size[0], original_pil.size[1]), Image.BILINEAR
        )
    ) / 255.0

    orig_rgb = np.array(original_pil.convert("RGB"), dtype=np.float32) / 255.0
    gray_np = np.array(original_pil.convert("L"), dtype=np.uint8)
    x1, y1, x2, y2 = _get_ink_bbox(gray_np, pad=10)

    ink_mask = np.zeros((gray_np.shape[0], gray_np.shape[1]), dtype=np.float32)
    ink_mask[y1:y2, x1:x2] = 1.0

    region = cam_up[y1:y2, x1:x2]
    if region.size > 0:
        lo, hi = np.percentile(region, 5), np.percentile(region, 95)
    else:
        lo, hi = cam_up.min(), cam_up.max()

    cam_norm = np.clip((cam_up - lo) / (hi - lo + 1e-8), 0, 1)
    cam_masked = cam_norm * ink_mask

    heatmap_rgb = _apply_jet_colormap(cam_masked).astype(np.float32) / 255.0
    mask_3ch = np.stack([ink_mask] * 3, axis=-1)
    blended = orig_rgb * (1.0 - mask_3ch * 0.5) + heatmap_rgb * mask_3ch * 0.5

    return cam_masked.astype(np.float32), (blended * 255).astype(np.uint8)


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
    ref_distances: List[float],
    ref_image_names: List[str],
    query_image_name: str,
) -> str:
    """Create a single landscape PDF report containing the Grad-CAM visual assets."""
    os.makedirs(results_dir, exist_ok=True)

    verdict_color = "#16A34A" if verdict == "GENUINE" else "#DC2626"

    canvas_width = 3600
    canvas_height = 2400
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(42, bold=True)
    header_font = _load_font(28, bold=True)
    label_font = _load_font(24, bold=False)

    title = (
        f"AVERA — Multi-Reference Grad-CAM | Verdict: {verdict} | "
        f"Avg Dist: {avg_distance:.4f} | Threshold: {threshold:.4f}"
    )
    draw.text((40, 24), title, fill=verdict_color, font=title_font)
    draw.text(
        (40, 82),
        "Col 1: Original Signature  |  Col 2: Grad-CAM Heatmap  |  Col 3: Heatmap Overlay  |  Col 4: Ink Bounding Box  |  Col 5: Forensic Stroke Map",
        fill="#444444",
        font=label_font,
    )

    col_titles = [
        "Original\nSignature",
        "Grad-CAM\nHeatmap",
        "Heatmap\nOverlay",
        "Ink Bounding\nBox",
        "Forensic\nStroke Map",
    ]
    cell_w = int((canvas_width - 160) / 5)
    cell_h = int((canvas_height - 220) / 5)
    header_h = 150

    for col_idx, title_text in enumerate(col_titles):
        x = 40 + col_idx * cell_w
        draw.multiline_text((x + 12, header_h - 42), title_text, fill="black", font=header_font, spacing=4)

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
            resized = img_rgb.resize((cell_w - 24, cell_h - 110), Image.LANCZOS)
            canvas.paste(resized, (x + 12, y + 44))

        if row_idx < 4:
            label = (
                f"Reference Specimen {row_idx + 1}\n"
                f"{os.path.basename(ref_image_names[row_idx]) if row_idx < len(ref_image_names) else 'n/a'}\n"
                f"Similarity Dist: {ref_distances[row_idx]:.4f}"
            )
        else:
            label = (
                f"Questioned Document\n"
                f"{os.path.basename(query_image_name)}\n"
                f"(vs consensus of 4 references)"
            )
        draw.multiline_text((40, y + cell_h - 92), label, fill="black", font=label_font, spacing=4)

    pdf_path = os.path.join(results_dir, "output.pdf")
    canvas.save(pdf_path, format="PDF")

    png_path = os.path.join(results_dir, "output.png")
    canvas.save(png_path, format="PNG")

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

        questioned_cam, questioned_blend = _build_heatmap_overlay(original_pil_image, cam_fullres)
        questioned_bbox = _generate_bounding_box_visualization(original_pil_image)

        safe_case = self._sanitize_case_name(case_name)
        export_dir = os.path.join(tempfile.gettempdir(), "gradcam-exports", safe_case)
        os.makedirs(export_dir, exist_ok=True)

        ref_images_paths = [f"{case_name}/{ref_id}" for ref_id in reference_image_ids]
        query_image_path = f"{case_name}/{questioned_image_id}"

        reference_gray_images: List[np.ndarray] = []
        ref_distances: List[float] = []
        reference_visuals: List[dict] = []

        for idx, ref_tensor in enumerate(reference_tensors):
            ref_pil = (
                reference_pil_images[idx]
                if idx < len(reference_pil_images)
                else Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            )
            ref_gray = np.asarray(ref_pil.convert("L"), dtype=np.uint8)
            reference_gray_images.append(ref_gray)

            ref_cam = self._compute_gradcam(ref_tensor)
            if ref_cam is None:
                ref_cam = np.zeros((self._settings.MODEL_INPUT_SIZE, self._settings.MODEL_INPUT_SIZE), dtype=np.float32)

            ref_cam_masked, ref_blend = _build_heatmap_overlay(ref_pil, ref_cam)
            ref_bbox = _generate_bounding_box_visualization(ref_pil)
            ref_distance = self._compute_pair_distance(ref_tensor, questioned_tensor)
            ref_distances.append(ref_distance)

            reference_visuals.append(
                {
                    "orig": ref_pil,
                    "cam": ref_cam_masked,
                    "blend": ref_blend,
                    "bbox": ref_bbox,
                }
            )

        questioned_gray = np.asarray(original_pil_image.convert("L"), dtype=np.uint8)
        global_top_markers = _compute_global_top_markers(reference_gray_images, questioned_gray)
        if not global_top_markers:
            global_top_markers = [
                (
                    1.0,
                    questioned_gray.shape[1] // 2,
                    questioned_gray.shape[0] // 2,
                )
            ]

        questioned_stroke_diff = _generate_stroke_difference_visualization(
            original_pil_image,
            global_top_markers,
            self._settings.MODEL_INPUT_SIZE,
        )

        all_orig_pils = [item["orig"] for item in reference_visuals] + [original_pil_image.convert("RGB")]
        all_cams = [item["cam"] for item in reference_visuals] + [cam_fullres]
        all_blends = [item["blend"] for item in reference_visuals] + [questioned_blend]
        all_bboxes = [item["bbox"] for item in reference_visuals] + [questioned_bbox]
        all_stroke_diffs = [
            _generate_stroke_difference_visualization(item["orig"], global_top_markers, self._settings.MODEL_INPUT_SIZE)
            for item in reference_visuals
        ] + [questioned_stroke_diff]

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
            ref_distances=ref_distances,
            ref_image_names=ref_images_paths,
            query_image_name=query_image_path,
        )

        exported_files.append(pdf_path)

        logger.info(
            "Grad-CAM visual assets generated",
            case_name=case_name,
            ref_images_paths=ref_images_paths,
            query_image_path=query_image_path,
            ref_distances=ref_distances,
            global_top_markers=global_top_markers,
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

        logger.info("gradcam tensor check", is_inference=tensor.is_inference())
        
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
            model.eval()
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

    def _compute_pair_distance(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
        """Compute the normalized L2 distance between two embeddings."""
        model = get_model()
        first_param = next(model.parameters())
        actual_device = first_param.device
        model_dtype = first_param.dtype

        if tensor_a.dim() == 3:
            tensor_a = tensor_a.unsqueeze(0)
        if tensor_b.dim() == 3:
            tensor_b = tensor_b.unsqueeze(0)

        tensor_a = tensor_a.to(device=actual_device, dtype=model_dtype)
        tensor_b = tensor_b.to(device=actual_device, dtype=model_dtype)

        with torch.inference_mode():
            embedding_a = model(tensor_a)
            embedding_b = model(tensor_b)
            normalized_a = F.normalize(embedding_a, p=2, dim=1)
            normalized_b = F.normalize(embedding_b, p=2, dim=1)
            return float(torch.norm(normalized_a - normalized_b, p=2, dim=1).item())

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
        """
        Try to upload generated PNG files to Azure Blob Storage.
        If upload fails, return local file paths instead so images are still accessible.
        """
        result_paths: List[str] = []

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

            try:
                with open(local_path, "rb") as handle:
                    data = handle.read()

                logger.info(
                    "Uploading Grad-CAM image",
                    file_path=local_path,
                    blob_id=blob_id,
                    size_bytes=len(data),
                )
                await blob_svc.upload_blob(blob_id=blob_id, data=data, content_type="image/png")
                result_paths.append(blob_id)
                logger.info("Uploaded blob", blob_id=blob_id)
            except Exception as upload_error:
                logger.warning(
                    "Blob upload failed; returning local path instead",
                    blob_id=blob_id,
                    local_path=local_path,
                    error=str(upload_error),
                )
                # Return local file path if upload fails
                result_paths.append(local_path)

        logger.info(
            "Completed Grad-CAM image processing",
            case_name=case_name,
            result_count=len(result_paths),
        )
        return result_paths

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