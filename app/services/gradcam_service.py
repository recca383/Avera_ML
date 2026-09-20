"""
Grad-CAM service.
Generates visualization images for the questioned signature, assembles the
compiled court-exhibit PDF report, and uploads assets to Azure Blob Storage.

Compiled PDF format
--------------------
`export_compiled_pdf()` mirrors the report layout built in the
AVERA_Pipeline7 notebook (Section 14, "Compiled Forensic PDF Report"):

    Cover page
    Section 1 -- Case Summary (verdict, confidence, F1-F7 at a glance)
    Section 2 -- Visual Forensic Analysis (signatures, Grad-CAM overlay,
                 overlay comparison, ink bounding box, forensic stroke map,
                 per-marker stroke-difference crop table)
    Section 3 -- Forensic Findings (F1-F7 full text report)
    Section 4 -- Understanding This Report + Disclaimer

The Grad-CAM page renders the heatmap blended over the actual signature
(the same array used for the individual "_heatmap.png" export), not the
raw masked activation map, so the signature stays visible under the
heatmap on the exhibit page.

F1-F7 findings are not computed in this service. `export_compiled_pdf()`
accepts an optional `forensic_findings` dict shaped like the notebook's
`forensic_json` output (`key_findings` + `modal_observations`, see
`_build_f1f7_report_lines()` below for the exact keys read). Pass the
real output of your F1-F7 service through that parameter; if omitted,
Section 3 is rendered with "N/A" placeholders instead of failing.
"""

import asyncio
import math
import os
import tempfile
import textwrap
import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import matplotlib
import matplotlib.colors
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.model_loader import get_model

logger = get_logger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_LAYER_NAME = "backbone.conv_layers.26"
MIN_MARKERS = 3
MAX_MARKERS = 9
LEGAL_FONT = "serif"


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


def _get_ink_mask(gray_np: np.ndarray) -> np.ndarray:
    """
    Binarized (not skeletonized) ink mask. Used only to check whether a
    marker sits on visible ink in a given specimen -- skeleton thinning can
    drift a pixel or two at stroke intersections, so this checks against
    what an examiner actually sees in the crop rather than the centerline.
    """
    gray_np = np.asarray(gray_np)
    if gray_np.dtype != np.uint8:
        gray_np = np.clip(gray_np, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(gray_np, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


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
    min_markers: int = MIN_MARKERS,
    max_markers: int = MAX_MARKERS,
    region: int = 12,
) -> List[tuple[float, int, int]]:
    """
    Returns (score, x, y) discrepancy markers, sorted left to right by
    x-coordinate, guaranteed between min_markers and max_markers when any
    discrepancy pixels exist at all.
    """
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

    ys, xs = np.where(combined > 0)
    if len(ys) == 0:
        return []

    scores = density[ys, xs]
    peak_score = float(scores.max())
    order = np.argsort(-scores)
    candidates: List[tuple[float, int, int]] = []
    taken = np.zeros((height, width), dtype=bool)

    for index in order:
        cy, cx = int(ys[index]), int(xs[index])
        if taken[cy, cx]:
            continue
        score = float(scores[index])
        candidates.append((score, cx, cy))
        y1, y2 = max(0, cy - region), min(height, cy + region)
        x1, x2 = max(0, cx - region), min(width, cx + region)
        taken[y1:y2, x1:x2] = True
        if len(candidates) >= max_markers:
            break

    score_threshold = min_score * peak_score
    kept = [candidate for candidate in candidates if candidate[0] >= score_threshold]
    target_floor = min(min_markers, len(candidates))
    if len(kept) < target_floor:
        kept = candidates[:target_floor]

    return sorted(kept[:max_markers], key=lambda candidate: candidate[1])


def _classify_local_discrepancy(
    skels_ref_list: List[np.ndarray],
    skel_query: np.ndarray,
    cx: int,
    cy: int,
    region: int = 12,
) -> str:
    """
    Classifies the discrepancy at (cx, cy) as an omission (reference has
    ink, query doesn't), an addition (query has ink, reference doesn't), or
    a structural deviation (both present but shaped/positioned differently).
    """
    height, width = skel_query.shape
    y1, y2 = max(0, cy - region), min(height, cy + region)
    x1, x2 = max(0, cx - region), min(width, cx + region)

    omission_count = 0
    addition_count = 0
    for skel_ref in skels_ref_list:
        window_ref = skel_ref[y1:y2, x1:x2]
        window_qry = skel_query[y1:y2, x1:x2]
        omission_count += int(((window_ref == 1) & (window_qry == 0)).sum())
        addition_count += int(((window_qry == 1) & (window_ref == 0)).sum())

    if omission_count > addition_count * 1.3:
        return "Omission - reference stroke missing in query"
    elif addition_count > omission_count * 1.3:
        return "Addition - extra stroke present in query"
    else:
        return "Structural deviation - shape/path mismatch"


def _which_refs_have_ink(
    ref_gray_list: List[np.ndarray],
    cx: int,
    cy: int,
    radius: int = 5,
) -> List[int]:
    """
    Returns 1-indexed reference numbers with visible ink within `radius`
    pixels of the marker point, checked against the raw ink mask rather
    than the thinned skeleton (which can drift a pixel or two).
    """
    refs = []
    for i, gray_np in enumerate(ref_gray_list):
        mask = _get_ink_mask(gray_np)
        height, width = mask.shape
        y1, y2 = max(0, cy - radius), min(height, cy + radius + 1)
        x1, x2 = max(0, cx - radius), min(width, cx + radius + 1)
        if mask[y1:y2, x1:x2].sum() > 0:
            refs.append(i + 1)
    return refs


def _crop_and_zoom(pil_img: Image.Image, cx: int, cy: int, crop_radius: int, zoom_size: int) -> np.ndarray:
    gray_np = np.array(pil_img.convert("L") if pil_img.mode != "L" else pil_img)
    height, width = gray_np.shape

    x1 = max(0, cx - crop_radius)
    y1 = max(0, cy - crop_radius)
    x2 = min(width, cx + crop_radius)
    y2 = min(height, cy + crop_radius)
    crop = gray_np[y1:y2, x1:x2]

    if crop.size == 0:
        crop = np.full((crop_radius * 2, crop_radius * 2), 255, dtype=np.uint8)
        local_cx, local_cy = crop_radius, crop_radius
    else:
        local_cx = cx - x1
        local_cy = cy - y1

    zoomed = cv2.resize(crop, (zoom_size, zoom_size), interpolation=cv2.INTER_CUBIC)
    scale_x = zoom_size / max(1, (x2 - x1))
    scale_y = zoom_size / max(1, (y2 - y1))
    zoomed_bgr = cv2.cvtColor(zoomed, cv2.COLOR_GRAY2BGR)

    marker_x = int(local_cx * scale_x)
    marker_y = int(local_cy * scale_y)
    cv2.circle(zoomed_bgr, (marker_x, marker_y), 6, (30, 30, 180), 2, cv2.LINE_AA)

    return cv2.cvtColor(zoomed_bgr, cv2.COLOR_BGR2RGB)


def build_stroke_crop_rows(
    reference_gray_images: List[np.ndarray],
    reference_pils: List[Image.Image],
    query_pil: Image.Image,
    skel_query: np.ndarray,
    skels_ref: List[np.ndarray],
    global_top_markers: List[tuple[float, int, int]],
    crop_radius: int = 30,
    zoom_size: int = 140,
) -> List[dict]:
    """
    Builds one row per discrepancy marker for the stroke-difference crop
    table: a zoomed crop of the marker location from all four references
    plus the query, a short caption classifying the discrepancy, and which
    reference(s) actually show ink at that point.
    """
    rows: List[dict] = []
    for idx, (_score, cx, cy) in enumerate(global_top_markers, 1):
        ref_crops = [_crop_and_zoom(p, cx, cy, crop_radius, zoom_size) for p in reference_pils]
        query_crop = _crop_and_zoom(query_pil, cx, cy, crop_radius, zoom_size)

        discrepancy_type = _classify_local_discrepancy(skels_ref, skel_query, cx, cy)
        refs_with_ink = _which_refs_have_ink(reference_gray_images, cx, cy)
        refs_note = (
            f"Present in Ref {', '.join(str(r) for r in refs_with_ink)}"
            if refs_with_ink
            else "Not present in any reference"
        )
        caption = f"Marker {idx} at ({cx},{cy}) - {discrepancy_type}\n{refs_note}"

        rows.append(
            {
                "marker_num": idx,
                "ref_crops": ref_crops,
                "query_crop": query_crop,
                "caption": caption,
                "position": (cx, cy),
                "refs_with_ink": refs_with_ink,
            }
        )
    return rows


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
    bubble_radius = 13
    min_gap = 2 * bubble_radius + 8
    raw_bubble_x = [cx + margin_px for _score, cx, _cy in global_top_markers]
    bubble_x = raw_bubble_x.copy()
    for index in range(1, len(bubble_x)):
        if bubble_x[index] - bubble_x[index - 1] < min_gap:
            bubble_x[index] = bubble_x[index - 1] + min_gap

    max_allowed = total_size - bubble_radius - 4
    if bubble_x and bubble_x[-1] > max_allowed:
        overflow = bubble_x[-1] - max_allowed
        bubble_x = [x - overflow for x in bubble_x]
    min_allowed = bubble_radius + 4
    if bubble_x and bubble_x[0] < min_allowed:
        shift = min_allowed - bubble_x[0]
        bubble_x = [x + shift for x in bubble_x]
    bubble_y = int(margin_px * 0.42)

    for idx, (_score, cx, cy) in enumerate(global_top_markers):
        ex = cx + margin_px
        ey = cy + margin_px

        bx = int(bubble_x[idx])
        by = bubble_y

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
    """
    Returns (cam_masked, blended). `blended` is the heatmap drawn over the
    actual signature pixels (restricted to the ink region) -- this is the
    array the compiled PDF's Grad-CAM page now displays, so the signature
    stays visible under the heatmap rather than showing a bare colormap.
    """
    cam_up = np.array(
        Image.fromarray((cam_fullres * 255).astype(np.uint8)).resize(
            (original_pil.size[0], original_pil.size[1]), Image.Resampling.BILINEAR
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
    """Apply a simple jet-like colormap without requiring Matplotlib's cm module."""
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


def _generate_overlay_comparison(ref_pil: Image.Image, query_pil: Image.Image) -> np.ndarray:
    """
    Colors the query's ink and the reference's ink separately, then blends
    them: blue = reference-only ink, red = query-only ink, purple = overlap.
    """
    ref_gray = np.asarray(ref_pil.convert("L"), dtype=np.uint8)
    query_gray = np.asarray(query_pil.convert("L"), dtype=np.uint8)
    _, ref_binary = cv2.threshold(
        ref_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, query_binary = cv2.threshold(
        query_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    ref_mask = ref_binary > 0
    query_mask = query_binary > 0
    canvas = np.ones((*ref_gray.shape, 3), dtype=np.uint8) * 255
    canvas[ref_mask & ~query_mask] = [59, 130, 246]
    canvas[query_mask & ~ref_mask] = [220, 38, 38]
    canvas[ref_mask & query_mask] = [88, 28, 135]
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Compiled court-exhibit PDF -- page builders
# Ported from AVERA_Pipeline7 notebook, Section 14 ("Compiled Forensic PDF
# Report"). Each helper takes the state it needs as explicit parameters
# instead of relying on notebook-cell globals.
# ─────────────────────────────────────────────────────────────────────────────

def _verdict_color(verdict: str) -> str:
    return "#16A34A" if verdict == "GENUINE" else "#DC2626"


def _stamp_footer(fig, page_counter: list, total_pages: Optional[int], case_id: str) -> None:
    """Stamps a running page number + case ID + confidentiality notice on
    every page. page_counter is a 1-element list so every helper shares
    the same running count."""
    page_counter[0] += 1
    fig.text(
        0.5,
        0.015,
        f"Page {page_counter[0]} of {total_pages}    |    AVERA Case {case_id}    |    "
        f"CONFIDENTIAL - Generated for Academic/Thesis Research Purposes",
        fontsize=7,
        color="#888888",
        ha="center",
        family=LEGAL_FONT,
    )


def _page_title(
    pdf,
    page_counter,
    total_pages,
    case_id,
    verdict,
    conf_genuine,
    conf_forged,
    avg_distance,
    threshold,
    query_image_name,
    ref_image_names,
    model_version_tag,
) -> None:

    verdict_color = _verdict_color(verdict)
    fig = plt.figure(figsize=(8.5, 11))

    banner_ax = fig.add_axes((0.0, 0.86, 1.0, 0.14))
    banner_ax.set_facecolor(verdict_color)
    banner_ax.set_xticks([]); banner_ax.set_yticks([])
    for spine in banner_ax.spines.values():
        spine.set_visible(False)
    banner_ax.text(0.5, 0.62, "AVERA", fontsize=40, fontweight="bold", color="white",
                    ha="center", va="center", family=LEGAL_FONT, transform=banner_ax.transAxes)
    banner_ax.text(0.5, 0.18, "Automated Verification & Explainable Recognition of Authorship",
                    fontsize=9.5, color="white", ha="center", va="center",
                    family=LEGAL_FONT, style="italic", transform=banner_ax.transAxes)

    fig.text(0.5, 0.75, "Signature Verification & Forensic Analysis Report",
              fontsize=17, fontweight="bold", ha="center", family=LEGAL_FONT)
    fig.text(0.5, 0.715, f"Case {case_id}", fontsize=12, ha="center",
              family=LEGAL_FONT, color="#555555")

    verdict_ax = fig.add_axes((0.20, 0.55, 0.60, 0.10))
    verdict_ax.set_facecolor(verdict_color)
    verdict_ax.set_facecolor(matplotlib.colors.to_rgba(verdict_color, 0.10))
    verdict_ax.set_xticks([]); verdict_ax.set_yticks([])
    for spine in verdict_ax.spines.values():
        spine.set_edgecolor(verdict_color)
        spine.set_linewidth(1.5)
    verdict_ax.text(0.5, 0.62, f"VERDICT: {verdict}", fontsize=16, fontweight="bold",
                     color=verdict_color, ha="center", va="center", family=LEGAL_FONT,
                     transform=verdict_ax.transAxes)
    verdict_ax.text(0.5, 0.22,
                     f"{conf_genuine:.1f}% genuine  /  {conf_forged:.1f}% forged\n"
                     f"distance {avg_distance:.4f}  (threshold {threshold:.4f})",
                     fontsize=8.5, ha="right", va="center", family=LEGAL_FONT, color="#444444",
                     transform=verdict_ax.transAxes)

    meta_lines = [
        f"Questioned Document  :  {os.path.basename(query_image_name)}",
        f"Reference Specimens  :  {len(ref_image_names)} genuine samples on file",
        f"Model / Pipeline     :  {model_version_tag}",
        f"Report Generated     :  {datetime.datetime.now().strftime('%B %d, %Y  %H:%M')}",
    ]
    fig.text(0.5, 0.40, "\n".join(meta_lines), fontsize=10, ha="center", va="top",
              family="monospace", color="#333333", linespacing=2.0)

    fig.text(0.5, 0.10,
              "This report was generated by AVERA, an offline signature verification\n"
              "system developed for academic thesis research. See the final section of\n"
              "this report for an explanation of these results and important disclaimers.",
              fontsize=8.7, ha="center", va="center", family=LEGAL_FONT, color="#666666", style="italic")

    _stamp_footer(fig, page_counter, total_pages, case_id)
    pdf.savefig(fig)
    plt.close(fig)


def _page_section_divider(section_no, title, subtitle, pdf, page_counter, total_pages, case_id, verdict) -> None:
    """Section-break divider page."""
    verdict_color = _verdict_color(verdict)
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.58, f"SECTION {section_no}", fontsize=14, color="#999999",
              ha="center", family=LEGAL_FONT)
    fig.text(0.5, 0.52, title, fontsize=26, fontweight="bold", ha="center",
              va="center", color=verdict_color, family=LEGAL_FONT)
    if subtitle:
        fig.text(0.5, 0.45, subtitle, fontsize=12, ha="center", va="center",
                  color="#444444", family=LEGAL_FONT)
    fig.add_artist(mlines.Line2D([0.25, 0.75], [0.60, 0.60], transform=getattr(fig, 'transFigure'),
                                  color=verdict_color, linewidth=1.5))
    _stamp_footer(fig, page_counter, total_pages, case_id)
    pdf.savefig(fig)
    plt.close(fig)


def _page_case_summary(pdf, page_counter, total_pages, case_id, verdict, conf_genuine, conf_forged,
                        avg_distance, threshold, query_image_name, ref_image_names,
                        model_version_tag, findings_rows) -> None:
    """Section 1 content -- case metadata, verdict block, F1-F7 at a glance."""
    verdict_color = _verdict_color(verdict)
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Case Summary", fontsize=16, fontweight="bold", family=LEGAL_FONT, y=0.965)

    fig.text(0.08, 0.90, "Case Information", fontsize=11, fontweight="bold", family=LEGAL_FONT)
    info_lines = [
        f"Case ID               {case_id}",
        f"Questioned Document   {os.path.basename(query_image_name)}",
        f"Reference Specimens   {', '.join(os.path.basename(r) for r in ref_image_names)}",
        f"Model / Pipeline      {model_version_tag}  (training ratio 5:4, inference ratio 4:1)",
        f"Report Generated      {datetime.datetime.now().strftime('%B %d, %Y  %H:%M')}",
    ]
    fig.text(0.08, 0.87, "\n".join(info_lines), fontsize=9, va="top",
              family="monospace", color="#333333", linespacing=1.9)

    fig.text(0.08, 0.68, "Verdict", fontsize=11, fontweight="bold", family=LEGAL_FONT)
    verdict_ax = fig.add_axes((0.08, 0.58, 0.84, 0.08))
    verdict_ax.set_facecolor(verdict_color); verdict_ax.set_facecolor(matplotlib.colors.to_rgba(verdict_color, 0.10))
    verdict_ax.set_xticks([]); verdict_ax.set_yticks([])
    for spine in verdict_ax.spines.values():
        spine.set_edgecolor(verdict_color); spine.set_linewidth(1.5)
    verdict_ax.text(0.03, 0.5, f"{verdict}", fontsize=15, fontweight="bold", color=verdict_color,
                     va="center", family=LEGAL_FONT, transform=verdict_ax.transAxes)
    verdict_ax.text(0.97, 0.5,
                     f"{conf_genuine:.1f}% genuine  /  {conf_forged:.1f}% forged\n"
                     f"distance {avg_distance:.4f}  (threshold {threshold:.4f})",
                     fontsize=8.5, ha="right", va="center", family=LEGAL_FONT, color="#444444",
                     transform=verdict_ax.transAxes)

    fig.text(0.08, 0.46, "Key Findings at a Glance (F1-F7)", fontsize=11,
              fontweight="bold", family=LEGAL_FONT)

    y = 0.42
    row_h = 0.032
    for code, name, label in findings_rows:
        fig.text(0.09, y, code, fontsize=9.5, fontweight="bold", family=LEGAL_FONT, color=verdict_color)
        fig.text(0.14, y, name, fontsize=9.5, family=LEGAL_FONT, color="#333333")
        fig.text(0.62, y, label, fontsize=9.5, family=LEGAL_FONT, color="#555555", ha="left")
        y -= row_h
    fig.add_artist(mlines.Line2D([0.08, 0.92], [0.445, 0.445], transform=getattr(fig, 'transFigure'),
                                  color="#cccccc", linewidth=0.8))

    fig.text(0.08, 0.16,
              "Full findings, measured data tables, and supporting visual evidence for\n"
              "each F1-F7 category are provided in Sections 2 and 3 of this report.",
              fontsize=9, family=LEGAL_FONT, color="#666666", style="italic")

    _stamp_footer(fig, page_counter, total_pages, case_id)
    pdf.savefig(fig)
    plt.close(fig)


def _page_signature_strip(pil_images, labels, title, pdf, case_id, verdict,
                           query_index=4, cmap: Optional[str] = None, vmin: Optional[float] = None, vmax: Optional[float] = None,
                           page_counter: Optional[list] = None, total_pages: int = 0) -> None:
    """Horizontal strip: references left to right, gap, then the QUESTIONED image."""
    verdict_color = _verdict_color(verdict)
    n_refs = query_index
    width_ratios = [1] * n_refs + [0.25, 1.15]
    fig = plt.figure(figsize=(3.2 * (n_refs + 1) + 1, 4.4))
    gs = fig.add_gridspec(1, n_refs + 2, width_ratios=width_ratios, wspace=0.15)

    fig.suptitle(title, fontsize=15, fontweight="bold", color=verdict_color, y=1.02, family=LEGAL_FONT)

    for i in range(n_refs):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(pil_images[i], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(labels[i], fontsize=10, fontweight="bold", family=LEGAL_FONT)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#999999")

    gap_ax = fig.add_subplot(gs[0, n_refs])
    gap_ax.axis("off")
    gap_ax.axvline(x=0.5, ymin=0.05, ymax=0.95, color="#999999", linewidth=1.2, linestyle="--")

    q_ax = fig.add_subplot(gs[0, n_refs + 1])
    q_ax.imshow(pil_images[query_index], cmap=cmap, vmin=vmin, vmax=vmax)
    q_ax.set_title(f"QUESTIONED\n{labels[query_index]}", fontsize=10,
                    fontweight="bold", color=verdict_color, family=LEGAL_FONT)
    q_ax.set_xticks([]); q_ax.set_yticks([])
    for spine in q_ax.spines.values():
        spine.set_edgecolor(verdict_color)
        spine.set_linewidth(2.5)

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))
    if page_counter is not None:
        _stamp_footer(fig, page_counter, total_pages, case_id)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_signature_stack(pil_images, labels, title, pdf, case_id, verdict,
                           query_index=4, cmap: Optional[str] = None, vmin: Optional[float] = None, vmax: Optional[float] = None,
                           colorbar=False, page_counter: Optional[list] = None, total_pages: int = 0) -> None:
    """Vertical stack: references stack first, divider, then the QUESTIONED row."""
    verdict_color = _verdict_color(verdict)
    n_refs = query_index
    height_ratios = [1] * n_refs + [0.18, 1.15]
    fig = plt.figure(figsize=(4.6, 3.0 * (n_refs + 1) + 1))
    gs = fig.add_gridspec(n_refs + 2, 1, height_ratios=height_ratios, hspace=0.12)

    fig.suptitle(title, fontsize=15, fontweight="bold", color=verdict_color, y=0.995, family=LEGAL_FONT)

    for i in range(n_refs):
        ax = fig.add_subplot(gs[i, 0])
        im = ax.imshow(pil_images[i], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_ylabel(labels[i], fontsize=9, fontweight="bold", rotation=0,
                       labelpad=45, va="center", family=LEGAL_FONT)
        ax.set_xticks([]); ax.set_yticks([])
        if colorbar:
            plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    gap_ax = fig.add_subplot(gs[n_refs, 0])
    gap_ax.axis("off")
    gap_ax.axhline(y=0.5, xmin=0.05, xmax=0.95, color="#999999", linewidth=1.2, linestyle="--")

    q_ax = fig.add_subplot(gs[n_refs + 1, 0])
    im = q_ax.imshow(pil_images[query_index], cmap=cmap, vmin=vmin, vmax=vmax)
    q_ax.set_ylabel(f"QUESTIONED\n{labels[query_index]}", fontsize=9,
                     fontweight="bold", color=verdict_color, rotation=0,
                     labelpad=45, va="center", family=LEGAL_FONT)
    q_ax.set_xticks([]); q_ax.set_yticks([])
    for spine in q_ax.spines.values():
        spine.set_edgecolor(verdict_color)
        spine.set_linewidth(2.5)
    if colorbar:
        plt.colorbar(im, ax=q_ax, fraction=0.035, pad=0.02)

    plt.tight_layout(rect=(0, 0.02, 1, 0.98))
    if page_counter is not None:
        _stamp_footer(fig, page_counter, total_pages, case_id)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_overlay_comparison(overlay_imgs, title, pdf, page_counter, total_pages, case_id, verdict) -> None:
    """Blue = reference-only ink, red = query-only ink, purple = overlap (agreement)."""
    verdict_color = _verdict_color(verdict)
    n = len(overlay_imgs)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n + 1, 4.6))
    axes = list(np.atleast_1d(axes).ravel())
    fig.suptitle(title, fontsize=15, fontweight="bold", color=verdict_color, y=1.02, family=LEGAL_FONT)

    for i, img in enumerate(overlay_imgs):
        axes[i].imshow(img)
        axes[i].set_title(f"vs. Reference {i + 1}", fontsize=10, fontweight="bold", family=LEGAL_FONT)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        for spine in axes[i].spines.values():
            spine.set_edgecolor("#999999")

    legend_handles = [
        mpatches.Patch(color="#3B82F6", label="Reference only (missing in query)"),
        mpatches.Patch(color="#DC2626", label="Query only (extra in query)"),
        mpatches.Patch(color="#581C87", label="Match (present in both)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=(0, 0.08, 1, 0.95))
    _stamp_footer(fig, page_counter, total_pages, case_id)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_stroke_crop_table(rows, title, pdf, page_counter, total_pages, case_id, verdict,
                             rows_per_page=3) -> None:
    """
    Stroke-difference table -- one row per marker (3-9), left to right:
    [Marker # | Ref 1 | Ref 2 | Ref 3 | Ref 4 | Questioned | Caption].
    Split across landscape pages so thumbnails and captions stay legible.
    """
    verdict_color = _verdict_color(verdict)
    n_total = len(rows)
    if n_total == 0:
        return
    n_pages = -(-n_total // rows_per_page)  # ceil division

    for page_idx in range(n_pages):
        chunk = rows[page_idx * rows_per_page: (page_idx + 1) * rows_per_page]
        n = len(chunk)

        fig = plt.figure(figsize=(14, 8.5))  # landscape
        gs = fig.add_gridspec(n, 6, width_ratios=[1, 1, 1, 1, 1, 1.6],
                               hspace=0.55, wspace=0.12,
                               left=0.06, right=0.97, top=0.85, bottom=0.06)

        page_title = title if n_pages == 1 else f"{title}  (page {page_idx + 1} of {n_pages})"
        fig.suptitle(page_title, fontsize=17, fontweight="bold", color=verdict_color,
                     y=0.965, family=LEGAL_FONT)
        fig.text(0.5, 0.915, "Green border = specimen has ink at this point   |   Gray dashed = specimen does not",
                  fontsize=9.5, ha="center", color="#666666", style="italic", family=LEGAL_FONT)

        for r, row in enumerate(chunk):
            for c in range(4):
                ax_ref = fig.add_subplot(gs[r, c])
                ax_ref.imshow(row["ref_crops"][c])
                ax_ref.set_xticks([]); ax_ref.set_yticks([])

                has_ink = (c + 1) in row.get("refs_with_ink", [])
                for spine in ax_ref.spines.values():
                    spine.set_edgecolor("#16A34A" if has_ink else "#9CA3AF")
                    spine.set_linewidth(2.6 if has_ink else 1.2)
                    spine.set_linestyle("solid" if has_ink else "dashed")

                if c == 0:
                    ax_ref.set_ylabel(f"Marker {row['marker_num']}", fontsize=13,
                                       fontweight="bold", rotation=0, labelpad=48,
                                       va="center", family=LEGAL_FONT)
                if r == 0:
                    ax_ref.set_title(f"Reference {c + 1}", fontsize=11, fontweight="bold", family=LEGAL_FONT)

            ax_qry = fig.add_subplot(gs[r, 4])
            ax_qry.imshow(row["query_crop"])
            ax_qry.set_xticks([]); ax_qry.set_yticks([])
            for spine in ax_qry.spines.values():
                spine.set_edgecolor(verdict_color)
                spine.set_linewidth(2.0)
            if r == 0:
                ax_qry.set_title("QUESTIONED", fontsize=11, fontweight="bold",
                                  color=verdict_color, family=LEGAL_FONT)

            ax_cap = fig.add_subplot(gs[r, 5])
            ax_cap.axis("off")
            cap_lines = row["caption"].split("\n")
            header = cap_lines[0].split("-", 1)
            title_txt = header[0].strip()
            detail = header[1].strip() if len(header) > 1 else ""
            refs_note = cap_lines[1] if len(cap_lines) > 1 else ""
            ax_cap.text(0.0, 0.78, title_txt, fontsize=11.5, fontweight="bold",
                        va="center", ha="left", wrap=True, family=LEGAL_FONT, transform=ax_cap.transAxes)
            ax_cap.text(0.0, 0.50, detail, fontsize=10.5, va="center", ha="left",
                        wrap=True, color="#333333", family=LEGAL_FONT, transform=ax_cap.transAxes)
            ax_cap.text(0.0, 0.22, refs_note, fontsize=10, va="center", ha="left",
                        wrap=True, color="#666666", style="italic", family=LEGAL_FONT, transform=ax_cap.transAxes)
            if r == 0:
                ax_cap.text(0.0, 1.0, "Discrepancy", fontsize=12, fontweight="bold",
                            va="bottom", ha="left", family=LEGAL_FONT, transform=ax_cap.transAxes)

        _stamp_footer(fig, page_counter, total_pages, case_id)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _page_text_block(lines, title, pdf, page_counter, total_pages, case_id,
                      lines_per_page=46, mono=True) -> None:
    """Paginated plain-text page(s), used for the F1-F7 findings report."""
    font = "monospace" if mono else LEGAL_FONT
    for start in range(0, len(lines), lines_per_page):
        chunk = lines[start:start + lines_per_page]
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97, family=LEGAL_FONT)
        fig.text(0.06, 0.935, "\n".join(chunk), fontsize=8.3, family=font, va="top", ha="left")
        _stamp_footer(fig, page_counter, total_pages, case_id)
        pdf.savefig(fig)
        plt.close(fig)


def _page_explanation(pdf, page_counter, total_pages, case_id) -> None:
    """Section 4 content -- plain-language explanation of what the results mean."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Understanding This Report", fontsize=16, fontweight="bold", family=LEGAL_FONT, y=0.97)

    body = [
        "WHAT AVERA DOES",
        "AVERA compares a questioned ('query') signature against four genuine",
        "reference specimens using a trained Siamese Neural Network (SNN). The",
        "network converts each signature into a numeric embedding, and measures",
        "the distance between the query's embedding and the reference cluster's",
        "embeddings. A shorter distance indicates greater structural similarity.",
        "",
        "HOW THE VERDICT IS DETERMINED",
        "The verdict (GENUINE or FORGED) is based on whether this distance falls",
        "below a decision threshold, calibrated from the Equal Error Rate (EER)",
        "during model evaluation. The confidence percentages reflect how far the",
        "distance sits from that threshold, not a probability in the statistical",
        "sense.",
        "",
        "WHAT F1-F7 REPRESENT",
        "F1-F7 translate the model's decision into forensic-examiner vocabulary:",
        "  F1  General structural features (strokes, pen lifts, terminal marks)",
        "  F2  Baseline angle consistency",
        "  F3  Line quality / tremor (stroke smoothness)",
        "  F4  Proportion & spacing (signature dimensions)",
        "  F5  Variation vs. the population-wide decision threshold",
        "  F6  Variation vs. this writer's OWN natural signature-to-signature range",
        "  F7  Ink deposition consistency (a pen-pressure proxy)",
        "F1-F4 and F7 are computed with classical image-processing techniques;",
        "F5 and F6 come directly from the trained model's own embedding space.",
        "",
        "WHAT THE VISUAL EVIDENCE MEANS",
        "Grad-CAM shows where the NEURAL NETWORK focused when making its",
        "decision -- it is a model-transparency check, not a forensic finding on",
        "its own. The Overlay Comparison and Forensic Stroke Map are classical,",
        "model-independent visual comparisons of the ink itself, closer to",
        "traditional forensic document examination practice.",
        "",
        "HOW THIS REPORT SHOULD BE USED",
        "AVERA is a decision-support and explainability tool developed for",
        "academic thesis research. Its numeric thresholds are current",
        "engineering defaults and have not yet been independently validated",
        "against a licensed forensic examiner's judgment on this dataset. This",
        "report should be treated as a structured summary of computational",
        "evidence to inform -- not replace -- expert human review.",
    ]
    fig.text(0.08, 0.90, "\n".join(body), fontsize=8.7, va="top", family=LEGAL_FONT,
              color="#222222", linespacing=1.55)

    _stamp_footer(fig, page_counter, total_pages, case_id)
    pdf.savefig(fig)
    plt.close(fig)


def _page_disclaimer(pdf, page_counter, total_pages, case_id) -> None:
    """Final page -- formal disclaimer."""
    fig = plt.figure(figsize=(8.5, 11))

    banner_ax = fig.add_axes((0.0, 0.90, 1.0, 0.06))
    banner_ax.set_facecolor("#7C2D12")
    banner_ax.set_xticks([]); banner_ax.set_yticks([])
    for spine in banner_ax.spines.values():
        spine.set_visible(False)
    banner_ax.text(0.5, 0.5, "DISCLAIMER", fontsize=15, fontweight="bold", color="white",
                    ha="center", va="center", family=LEGAL_FONT, transform=banner_ax.transAxes)

    body = [
        "1.  AVERA is an automated, offline signature verification system",
        "    developed as part of an academic thesis project. It is a research",
        "    prototype, not a certified or legally-accredited forensic tool.",
        "",
        "2.  The findings in this report (F1-F7, verdict, and confidence",
        "    percentages) are the output of computational heuristics and a",
        "    trained neural network. Several numeric thresholds used to label",
        "    these findings are current engineering defaults and have not yet",
        "    been independently validated against a licensed forensic document",
        "    examiner's judgment on this specific dataset.",
        "",
        "3.  This report is intended to serve as a decision-support and",
        "    explainability aid -- it summarizes computational evidence to help",
        "    a human examiner reason about a case. It is NOT a substitute for",
        "    review and certification by a qualified, licensed Questioned",
        "    Document Examiner (QDE), and should not be submitted as",
        "    standalone evidence in any legal or administrative proceeding.",
        "",
        "4.  Results may vary with scan quality, signature complexity, and",
        "    writer-specific variability. A GENUINE or FORGED verdict reflects",
        "    the balance of computed evidence at the time of analysis, and",
        "    does not constitute a certainty claim.",
        "",
        "5.  This system and report were developed for academic research",
        "    purposes by the AVERA thesis research team.",
    ]
    fig.text(0.08, 0.85, "\n".join(body), fontsize=9, va="top", family=LEGAL_FONT,
              color="#222222", linespacing=1.7)

    fig.text(0.5, 0.06, f"AVERA Case {case_id} - End of Report",
              fontsize=9, ha="center", family=LEGAL_FONT, color="#888888", style="italic")

    _stamp_footer(fig, page_counter, total_pages, case_id)
    pdf.savefig(fig)
    plt.close(fig)


def _build_f1f7_report_lines(
    findings: Optional[dict],
    case_id: str,
    verdict: str,
    conf_genuine: float,
    conf_forged: float,
    avg_distance: float,
    threshold: float,
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """
    Builds the F1-F7 findings text report and the (code, name, label) rows
    used on the Case Summary page.

    `findings` is expected to be shaped like the notebook's `forensic_json`:
        {
          "key_findings": {"f1_label": ..., ..., "f7_label": ...},
          "modal_observations": {
              "f1_observation": ..., ..., "f7_observation": ...,
              "f2_angle": ..., "f3_variance": ..., "f4_width_px": ...,
              "f4_height_px": ..., "f4_ratio": ..., "f5_distance": ...,
              "f5_percent_threshold": ..., "f6_min": ..., "f6_max": ...,
              "f6_mean": ..., "f7_darkness": ..., "f7_width_variance": ...,
          },
        }
    Missing keys render as "N/A" instead of raising, since F1-F7 is
    computed by a separate service and this parameter may not be wired
    through on every call site yet.
    """
    findings = findings or {}
    key = findings.get("key_findings", {})
    modal = findings.get("modal_observations", {})

    def kf(code: str) -> str:
        return key.get(f"{code}_label", "N/A")

    def mo(field: str) -> str:
        value = modal.get(field, "N/A")
        return value if isinstance(value, str) else str(value)

    names = {
        "f1": "General Information",
        "f2": "Relation to Baseline",
        "f3": "Line Quality",
        "f4": "Proportion & Spacing",
        "f5": "Variation",
        "f6": "Natural Variation Range",
        "f7": "Stroke Density (Ink Deposition)",
    }

    findings_rows = [(code.upper(), name, kf(code)) for code, name in names.items()]

    lines: List[str] = []
    sep = "=" * 65
    lines.append(sep)
    lines.append("  AVERA FORENSIC FINDINGS REPORT  (F1-F7)")
    lines.append(sep)
    lines.append("")
    lines.append(f"  Case            : {case_id}")
    lines.append(f"  Verdict         : {verdict}")
    lines.append(f"  Confidence      : {conf_genuine:.1f}% genuine / {conf_forged:.1f}% forged")
    lines.append(f"  Avg distance    : {avg_distance:.4f}   (threshold: {threshold:.4f})")
    lines.append("")
    lines.append("  KEY FINDINGS")
    lines.append("  " + "-" * 61)
    for code, name, label in findings_rows:
        lines.append(f"  {code}  {name:<32}: {label}")
    lines.append("")

    lines.append("  OBSERVATIONS")
    lines.append("  " + "-" * 61)
    for code, name in names.items():
        lines.append(f"  {code.upper()} - {name}")
        obs = mo(f"{code}_observation")
        for wrapped in textwrap.wrap(obs, width=72):
            lines.append(f"      {wrapped}")
        lines.append("")

    lines.append("  RAW MEASURED VALUES")
    lines.append("  " + "-" * 61)
    lines.append(f"  F2 angle (deg)         : {mo('f2_angle')}")
    lines.append(f"  F3 curvature variance  : {mo('f3_variance')}")
    lines.append(
        f"  F4 W x H / ratio       : {mo('f4_width_px')}px x {mo('f4_height_px')}px / {mo('f4_ratio')}"
    )
    lines.append(f"  F5 distance / % thresh : {mo('f5_distance')} / {mo('f5_percent_threshold')}%")
    lines.append(f"  F6 natural range       : {mo('f6_min')} - {mo('f6_max')}  (mean {mo('f6_mean')})")
    lines.append(f"  F7 darkness / width var: {mo('f7_darkness')} / {mo('f7_width_variance')}")
    lines.append("")
    lines.append(sep)

    return lines, findings_rows


def export_compiled_pdf(
    case_id: str,
    verdict: str,
    conf_genuine: float,
    conf_forged: float,
    avg_distance: float,
    threshold: float,
    orig_pils: List[Image.Image],
    blends: List[np.ndarray],
    bboxes: List[np.ndarray],
    stroke_diffs: List[np.ndarray],
    overlay_comparisons: List[np.ndarray],
    stroke_crop_rows: List[dict],
    ref_image_names: List[str],
    query_image_name: str,
    results_dir: str,
    model_version_tag: str = "AVERA SNN",
    forensic_findings: Optional[dict] = None,
) -> str:
    """
    Assemble the single, court-exhibit-style compiled PDF: cover page,
    Section 1 (case summary), Section 2 (signatures, Grad-CAM heatmap
    overlaid on the signature, overlay comparison, ink bounding box,
    forensic stroke map, per-marker stroke-difference crop table),
    Section 3 (F1-F7 findings), Section 4 (explanation + disclaimer).

    `blends` must be the heatmap-over-signature arrays (see
    `_build_heatmap_overlay`), not the raw masked CAM, so the Grad-CAM
    page shows the signature underneath the heatmap.
    """
    os.makedirs(results_dir, exist_ok=True)
    compiled_pdf_path = os.path.join(results_dir, f"AVERA_compiled_report_{case_id}.pdf")
    image_labels_full = ["Ref 1", "Ref 2", "Ref 3", "Ref 4", "Query"]

    report_lines, findings_rows = _build_f1f7_report_lines(
        forensic_findings, case_id, verdict, conf_genuine, conf_forged, avg_distance, threshold
    )
    lines_per_page = 70
    findings_pages = -(-len(report_lines) // lines_per_page)  # ceil division

    rows_per_page_stroke_table = 3
    stroke_table_pages = -(-max(len(stroke_crop_rows), 1) // rows_per_page_stroke_table)

    # Fixed (non-findings, non-stroke-table) page count:
    #   Cover(1) + [S1 divider(1) + Case Summary(1)] + [S2 divider(1) + Signatures(1)
    #   + Grad-CAM(1) + Overlay(1) + BBox(1) + Stroke Map(1)]
    #   + [S3 divider(1)] + [S4 divider(1) + Explanation(1) + Disclaimer(1)] = 13
    fixed_pages = 13
    total_pages = fixed_pages + stroke_table_pages + findings_pages
    page_counter = [0]

    logger.info("Building compiled PDF", path=compiled_pdf_path, total_pages=total_pages)

    with PdfPages(compiled_pdf_path) as pdf:

        # ── Cover Page ───────────────────────────────────────────────────
        _page_title(pdf, page_counter, total_pages, case_id, verdict, conf_genuine, conf_forged,
                avg_distance, threshold, query_image_name, ref_image_names, model_version_tag)

        # ── SECTION 1 — Case Summary ────────────────────────────────────
        _page_section_divider(1, "Case Summary",
                               "Verdict, confidence, and key findings at a glance",
                               pdf, page_counter, total_pages, case_id, verdict)
        _page_case_summary(pdf, page_counter, total_pages, case_id, verdict, conf_genuine, conf_forged,
                            avg_distance, threshold, query_image_name, ref_image_names,
                            model_version_tag, findings_rows)

        # ── SECTION 2 — Visual Forensic Analysis ─────────────────────────
        _page_section_divider(2, "Visual Forensic Analysis",
                               "Explainable AI and classical image-comparison evidence",
                               pdf, page_counter, total_pages, case_id, verdict)

        _page_signature_strip(
            pil_images=orig_pils, labels=image_labels_full,
            title=f"Signature Overview  |  Case {case_id}  |  Verdict: {verdict}",
            pdf=pdf, case_id=case_id, verdict=verdict, cmap="gray",
            page_counter=page_counter, total_pages=total_pages,
        )

        # Grad-CAM page: heatmap blended over the actual signature (all_blend
        # in the notebook), not the raw masked activation map, so the
        # signature is still visible under the heatmap.
        _page_signature_stack(
            pil_images=blends, labels=image_labels_full,
            title="Grad-CAM Attention Heatmaps (overlaid on signature)",
            pdf=pdf, case_id=case_id, verdict=verdict,
            page_counter=page_counter, total_pages=total_pages,
        )

        _page_overlay_comparison(
            overlay_imgs=overlay_comparisons,
            title="Overlay Comparison - Questioned vs. Each Reference",
            pdf=pdf, page_counter=page_counter, total_pages=total_pages, case_id=case_id, verdict=verdict,
        )

        _page_signature_strip(
            pil_images=bboxes, labels=image_labels_full,
            title="Ink Bounding Box (stroke extents)",
            pdf=pdf, case_id=case_id, verdict=verdict,
            page_counter=page_counter, total_pages=total_pages,
        )

        _page_signature_stack(
            pil_images=stroke_diffs, labels=image_labels_full,
            title="Forensic Stroke Map (numbered discrepancies)",
            pdf=pdf, case_id=case_id, verdict=verdict,
            page_counter=page_counter, total_pages=total_pages,
        )

        _page_stroke_crop_table(
            rows=stroke_crop_rows,
            title=f"Forensic Stroke-Difference Table  ({len(stroke_crop_rows)} markers, left to right)",
            pdf=pdf, page_counter=page_counter, total_pages=total_pages, case_id=case_id, verdict=verdict,
            rows_per_page=rows_per_page_stroke_table,
        )

        # ── SECTION 3 — Forensic Findings (F1-F7) ────────────────────────
        _page_section_divider(3, "Forensic Findings (F1-F7)",
                               "Detailed observations and measured data tables",
                               pdf, page_counter, total_pages, case_id, verdict)
        _page_text_block(
            lines=report_lines,
            title=f"AVERA - Forensic Findings Report  |  Case {case_id}",
            pdf=pdf, page_counter=page_counter, total_pages=total_pages, case_id=case_id,
            lines_per_page=lines_per_page,
        )

        # ── SECTION 4 — Understanding & Disclaimer ───────────────────────
        _page_section_divider(4, "Understanding This Report",
                               "Plain-language explanation and disclaimers",
                               pdf, page_counter, total_pages, case_id, verdict)
        _page_explanation(pdf, page_counter, total_pages, case_id)
        _page_disclaimer(pdf, page_counter, total_pages, case_id)

    logger.info("Compiled PDF saved", path=compiled_pdf_path, pages_written=page_counter[0])

    return compiled_pdf_path


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

    exported_files: List[str] = []

    for i, path in enumerate(all_paths):
        base_name = os.path.splitext(os.path.basename(path))[0]
        prefix = f"genuine_{i + 1}" if i < len(ref_images_paths) else "suspected"

        orig_pil = orig_pils[i] if i < len(orig_pils) else Image.new("RGB", (224, 224), "white")
        blend = blends[i] if i < len(blends) else np.zeros((224, 224, 3), dtype=np.uint8)
        bbox = bboxes[i] if i < len(bboxes) else np.zeros((224, 224, 3), dtype=np.uint8)
        stroke_diff = stroke_diffs[i] if i < len(stroke_diffs) else np.zeros((224, 224, 3), dtype=np.uint8)
        blend_pil = Image.fromarray(np.asarray(blend, dtype=np.uint8))

        orig_filename = os.path.join(export_base_dir, f"{prefix}_{base_name}_original.png")
        orig_pil.save(orig_filename)
        exported_files.append(orig_filename)

        # Keep the signature visible in every Grad-CAM export. The raw CAM is
        # retained for numeric/report processing, while the image artifact uses
        # the same heatmap-over-signature blend as the compiled PDF.
        cam_pil = blend_pil
        cam_filename = os.path.join(export_base_dir, f"{prefix}_{base_name}_heatmap.png")
        cam_pil.save(cam_filename)
        exported_files.append(cam_filename)

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


class GradCAMService:
    """Produces per-image Grad-CAM visualization assets and the compiled PDF report."""

    def __init__(self) -> None:
        self._settings = get_settings()

    async def generate(
        self,
        questioned_tensor: torch.Tensor,
        original_pil_image: Image.Image,
        case_name: str,
        reference_image_ids: Optional[List[str]] = None,
        questioned_image_id: Optional[str] = None,
        reference_tensors: Optional[List[torch.Tensor]] = None,
        reference_pil_images: Optional[List[Image.Image]] = None,
        verdict: Optional[str] = None,
        avg_distance: Optional[float] = None,
        threshold: Optional[float] = None,
        conf_genuine: Optional[float] = None,
        conf_forged: Optional[float] = None,
        model_version_tag: str = "AVERA SNN",
        forensic_findings: Optional[dict] = None,
        blob_svc=None,
        upload_to_blob: bool = False,
    ) -> List[str]:
        """Generate and optionally upload visualization images and the compiled PDF report."""
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
            threshold or 0.0,
            conf_genuine,
            conf_forged,
            model_version_tag,
            forensic_findings,
        )

        if upload_to_blob and blob_svc is not None:
            return await self._upload_visuals(
                local_paths=exported_files,
                case_name=case_name,
                blob_svc=blob_svc,
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
        threshold: float,
        conf_genuine: Optional[float],
        conf_forged: Optional[float],
        model_version_tag: str,
        forensic_findings: Optional[dict],
    ) -> List[str]:
        """Compute the Grad-CAM map, create visual images, and return local file paths."""
        model = get_model()
        first_param = next(model.parameters())
        actual_device = first_param.device
        model_dtype = first_param.dtype

        # Confidence falls back to the same distance-to-threshold sigmoid used
        # by InferenceService._classify() when the caller doesn't pass it in.
        if conf_genuine is None or conf_forged is None:
            scale = 6.0 / threshold if threshold > 0 else 6.0
            conf_genuine_raw = 1.0 / (1.0 + math.exp(scale * (avg_distance - threshold)))
            conf_genuine = round(conf_genuine_raw * 100.0, 4)
            conf_forged = round(100.0 - conf_genuine, 4)

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

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        export_dir = os.path.join(
            project_root,
            "results",
            "gradcam-exports",
            safe_case,
        )
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
        all_cams = [item["cam"] for item in reference_visuals] + [questioned_cam]
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

        # Overlay comparison (query vs. each reference) for the compiled PDF.
        reference_pils_only = [item["orig"] for item in reference_visuals]
        overlay_comparisons = [
            _generate_overlay_comparison(ref_pil, original_pil_image)
            for ref_pil in reference_pils_only
        ]

        # Per-marker stroke-difference crop table for the compiled PDF.
        skel_query = _get_skeleton(questioned_gray)
        skels_ref = [_get_skeleton(gray) for gray in reference_gray_images]
        stroke_crop_rows = build_stroke_crop_rows(
            reference_gray_images=reference_gray_images,
            reference_pils=reference_pils_only,
            query_pil=original_pil_image,
            skel_query=skel_query,
            skels_ref=skels_ref,
            global_top_markers=global_top_markers,
        )

        case_id = safe_case or case_name

        pdf_path = export_compiled_pdf(
            case_id=case_id,
            verdict=verdict,
            conf_genuine=conf_genuine,
            conf_forged=conf_forged,
            avg_distance=avg_distance,
            threshold=threshold,
            orig_pils=all_orig_pils,
            blends=all_blends,
            bboxes=all_bboxes,
            stroke_diffs=all_stroke_diffs,
            overlay_comparisons=overlay_comparisons,
            stroke_crop_rows=stroke_crop_rows,
            ref_image_names=ref_images_paths,
            query_image_name=query_image_path,
            results_dir=export_dir,
            model_version_tag=model_version_tag,
            forensic_findings=forensic_findings,
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
        """Compute the Grad-CAM activation map (full resolution 224x224)."""
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

        original_training = model.training
        try:
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

            cam_np = cam.squeeze().cpu().numpy()
            cam_min, cam_max = cam_np.min(), cam_np.max()
            if cam_max - cam_min > 1e-8:
                cam_np = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                cam_np = np.zeros_like(cam_np)

            target_size = self._settings.MODEL_INPUT_SIZE
            cam_fullres = np.array(
                Image.fromarray((cam_np * 255).astype(np.uint8)).resize(
                    (target_size, target_size), Image.Resampling.BILINEAR
                )
            ) / 255.0

            return cam_fullres.astype(np.float32)

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

    async def _upload_visuals(
        self,
        local_paths: List[str],
        case_name: str,
        blob_svc
    ) -> List[str]:
        """
        Try to upload generated PNG/PDF files to Azure Blob Storage.
        If upload fails, return local file paths instead so files are still accessible.
        """
        result_paths: List[str] = []

        logger.info(
            "Starting Grad-CAM asset upload",
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
                    "Uploading Grad-CAM asset",
                    file_path=local_path,
                    blob_id=blob_id,
                    size_bytes=len(data),
                )
                content_type = "application/pdf" if local_path.lower().endswith(".pdf") else "image/png"
                await blob_svc.upload_blob(blob_id=blob_id, data=data, content_type=content_type)
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
            "Completed Grad-CAM asset processing",
            case_name=case_name,
            result_count=len(result_paths),
        )
        return result_paths

    def _parse_export_filename(self, filename: str) -> tuple:
        """
        Parse a filename like 'genuine_1_G1_original.png' into (prefix, rest).

        Returns:
            (prefix, rest_filename)
            e.g., ('genuine_1', 'G1_original.png')
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
            # Fallback: no prefix parsing (e.g. the compiled PDF)
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


# ── Module-level singleton ────────────────────────────────────────────────────
gradcam_service = GradCAMService()


def get_gradcam_service() -> GradCAMService:
    """FastAPI dependency: returns the application-scoped GradCAMService."""
    return gradcam_service