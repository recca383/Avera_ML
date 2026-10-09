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
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.model_loader import get_model
from app.services import report_pages as rp
from app.services.finding_visuals import build_finding_visuals
from app.services.forensic_findings_service import (
    compute_forensic_findings,
    embed_tensors,
)

logger = get_logger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_LAYER_NAME = "backbone.conv_layers.26"
MIN_MARKERS = 3
MAX_MARKERS = 9
# How far (px, at model input size) a shared marker may move to land on a
# specimen's own ink. Beyond this, the specimen is treated as having no ink
# at that marker.
MARKER_SNAP_RADIUS = 8
LEGAL_FONT = rp.FONT
plt.rcParams["font.family"] = rp.FONT
plt.rcParams["pdf.fonttype"] = 42  # embed TrueType so report text stays selectable


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


def _skeleton_mismatch(
    skel_ref: np.ndarray,
    skel_query: np.ndarray,
    tolerance: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    (ref_only, query_only) skeleton pixels, where a pixel only counts as
    unmatched if the other skeleton has no stroke within `tolerance` px.
    Exact 1-px skeleton comparison flags strokes both specimens share as
    soon as they are a pixel apart, which buries real differences.
    """
    size = 2 * tolerance + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    ref_near = cv2.dilate(skel_ref, kernel)
    query_near = cv2.dilate(skel_query, kernel)
    ref_only = (skel_ref == 1) & (query_near == 0)
    query_only = (skel_query == 1) & (ref_near == 0)
    return ref_only, query_only


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
        ref_only, query_only = _skeleton_mismatch(skel_ref, skel_query)
        combined += ref_only.astype(np.float32) + query_only.astype(np.float32)

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
        ref_only, query_only = _skeleton_mismatch(skel_ref, skel_query)
        omission_count += int(ref_only[y1:y2, x1:x2].sum())
        addition_count += int(query_only[y1:y2, x1:x2].sum())

    if omission_count > addition_count * 1.3:
        return "Omission - reference stroke missing in query"
    elif addition_count > omission_count * 1.3:
        return "Addition - extra stroke present in query"
    else:
        return "Structural deviation - shape/path mismatch"


def _snap_to_ink(
    skeleton: np.ndarray,
    ink_mask: np.ndarray,
    cx: int,
    cy: int,
    radius: int = MARKER_SNAP_RADIUS,
) -> Optional[tuple[int, int]]:
    """
    Moves a shared marker onto the nearest stroke of one specimen. Global
    markers sit on a pixel where *some* image differs, so in any other image
    that exact pixel may be blank paper. Prefers the skeleton (stroke
    centerline) and falls back to the raw ink mask, since thinning can drop
    short or faint strokes. Returns None when the specimen has no ink within
    `radius` of the marker.
    """
    height, width = ink_mask.shape
    y1, y2 = max(0, cy - radius), min(height, cy + radius + 1)
    x1, x2 = max(0, cx - radius), min(width, cx + radius + 1)

    for mask in (skeleton, ink_mask):
        ys, xs = np.nonzero(mask[y1:y2, x1:x2])
        if len(ys) == 0:
            continue
        dist_sq = (xs + x1 - cx) ** 2 + (ys + y1 - cy) ** 2
        best = int(np.argmin(dist_sq))
        if dist_sq[best] <= radius * radius:
            return int(xs[best] + x1), int(ys[best] + y1)
    return None


def _ink_density(gray_np: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Soft ink map (0 = paper, 1 = ink), blurred so alignment has a smooth basin."""
    ink = 1.0 - np.asarray(gray_np, dtype=np.float32) / 255.0
    return cv2.GaussianBlur(ink, (0, 0), sigma)


def _warp_reference(ref_gray: np.ndarray, warp: np.ndarray) -> np.ndarray:
    """Resamples a reference into questioned-image coordinates (paper-white border)."""
    height, width = ref_gray.shape
    return cv2.warpAffine(
        ref_gray,
        warp,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255.0,),
    )


def _estimate_alignment(ref_gray: np.ndarray, query_gray: np.ndarray) -> np.ndarray:
    """
    Returns a 2x3 affine warp mapping questioned-image coordinates to
    reference coordinates, so the same letter lands at the same position.
    Candidates are identity, a moment-based fit (ink centroid + spread),
    and an ECC refinement of that fit; the one with the best ink overlap
    wins, so a failed or implausible refinement can never make things worse.
    """
    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    ref_mask = _get_ink_mask(ref_gray)
    query_mask = _get_ink_mask(query_gray)
    ref_pts = np.argwhere(ref_mask > 0)[:, ::-1].astype(np.float32)
    query_pts = np.argwhere(query_mask > 0)[:, ::-1].astype(np.float32)
    if len(ref_pts) < 10 or len(query_pts) < 10:
        return identity

    ref_c, query_c = ref_pts.mean(axis=0), query_pts.mean(axis=0)
    ref_spread = float(np.sqrt(((ref_pts - ref_c) ** 2).sum(axis=1).mean()))
    query_spread = float(np.sqrt(((query_pts - query_c) ** 2).sum(axis=1).mean()))
    scale = ref_spread / max(query_spread, 1e-6)
    moment_fit = np.array(
        [
            [scale, 0, ref_c[0] - scale * query_c[0]],
            [0, scale, ref_c[1] - scale * query_c[1]],
        ],
        dtype=np.float32,
    )

    query_ink = _ink_density(query_gray)
    ref_ink = _ink_density(ref_gray)
    candidates = [identity, moment_fit]
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
        full_mask = np.ones(query_ink.shape, dtype=np.uint8)  # use every pixel
        _, ecc_raw = cv2.findTransformECC(
            query_ink, ref_ink, moment_fit.copy(), cv2.MOTION_AFFINE, criteria, full_mask, 5
        )
        ecc_fit = np.asarray(ecc_raw, dtype=np.float32)
        singular_values = np.linalg.svd(ecc_fit[:, :2], compute_uv=False)
        if singular_values.min() > 0.6 and singular_values.max() < 1.6:
            candidates.append(ecc_fit)
    except cv2.error:
        pass

    def overlap(warp: np.ndarray) -> float:
        aligned = _ink_density(_warp_reference(ref_gray, warp))
        denom = np.sqrt(float((aligned * aligned).sum()) * float((query_ink * query_ink).sum()))
        return float((aligned * query_ink).sum()) / (denom + 1e-8)

    return max(candidates, key=overlap)


def _column_correspondence(
    query_gray: np.ndarray,
    aligned_ref_gray: np.ndarray,
    band_frac: float = 0.2,
    bins: int = 16,
) -> np.ndarray:
    """
    Dynamic time warping over columns, exploiting that signatures are
    written left to right: x_map[qx] is the x in the aligned reference that
    corresponds to column qx of the questioned signature, so a letter that is
    wider or narrower in one specimen still maps onto itself. Each column is
    described by its vertical ink distribution, and the warp is confined to
    a band so it can only refine the affine alignment, not replace it.
    """

    def column_features(gray: np.ndarray) -> np.ndarray:
        mask = cv2.GaussianBlur(_get_ink_mask(gray).astype(np.float32), (0, 0), 2.0)
        return cv2.resize(mask, (mask.shape[1], bins), interpolation=cv2.INTER_AREA).T

    q_feat = column_features(query_gray)
    r_feat = column_features(aligned_ref_gray)
    n, m = len(q_feat), len(r_feat)
    cost = np.sqrt(((q_feat[:, None, :] - r_feat[None, :, :]) ** 2).sum(axis=-1))

    band = max(abs(n - m) + 1, int(band_frac * max(n, m)))
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])

    sums = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    i, j = n, m
    while i > 0 and j > 0:
        sums[i - 1] += j - 1
        counts[i - 1] += 1
        step = int(np.argmin((acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return sums / np.maximum(counts, 1)


MarkerLocation = tuple[tuple[int, int], Optional[tuple[int, int]]]


def _locate_markers(
    gray_np: np.ndarray,
    global_top_markers: List[tuple[float, int, int]],
    warp: Optional[np.ndarray] = None,
    x_map: Optional[np.ndarray] = None,
    skeleton: Optional[np.ndarray] = None,
) -> List[MarkerLocation]:
    """
    Per-specimen (anchor, snapped) positions, index-for-index with
    global_top_markers. Markers live in questioned-image coordinates; for a
    reference, x_map moves each one to the corresponding column of the
    aligned reference and `warp` carries it back into the reference's own
    (unwarped) pixels, so the exhibit shows original images, not resampled
    ones. `snapped` is the nearest ink to the anchor, or None if there is none.
    """
    if skeleton is None:
        skeleton = _get_skeleton(gray_np)
    ink_mask = _get_ink_mask(gray_np)
    height, width = ink_mask.shape

    located: List[MarkerLocation] = []
    for _score, cx, cy in global_top_markers:
        x = float(x_map[min(cx, len(x_map) - 1)]) if x_map is not None else float(cx)
        y = float(cy)
        if warp is not None:
            x, y = (warp @ np.array([x, y, 1.0], dtype=np.float32)).tolist()
        anchor = (int(np.clip(round(x), 0, width - 1)), int(np.clip(round(y), 0, height - 1)))
        located.append((anchor, _snap_to_ink(skeleton, ink_mask, *anchor)))
    return located


def _crop_and_zoom(
    pil_img: Image.Image,
    cx: int,
    cy: int,
    crop_radius: int,
    zoom_size: int,
    marker: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Crops around the specimen's marker anchor (cx, cy) -- the corresponding
    part of the signature in that specimen -- then rings its snapped ink
    point. With no snapped point, a thin gray ring marks the empty location.
    """
    gray_np = np.array(pil_img.convert("L") if pil_img.mode != "L" else pil_img)
    height, width = gray_np.shape

    x1 = max(0, cx - crop_radius)
    y1 = max(0, cy - crop_radius)
    x2 = min(width, cx + crop_radius)
    y2 = min(height, cy + crop_radius)
    crop = gray_np[y1:y2, x1:x2]

    point = marker if marker is not None else (cx, cy)
    if crop.size == 0:
        crop = np.full((crop_radius * 2, crop_radius * 2), 255, dtype=np.uint8)
        x1, y1, x2, y2 = cx - crop_radius, cy - crop_radius, cx + crop_radius, cy + crop_radius
    local_cx = point[0] - x1
    local_cy = point[1] - y1

    zoomed = cv2.resize(crop, (zoom_size, zoom_size), interpolation=cv2.INTER_CUBIC)
    scale_x = zoom_size / max(1, (x2 - x1))
    scale_y = zoom_size / max(1, (y2 - y1))
    zoomed_bgr = cv2.cvtColor(zoomed, cv2.COLOR_GRAY2BGR)

    marker_x = int(local_cx * scale_x)
    marker_y = int(local_cy * scale_y)
    if marker is not None:
        cv2.circle(zoomed_bgr, (marker_x, marker_y), 6, (30, 30, 180), 2, cv2.LINE_AA)
    else:
        cv2.circle(zoomed_bgr, (marker_x, marker_y), 8, (150, 150, 150), 1, cv2.LINE_AA)

    return cv2.cvtColor(zoomed_bgr, cv2.COLOR_BGR2RGB)


def build_stroke_crop_rows(
    reference_pils: List[Image.Image],
    query_pil: Image.Image,
    skel_query: np.ndarray,
    skels_ref_aligned: List[np.ndarray],
    global_top_markers: List[tuple[float, int, int]],
    ref_locations: List[List[MarkerLocation]],
    query_locations: List[MarkerLocation],
    crop_radius: int = 30,
    zoom_size: int = 140,
) -> List[dict]:
    """
    Builds one row per discrepancy marker for the stroke-difference crop
    table: a zoomed crop of the corresponding part of the signature from all
    four references plus the query, a short caption classifying the
    discrepancy, and which reference(s) actually show ink there.
    `skels_ref_aligned` are reference skeletons in questioned-image
    coordinates, matching how the markers were found.
    """
    rows: List[dict] = []
    for idx, (_score, cx, cy) in enumerate(global_top_markers, 1):
        m = idx - 1
        ref_crops = [
            _crop_and_zoom(p, *locs[m][0], crop_radius, zoom_size, locs[m][1])
            for p, locs in zip(reference_pils, ref_locations)
        ]
        query_crop = _crop_and_zoom(
            query_pil, *query_locations[m][0], crop_radius, zoom_size, query_locations[m][1]
        )

        discrepancy_type = _classify_local_discrepancy(skels_ref_aligned, skel_query, cx, cy)
        refs_with_ink = [i + 1 for i, locs in enumerate(ref_locations) if locs[m][1] is not None]
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
                "query_has_ink": query_locations[m][1] is not None,
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
    marker_locations: List[MarkerLocation],
    image_size: int,
) -> np.ndarray:
    """
    Numbered marker map for one specimen. `marker_locations` come from
    `_locate_markers()` for this specimen, so arrows point at the
    corresponding part of *its* signature; the bubble numbering is shared.
    """
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
    raw_bubble_x = [anchor[0] + margin_px for anchor, _snapped in marker_locations]
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

    no_ink_gray = (150, 150, 150)

    for idx, (anchor, point) in enumerate(marker_locations):
        target = point if point is not None else anchor
        ex, ey = target[0] + margin_px, target[1] + margin_px

        bx = int(bubble_x[idx])
        by = bubble_y

        cv2.arrowedLine(
            expanded,
            (bx, by),
            (ex, ey),
            dark_red if point is not None else no_ink_gray,
            thickness=1,
            tipLength=0.05,
            line_type=cv2.LINE_AA,
        )
        if point is None:
            cv2.circle(expanded, (ex, ey), 7, no_ink_gray, 1, cv2.LINE_AA)
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

    legend_y = total_size - 52
    cv2.rectangle(expanded, (0, legend_y), (total_size, total_size), (235, 235, 235), -1)
    cv2.putText(
        expanded,
        f"Markers 1-{len(marker_locations)}: key discrepancy locations",
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
    cv2.putText(
        expanded,
        "Gray ring = no ink in this specimen there",
        (6, legend_y + 46),
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

    cam_norm = np.clip((cam_up - float(lo)) / (float(hi) - float(lo) + 1e-8), 0, 1)
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
    return rp.verdict_color(verdict)


def _stamp_footer(fig, page_counter: list, total_pages: Optional[int], case_id: str) -> None:
    """Running page number + case ID + confidentiality notice (text only)."""
    rp.footer(fig, page_counter, total_pages, case_id, rule=False)


_LAND_W, _LAND_H = 11.0, 8.5
_LM = 0.88  # landscape side margin (inches)


def _cell_axes(fig, x_in: float, y_in: float, w_in: float, h_in: float, W: float, H: float):
    """Axes placed by inches, measured from the top-left of the page."""
    return fig.add_axes([x_in / W, 1 - (y_in + h_in) / H, w_in / W, h_in / H])


def _frame(ax, color: str, lw: float = 1.0, dashed: bool = False) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)
        spine.set_linestyle("--" if dashed else "-")


def _panel(fig, x_in, y_in, w_in, h_in, W, H, heading: str, body: str) -> None:
    """Bordered note panel used in empty grid cells."""
    rp.rect(fig, x_in / W, 1 - (y_in + h_in) / H, w_in / W, h_in / H, rp.SURFACE, rp.BORDER, 1.0)
    rp.text(fig, (x_in + 0.18) / W, 1 - (y_in + 0.28) / H, heading, 10, rp.INK, "bold", va="center")
    rp.para(fig, (x_in + 0.18) / W, 1 - (y_in + 0.50) / H, body, w_in - 0.36, 8.8, rp.TEXT, spacing=1.45)


def _page_grid(pdf, page_counter, total_pages, case_id, section, title, subtitle,
               images, labels, note, legend=None, cmap=None) -> None:
    """
    Landscape 3 x 2 grid: references 1-3, then reference 4 + QUESTIONED + a note panel.
    `images` holds 5 arrays/PIL images (4 refs then the questioned one), or 4
    overlays (the fifth cell then holds `legend`).
    """
    W, H = _LAND_W, _LAND_H
    fig = plt.figure(figsize=(W, H), facecolor=rp.WHITE)
    rp.chrome(fig, case_id, section)
    y0 = rp.page_title(fig, title, subtitle)
    top = (1 - y0) * H + 0.10
    avail_h = H - top - 0.65
    cw = (W - 2 * _LM) / 3
    rh = avail_h / 2
    label_h = 0.30
    side = min(cw - 0.30, rh - label_h - 0.12)
    n_img = len(images)

    for i in range(6):
        r, c = divmod(i, 3)
        cx = _LM + c * cw
        cy = top + r * rh
        if i < n_img:
            is_q = labels[i].upper().startswith("QUESTIONED") or (n_img == 5 and i == 4)
            col = rp.BRAND if is_q else rp.MUTED
            rp.text(fig, (cx + (cw - side) / 2) / W, 1 - (cy + 0.16) / H, labels[i], 9.5,
                    rp.BRAND if is_q else rp.INK, "bold", va="center")
            ax = _cell_axes(fig, cx + (cw - side) / 2, cy + label_h, side, side, W, H)
            im = images[i]
            ax.imshow(im, cmap=cmap) if cmap and not isinstance(im, Image.Image) else ax.imshow(im)
            _frame(ax, col, 2.6 if is_q else 1.0)
        elif legend is not None and i == n_img:
            legend(fig, cx, cy + label_h, cw - 0.25, rh - label_h - 0.1, W, H)
        elif i == 5 or (i == n_img and legend is None):
            _panel(fig, cx, cy + label_h, cw - 0.25, rh - label_h - 0.1, W, H, "How to read this page", note)

    if legend is not None and n_img == 4:
        # 4 overlays fill cells 0-3; legend sits in cell 4, note in cell 5.
        r, c = divmod(5, 3)
        _panel(fig, _LM + c * cw, top + r * rh + label_h, cw - 0.25, rh - label_h - 0.1, W, H,
               "How to read this page", note)

    rp.footer(fig, page_counter, total_pages, case_id, rule=True)
    pdf.savefig(fig)
    plt.close(fig)


def _overlay_legend(fig, x_in, y_in, w_in, h_in, W, H) -> None:
    rp.rect(fig, x_in / W, 1 - (y_in + h_in) / H, w_in / W, h_in / H, rp.WHITE, rp.BORDER, 1.0)
    rp.text(fig, (x_in + 0.18) / W, 1 - (y_in + 0.28) / H, "Legend", 10, rp.INK, "bold", va="center")
    items = [("#3B82F6", "Reference only (missing in the questioned signature)"),
             ("#DC2626", "Questioned only (extra in the questioned signature)"),
             ("#581C87", "Match (present in both)")]
    for k, (col, lab) in enumerate(items):
        yy = y_in + 0.62 + k * 0.42
        rp.rect(fig, (x_in + 0.18) / W, 1 - (yy + 0.14) / H, 0.28 / W, 0.14 / H, col, col, 0.8)
        for j, ln in enumerate(rp.wrap(lab, w_in - 1.0, 8.5)[:2]):
            rp.text(fig, (x_in + 0.62) / W, 1 - (yy + 0.06 + j * 0.16) / H, ln, 8.5, rp.TEXT, va="center")


def _page_stroke_crop_table(rows, title, pdf, page_counter, total_pages, case_id, verdict,
                             rows_per_page=3) -> None:
    """One row per marker: Marker | Ref 1-4 | Questioned | description. Landscape pages."""
    n_total = len(rows)
    if n_total == 0:
        return
    n_pages = -(-n_total // rows_per_page)
    W, H = _LAND_W, _LAND_H

    for page_idx in range(n_pages):
        chunk = rows[page_idx * rows_per_page:(page_idx + 1) * rows_per_page]
        fig = plt.figure(figsize=(W, H), facecolor=rp.WHITE)
        rp.chrome(fig, case_id, "Visual Evidence")
        page_t = title if n_pages == 1 else f"{title} (page {page_idx + 1} of {n_pages})"
        y0 = rp.page_title(fig, page_t,
                           "Solid gray border = the specimen has ink at this point. "
                           "Dashed border = no ink there. Blue border = questioned signature.")
        top = (1 - y0) * H + 0.05
        img, gap = 1.2, 0.10
        x_img0 = _LM + 0.80
        x_cap = x_img0 + 5 * img + 4 * gap + 0.22
        row_h = (H - top - 0.70 - 0.30) / rows_per_page

        for c in range(4):
            rp.text(fig, (x_img0 + c * (img + gap) + img / 2) / W, 1 - (top + 0.12) / H,
                    f"Reference {c + 1}", 8.5, rp.INK, "bold", ha="center", va="center")
        rp.text(fig, (x_img0 + 4 * (img + gap) + img / 2) / W, 1 - (top + 0.12) / H, "Questioned",
                8.5, rp.BRAND, "bold", ha="center", va="center")
        rp.text(fig, x_cap / W, 1 - (top + 0.12) / H, "Discrepancy", 8.5, rp.INK, "bold", va="center")

        for r, row in enumerate(chunk):
            ry = top + 0.30 + r * row_h
            rp.text(fig, _LM / W, 1 - (ry + img / 2) / H, f"Marker {row['marker_num']}", 10, rp.INK,
                    "bold", va="center")
            for c in range(4):
                ax = _cell_axes(fig, x_img0 + c * (img + gap), ry, img, img, W, H)
                ax.imshow(row["ref_crops"][c])
                has_ink = (c + 1) in row.get("refs_with_ink", [])
                _frame(ax, rp.MUTED if has_ink else rp.BORDER, 1.6 if has_ink else 1.2, dashed=not has_ink)
            axq = _cell_axes(fig, x_img0 + 4 * (img + gap), ry, img, img, W, H)
            axq.imshow(row["query_crop"])
            _frame(axq, rp.BRAND, 2.6, dashed=not row.get("query_has_ink", True))

            cap_lines = row["caption"].split("\n")
            header = cap_lines[0].split("-", 1)
            head_txt = header[0].strip()
            detail = header[1].strip() if len(header) > 1 else ""
            refs_note = cap_lines[1] if len(cap_lines) > 1 else ""
            cw_in = W - _LM - x_cap
            yy = 1 - (ry + 0.05) / H
            yy = rp.para(fig, x_cap / W, yy, head_txt, cw_in, 9.5, rp.INK, "bold", 1.3) - 0.05 / H
            yy = rp.para(fig, x_cap / W, yy, detail, cw_in, 8.5, rp.TEXT, "normal", 1.35) - 0.05 / H
            rp.para(fig, x_cap / W, yy, refs_note, cw_in, 8, rp.MUTED, "normal", 1.3)
            if r < len(chunk) - 1:
                rp.hline(fig, _LM / W, 1 - _LM / W, 1 - (ry + row_h - 0.10) / H, rp.BORDER, 0.8)

        rp.footer(fig, page_counter, total_pages, case_id, rule=True)
        pdf.savefig(fig)
        plt.close(fig)


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
    finding_visuals: Optional[dict] = None,
) -> str:
    """
    Assemble the compiled report PDF (Letter, Sora, AVERA brand colour):
    cover, case summary, forensic findings (F1-F7 cards), visual evidence
    (signatures, Grad-CAM, overlay, bounding box, stroke map, stroke table),
    explanation + glossary, disclaimer.

    `blends` must be the heatmap-over-signature arrays (see `_build_heatmap_overlay`).
    `forensic_findings` is the dict from forensic_findings_service.compute_forensic_findings.
    `finding_visuals` is the dict from finding_visuals.build_finding_visuals (optional;
    cards fall back to numbers only without it).
    """
    os.makedirs(results_dir, exist_ok=True)
    compiled_pdf_path = os.path.join(results_dir, f"output.pdf")
    image_labels = ["Reference 1", "Reference 2", "Reference 3", "Reference 4", "QUESTIONED"]

    rows_per_page_stroke_table = 3
    stroke_table_pages = -(-max(len(stroke_crop_rows), 1) // rows_per_page_stroke_table)
    findings_pages = rp.plan_findings_pages(forensic_findings, finding_visuals)
    # cover + summary + findings pages + 5 visual grids + explanation + disclaimer
    total_pages = 9 + len(findings_pages) + (stroke_table_pages if stroke_crop_rows else 0)
    page_counter = [0]

    logger.info("Building compiled PDF", path=compiled_pdf_path, total_pages=total_pages)

    with PdfPages(compiled_pdf_path) as pdf:
        rp.page_cover(pdf, page_counter, total_pages, case_id, verdict, conf_genuine, conf_forged,
                      avg_distance, threshold, query_image_name, ref_image_names, model_version_tag)
        rp.page_summary(pdf, page_counter, total_pages, case_id, verdict, avg_distance, threshold,
                        forensic_findings)
        for i, codes in enumerate(findings_pages):
            rp.page_findings(pdf, page_counter, total_pages, case_id, forensic_findings, codes,
                             visuals=finding_visuals, first=i == 0, last=i == len(findings_pages) - 1)

        section = "Visual Evidence"
        _page_grid(pdf, page_counter, total_pages, case_id, section, "Signature Overview",
                   "The four genuine references and the questioned signature side by side",
                   orig_pils, image_labels,
                   "Compare overall shape, size, slant and pen pressure by eye first. The pages that follow "
                   "show where AVERA found differences.", cmap="gray")
        _page_grid(pdf, page_counter, total_pages, case_id, section, "Grad-CAM Attention Heatmaps",
                   "Where the AI model paid attention when comparing the signatures",
                   blends, image_labels,
                   "Warm colors (red, yellow) mark areas that influenced the model most; cool colors (blue) "
                   "influenced it least. This explains the model's focus. It is not proof of forgery on its own.")
        _page_grid(pdf, page_counter, total_pages, case_id, section, "Overlay Comparison",
                   "The questioned signature laid over each reference signature",
                   overlay_comparisons, ["Versus Reference 1", "Versus Reference 2",
                                         "Versus Reference 3", "Versus Reference 4"],
                   "Purple shows where strokes coincide. Large blue or red areas are strokes found in only one of "
                   "the two signatures.", legend=_overlay_legend)
        _page_grid(pdf, page_counter, total_pages, case_id, section, "Ink Bounding Box",
                   "The outer boundary of the ink on each signature",
                   bboxes, image_labels,
                   "W = width and H = height in pixels; R = width divided by height. Compare the questioned "
                   "signature's proportions with the references (see F4 Proportion & Spacing).")
        _page_grid(pdf, page_counter, total_pages, case_id, section, "Forensic Stroke Map",
                   "Numbered places where the questioned and reference strokes differ most",
                   stroke_diffs, image_labels,
                   "Each numbered marker points to one location, and the same number marks the same place on "
                   "every signature. The next pages zoom in on each marker.")

        _page_stroke_crop_table(
            rows=stroke_crop_rows,
            title="Stroke-Difference Table",
            pdf=pdf, page_counter=page_counter, total_pages=total_pages, case_id=case_id, verdict=verdict,
            rows_per_page=rows_per_page_stroke_table,
        )

        rp.page_explanation(pdf, page_counter, total_pages, case_id)
        rp.page_disclaimer(pdf, page_counter, total_pages, case_id)

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

        # Align each reference to the questioned signature before diffing, so
        # markers reflect real stroke differences rather than shifts in size
        # or position, and the same marker lands on the same letter everywhere.
        questioned_gray = np.asarray(original_pil_image.convert("L"), dtype=np.uint8)
        ref_warps = [_estimate_alignment(gray, questioned_gray) for gray in reference_gray_images]
        aligned_ref_grays = [
            _warp_reference(gray, warp) for gray, warp in zip(reference_gray_images, ref_warps)
        ]
        global_top_markers = _compute_global_top_markers(aligned_ref_grays, questioned_gray)
        if not global_top_markers:
            global_top_markers = [
                (
                    1.0,
                    questioned_gray.shape[1] // 2,
                    questioned_gray.shape[0] // 2,
                )
            ]

        skel_query = _get_skeleton(questioned_gray)
        query_locations = _locate_markers(questioned_gray, global_top_markers, skeleton=skel_query)
        ref_locations = [
            _locate_markers(
                gray,
                global_top_markers,
                warp=warp,
                x_map=_column_correspondence(questioned_gray, aligned),
            )
            for gray, warp, aligned in zip(reference_gray_images, ref_warps, aligned_ref_grays)
        ]

        questioned_stroke_diff = _generate_stroke_difference_visualization(
            original_pil_image,
            query_locations,
            self._settings.MODEL_INPUT_SIZE,
        )

        all_orig_pils = [item["orig"] for item in reference_visuals] + [original_pil_image.convert("RGB")]
        all_cams = [item["cam"] for item in reference_visuals] + [questioned_cam]
        all_blends = [item["blend"] for item in reference_visuals] + [questioned_blend]
        all_bboxes = [item["bbox"] for item in reference_visuals] + [questioned_bbox]
        all_stroke_diffs = [
            _generate_stroke_difference_visualization(item["orig"], locations, self._settings.MODEL_INPUT_SIZE)
            for item, locations in zip(reference_visuals, ref_locations)
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
        stroke_crop_rows = build_stroke_crop_rows(
            reference_pils=reference_pils_only,
            query_pil=original_pil_image,
            skel_query=skel_query,
            skels_ref_aligned=[_get_skeleton(gray) for gray in aligned_ref_grays],
            global_top_markers=global_top_markers,
            ref_locations=ref_locations,
            query_locations=query_locations,
        )

        if forensic_findings is None:
            forensic_findings = self._compute_findings(
                reference_pils=reference_pils_only,
                questioned_pil=original_pil_image,
                reference_tensors=reference_tensors,
                questioned_tensor=questioned_tensor,
                distance=avg_distance,
                threshold=threshold,
                case_name=case_name,
            )
            
        case_id = safe_case or case_name

        finding_visuals = None
        try:
            finding_visuals = build_finding_visuals(reference_pils_only, original_pil_image)
        except Exception as exc:
            logger.exception("Finding illustrations failed; cards will show numbers only",
                             case_name=case_name, error=str(exc))

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
            finding_visuals=finding_visuals,
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

    def _compute_findings(
        self,
        reference_pils: List[Image.Image],
        questioned_pil: Image.Image,
        reference_tensors: List[torch.Tensor],
        questioned_tensor: torch.Tensor,
        distance: float,
        threshold: float,
        case_name: str,
    ) -> Optional[dict]:
        """
        Compute F1-F7 for the compiled PDF. Returns None on failure so the
        report still renders (with N/A) instead of failing the whole request.
        """
        if not reference_pils:
            logger.warning("No reference images; skipping forensic findings", case_name=case_name)
            return None

        model = get_model()
        was_training = model.training
        try:
            model.eval()  # embeddings must not touch BatchNorm running stats
            ref_emb = embed_tensors(reference_tensors) if reference_tensors else None
            q_emb = embed_tensors([questioned_tensor])[0]

            findings = compute_forensic_findings(
                reference_images=reference_pils,
                questioned_image=questioned_pil,
                reference_embeddings=ref_emb,
                questioned_embedding=q_emb,
                distance=distance,
                threshold=threshold,
            )
            logger.info(
                "Forensic findings computed",
                case_name=case_name,
                key_findings=findings.get("key_findings"),
            )
            return findings
        except Exception as exc:
            logger.exception(
                "Forensic findings failed; report will show N/A",
                case_name=case_name,
                error=str(exc),
            )
            return None
        finally:
            model.train(was_training)

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
                embedding = model.get_embedding(tensor)
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
            embedding_a = model.get_embedding(tensor_a)
            embedding_b = model.get_embedding(tensor_b)
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