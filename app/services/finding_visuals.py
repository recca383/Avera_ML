"""
Finding illustrations (F1-F4, F7) for the compiled PDF.

Each finding card shows a reference and the questioned signature with the
measurement drawn on the ink, so a reader can see *what* was measured rather
than read an abstract number:

    F1  each separate stroke in its own colour; loops filled, dots ringed
    F2  the fitted baseline through the signature, with its angle
    F3  the stroke centreline coloured from smooth to shaky
    F4  the ink bounding box with width, height and ratio
    F7  the stroke centreline coloured from thin to thick line width

The drawing reuses the measurement helpers from forensic_findings_service so
the pictures always match the numbers on the card. Colour scales for F3 and
F7 are shared between the two images, so equal colours mean equal values.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from app.services.forensic_findings_service import (
    THRESHOLDS,
    _ink_mask,
    _skeleton,
    _to_gray,
    measure_f1,
    measure_f2,
    measure_f3,
    measure_f4,
    measure_f7,
)

SCALE = 3          # upsampling factor so overlay lines stay crisp in the PDF
PAD = 10           # px (at model size) of paper kept around the ink when cropping

# Colours (RGB). Stroke palette: distinct but muted, readable on white.
_BRAND = (30, 111, 217)
_FADED_INK = 0.30  # how much of the original ink darkness the faded base keeps
_STROKE_PALETTE = [
    (30, 111, 217), (13, 148, 136), (124, 58, 237), (217, 119, 6),
    (219, 39, 119), (21, 128, 61), (71, 85, 105), (234, 88, 12),
]
_LOOP_FILL = (191, 219, 254)
_SMOOTH_SCALE = [(203, 213, 225), (245, 158, 11), (220, 38, 38)]   # light slate -> amber -> red
_WIDTH_SCALE = [(186, 230, 253), (59, 130, 246), (30, 58, 138)]    # sky -> blue -> navy


def _ink_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w, h
    return (max(0, int(xs.min()) - PAD), max(0, int(ys.min()) - PAD),
            min(w, int(xs.max()) + PAD + 1), min(h, int(ys.max()) + PAD + 1))


def _base(gray: np.ndarray, faded: bool = True) -> np.ndarray:
    """Upsampled RGB canvas of the signature; faded so overlays stand out."""
    g = gray.astype(np.float32)
    if faded:
        g = 255.0 - (255.0 - g) * _FADED_INK
    up = cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    up = np.clip(up, 0, 255).astype(np.uint8)
    return cv2.cvtColor(up, cv2.COLOR_GRAY2RGB)


def _ramp(stops: List[Tuple[int, int, int]], t: float) -> Tuple[int, int, int]:
    """Piecewise-linear colour scale through `stops`, t in [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0)) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    f = t - i
    c0, c1 = stops[i], stops[i + 1]
    return tuple(int(round(a + (b - a) * f)) for a, b in zip(c0, c1))  # type: ignore[return-value]


def _crop(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = (v * SCALE for v in bbox)
    return np.ascontiguousarray(img[y1:y2, x1:x2])


def _up(pt: Tuple[float, float]) -> Tuple[int, int]:
    return int(round(pt[0] * SCALE + SCALE / 2)), int(round(pt[1] * SCALE + SCALE / 2))


# ── F1: strokes, loops, dots ──────────────────────────────────────────────────
def _draw_f1(gray: np.ndarray) -> Tuple[np.ndarray, str]:
    mask = _ink_mask(gray)
    m8 = mask.astype(np.uint8)
    canvas = _base(gray, faded=True)

    # Loops first so the stroke colour sits on top of the fill edge.
    contours, hierarchy = cv2.findContours(m8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for idx, cnt in enumerate(contours):
            if hierarchy[0][idx][3] != -1 and cv2.contourArea(cnt) >= 12:
                cv2.drawContours(canvas, [cnt * SCALE], -1, _LOOP_FILL, -1, cv2.LINE_AA)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(m8, connectivity=8)
    big = cv2.resize(labels.astype(np.int32).astype(np.float32), None, fx=SCALE, fy=SCALE,
                     interpolation=cv2.INTER_NEAREST).astype(np.int32)
    colour_idx = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < THRESHOLDS["min_component_area"]:
            continue
        colour = _STROKE_PALETTE[colour_idx % len(_STROKE_PALETTE)]
        colour_idx += 1
        canvas[big == i] = colour
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if area <= THRESHOLDS["dot_max_area"] and max(w, h) <= 2.5 * max(1, min(w, h)):
            cv2.circle(canvas, _up(tuple(centroids[i])), 6 * SCALE // 2 + 4, colour, 2, cv2.LINE_AA)

    m = measure_f1(gray)
    caption = f"{m['strokes']} strokes  ·  {m['bowls']} loops  ·  {m['dots']} dots"
    return _crop(canvas, _ink_bbox(mask)), caption


# ── F2: baseline ──────────────────────────────────────────────────────────────
def _draw_f2(gray: np.ndarray) -> Tuple[np.ndarray, str]:
    mask = _ink_mask(gray)
    canvas = _base(gray, faded=True)
    angle = measure_f2(gray)
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) >= 3:
        ys = np.array([np.mean(np.where(mask[:, c])[0]) for c in cols], dtype=np.float64)
        slope, icpt = np.polyfit(cols.astype(np.float64), ys, 1)
        x0, x1 = float(cols.min()), float(cols.max())
        p0, p1 = _up((x0, slope * x0 + icpt)), _up((x1, slope * x1 + icpt))
        # Dashed level guide from the start of the baseline, for comparison.
        dash = 6 * SCALE
        for xs in range(p0[0], p1[0], dash * 2):
            cv2.line(canvas, (xs, p0[1]), (min(xs + dash, p1[0]), p0[1]), (156, 163, 175), 1, cv2.LINE_AA)
        cv2.line(canvas, p0, p1, _BRAND, 2 * SCALE // 2 + 1, cv2.LINE_AA)
    return _crop(canvas, _ink_bbox(mask)), f"Slope {angle:+.1f}°"


# ── F3: line smoothness ───────────────────────────────────────────────────────
def _turning_points(gray: np.ndarray) -> List[Tuple[int, int, float]]:
    """(x, y, |turn|) along the skeleton, computed the same way as measure_f3."""
    skel = _skeleton(_ink_mask(gray)).astype(np.uint8)
    contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out: List[Tuple[int, int, float]] = []
    k = 5
    kernel = np.ones(k) / k
    for cnt in contours:
        pts = cnt[:, 0, :].astype(np.float64)
        if len(pts) < 15:
            continue
        xs = np.convolve(pts[:, 0], kernel, mode="valid")
        ys = np.convolve(pts[:, 1], kernel, mode="valid")
        heading = np.unwrap(np.arctan2(np.diff(ys), np.diff(xs)))
        d = np.diff(heading)
        d = np.abs((d + np.pi) % (2 * np.pi) - np.pi)
        offset = (k - 1) // 2 + 1  # centre the value on the point it describes
        for i, v in enumerate(d):
            x, y = pts[min(i + offset, len(pts) - 1)]
            out.append((int(x), int(y), float(v)))
    return out


def _draw_points(gray: np.ndarray, pts: List[Tuple[int, int, float]], vmin: float, vmax: float,
                 stops: List[Tuple[int, int, int]]) -> np.ndarray:
    canvas = _base(gray, faded=True)
    span = max(vmax - vmin, 1e-9)
    # Low values first so the high (interesting) ones are drawn on top.
    for x, y, v in sorted(pts, key=lambda p: p[2]):
        cv2.circle(canvas, _up((x, y)), SCALE, _ramp(stops, (v - vmin) / span), -1, cv2.LINE_AA)
    return canvas


def _shared_scale(values: List[float], lo_pct: float, hi_pct: float) -> Tuple[float, float]:
    """Colour-scale limits shared by both images, so equal colours mean equal values."""
    if not values:
        return 0.0, 1.0
    return float(np.percentile(values, lo_pct)), float(np.percentile(values, hi_pct))


# ── F4: proportions ───────────────────────────────────────────────────────────
def _draw_f4(gray: np.ndarray) -> Tuple[np.ndarray, str]:
    mask = _ink_mask(gray)
    canvas = _base(gray, faded=False)
    ys, xs = np.where(mask)
    if len(xs):
        p0 = (int(xs.min()) * SCALE, int(ys.min()) * SCALE)
        p1 = (int(xs.max() + 1) * SCALE, int(ys.max() + 1) * SCALE)
        cv2.rectangle(canvas, p0, p1, _BRAND, 2, cv2.LINE_AA)
    m = measure_f4(gray)
    return _crop(canvas, _ink_bbox(mask)), f"{int(m['w'])} × {int(m['h'])} px  ·  {m['ratio']:.2f} : 1"


# ── F7: line width ────────────────────────────────────────────────────────────
def _width_points(gray: np.ndarray) -> List[Tuple[int, int, float]]:
    mask = _ink_mask(gray)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    ys, xs = np.where(_skeleton(mask))
    return [(int(x), int(y), float(2.0 * dist[y, x])) for y, x in zip(ys, xs)]


# ── Public API ────────────────────────────────────────────────────────────────
def build_finding_visuals(
    reference_images: Sequence[Image.Image | np.ndarray],
    questioned_image: Image.Image | np.ndarray,
    reference_index: Optional[int] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Returns {"f1": {"ref": RGB array, "q": RGB array, "ref_caption": str,
    "q_caption": str, "ref_label": str, "legend": str|None}, ...} for F1-F4
    and F7. The reference shown is the most typical one (closest to the
    others in ink-box proportion) unless `reference_index` is given.
    """
    refs = [_to_gray(r) for r in reference_images]
    q = _to_gray(questioned_image)
    if not refs:
        return {}

    if reference_index is None:
        ratios = np.array([measure_f4(g)["ratio"] for g in refs])
        reference_index = int(np.argmin(np.abs(ratios - np.median(ratios))))
    ref = refs[reference_index]
    ref_label = f"Reference {reference_index + 1}"

    visuals: Dict[str, Dict[str, object]] = {}

    def add(code: str, pair: Tuple[Tuple[np.ndarray, str], Tuple[np.ndarray, str]], legend: Optional[str]) -> None:
        (r_img, r_cap), (q_img, q_cap) = pair
        visuals[code] = {"ref": r_img, "q": q_img, "ref_caption": r_cap, "q_caption": q_cap,
                         "ref_label": ref_label, "legend": legend}

    add("f1", (_draw_f1(ref), _draw_f1(q)), "Each colour = one separate stroke · shaded = closed loop · ring = dot")
    add("f2", (_draw_f2(ref), _draw_f2(q)), "Blue line = fitted baseline · dashed = level")

    # Turning is near zero along most of a stroke, so the scale starts at the
    # median: ordinary curves stay light and only sharp direction changes show.
    r_pts, q_pts = _turning_points(ref), _turning_points(q)
    t_lo, t_hi = _shared_scale([p[2] for p in r_pts + q_pts], 50, 99)
    add("f3", (
        (_crop(_draw_points(ref, r_pts, t_lo, t_hi, _SMOOTH_SCALE), _ink_bbox(_ink_mask(ref))),
         f"Wobble {measure_f3(ref):.3f}"),
        (_crop(_draw_points(q, q_pts, t_lo, t_hi, _SMOOTH_SCALE), _ink_bbox(_ink_mask(q))),
         f"Wobble {measure_f3(q):.3f}"),
    ), "Light = smooth line · amber to red = sharp changes of direction (wobble)")

    add("f4", (_draw_f4(ref), _draw_f4(q)), "Blue box = outer edge of the ink · ratio = width ÷ height")

    r_w, q_w = _width_points(ref), _width_points(q)
    w_lo, w_hi = _shared_scale([p[2] for p in r_w + q_w], 2, 98)
    m_r, m_q = measure_f7(ref), measure_f7(q)
    add("f7", (
        (_crop(_draw_points(ref, r_w, w_lo, w_hi, _WIDTH_SCALE), _ink_bbox(_ink_mask(ref))),
         f"Ink {m_r['darkness'] * 100:.0f}%  ·  width var {m_r['width_var']:.2f}"),
        (_crop(_draw_points(q, q_w, w_lo, w_hi, _WIDTH_SCALE), _ink_bbox(_ink_mask(q))),
         f"Ink {m_q['darkness'] * 100:.0f}%  ·  width var {m_q['width_var']:.2f}"),
    ), "Ink = average darkness · light blue = thin line · dark blue = thick line")

    return visuals
