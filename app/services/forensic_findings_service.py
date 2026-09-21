"""
Forensic findings service (F1-F7).

Computes the seven examiner-style findings described in Chapter 3 and returns
them in the exact shape `gradcam_service.export_compiled_pdf(forensic_findings=...)`
reads:

    {
      "key_findings":       {"f1_label": ..., ..., "f7_label": ...},
      "modal_observations": {"f1_observation": ..., ..., "f2_angle": ...,
                             "f3_variance": ..., "f4_width_px": ..., ...},
    }

F1-F4 and F7 use classical image processing (OpenCV / scikit-image, incl. the
Zhang-Suen skeleton). F5 and F6 use the trained model's embedding space.

All comparison thresholds below are ENGINEERING DEFAULTS (see THRESHOLDS) and
have not been validated against a forensic examiner. The PDF disclaimer says so.

Inputs are the preprocessed PIL images (ink dark on white, as returned by
PreprocessingService.bytes_to_pil), NOT the inverted model tensors.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

# ── Engineering defaults (tune / validate against examiner judgment) ──────────
THRESHOLDS = {
    "min_component_area": 12,     # px; ignore specks smaller than this
    "dot_max_area": 45,           # px; a component this small (and roundish) = terminal dot
    "count_tolerance": 1,         # allowed +/- difference in F1 counts vs. reference range
    "angle_tolerance_deg": 5.0,   # F2 minimum tolerance
    "angle_sigma": 2.0,           # F2 tolerance = max(min tol, sigma * ref std)
    "curvature_ratio": 1.5,       # F3 query/ref curvature variance ratio
    "ratio_tolerance": 0.25,      # F4 relative aspect-ratio deviation
    "natural_range_margin": 1.10, # F6 margin on the reference max pairwise distance
    "darkness_tolerance": 0.25,   # F7 relative mean-darkness deviation
    "width_var_ratio": 2.0,       # F7 query/ref stroke-width-variance ratio
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _to_gray(img: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("L"), dtype=np.uint8)
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return arr.astype(np.uint8)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Boolean ink mask (ink = True) via Otsu, same convention as gradcam_service."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary > 0


def _skeleton(mask: np.ndarray) -> np.ndarray:
    """1-px centerline via Zhang-Suen thinning (as stated in Chapter 3)."""
    return skeletonize(mask, method="zhang")


def _fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


# ── Per-image measurements ────────────────────────────────────────────────────
def measure_f1(gray: np.ndarray) -> Dict[str, int]:
    """Stroke count, bowls (closed loops), terminal dots, pen lifts."""
    mask = _ink_mask(gray)
    m8 = mask.astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)

    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    comps = [(i, a) for i, a in zip(range(1, n), areas) if a >= THRESHOLDS["min_component_area"]]

    dots = 0
    for i, a in comps:
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if a <= THRESHOLDS["dot_max_area"] and max(w, h) <= 2.5 * max(1, min(w, h)):
            dots += 1

    stroke_count = len(comps)
    pen_lifts = max(stroke_count - 1, 0)

    # Bowls = enclosed holes (child contours) with a minimum area.
    contours, hierarchy = cv2.findContours(m8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    bowls = 0
    if hierarchy is not None:
        for idx, cnt in enumerate(contours):
            has_parent = hierarchy[0][idx][3] != -1
            if has_parent and cv2.contourArea(cnt) >= 12:
                bowls += 1

    return {"strokes": stroke_count, "bowls": bowls, "dots": dots, "pen_lifts": pen_lifts}


def measure_f2(gray: np.ndarray) -> float:
    """Baseline angle in degrees (positive = rising to the right)."""
    mask = _ink_mask(gray)
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) < 3:
        return 0.0
    ys = np.array([np.mean(np.where(mask[:, c])[0]) for c in cols], dtype=np.float64)
    slope, _ = np.polyfit(cols.astype(np.float64), ys, 1)
    return float(-math.degrees(math.atan(slope)))  # image y grows downward


def measure_f3(gray: np.ndarray) -> float:
    """Curvature variance of the skeleton (higher = less smooth / more tremor)."""
    skel = _skeleton(_ink_mask(gray)).astype(np.uint8)
    contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    deltas: List[np.ndarray] = []
    for cnt in contours:
        pts = cnt[:, 0, :].astype(np.float64)
        if len(pts) < 15:
            continue
        k = 5
        kernel = np.ones(k) / k
        xs = np.convolve(pts[:, 0], kernel, mode="valid")
        ys = np.convolve(pts[:, 1], kernel, mode="valid")
        heading = np.unwrap(np.arctan2(np.diff(ys), np.diff(xs)))
        d = np.diff(heading)
        d = (d + np.pi) % (2 * np.pi) - np.pi
        deltas.append(d)
    if not deltas:
        return 0.0
    return float(np.var(np.concatenate(deltas)))


def measure_f4(gray: np.ndarray) -> Dict[str, float]:
    """Ink bounding-box width, height and aspect ratio."""
    mask = _ink_mask(gray)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {"w": 0.0, "h": 0.0, "ratio": 0.0}
    w = float(xs.max() - xs.min() + 1)
    h = float(ys.max() - ys.min() + 1)
    return {"w": w, "h": h, "ratio": w / max(h, 1.0)}


def measure_f7(gray: np.ndarray) -> Dict[str, float]:
    """Mean ink darkness (0-1) and stroke-width variance (px^2) as pen-pressure proxies."""
    mask = _ink_mask(gray)
    if not mask.any():
        return {"darkness": 0.0, "width_var": 0.0}
    darkness = float(np.mean((255 - gray[mask]).astype(np.float64)) / 255.0)

    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    skel = _skeleton(mask)
    widths = 2.0 * dist[skel]
    width_var = float(np.var(widths)) if widths.size else 0.0
    return {"darkness": darkness, "width_var": width_var}


# ── Embedding helpers (F5 / F6) ───────────────────────────────────────────────
def embed_tensors(tensors: Sequence[Any]) -> np.ndarray:
    """L2-normalised embeddings, shape (N, D), from (1,H,W) or (1,1,H,W) tensors."""
    import torch
    import torch.nn.functional as F
    from app.ml.model_loader import get_model

    model = get_model()
    p = next(model.parameters())
    outs = []
    with torch.inference_mode():
        for t in tensors:
            if t.dim() == 3:
                t = t.unsqueeze(0)
            e = model(t.to(device=p.device, dtype=p.dtype))
            outs.append(F.normalize(e, p=2, dim=1).squeeze(0).cpu().numpy())
    return np.stack(outs, axis=0)


# ── Public API ────────────────────────────────────────────────────────────────
def compute_forensic_findings(
    reference_images: Sequence[Image.Image | np.ndarray],
    questioned_image: Image.Image | np.ndarray,
    reference_embeddings: Optional[np.ndarray],
    questioned_embedding: Optional[np.ndarray],
    distance: float,
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute F1-F7.

    reference_embeddings : (N, D) L2-normalised, from embed_tensors(); may be None
    questioned_embedding : (D,)  L2-normalised; may be None
    distance / threshold : the verdict distance and the EER threshold
    """
    T = THRESHOLDS
    refs = [_to_gray(r) for r in reference_images]
    q = _to_gray(questioned_image)

    def _plural(n: int, word: str) -> str:
        return f"{n} {word}" if n == 1 else f"{n} {word}s"

    # ---- F1 General information ------------------------------------------------
    r1 = [measure_f1(g) for g in refs]
    q1 = measure_f1(q)
    f1_counts: Dict[str, Dict[str, int]] = {}
    diffs = []
    names1 = {"strokes": "strokes", "bowls": "closed loops", "dots": "dots", "pen_lifts": "pen lifts"}
    for key in ("strokes", "bowls", "dots", "pen_lifts"):
        lo, hi = min(x[key] for x in r1), max(x[key] for x in r1)
        f1_counts[key] = {"q": q1[key], "min": lo, "max": hi}
        if q1[key] < lo - T["count_tolerance"] or q1[key] > hi + T["count_tolerance"]:
            diffs.append(f"{names1[key]} ({q1[key]} vs. {lo}-{hi})")
    f1_label = "Consistent" if not diffs else "Different"
    f1_obs = (
        f"The questioned signature has {_plural(q1['strokes'], 'separate stroke')}, "
        f"{_plural(q1['bowls'], 'closed loop')}, {_plural(q1['dots'], 'dot')} and "
        f"{_plural(q1['pen_lifts'], 'pen lift')}. "
        + ("All of these fall within the range seen in the reference signatures."
           if not diffs else "These differ from the references: " + "; ".join(diffs) + ".")
    )

    # ---- F2 Relation to baseline -----------------------------------------------
    r_ang = np.array([measure_f2(g) for g in refs])
    q_ang = measure_f2(q)
    tol = max(T["angle_tolerance_deg"], T["angle_sigma"] * float(r_ang.std()))
    ang_diff = abs(q_ang - float(r_ang.mean()))
    f2_label = "Consistent" if ang_diff <= tol else "Different"
    f2_obs = (
        f"The questioned signature slopes at {_fmt(q_ang)} degrees (positive means rising to the right). "
        f"The references average {_fmt(float(r_ang.mean()))} degrees, a difference of {_fmt(ang_diff)} degrees. "
        f"Differences up to {_fmt(tol)} degrees are treated as normal."
    )

    # ---- F3 Line quality --------------------------------------------------------
    r_var = np.array([measure_f3(g) for g in refs])
    q_var = measure_f3(q)
    ref_mean_var = float(r_var.mean()) or 1e-9
    ratio3 = q_var / ref_mean_var
    if ratio3 > T["curvature_ratio"]:
        f3_label = "Less smooth"
        f3_desc = "shakier than the references, which can happen when a signature is drawn slowly and carefully"
    elif ratio3 < 1.0 / T["curvature_ratio"]:
        f3_label = "Smoother"
        f3_desc = "smoother than the references"
    else:
        f3_label = "Consistent"
        f3_desc = "similar in smoothness to the references"
    f3_obs = (
        f"The line-wobble score is {q_var:.3f} for the questioned signature and {ref_mean_var:.3f} on average "
        f"for the references (higher means a more uneven pen line). The questioned signature is {f3_desc}."
    )

    # ---- F4 Proportion & spacing ------------------------------------------------
    r4 = [measure_f4(g) for g in refs]
    q4 = measure_f4(q)
    r_ratios = [x["ratio"] for x in r4]
    r_ratio = float(np.mean(r_ratios))
    rel = abs(q4["ratio"] - r_ratio) / max(r_ratio, 1e-9)
    f4_label = "Consistent" if rel <= T["ratio_tolerance"] else "Different"
    f4_obs = (
        f"The questioned signature measures {int(q4['w'])} x {int(q4['h'])} pixels, about "
        f"{_fmt(q4['ratio'])} times wider than tall. The references average {_fmt(r_ratio)} times, "
        f"a difference of {_fmt(rel * 100, 0)}%."
    )

    # ---- F5 Variation (model distance vs. threshold) ---------------------------
    pct = 100.0 * distance / threshold if threshold > 0 else float("inf")
    if pct <= 75:
        f5_label = "Well within threshold"
    elif pct <= 100:
        f5_label = "Within threshold"
    elif pct <= 150:
        f5_label = "Exceeds threshold"
    else:
        f5_label = "Far exceeds threshold"
    f5_obs = (
        f"The model measured a distance of {distance:.4f} between the questioned signature and the "
        f"combined references. The decision threshold is {threshold:.4f}. A distance above the threshold "
        f"is read as a different writer; here it is {_fmt(pct, 0)}% of the threshold."
    )

    # ---- F6 Natural variation range (writer's own spread) ----------------------
    f6_min = f6_max = f6_mean = q_to_ref = None
    if reference_embeddings is not None and len(reference_embeddings) >= 2:
        E = np.asarray(reference_embeddings)
        pair = [float(np.linalg.norm(E[i] - E[j]))
                for i in range(len(E)) for j in range(i + 1, len(E))]
        f6_min, f6_max, f6_mean = min(pair), max(pair), float(np.mean(pair))
        if questioned_embedding is not None:
            q_to_ref = float(np.mean([np.linalg.norm(questioned_embedding - e) for e in E]))
        else:
            q_to_ref = distance
        inside = q_to_ref <= f6_max * T["natural_range_margin"]
        f6_label = "Within natural range" if inside else "Outside natural range"
        f6_obs = (
            f"This writer's own genuine signatures differ from one another by {f6_min:.2f} to {f6_max:.2f} "
            f"(average {f6_mean:.2f}). The questioned signature is {q_to_ref:.2f} away from the references "
            f"on average, which is {'inside' if inside else 'outside'} the writer's normal range."
        )
    else:
        f6_label = "N/A"
        f6_obs = "Reference embeddings were not provided, so the writer's natural range could not be calculated."

    # ---- F7 Stroke density (ink deposition) ------------------------------------
    r7 = [measure_f7(g) for g in refs]
    q7 = measure_f7(q)
    r_dark = float(np.mean([x["darkness"] for x in r7]))
    r_wv = float(np.mean([x["width_var"] for x in r7])) or 1e-9
    dark_rel = abs(q7["darkness"] - r_dark) / max(r_dark, 1e-9)
    wv_ratio = q7["width_var"] / r_wv
    dark_ok = dark_rel <= T["darkness_tolerance"]
    wv_ok = (1.0 / T["width_var_ratio"]) <= wv_ratio <= T["width_var_ratio"]
    f7_label = "Consistent" if (dark_ok and wv_ok) else "Different"
    f7_obs = (
        f"Average ink darkness is {q7['darkness'] * 100:.0f}% for the questioned signature and "
        f"{r_dark * 100:.0f}% for the references. Line-width variation is {q7['width_var']:.2f} versus "
        f"{r_wv:.2f}. Pen pressure cannot be measured directly from an image, so this is an estimate."
    )

    ranges = {
        "f1": f1_counts,
        "f2": {"q": q_ang, "min": float(r_ang.min()), "max": float(r_ang.max())},
        "f3": {"q": q_var, "min": float(r_var.min()), "max": float(r_var.max())},
        "f4": {"q": q4["ratio"], "min": float(min(r_ratios)), "max": float(max(r_ratios))},
        "f5": {"distance": float(distance), "threshold": float(threshold)},
        "f6": {"q": q_to_ref, "min": f6_min, "max": f6_max, "mean": f6_mean},
        "f7_darkness": {"q": q7["darkness"], "min": float(min(x["darkness"] for x in r7)),
                        "max": float(max(x["darkness"] for x in r7))},
        "f7_width": {"q": q7["width_var"], "min": float(min(x["width_var"] for x in r7)),
                     "max": float(max(x["width_var"] for x in r7))},
    }

    def _r(x: Optional[float], nd: int = 4) -> Any:
        return "N/A" if x is None else round(float(x), nd)

    return {
        "ranges": ranges,
        "key_findings": {
            "f1_label": f1_label, "f2_label": f2_label, "f3_label": f3_label,
            "f4_label": f4_label, "f5_label": f5_label, "f6_label": f6_label,
            "f7_label": f7_label,
        },
        "modal_observations": {
            "f1_observation": f1_obs, "f2_observation": f2_obs, "f3_observation": f3_obs,
            "f4_observation": f4_obs, "f5_observation": f5_obs, "f6_observation": f6_obs,
            "f7_observation": f7_obs,
            "f2_angle": round(q_ang, 2),
            "f3_variance": round(q_var, 4),
            "f4_width_px": int(q4["w"]), "f4_height_px": int(q4["h"]),
            "f4_ratio": round(q4["ratio"], 2),
            "f5_distance": round(distance, 4),
            "f5_percent_threshold": round(pct, 1) if math.isfinite(pct) else "N/A",
            "f6_min": _r(f6_min), "f6_max": _r(f6_max), "f6_mean": _r(f6_mean),
            "f7_darkness": round(q7["darkness"], 3),
            "f7_width_variance": round(q7["width_var"], 3),
        },
    }