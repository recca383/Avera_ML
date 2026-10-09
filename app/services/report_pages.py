"""
Report design system + portrait pages for the AVERA compiled PDF.

Design rules
------------
* Font: Arial (falls back to Liberation Sans / DejaVu Sans if Arial is not installed).
* Brand colour #1E6FD9 for headers, accents and neutral highlights.
* Solid fills and borders only: no gradients, shadows or transparency.
* Status colours are used sparingly (green = consistent, amber = differs,
  red = far outside), always alongside a text label, never colour alone.

This module never imports gradcam_service (no circular import).
"""

from __future__ import annotations

import datetime
import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Polygon, Rectangle

# ── Design tokens ─────────────────────────────────────────────────────────────
def _pick_font() -> List[str]:
    """Arial if installed, else the closest metric-compatible font (no missing-font warnings)."""
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Arial", "Liberation Sans", "Helvetica", "Nimbus Sans"):
        if name in installed:
            return [name, "DejaVu Sans"]
    return ["DejaVu Sans"]


FONT = _pick_font()

BRAND = "#1E6FD9"
BRAND_LIGHT = "#D6E6FB"     # solid light blue (reference bands)
INK = "#111827"
TEXT = "#1F2937"
MUTED = "#6B7280"
BORDER = "#D1D5DB"
SURFACE = "#F3F6FB"
WHITE = "#FFFFFF"

OK = "#15803D"
WARN = "#B45309"
BAD = "#B91C1C"
NA = "#6B7280"
OK_FILL = "#E8F3EC"
BAD_FILL = "#FBEAEA"

VERDICT_LABELS = {"GENUINE": "GENUINE", "FORGED": "SUSPECTED"}
VERDICT_SUBTITLES = {
    "GENUINE": "Written by one and the same person",
    "FORGED": "Not written by one and the same person",
}

PAGE_W, PAGE_H = 8.5, 11.0
MX = 0.08                    # left/right margin (fraction of page width)
CONTENT_W = 1 - 2 * MX

FINDING_ORDER = ["f1", "f2", "f3", "f4", "f5", "f6", "f7"]
FINDING_NAMES = {
    "f1": "General Information",
    "f2": "Relation to Baseline",
    "f3": "Line Quality",
    "f4": "Proportion & Spacing",
    "f5": "Variation",
    "f6": "Natural Variation Range",
    "f7": "Stroke Density",
}
FINDING_EXPLAIN = {
    "f1": "Compares the basic building blocks of the signature: how many separate strokes it has, "
          "closed loops (bowls), dots, and pen lifts (places where the pen left the paper).",
    "f2": "Checks the overall slope of the signature line, whether it rises, falls or stays level, "
          "and compares it with the reference signatures.",
    "f3": "Checks how smooth the pen line is. A shaky or uneven line can suggest slow, careful drawing, "
          "which is common when someone copies another person's signature.",
    "f4": "Compares the overall shape and proportions of the signature: how wide it is compared with how tall it is.",
    "f5": "Shows how different the questioned signature is from the references according to the AI model, "
          "measured against the model's decision limit (the threshold).",
    "f6": "No one signs exactly the same way twice. This compares the questioned signature with how much the "
          "writer's own genuine signatures vary from one another.",
    "f7": "Estimates pen pressure from ink darkness and how evenly the line width changes. Pressure cannot be "
          "measured directly from a scan or photo, so this is an approximation.",
}

FINDING_SHORT = {
    "f1": "Strokes, closed loops, dots and pen lifts.",
    "f2": "Overall slope of the signature line.",
    "f3": "How smooth or shaky the pen line is.",
    "f4": "Width of the signature compared with its height.",
    "f5": "The AI model's distance versus its decision limit.",
    "f6": "Whether the difference is normal for this writer.",
    "f7": "Estimated pen pressure from ink darkness.",
}

_OK_LABELS = {"Consistent", "Well within threshold", "Within threshold", "Within natural range"}
_BAD_LABELS = {"Far exceeds threshold"}


def label_color(label: str) -> str:
    if label in _OK_LABELS:
        return OK
    if label in _BAD_LABELS:
        return BAD
    if label in ("", "N/A"):
        return NA
    return WARN


def verdict_color(verdict: str) -> str:
    return OK if verdict == "GENUINE" else BAD


# ── Low-level drawing helpers (figure-fraction coordinates) ───────────────────
def _tf(fig):
    return getattr(fig, "transFigure")


def _size(fig) -> Tuple[float, float]:
    w, h = fig.get_size_inches()
    return float(w), float(h)


def rect(fig, x, y, w, h, face="none", edge="none", lw=0.8, z=0):
    fig.add_artist(Rectangle((x, y), w, h, transform=_tf(fig), facecolor=face,
                             edgecolor=edge, linewidth=lw, zorder=z))


def hline(fig, x0, x1, y, color=BORDER, lw=0.8, z=1):
    fig.add_artist(Line2D([x0, x1], [y, y], transform=_tf(fig), color=color, linewidth=lw, zorder=z))


def vline(fig, x, y0, y1, color=BORDER, lw=0.8, z=1):
    fig.add_artist(Line2D([x, x], [y0, y1], transform=_tf(fig), color=color, linewidth=lw, zorder=z))


def dot(fig, cx, cy, diameter_in, color):
    w, h = _size(fig)
    fig.add_artist(Ellipse((cx, cy), diameter_in / w, diameter_in / h, transform=_tf(fig),
                           facecolor=color, edgecolor="none", zorder=4))


def text(fig, x, y, s, size=9.0, color=TEXT, weight="normal", ha="left", va="baseline", **kw):
    return fig.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
                    family=FONT, zorder=5, **kw)


def wrap(s: str, width_in: float, size: float) -> List[str]:
    chars = max(12, int(width_in / (size / 72.0 * 0.49)))
    return textwrap.wrap(s, width=chars) or [""]


def para(fig, x, y_top, s, width_in, size=9.0, color=TEXT, weight="normal", spacing=1.4) -> float:
    """Draw wrapped text with its top at y_top; returns the y of the paragraph bottom."""
    _, H = _size(fig)
    lines = wrap(s, width_in, size)
    text(fig, x, y_top, "\n".join(lines), size, color, weight, va="top", linespacing=spacing)
    return y_top - (len(lines) * size * spacing / 72.0) / H


def para_height_in(s: str, width_in: float, size: float, spacing: float = 1.4) -> float:
    return len(wrap(s, width_in, size)) * size * spacing / 72.0


def chip(fig, x_anchor, y_center, label: str, color: str, size=8.5, align: str = "right") -> None:
    """Bordered status label with a colour dot, anchored at x_anchor (left or right edge)."""
    W, H = _size(fig)
    w_in = len(label) * size / 72.0 * 0.56 + 0.42
    h_in = 0.26
    x0 = x_anchor - w_in / W if align == "right" else x_anchor
    rect(fig, x0, y_center - h_in / 2 / H, w_in / W, h_in / H, WHITE, color, 1.2)
    dot(fig, x0 + 0.15 / W, y_center, 0.09, color)
    text(fig, x0 + 0.28 / W, y_center, label, size, color, "bold", va="center")


# ── Page chrome ───────────────────────────────────────────────────────────────
def chrome(fig, case_id: str, section: str = "") -> None:
    """Brand bar + running header for portrait pages."""
    W, H = _size(fig)
    rect(fig, 0, 1 - 0.12 / H, 1, 0.12 / H, BRAND)
    text(fig, MX, 1 - 0.42 / H, "AVERA", 11, BRAND, "bold", va="center")
    text(fig, 1 - MX, 1 - 0.42 / H, f"{section}    Case {case_id}" if section else f"Case {case_id}",
         8.5, MUTED, ha="right", va="center")
    hline(fig, MX, 1 - MX, 1 - 0.62 / H, BORDER, 0.8)


def page_title(fig, title: str, subtitle: str = "") -> float:
    """Left-aligned page title; returns the y (fraction) where content may start."""
    _, H = _size(fig)
    y = 1 - 1.02 / H
    text(fig, MX, y, title, 19, INK, "bold", va="center")
    y_content = y - 0.30 / H
    if subtitle:
        text(fig, MX, y - 0.34 / H, subtitle, 9.5, MUTED, va="center")
        y_content = y - 0.62 / H
    return y_content


def footer(fig, page_counter: list, total_pages: Optional[int], case_id: str, rule: bool = True) -> None:
    page_counter[0] += 1
    W, H = _size(fig)
    if rule:
        hline(fig, MX, 1 - MX, 0.50 / H, BORDER, 0.8)
    y = 0.28 / H if rule else 0.015
    text(fig, MX, y, f"AVERA Case {case_id}   |   Confidential: generated for academic/thesis research",
         7.5, MUTED, va="center")
    text(fig, 1 - MX, y, f"Page {page_counter[0]} of {total_pages}", 7.5, MUTED, ha="right", va="center")


def _new_page(case_id: str, section: str):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=WHITE)
    chrome(fig, case_id, section)
    return fig


def _finish(fig, pdf, page_counter, total_pages, case_id) -> None:
    footer(fig, page_counter, total_pages, case_id, rule=True)
    pdf.savefig(fig)
    plt.close(fig)


# ── Visual components ─────────────────────────────────────────────────────────
def distance_gauge(fig, x, y, w, distance: float, threshold: float, color: str, show_zones=True) -> None:
    """Track split at the threshold; marker shows the questioned distance."""
    W, H = _size(fig)
    dom = max(threshold * 1.6, distance * 1.15, 1e-6)
    th = 0.17 / H                                     # track height (fraction)
    tx = x + w * (threshold / dom)
    rect(fig, x, y - th / 2, tx - x, th, OK_FILL, BORDER, 0.8)
    rect(fig, tx, y - th / 2, x + w - tx, th, BAD_FILL, BORDER, 0.8)
    if show_zones:
        if (tx - x) * W > 1.0:
            text(fig, (x + tx) / 2, y, "Same writer", 7.5, OK, ha="center", va="center")
        if (x + w - tx) * W > 1.1:
            text(fig, (tx + x + w) / 2, y, "Different writer", 7.5, BAD, ha="center", va="center")
    vline(fig, tx, y - th / 2 - 0.06 / H, y + th / 2 + 0.06 / H, BRAND, 2.2, z=3)
    mx = x + w * (min(distance, dom) / dom)
    fig.add_artist(Polygon([[mx - 0.07 / W, y + th / 2 + 0.16 / H], [mx + 0.07 / W, y + th / 2 + 0.16 / H],
                            [mx, y + th / 2 + 0.02 / H]], closed=True, transform=_tf(fig),
                           facecolor=color, edgecolor="none", zorder=4))
    ha = "right" if mx > x + w * 0.55 else "left"
    text(fig, mx, y + th / 2 + 0.24 / H, f"Distance {distance:.4f}", 8.5, color, "bold", ha=ha, va="bottom")
    text(fig, tx, y - th / 2 - 0.09 / H, f"Threshold {threshold:.4f}", 8, BRAND, "bold", ha="center", va="top")
    text(fig, x, y - th / 2 - 0.09 / H, "0", 7.5, MUTED, va="top")


def range_bar(fig, x, y, w, value: Optional[float], ref_min: Optional[float], ref_max: Optional[float],
              color: str, fmt: str = "{:.2f}", ok_lo: Optional[float] = None,
              ok_hi: Optional[float] = None) -> None:
    """
    Green zone = what the rule accepts as normal for this writer; blue band =
    the range actually seen in the references; marker = questioned signature.
    The green zone matters because a marker just outside the blue band can
    still be normal variation.
    """
    W, H = _size(fig)
    if value is None or ref_min is None or ref_max is None:
        text(fig, x, y, "Not available", 8, MUTED, va="center")
        return
    if ok_lo is not None and ok_hi is not None:
        # Colour by this bar's own zone: a card can combine several measures
        # (F7), and one can be normal while the card as a whole differs.
        color = OK if ok_lo <= value <= ok_hi else WARN
    pts = [value, ref_min, ref_max] + [v for v in (ok_lo, ok_hi) if v is not None]
    lo, hi = min(pts), max(pts)
    span = (hi - lo) or max(abs(hi), 1.0) * 0.2
    dom_lo, dom_hi = lo - span * 0.12, hi + span * 0.12
    px = lambda v: x + w * ((v - dom_lo) / (dom_hi - dom_lo))
    th = 0.16 / H
    rect(fig, x, y - th / 2, w, th, SURFACE, BORDER, 0.8)
    if ok_lo is not None and ok_hi is not None:
        rect(fig, px(ok_lo), y - th / 2, px(ok_hi) - px(ok_lo), th, OK_FILL, "none", z=1)
    bh = th * 0.5
    rect(fig, px(ref_min), y - bh / 2, max(px(ref_max) - px(ref_min), 0.004), bh, BRAND_LIGHT, BRAND, 0.8, z=2)
    mx = px(value)
    vline(fig, mx, y - 0.15 / H, y + 0.15 / H, color, 3.0, z=4)
    ha = "right" if mx > x + w * 0.75 else ("left" if mx < x + w * 0.25 else "center")
    text(fig, mx, y + 0.20 / H, f"Questioned {fmt.format(value)}", 8, color, "bold", ha=ha, va="bottom")
    cap = f"References {fmt.format(ref_min)} to {fmt.format(ref_max)}"
    if ok_lo is not None and ok_hi is not None:
        cap += f"   ·   Normal {fmt.format(ok_lo)} to {fmt.format(ok_hi)}"
    text(fig, x, y - 0.22 / H, cap, 7.5, MUTED, va="top")


# Test-split AUC of each supporting check: how well it separated genuine from
# forged signatures on unseen Pipeline 32 data (0.5 = coin flip, 1.0 = perfect).
# Source: docs/forensic_calibration_final.json, "test_results". F5 is the model
# result itself and is not graded here.
FINDING_AUC = {"f1": 0.565, "f2": 0.616, "f3": 0.691, "f4": 0.751, "f6": 0.881, "f7": 0.871}


def strength(code: str) -> Optional[Tuple[int, str]]:
    auc = FINDING_AUC.get(code)
    if auc is None:
        return None
    if auc >= 0.80:
        return 3, "Strong"
    if auc >= 0.65:
        return 2, "Moderate"
    return 1, "Weak"


def text_width_in(fig, t) -> float:
    return t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi


def strength_badge(fig, x, y, code: str, suffix: str = " evidence", size: float = 7.5) -> None:
    """Three dots (filled = stronger evidence) plus a word, left-aligned at x, centred on y."""
    W, _ = _size(fig)
    s = strength(code)
    if s is None:
        text(fig, x, y, "Decides the result", size, MUTED, "bold", va="center")
        return
    n, word = s
    for i in range(3):
        dot(fig, x + (0.05 + i * 0.12) / W, y, 0.085, BRAND if i < n else BORDER)
    text(fig, x + 0.40 / W, y, word + suffix, size, MUTED, "bold", va="center")


DEV_MAX = 3.0   # deviation tracks run from 0 (typical) to 3x the edge of normal


def deviation_track(fig, x, y, w, dev: Optional[float], color: str) -> None:
    """0 = typical for this writer, 1 = edge of normal (green zone), beyond = outside normal."""
    W, H = _size(fig)
    th = 0.14 / H
    edge = x + w / DEV_MAX
    rect(fig, x, y - th / 2, w, th, SURFACE, BORDER, 0.8)
    rect(fig, x, y - th / 2, edge - x, th, OK_FILL, BORDER, 0.8)
    vline(fig, edge, y - th / 2 - 0.04 / H, y + th / 2 + 0.04 / H, OK, 1.2, z=3)
    if dev is None:
        text(fig, x + w / 2, y, "Not available", 7, MUTED, ha="center", va="center")
        return
    mx = x + w * min(dev, DEV_MAX) / DEV_MAX
    dot(fig, mx, y, 0.15, color)
    if dev > DEV_MAX:
        text(fig, x + w + 0.05 / W, y, f"{dev:.1f}×", 7, color, "bold", va="center")


def dot_strip(fig, x, y, w, pairs: List[float], value: Optional[float], ok_hi: Optional[float],
              color: str, fmt: str = "{:.2f}") -> None:
    """F6: each genuine-vs-genuine distance as a dot, the questioned distance as a marker."""
    W, H = _size(fig)
    if value is None or ok_hi is None or not pairs:
        text(fig, x, y, "Not available", 8, MUTED, va="center")
        return
    color = OK if value <= ok_hi else WARN
    dom = max(ok_hi, value, max(pairs)) * 1.08
    px = lambda v: x + w * (v / dom)
    th = 0.16 / H
    rect(fig, x, y - th / 2, w, th, SURFACE, BORDER, 0.8)
    rect(fig, x, y - th / 2, px(ok_hi) - x, th, OK_FILL, "none", z=1)
    for p in pairs:
        dot(fig, px(p), y, 0.09, BRAND)
    mx = px(value)
    vline(fig, mx, y - 0.15 / H, y + 0.15 / H, color, 3.0, z=4)
    ha = "right" if mx > x + w * 0.75 else ("left" if mx < x + w * 0.25 else "center")
    text(fig, mx, y + 0.20 / H, f"Questioned {fmt.format(value)}", 8, color, "bold", ha=ha, va="bottom")
    text(fig, x, y - 0.22 / H, f"Blue dots = genuine vs. genuine   ·   Normal up to {fmt.format(ok_hi)}",
         7.5, MUTED, va="top")


def image_cell(fig, x, y_top, w_in, h_in, img, label: str, caption: str, label_color: str = MUTED) -> None:
    """Labelled image (letterboxed, thin border) with a caption underneath."""
    W, H = _size(fig)
    text(fig, x, y_top, label, 7.5, label_color, "bold", va="top")
    ax = fig.add_axes((x, y_top - (0.17 + h_in) / H, w_in / W, h_in / H))
    ax.imshow(img, interpolation="lanczos")
    ax.set_anchor("W")              # letterbox to the left so the image lines up with its label
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.8)
    text(fig, x, y_top - (0.17 + h_in + 0.06) / H, caption, 7.5, TEXT, va="top")


# ── Cover page ────────────────────────────────────────────────────────────────
def page_cover(pdf, page_counter, total_pages, case_id, verdict, conf_genuine, conf_forged,
               avg_distance, threshold, query_image_name, ref_image_names, model_version_tag) -> None:
    W, H = PAGE_W, PAGE_H
    fig = plt.figure(figsize=(W, H), facecolor=WHITE)
    vcol = verdict_color(verdict)

    rect(fig, 0, 0.80, 1, 0.20, BRAND)
    text(fig, MX, 0.915, "AVERA", 38, WHITE, "bold", va="center")
    text(fig, MX, 0.865, "Automated Verification & Explainable Recognition of Authorship", 10.5, WHITE, va="center")

    text(fig, MX, 0.745, "Signature Verification & Forensic Analysis Report", 20, INK, "bold", va="center")
    text(fig, MX, 0.715, f"Case {case_id}", 12, MUTED, va="center")
    rect(fig, MX, 0.700, 0.08, 0.003, BRAND)

    # Verdict card
    top, ch = 0.665, 0.195
    rect(fig, MX, top - ch, CONTENT_W, ch, WHITE, vcol, 2.0)
    text(fig, MX + 0.03, top - 0.030, "MODEL RESULT", 8, MUTED, "bold", va="center")
    text(fig, MX + 0.03, top - 0.078, VERDICT_LABELS.get(verdict, verdict), 26, vcol, "bold", va="center")
    text(fig, MX + 0.03, top - 0.103, VERDICT_SUBTITLES.get(verdict, ""), 9, TEXT, va="center")
    body = (f"The model measured a distance of {avg_distance:.4f} between the questioned signature and the "
            f"reference signatures. The decision threshold is {threshold:.4f}. "
            f"A distance above the threshold is read as a different writer.")
    para(fig, MX + 0.03, top - 0.130, body, CONTENT_W * W - 0.5, 9, TEXT)
    text(fig, MX + 0.03, top - ch + 0.018,
         f"Model confidence score: {conf_genuine:.1f}% same writer / {conf_forged:.1f}% different writer "
         f"(a score, not a statistical probability)", 7.5, MUTED, va="center")

    # Case details table
    rows = [
        ("Case ID", case_id),
        ("Questioned document", os.path.basename(query_image_name)),
        ("Reference specimens", f"{len(ref_image_names)} genuine samples on file"),
        ("Model / pipeline", model_version_tag),
        ("Report generated", datetime.datetime.now().strftime("%B %d, %Y  %H:%M")),
    ]
    y = 0.440
    text(fig, MX, y, "Case Details", 11, INK, "bold", va="center")
    y -= 0.022
    rh = 0.034
    hline(fig, MX, 1 - MX, y, INK, 1.0)
    for label, value in rows:
        y -= rh
        text(fig, MX + 0.01, y + rh / 2, label, 9, MUTED, va="center")
        text(fig, MX + 0.27, y + rh / 2, value, 10, INK, va="center")
        hline(fig, MX, 1 - MX, y, BORDER, 0.8)

    note = ("This report was generated by AVERA, an offline signature verification system developed for "
            "academic thesis research. It supports, and does not replace, the judgment of a qualified "
            "document examiner. Please read the final pages for an explanation of the results and the "
            "disclaimer.")
    note_bottom = para(fig, MX + 0.02, 0.157, note, CONTENT_W * W - 0.3, 8.5, MUTED)
    rect(fig, MX, note_bottom, 0.004, 0.157 - note_bottom, BRAND)

    footer(fig, page_counter, total_pages, case_id, rule=True)
    pdf.savefig(fig)
    plt.close(fig)


# ── Case summary ──────────────────────────────────────────────────────────────
def _support_stats(key_findings: Dict[str, str]) -> Tuple[int, int]:
    codes = ["f1", "f2", "f3", "f4", "f6", "f7"]
    labels = [key_findings.get(f"{c}_label", "N/A") for c in codes]
    evaluated = [l for l in labels if l != "N/A"]
    return sum(1 for l in evaluated if l in _OK_LABELS), len(evaluated)


def page_summary(pdf, page_counter, total_pages, case_id, verdict, avg_distance, threshold,
                 findings: Optional[dict]) -> None:
    W, H = PAGE_W, PAGE_H
    findings = findings or {}
    key = findings.get("key_findings", {})
    fig = _new_page(case_id, "Case Summary")
    y = page_title(fig, "Case Summary", "The model result, supporting checks and key findings at a glance")
    vcol = verdict_color(verdict)

    # Model result card
    ch = 1.45 / H
    top = y
    rect(fig, MX, top - ch, CONTENT_W, ch, WHITE, BORDER, 1.0)
    text(fig, MX + 0.02, top - 0.20 / H, "MODEL RESULT", 8, MUTED, "bold", va="center")
    text(fig, MX + 0.02, top - 0.55 / H, VERDICT_LABELS.get(verdict, verdict), 22, vcol, "bold", va="center")
    text(fig, MX + 0.02, top - 0.76 / H, VERDICT_SUBTITLES.get(verdict, ""), 8.5, TEXT, va="center")
    pct = 100.0 * avg_distance / threshold if threshold > 0 else 0.0
    para(fig, MX + 0.02, top - 0.86 / H,
         f"Distance is {pct:.0f}% of the threshold. Lower distance means more similar signatures.",
         2.3, 8.5, MUTED)
    distance_gauge(fig, 0.42, top - 0.72 / H, 0.47, avg_distance, threshold, vcol)
    y = top - ch - 0.22 / H

    # Supporting checks headline
    n_ok, n_eval = _support_stats(key)
    strong = [c for c in ("f1", "f2", "f3", "f4", "f6", "f7")
              if (strength(c) or (0,))[0] == 3 and key.get(f"{c}_label", "N/A") != "N/A"]
    strong_ok = sum(1 for c in strong if key.get(f"{c}_label") in _OK_LABELS)
    text(fig, MX, y, "Supporting checks", 11, INK, "bold", va="center")
    if n_eval:
        summary = f"{n_ok} of {n_eval} are consistent with the references"
        if strong:
            summary += (f", including {strong_ok} of the {len(strong)} strongest "
                        f"({', '.join(c.upper() for c in strong)})")
        summary += "."
    else:
        summary = "Supporting checks are not available for this case."
    text(fig, MX + 0.215, y, summary, 9, TEXT, va="center")
    y -= 0.30 / H

    # Agreement note
    if n_eval:
        same = verdict == "GENUINE"
        ratio = n_ok / n_eval
        if (not same and ratio >= 0.67) or (same and ratio <= 0.33):
            msg = ("The model result and the supporting checks point in different directions. "
                   "This case should be reviewed by a qualified document examiner before any conclusion is drawn.")
            rect(fig, MX, y - 0.62 / H, CONTENT_W, 0.62 / H, WHITE, WARN, 1.4)
            text(fig, MX + 0.02, y - 0.16 / H, "REVIEW RECOMMENDED", 8, WARN, "bold", va="center")
            para(fig, MX + 0.02, y - 0.27 / H, msg, CONTENT_W * W - 0.3, 9, TEXT)
            y -= 0.62 / H + 0.30 / H

    # Findings table: result, evidence strength and distance from the writer's normal
    deviation = findings.get("deviation", {}) or {}
    cols = {"code": MX + 0.010, "name": MX + 0.055, "chip": MX + 0.262, "str": MX + 0.462, "track": MX + 0.615}
    track_w = 1 - MX - 0.035 - cols["track"]
    hh = 0.46 / H
    rect(fig, MX, y - hh, CONTENT_W, hh, SURFACE, BORDER, 0.8)
    yh = y - 0.15 / H
    text(fig, cols["code"], y - hh / 2, "Code", 8, MUTED, "bold", va="center")
    text(fig, cols["name"], y - hh / 2, "Finding", 8, MUTED, "bold", va="center")
    text(fig, cols["chip"], y - hh / 2, "Result", 8, MUTED, "bold", va="center")
    text(fig, cols["str"], y - hh / 2, "Evidence strength", 8, MUTED, "bold", va="center")
    text(fig, cols["track"], yh, "Distance from this writer's normal", 8, MUTED, "bold", va="center")
    yt = y - 0.33 / H
    text(fig, cols["track"], yt, "0", 7, MUTED, va="center")
    text(fig, cols["track"] + track_w / DEV_MAX, yt, "normal limit", 7, OK, "bold", ha="center", va="center")
    text(fig, cols["track"] + track_w, yt, "3×", 7, MUTED, ha="right", va="center")
    y -= hh
    rh = 0.46 / H
    for code in FINDING_ORDER:
        lab = key.get(f"{code}_label", "N/A")
        col = label_color(lab)
        yc = y - rh / 2
        text(fig, cols["code"], yc, code.upper(), 9.5, BRAND, "bold", va="center")
        text(fig, cols["name"], yc, FINDING_NAMES[code], 9, INK, "bold", va="center")
        chip(fig, cols["chip"], yc, lab, col, size=7.5, align="left")
        strength_badge(fig, cols["str"], yc, code, suffix="", size=7.5)
        dev = deviation.get(code)
        deviation_track(fig, cols["track"], yc, track_w, None if lab == "N/A" else dev, col)
        y -= rh
        hline(fig, MX, 1 - MX, y, BORDER, 0.8)

    notes = ("Distance from normal: 0 is typical for this writer and the green zone is the normal range "
             "(for F5, the model's threshold). Evidence strength: how well each check told genuine from forged "
             "signatures in testing; the strongest checks deserve the most weight.")
    y = para(fig, MX, y - 0.14 / H, notes, CONTENT_W * W, 8, MUTED, spacing=1.35)

    # In plain words: one everyday sentence per finding, strongest evidence first
    plain = findings.get("plain", {}) or {}
    if plain:
        y -= 0.34 / H
        text(fig, MX, y, "In plain words", 11, INK, "bold", va="center")
        y -= 0.20 / H
        order = sorted(FINDING_ORDER, key=lambda c: (c != "f5", -(strength(c) or (0,))[0]))
        for code in order:
            if not plain.get(code):
                continue
            col = label_color(key.get(f"{code}_label", "N/A"))
            dot(fig, MX + 0.06 / W, y - 0.07 / H, 0.08, col)
            text(fig, MX + 0.16 / W, y, code.upper(), 8.5, BRAND, "bold", va="top")
            y = para(fig, MX + 0.50 / W, y, plain[code], CONTENT_W * W - 0.55, 8.5, TEXT, spacing=1.35) - 0.07 / H
    text(fig, MX, y - 0.14 / H, "The Findings pages show what each check measured, with pictures.",
         8, MUTED, va="center")

    _finish(fig, pdf, page_counter, total_pages, case_id)


# ── Findings cards ────────────────────────────────────────────────────────────
# Card layout, in inches. Cards with an illustration show it on the left and
# the numbers on the right; F5/F6 (and any card whose illustration is missing)
# show the numbers on the left instead.
_CARD_W = CONTENT_W * PAGE_W
_PADX = 0.18
_HEADER_H = 0.48
_LEFT_W = 3.3
_COL_GAP = 0.28
_RIGHT_W = _CARD_W - 2 * _PADX - _LEFT_W - _COL_GAP
_IMG_H = 0.82
_IMG_GAP = 0.14
_CARD_GAP = 0.14
_FINDINGS_TOP_IN = 1.64        # where page_title() lets content start (with subtitle)
_FINDINGS_BOTTOM_IN = 0.68     # keep clear of the footer rule
_HOW_TO_READ_H = 1.55
_VISUAL_CODES = ("f1", "f2", "f3", "f4", "f7")
_F1_ROWS = [("strokes", "Strokes"), ("bowls", "Closed loops"), ("dots", "Dots"), ("pen_lifts", "Pen lifts")]


def _numbers_h(code: str) -> float:
    return {"f1": 1.18, "f5": 0.92, "f7": 1.96}.get(code, 0.80)


def _card_parts(code: str, findings: dict, visuals: Optional[dict]) -> dict:
    """Text and section heights for one card; shared by layout planning and drawing."""
    modal = findings.get("modal_observations", {}) or {}
    plain = (findings.get("plain", {}) or {}).get(code, "")
    obs = str(modal.get(f"{code}_observation", "Not available for this case."))
    vis = (visuals or {}).get(code) if code in _VISUAL_CODES else None
    full_w = _CARD_W - 2 * _PADX

    explain_h = para_height_in(FINDING_EXPLAIN[code], full_w, 8, 1.35)
    plain_h = para_height_in(plain, full_w, 10.5, 1.3) if plain else 0.0
    details_h = 0.17 + para_height_in(obs, _RIGHT_W, 8, 1.35)
    if vis:
        legend = str(vis.get("legend") or "")
        left_h = 0.17 + _IMG_H + 0.24 + (para_height_in(legend, _LEFT_W, 7, 1.3) if legend else 0)
        right_h = _numbers_h(code) + 0.10 + details_h
    else:
        left_h = _numbers_h(code)
        right_h = details_h
    body_top = _HEADER_H + explain_h + 0.08 + plain_h + 0.16
    return {"plain": plain, "obs": obs, "vis": vis, "explain_h": explain_h, "plain_h": plain_h,
            "body_top": body_top, "height": body_top + max(left_h, right_h) + 0.16}


def _card_height_in(code: str, findings: dict, visuals: Optional[dict] = None) -> float:
    return _card_parts(code, findings, visuals)["height"]


def _draw_numbers(fig, code: str, x: float, top: float, w_in: float, ranges: dict, col: str) -> None:
    """The measured values for one finding, drawn from `top` (figure fraction) downward."""
    W, H = _size(fig)
    w = w_in / W
    if code == "f1":
        counts = ranges.get("f1") or {}
        cx = [0.0, 1.05, 1.75, 2.50]
        for off, head in zip(cx, ("Feature", "Questioned", "References", "Normal")):
            text(fig, x + off / W, top - 0.10 / H, head, 7.5, MUTED, "bold", va="center")
        for i, (k, nm) in enumerate(_F1_ROWS):
            ry = top - (0.36 + i * 0.22) / H
            hline(fig, x, x + w, ry + 0.11 / H, BORDER, 0.6)
            text(fig, x, ry, nm, 8.5, TEXT, va="center")
            c = counts.get(k)
            if not c:
                continue
            lo, hi = c.get("ok_lo", c["min"] - c.get("tol", 1)), c.get("ok_hi", c["max"] + c.get("tol", 1))
            inside = lo <= c["q"] <= hi
            text(fig, x + cx[1] / W, ry, str(c["q"]), 9, OK if inside else WARN, "bold", va="center")
            rng = f"{c['min']}" if c["min"] == c["max"] else f"{c['min']} to {c['max']}"
            text(fig, x + cx[2] / W, ry, rng, 8.5, TEXT, va="center")
            text(fig, x + cx[3] / W, ry, f"{max(lo, 0)} to {hi}", 8.5, MUTED, va="center")
    elif code == "f5":
        r = ranges.get("f5") or {}
        if r:
            distance_gauge(fig, x, top - 0.50 / H, w, r["distance"], r["threshold"], col)
    elif code == "f6":
        r = ranges.get("f6") or {}
        dot_strip(fig, x, top - 0.42 / H, w, r.get("pairs") or [], r.get("q"), r.get("ok_hi"), col)
    elif code == "f7":
        for i, (key, title, fmt) in enumerate((("f7_darkness", "Ink darkness", "{:.2f}"),
                                               ("f7_width", "Line-width variation", "{:.2f}"))):
            r = ranges.get(key) or {}
            t0 = top - i * 1.0 / H
            text(fig, x, t0 - 0.06 / H, title, 7.5, MUTED, "bold", va="center")
            range_bar(fig, x, t0 - 0.56 / H, w, r.get("q"), r.get("min"), r.get("max"), col, fmt,
                      r.get("ok_lo"), r.get("ok_hi"))
    else:
        r = ranges.get(code) or {}
        fmt = {"f2": "{:+.1f}°", "f3": "{:.3f}", "f4": "{:.2f}"}[code]
        range_bar(fig, x, top - 0.42 / H, w, r.get("q"), r.get("min"), r.get("max"), col, fmt,
                  r.get("ok_lo"), r.get("ok_hi"))


def _card(fig, top: float, code: str, findings: dict, visuals: Optional[dict] = None) -> float:
    """Draws one finding card with its top at `top`; returns the y of the card bottom."""
    W, H = _size(fig)
    parts = _card_parts(code, findings, visuals)
    key = findings.get("key_findings", {})
    ranges = findings.get("ranges", {}) or {}
    label = key.get(f"{code}_label", "N/A")
    col = label_color(label)
    h = parts["height"] / H
    rect(fig, MX, top - h, CONTENT_W, h, WHITE, BORDER, 1.0)
    rect(fig, MX, top - h, 0.05 / W, h, col, "none", z=1)      # status accent stripe

    x0 = MX + _PADX / W
    yh = top - 0.24 / H
    text(fig, x0, yh, code.upper(), 12, BRAND, "bold", va="center")
    t_name = text(fig, x0 + 0.42 / W, yh, FINDING_NAMES[code], 12, INK, "bold", va="center")
    strength_badge(fig, x0 + (0.42 + text_width_in(fig, t_name) + 0.22) / W, yh, code)
    chip(fig, 1 - MX - _PADX / W, yh, label, col)

    full_w = _CARD_W - 2 * _PADX
    y = top - _HEADER_H / H
    para(fig, x0, y, FINDING_EXPLAIN[code], full_w, 8, MUTED, spacing=1.35)
    y -= (parts["explain_h"] + 0.08) / H
    if parts["plain"]:
        para(fig, x0, y, parts["plain"], full_w, 10.5, INK, spacing=1.3)
    hline(fig, x0, 1 - MX - _PADX / W, top - (parts["body_top"] - 0.08) / H, BORDER, 0.6)

    body = top - parts["body_top"] / H
    xr = x0 + (_LEFT_W + _COL_GAP) / W
    vis = parts["vis"]
    if vis:
        cell_w = (_LEFT_W - _IMG_GAP) / 2
        image_cell(fig, x0, body, cell_w, _IMG_H, vis["ref"], str(vis["ref_label"]), str(vis["ref_caption"]))
        image_cell(fig, x0 + (cell_w + _IMG_GAP) / W, body, cell_w, _IMG_H, vis["q"], "Questioned",
                   str(vis["q_caption"]), BRAND)
        if vis.get("legend"):
            para(fig, x0, body - (0.17 + _IMG_H + 0.26) / H, str(vis["legend"]), _LEFT_W, 7, MUTED, spacing=1.3)
        _draw_numbers(fig, code, xr, body, _RIGHT_W, ranges, col)
        details_top = body - (_numbers_h(code) + 0.10) / H
    else:
        _draw_numbers(fig, code, x0, body, _LEFT_W, ranges, col)
        details_top = body
    text(fig, xr, details_top, "MEASUREMENT DETAILS", 7, MUTED, "bold", va="top")
    para(fig, xr, details_top - 0.17 / H, parts["obs"], _RIGHT_W, 8, TEXT, spacing=1.35)
    return top - h


def plan_findings_pages(findings: Optional[dict], visuals: Optional[dict] = None) -> List[List[str]]:
    """
    Flows the F1-F7 cards onto as many pages as they need (cards never split),
    and reserves room for the "How to read" box after the last card. An empty
    list at the end means that box gets a page of its own.
    """
    findings = findings or {}
    avail = PAGE_H - _FINDINGS_TOP_IN - _FINDINGS_BOTTOM_IN
    pages: List[List[str]] = [[]]
    used = 0.0
    for code in FINDING_ORDER:
        hc = _card_height_in(code, findings, visuals)
        if pages[-1] and used + hc > avail:
            pages.append([])
            used = 0.0
        pages[-1].append(code)
        used += hc + _CARD_GAP
    if used + _HOW_TO_READ_H > avail:
        pages.append([])
    return pages


def _how_to_read(fig, y: float) -> None:
    W, H = _size(fig)
    rect(fig, MX, y - _HOW_TO_READ_H / H, CONTENT_W, _HOW_TO_READ_H / H, SURFACE, BORDER, 0.8)
    text(fig, MX + 0.02, y - 0.20 / H, "How to read these cards", 9.5, INK, "bold", va="center")
    notes = [
        "Bars: the green zone is what counts as normal for this writer, the blue band is the range seen in "
        "the four references, and the marker is the questioned signature. A marker inside the green zone is "
        "normal variation even if it sits outside the blue band.",
        "Evidence strength (dots next to each title) shows how well that check told genuine from forged "
        "signatures when tested on signatures the system had never seen. A weak check is easily fooled, so "
        "give it less weight than a strong one.",
        "Pictures: the reference shown is the most typical of the four. F5 and F6 both use the AI model: F5 "
        "compares with the combined references, F6 with the writer's own variation, so their numbers differ.",
    ]
    yy = y - 0.36 / H
    for n in notes:
        yy = para(fig, MX + 0.02, yy, n, CONTENT_W * W - 0.3, 8, TEXT, spacing=1.35) - 0.06 / H


def page_findings(pdf, page_counter, total_pages, case_id, findings: Optional[dict], codes: List[str],
                  visuals: Optional[dict] = None, first: bool = True, last: bool = False) -> None:
    """One page of finding cards (see plan_findings_pages); the last page ends with the reading guide."""
    _, H = PAGE_W, PAGE_H
    findings = findings or {}
    fig = _new_page(case_id, "Forensic Findings")
    y = page_title(fig, "Forensic Findings (F1-F7)" + ("" if first else "  continued"),
                   "What each check measured, what it found, and how much weight it deserves")
    for code in codes:
        y = _card(fig, y, code, findings, visuals) - _CARD_GAP / H
    if last:
        _how_to_read(fig, y)
    _finish(fig, pdf, page_counter, total_pages, case_id)


# ── Explanation + glossary ────────────────────────────────────────────────────
def page_explanation(pdf, page_counter, total_pages, case_id) -> None:
    W, H = PAGE_W, PAGE_H
    fig = _new_page(case_id, "Understanding This Report")
    y = page_title(fig, "Understanding This Report", "A plain-language guide to the results")

    sections = [
        ("What AVERA does",
         "AVERA compares one questioned signature against four genuine reference signatures from the same "
         "person. A trained neural network turns each signature into a set of numbers, then measures how far "
         "apart they are. A short distance means the signatures are structurally similar."),
        ("How the result is decided",
         "The result is based only on the model's distance (F5). If the distance is below the decision "
         "threshold, AVERA reads the signatures as written by the same person. The threshold was set during "
         "testing at the point where wrongly accepted and wrongly rejected signatures were equally common, so "
         "results close to the threshold are less certain."),
        ("What the supporting checks are",
         "F1 to F4 and F7 use classical image measurements, the way an examiner might compare shape, slope, "
         "smoothness and ink. F6 uses the model to compare the questioned signature with the writer's own "
         "natural variation. These checks add context but do not change the model result."),
        ("What the visual pages show",
         "The Grad-CAM heatmap shows where the model paid the most attention. It explains the model, and is "
         "not proof of forgery on its own. The overlay, bounding box and stroke map are direct comparisons of "
         "the ink and are closer to traditional document examination."),
    ]
    tw = CONTENT_W * W
    for head, body in sections:
        rect(fig, MX, y - 0.19 / H, 0.004, 0.19 / H, BRAND)
        text(fig, MX + 0.014, y - 0.09 / H, head, 10.5, INK, "bold", va="center")
        y = para(fig, MX, y - 0.28 / H, body, tw, 9, TEXT, spacing=1.45) - 0.16 / H

    text(fig, MX, y, "Glossary", 11, INK, "bold", va="center")
    y -= 0.14 / H
    terms = [
        ("Pen lift", "A place where the pen left the paper, so the signature has a break."),
        ("Bowl", "A closed loop in a letter, such as the round part of an 'o' or 'a'."),
        ("Terminal dot", "A small dot at the end of a stroke or above a letter."),
        ("Baseline", "The invisible line a signature is written along. Its slope is the baseline angle."),
        ("Threshold", "The distance limit the model uses to separate 'same writer' from 'different writer'."),
        ("Distance", "How far apart two signatures are according to the model. Lower means more similar."),
        ("Grad-CAM", "A heatmap showing which parts of the signature the model relied on most."),
        ("Normal range", "The green zone on each bar: how much this writer's signatures can vary and still "
                         "count as consistent."),
        ("Evidence strength", "How well a check told genuine from forged signatures in testing. Weak checks "
                              "are easily fooled; strong ones deserve more weight."),
    ]
    rh = 0.40 / H
    hline(fig, MX, 1 - MX, y, INK, 1.0)
    for term, meaning in terms:
        y -= rh
        text(fig, MX + 0.01, y + rh / 2, term, 9, INK, "bold", va="center")
        lines = wrap(meaning, (CONTENT_W - 0.22) * W, 8.5)[:2]
        text(fig, MX + 0.20, y + rh / 2, "\n".join(lines), 8.5, TEXT, va="center", linespacing=1.3)
        hline(fig, MX, 1 - MX, y, BORDER, 0.8)

    _finish(fig, pdf, page_counter, total_pages, case_id)


# ── Disclaimer ────────────────────────────────────────────────────────────────
def page_disclaimer(pdf, page_counter, total_pages, case_id) -> None:
    W, H = PAGE_W, PAGE_H
    fig = _new_page(case_id, "Disclaimer")
    y = page_title(fig, "Disclaimer", "Please read before relying on this report")
    items = [
        "AVERA is an automated, offline signature verification system developed as part of an academic thesis "
        "project. It is a research prototype, not a certified or legally accredited forensic tool.",
        "The findings in this report (F1 to F7, the model result and the confidence score) come from computer "
        "measurements and a trained neural network. The cut-offs used to label F1 to F7 were calibrated "
        "statistically on validation data so that about 1 in 20 genuine signatures is flagged on each check; "
        "on new signatures some checks flag genuine signatures more often than that. A flagged check is a "
        "prompt for closer review, not proof of forgery. These cut-offs have not been validated against a "
        "licensed forensic document examiner's judgment.",
        "This report is a decision-support and explainability aid. It summarizes computational evidence to help "
        "a human examiner reason about a case. It is not a substitute for review and certification by a "
        "qualified, licensed Questioned Document Examiner, and should not be submitted as standalone evidence "
        "in any legal or administrative proceeding.",
        "Results can vary with scan or photo quality, signature complexity, and the writer's natural "
        "variability. A result reflects the balance of computed evidence at the time of analysis and does not "
        "express certainty.",
        "This system and report were developed for academic research by the AVERA thesis research team.",
    ]
    tw = CONTENT_W * W - 0.75
    total_h = sum(para_height_in(t, tw, 9.5, 1.5) + 0.20 for t in items) + 0.25
    rect(fig, MX, y - total_h / H, CONTENT_W, total_h / H, WHITE, BORDER, 1.0)
    yy = y - 0.20 / H
    for i, t in enumerate(items, 1):
        rect(fig, MX + 0.02, yy - 0.24 / H, 0.24 / W, 0.24 / H, BRAND, BRAND, 0.8)
        text(fig, MX + 0.02 + 0.12 / W, yy - 0.12 / H, str(i), 9, WHITE, "bold", ha="center", va="center")
        yy = para(fig, MX + 0.075, yy, t, tw, 9.5, TEXT, spacing=1.5) - 0.20 / H
    text(fig, 0.5, 0.10, f"AVERA Case {case_id}  -  End of Report", 8.5, MUTED, ha="center", va="center")
    _finish(fig, pdf, page_counter, total_pages, case_id)