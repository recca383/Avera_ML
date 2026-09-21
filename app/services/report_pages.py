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

# Change these strings if the examiner-approved wording must replace GENUINE/FORGED
# (e.g. "Written by one and the same person").
VERDICT_LABELS = {"GENUINE": "GENUINE", "FORGED": "FORGED"}

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
              color: str, fmt: str = "{:.2f}", caption: str = "") -> None:
    """Blue band = writer's reference range; marker = questioned signature."""
    W, H = _size(fig)
    if value is None or ref_min is None or ref_max is None:
        text(fig, x, y, "Not available", 8, MUTED, va="center")
        return
    lo, hi = min(ref_min, value), max(ref_max, value)
    span = (hi - lo) or max(abs(hi), 1.0) * 0.2
    dom_lo, dom_hi = lo - span * 0.35, hi + span * 0.35
    px = lambda v: x + w * ((v - dom_lo) / (dom_hi - dom_lo))
    th = 0.15 / H
    rect(fig, x, y - th / 2, w, th, SURFACE, BORDER, 0.8)
    rect(fig, px(ref_min), y - th / 2, max(px(ref_max) - px(ref_min), 0.004), th, BRAND_LIGHT, BRAND, 0.8)
    mx = px(value)
    vline(fig, mx, y - 0.14 / H, y + 0.14 / H, color, 3.0, z=4)
    ha = "right" if mx > x + w * 0.75 else ("left" if mx < x + w * 0.25 else "center")
    text(fig, mx, y + 0.19 / H, f"Questioned {fmt.format(value)}", 8, color, "bold", ha=ha, va="bottom")
    text(fig, x, y - 0.24 / H, f"Reference range {fmt.format(ref_min)} to {fmt.format(ref_max)}"
         + (f"   {caption}" if caption else ""), 7.5, MUTED, va="top")


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
    body = (f"The model measured a distance of {avg_distance:.4f} between the questioned signature and the "
            f"reference signatures. The decision threshold is {threshold:.4f}. "
            f"A distance above the threshold is read as a different writer.")
    para(fig, MX + 0.03, top - 0.108, body, CONTENT_W * W - 0.5, 9, TEXT)
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
    pct = 100.0 * avg_distance / threshold if threshold > 0 else 0.0
    para(fig, MX + 0.02, top - 0.86 / H,
         f"Distance is {pct:.0f}% of the threshold. Lower distance means more similar signatures.",
         2.3, 8.5, MUTED)
    distance_gauge(fig, 0.42, top - 0.72 / H, 0.47, avg_distance, threshold, vcol)
    y = top - ch - 0.22 / H

    # Supporting checks strip
    n_ok, n_eval = _support_stats(key)
    text(fig, MX, y, "Supporting checks", 11, INK, "bold", va="center")
    summary = (f"{n_ok} of {n_eval} supporting checks are consistent with the reference signatures."
               if n_eval else "Supporting checks are not available for this case.")
    text(fig, MX + 0.215, y, summary, 9, TEXT, va="center")
    y -= 0.36 / H
    sq = 0.30
    for i, code in enumerate(["f1", "f2", "f3", "f4", "f6", "f7"]):
        lab = key.get(f"{code}_label", "N/A")
        col = label_color(lab)
        x0 = MX + i * 0.50 / W
        rect(fig, x0, y - sq / H / 2, sq / W, sq / H, col, col, 0.8)
        text(fig, x0 + sq / W / 2, y, code.upper(), 8.5, WHITE, "bold", ha="center", va="center")
    text(fig, MX + 6 * 0.50 / W + 0.02, y, "Green = consistent    Amber = differs", 8, MUTED, va="center")
    y -= 0.32 / H

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
            y -= 0.62 / H + 0.34 / H
        else:
            y -= 0.05 / H

    # Key findings table
    text(fig, MX, y, "Key Findings", 11, INK, "bold", va="center")
    y -= 0.16 / H
    cols = {"code": MX + 0.010, "name": MX + 0.060, "chip": MX + 0.300, "what": MX + 0.505}
    hh = 0.30 / H
    rect(fig, MX, y - hh, CONTENT_W, hh, SURFACE, BORDER, 0.8)
    text(fig, cols["code"], y - hh / 2, "Code", 8, MUTED, "bold", va="center")
    text(fig, cols["name"], y - hh / 2, "Finding", 8, MUTED, "bold", va="center")
    text(fig, cols["chip"], y - hh / 2, "Result", 8, MUTED, "bold", va="center")
    text(fig, cols["what"], y - hh / 2, "What it checks", 8, MUTED, "bold", va="center")
    y -= hh
    rh = 0.50 / H
    for code in FINDING_ORDER:
        lab = key.get(f"{code}_label", "N/A")
        col = label_color(lab)
        yc = y - rh / 2
        text(fig, cols["code"], yc, code.upper(), 9.5, BRAND, "bold", va="center")
        text(fig, cols["name"], yc, FINDING_NAMES[code], 9.5, INK, "bold", va="center")
        chip(fig, cols["chip"], yc, lab, col, size=7.5, align="left")
        text(fig, cols["what"], yc, FINDING_SHORT[code], 8, MUTED, va="center")
        y -= rh
        hline(fig, MX, 1 - MX, y, BORDER, 0.8)
    text(fig, MX, y - 0.22 / H, "See the Findings pages for what each result means and how it was measured.",
         8, MUTED, va="center")

    _finish(fig, pdf, page_counter, total_pages, case_id)


# ── Findings cards ────────────────────────────────────────────────────────────
_MIN_CARD_IN = {"f1": 2.05, "f2": 1.65, "f3": 1.65, "f4": 1.65, "f5": 1.85, "f6": 1.85, "f7": 2.55}
_TEXT_COL_IN = 4.0


def _card_height_in(code: str, findings: dict) -> float:
    obs = str((findings.get("modal_observations", {}) or {}).get(f"{code}_observation", "Not available."))
    need = (0.50 + 0.14 + para_height_in(FINDING_EXPLAIN[code], _TEXT_COL_IN, 8.5) + 0.10 + 0.14
            + para_height_in(obs, _TEXT_COL_IN, 8.5) + 0.22)
    return max(need, _MIN_CARD_IN[code])


def _card(fig, top: float, height_in: float, code: str, findings: dict) -> float:
    """Draws one finding card; returns the y of the card bottom."""
    W, H = _size(fig)
    key = findings.get("key_findings", {})
    modal = findings.get("modal_observations", {})
    ranges = findings.get("ranges", {}) or {}
    label = key.get(f"{code}_label", "N/A")
    col = label_color(label)
    h = height_in / H
    rect(fig, MX, top - h, CONTENT_W, h, WHITE, BORDER, 1.0)

    text(fig, MX + 0.02, top - 0.22 / H, code.upper(), 12, BRAND, "bold", va="center")
    text(fig, MX + 0.055, top - 0.22 / H, FINDING_NAMES[code], 12, INK, "bold", va="center")
    chip(fig, 1 - MX - 0.015, top - 0.22 / H, label, col)
    hline(fig, MX + 0.015, 1 - MX - 0.015, top - 0.42 / H, BORDER, 0.8)

    tx = MX + 0.02
    tw = _TEXT_COL_IN
    y = top - 0.50 / H
    text(fig, tx, y, "WHAT THIS CHECKS", 7, MUTED, "bold", va="top")
    y = para(fig, tx, y - 0.14 / H, FINDING_EXPLAIN[code], tw, 8.5, TEXT) - 0.10 / H
    text(fig, tx, y, "WHAT WE FOUND", 7, MUTED, "bold", va="top")
    obs = modal.get(f"{code}_observation", "Not available for this case.")
    para(fig, tx, y - 0.14 / H, str(obs), tw, 8.5, INK)

    # Right-hand visual
    vx, vw = MX + 0.535, 0.29
    vy_mid = top - h * 0.55
    cap = "Blue band = writer's reference range"
    if code == "f1":
        counts = ranges.get("f1") or {}
        cy = top - 0.62 / H
        text(fig, vx, cy, "Feature", 7.5, MUTED, "bold", va="center")
        text(fig, vx + 0.115, cy, "Questioned", 7.5, MUTED, "bold", va="center")
        text(fig, vx + 0.20, cy, "References", 7.5, MUTED, "bold", va="center")
        names = [("strokes", "Strokes"), ("bowls", "Closed loops"), ("dots", "Dots"), ("pen_lifts", "Pen lifts")]
        for i, (k, nm) in enumerate(names):
            ry = cy - (i + 1) * 0.30 / H
            c = counts.get(k)
            hline(fig, vx, vx + vw, ry + 0.15 / H, BORDER, 0.6)
            text(fig, vx, ry, nm, 8.5, TEXT, va="center")
            if c:
                inside = (c["min"] - 1) <= c["q"] <= (c["max"] + 1)
                text(fig, vx + 0.115, ry, str(c["q"]), 9, OK if inside else WARN, "bold", va="center")
                rng = f"{c['min']}" if c["min"] == c["max"] else f"{c['min']} to {c['max']}"
                text(fig, vx + 0.20, ry, rng, 8.5, TEXT, va="center")
    elif code == "f2":
        r = ranges.get("f2") or {}
        range_bar(fig, vx, vy_mid, vw, r.get("q"), r.get("min"), r.get("max"), col, "{:.1f} deg")
        text(fig, vx, vy_mid - 0.62 / H, cap, 7.5, MUTED, va="top")
    elif code == "f3":
        r = ranges.get("f3") or {}
        range_bar(fig, vx, vy_mid, vw, r.get("q"), r.get("min"), r.get("max"), col, "{:.3f}")
        text(fig, vx, vy_mid - 0.62 / H, "Lower = smoother line, higher = more wobble", 7.5, MUTED, va="top")
    elif code == "f4":
        r = ranges.get("f4") or {}
        range_bar(fig, vx, vy_mid, vw, r.get("q"), r.get("min"), r.get("max"), col, "{:.2f}")
        text(fig, vx, vy_mid - 0.62 / H, "Ratio = width divided by height", 7.5, MUTED, va="top")
    elif code == "f5":
        r = ranges.get("f5") or {}
        if r:
            distance_gauge(fig, vx, vy_mid + 0.05 / H, vw, r["distance"], r["threshold"], col)
    elif code == "f6":
        r = ranges.get("f6") or {}
        range_bar(fig, vx, vy_mid, vw, r.get("q"), r.get("min"), r.get("max"), col, "{:.2f}")
        text(fig, vx, vy_mid - 0.62 / H, "Distance between signatures, lower = more alike", 7.5, MUTED, va="top")
    elif code == "f7":
        d = ranges.get("f7_darkness") or {}
        wv = ranges.get("f7_width") or {}
        text(fig, vx, top - 0.60 / H, "Ink darkness", 7.5, MUTED, "bold", va="center")
        range_bar(fig, vx, top - 0.98 / H, vw, d.get("q"), d.get("min"), d.get("max"), col, "{:.2f}")
        text(fig, vx, top - 1.58 / H, "Line-width variation", 7.5, MUTED, "bold", va="center")
        range_bar(fig, vx, top - 1.96 / H, vw, wv.get("q"), wv.get("min"), wv.get("max"), col, "{:.2f}")
    return top - h


def page_findings(pdf, page_counter, total_pages, case_id, findings: Optional[dict], part: int) -> None:
    """part 1 = F1-F4, part 2 = F5-F7 + notes."""
    W, H = PAGE_W, PAGE_H
    findings = findings or {}
    fig = _new_page(case_id, "Forensic Findings")
    codes = ["f1", "f2", "f3", "f4"] if part == 1 else ["f5", "f6", "f7"]
    y = page_title(fig, "Forensic Findings (F1-F7)" + ("" if part == 1 else "  continued"),
                   "Each finding explains what was checked, what was found, and how it compares with the references")
    for code in codes:
        y = _card(fig, y, _card_height_in(code, findings), code, findings) - 0.10 / H

    if part == 2:
        rect(fig, MX, y - 1.05 / H, CONTENT_W, 1.05 / H, SURFACE, BORDER, 0.8)
        text(fig, MX + 0.02, y - 0.20 / H, "How to read this page", 9, INK, "bold", va="center")
        notes = [
            "Blue band = the range seen in this writer's own reference signatures. The marker shows where the "
            "questioned signature falls. A marker inside the band means it looks like the writer's normal variation.",
            "F5 and F6 both use the model but measure different things. F5 compares the questioned signature with "
            "the combined (average) references. F6 compares it with each reference one by one, so the two "
            "distances can differ.",
        ]
        yy = y - 0.34 / H
        for n in notes:
            yy = para(fig, MX + 0.02, yy, n, CONTENT_W * W - 0.3, 8, TEXT, spacing=1.35) - 0.06 / H

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
        "measurements and a trained neural network. Several numeric cut-offs used to label the findings are "
        "engineering defaults and have not yet been validated against a licensed forensic document examiner's "
        "judgment on this dataset.",
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