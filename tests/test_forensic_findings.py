"""
F1-F7 presentation data must agree with the labels.

The summary page draws every check on one "distance from normal" scale where
1.0 is the edge of normal, and colours range-bar markers by the allowed zone.
Both only make sense if deviation <= 1 exactly when a check is Consistent.

Run from the project root:  python -m unittest discover -s tests
"""

import unittest

import cv2
import numpy as np

from app.services.forensic_findings_service import compute_forensic_findings

CONSISTENT = {"Consistent", "Within natural range"}


def _signature(seed: int, stretch: float = 1.0, slope: float = 0.0, width: int = 3,
               ink: int = 0, extra_strokes: int = 0, jitter: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((224, 224), 255, dtype=np.uint8)
    xs = np.linspace(20, 20 + 180 * stretch, 9)
    ys = 112 - slope * (xs - 112) + rng.normal(0, 6, len(xs))
    pts = np.stack([xs, ys], axis=1)
    if jitter:
        dense = np.linspace(0, len(pts) - 1, 120)
        pts = np.stack([np.interp(dense, np.arange(len(pts)), pts[:, i]) for i in (0, 1)], axis=1)
        pts += rng.normal(0, jitter, pts.shape)
    cv2.polylines(img, [pts.astype(np.int32)], False, ink, width, cv2.LINE_AA)
    cv2.ellipse(img, (70, 100), (16, 11), 0, 0, 360, ink, width - 1, cv2.LINE_AA)
    for k in range(extra_strokes):
        cv2.line(img, (40 + 30 * k, 50), (55 + 30 * k, 45), ink, width - 1, cv2.LINE_AA)
    return img


class DeviationMatchesLabelTest(unittest.TestCase):
    def test_deviation_agrees_with_label(self) -> None:
        refs = [_signature(s) for s in range(4)]
        queries = {
            "genuine-like": _signature(10),
            "narrow": _signature(11, stretch=0.55),
            "sloped": _signature(12, slope=0.35),
            "thick-faint": _signature(13, width=6, ink=110),
            "extra strokes": _signature(14, extra_strokes=4),
            "shaky": _signature(15, jitter=2.0),
        }
        for name, q in queries.items():
            f = compute_forensic_findings(refs, q, None, None, distance=0.5, threshold=0.6851)
            labels, dev = f["key_findings"], f["deviation"]
            for code in ("f1", "f2", "f3", "f4", "f7"):
                with self.subTest(query=name, check=code):
                    self.assertEqual(labels[f"{code}_label"] in CONSISTENT, dev[code] <= 1.0 + 1e-9,
                                     f"label={labels[f'{code}_label']} deviation={dev[code]:.3f}")
            self.assertTrue(f["plain"]["f1"])

    def test_allowed_zone_contains_value_when_consistent(self) -> None:
        refs = [_signature(s) for s in range(4)]
        f = compute_forensic_findings(refs, _signature(11, stretch=0.55), None, None, 0.5, 0.6851)
        for code in ("f2", "f3", "f4"):
            r = f["ranges"][code]
            inside = r["ok_lo"] <= r["q"] <= r["ok_hi"]
            with self.subTest(check=code):
                self.assertEqual(inside, f["key_findings"][f"{code}_label"] == "Consistent")


class F5MatchesVerdictTest(unittest.TestCase):
    """F5 reads as within the threshold exactly when the verdict is GENUINE (distance < threshold)."""

    def test_f5_bands(self) -> None:
        refs = [_signature(s) for s in range(4)]
        q = _signature(10)
        t = 0.6851
        cases = {
            0.50 * t: "Well within threshold",
            0.90 * t: "Within threshold",
            t: "Exceeds threshold",            # verdict is FORGED at distance == threshold
            1.40 * t: "Exceeds threshold",
            1.60 * t: "Far exceeds threshold",
        }
        for distance, expected in cases.items():
            with self.subTest(distance=distance):
                f = compute_forensic_findings(refs, q, None, None, distance=distance, threshold=t)
                self.assertEqual(f["key_findings"]["f5_label"], expected)
                close = "close to the references" in f["plain"]["f5"]
                self.assertEqual(close, distance < t)


if __name__ == "__main__":
    unittest.main()
