"""
Report text must fit the space it is laid out for.

Wrapping and block heights are measured with the real report font (Sora), so
a wrapped line never runs past its column and a paragraph's reserved height
matches what matplotlib draws.

Run from the project root:  python -m unittest discover -s tests
"""

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from app.services import report_pages as rp  # noqa: E402

SAMPLE = (
    "Shows how different the questioned signature is from the references according to the AI model, "
    "measured against the model's decision limit (the threshold). The line-wobble score is 0.123 for "
    "the questioned signature and 0.098 on average for the references."
)


class WrapFitsWidthTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dpi = 300
        self.fig = plt.figure(figsize=(rp.PAGE_W, rp.PAGE_H), dpi=self.dpi)
        self.renderer = self.fig.canvas.get_renderer()

    def tearDown(self) -> None:
        plt.close(self.fig)

    def test_wrapped_lines_fit(self) -> None:
        for size, weight in ((7.0, "normal"), (8.5, "normal"), (9.5, "bold")):
            for width_in in (2.0, 3.3, 6.5):
                for line in rp.wrap(SAMPLE, width_in, size, weight):
                    t = rp.text(self.fig, 0.1, 0.5, line, size, weight=weight)
                    drawn = t.get_window_extent(self.renderer).width / self.dpi
                    t.remove()
                    with self.subTest(size=size, weight=weight, width=width_in, line=line):
                        self.assertLessEqual(drawn, width_in)

    def test_paragraph_height_matches_drawing(self) -> None:
        for size, spacing in ((8.0, 1.35), (8.5, 1.35), (9.5, 1.5)):
            width_in = 3.0
            lines = rp.wrap(SAMPLE, width_in, size)
            t = rp.text(self.fig, 0.1, 0.9, "\n".join(lines), size, va="top", linespacing=spacing)
            drawn = t.get_window_extent(self.renderer).height / self.dpi
            t.remove()
            with self.subTest(size=size, spacing=spacing):
                self.assertAlmostEqual(rp.para_height_in(SAMPLE, width_in, size, spacing), drawn, delta=0.02)


if __name__ == "__main__":
    unittest.main()
