"""
ProcessResponse stays backward compatible with the .NET API.

Run from the project root:  python -m unittest discover -s tests
"""

import unittest

from app.models.response_models import ProcessResponse

BASE = dict(case_name="Case 001", verdict="GENUINE", confidence_genuine=60.0, confidence_forged=40.0,
            distance=0.55, threshold=0.6851, gradcam_blob_ids=[])


class ReviewRecommendedTest(unittest.TestCase):
    def test_defaults_to_false_when_omitted(self) -> None:
        self.assertFalse(ProcessResponse(**BASE).review_recommended)

    def test_borderline_keeps_genuine_verdict(self) -> None:
        r = ProcessResponse(**BASE, review_recommended=True)
        self.assertEqual(r.verdict, "GENUINE")
        self.assertTrue(r.model_dump()["review_recommended"])


if __name__ == "__main__":
    unittest.main()
