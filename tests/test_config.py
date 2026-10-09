"""
Settings tests.

Run from the project root:  python -m unittest discover -s tests
"""

import os
import unittest
from unittest import mock

from app.core.config import Settings


def _env_without(*names: str) -> dict:
    return {k: v for k, v in os.environ.items() if k not in names}


class SettingsTest(unittest.TestCase):
    def test_settings_allows_missing_azure_storage_config(self):
        env = _env_without("AZURE_STORAGE_ACCOUNT_URL", "AZURE_STORAGE_CONTAINER")
        with mock.patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.AZURE_STORAGE_ACCOUNT_URL, "")
        self.assertEqual(settings.AZURE_STORAGE_CONTAINER, "")

    def test_inference_threshold_default_is_pipeline32_eer(self):
        env = _env_without("INFERENCE_THRESHOLD")
        with mock.patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.INFERENCE_THRESHOLD, 0.6851)


if __name__ == "__main__":
    unittest.main()
