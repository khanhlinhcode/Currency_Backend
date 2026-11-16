"""The public staging demo must not create private prediction records."""

import asyncio
import io
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image
from starlette.datastructures import Headers
from starlette.requests import Request

from api.inference import CLASS_NAMES


class FakeModel:
    def predict(self, _tensor, verbose=0):
        probabilities = np.zeros((1, len(CLASS_NAMES)), dtype=np.float32)
        probabilities[0, 0] = 1.0
        return probabilities


with patch("api.runtime_release.load_model", return_value=FakeModel()):
    from api import main


class NoDatabaseWrites:
    def __getattr__(self, name):
        raise AssertionError(f"Anonymous demo unexpectedly used database: {name}")


class PublicDemoTest(unittest.TestCase):
    def setUp(self):
        main.demo_request_times.clear()

    def tearDown(self):
        main.demo_request_times.clear()

    def test_guest_prediction_returns_result_without_saving_image_or_history(self):
        image_bytes = io.BytesIO()
        Image.new("RGB", (16, 16), "red").save(image_bytes, format="JPEG")
        upload = UploadFile(file=io.BytesIO(image_bytes.getvalue()), filename="sample.jpg",
                            headers=Headers({"content-type": "image/jpeg"}))
        request = Request({"type": "http", "method": "POST", "path": "/predict", "headers": []})
        with patch.object(main, "ENVIRONMENT", "staging"), \
             patch.object(main.runtime_release, "snapshot", return_value=(FakeModel(), {"version": "demo"})), \
             patch.object(main, "save_image") as save_image:
            result = asyncio.run(main.predict(request, upload, None, NoDatabaseWrites()))
        self.assertEqual(result["prediction"], CLASS_NAMES[0])
        self.assertEqual(result["model_version"], "demo")
        self.assertGreaterEqual(result["processing_time"], 0)
        save_image.assert_not_called()

    def test_guest_rate_limit_returns_retry_after(self):
        with patch.object(main.time, "monotonic", return_value=100.0):
            for _ in range(main.DEMO_REQUEST_LIMIT):
                main.check_demo_rate_limit()
            with self.assertRaises(HTTPException) as blocked:
                main.check_demo_rate_limit()
        self.assertEqual(blocked.exception.status_code, 429)
        self.assertEqual(blocked.exception.headers["Retry-After"], "60")
        with patch.object(main.time, "monotonic", return_value=161.0):
            main.check_demo_rate_limit()


if __name__ == "__main__":
    unittest.main()
