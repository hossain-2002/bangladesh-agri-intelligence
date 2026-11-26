"""API tests. Run from the backend/ folder:  python -m pytest test_main.py -v"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
from main import app

client = TestClient(app)


def _png_bytes(size=(8, 8)):
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, (100, 150, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_diagnose_rejects_non_image():
    response = client.post(
        "/diagnose", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400
    assert "message_bn" in response.json()


def test_diagnose_returns_503_when_model_missing(monkeypatch):
    monkeypatch.setattr(main, "MODEL_PATH", main.Path("/nonexistent/model.tflite"))
    monkeypatch.setattr(main, "_interpreter", None)
    monkeypatch.setattr(main, "_model_unavailable", False)
    response = client.post(
        "/diagnose", files={"file": ("leaf.png", _png_bytes(), "image/png")}
    )
    assert response.status_code == 503
    body = response.json()
    assert "message_bn" in body and "message_en" in body


class _FakeInterpreter:
    """Fake TFLite interpreter: 128x128 int8 input, int8 4-class output."""

    def get_input_details(self):
        return [
            {
                "index": 0,
                "shape": np.array([1, 128, 128, 3]),
                "dtype": np.int8,
                "quantization": (1.0 / 255.0, -128),
            }
        ]

    def get_output_details(self):
        return [
            {
                "index": 0,
                "dtype": np.int8,
                "quantization": (1.0 / 255.0, -128),
            }
        ]

    def allocate_tensors(self):
        pass

    def set_tensor(self, index, value):
        self._input = value

    def invoke(self):
        assert self._input.shape == (128, 128, 3)
        assert self._input.dtype == np.int8
        self._output = np.array([10, 90, 30, 40], dtype=np.int8)

    def get_tensor(self, index):
        return self._output.reshape(1, 4)


def test_diagnose_success(monkeypatch):
    fake = _FakeInterpreter()
    monkeypatch.setattr(main, "_interpreter", fake)
    monkeypatch.setattr(main, "_model_unavailable", False)
    response = client.post(
        "/diagnose", files={"file": ("leaf.png", _png_bytes(), "image/png")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["disease_en"] == "Brown Spot"
    assert body["disease_bn"] == "ব্রাউন স্পট (বাদামি দাগ) রোগ"
    assert body["confidence_percent"] == pytest.approx((90 + 128) / 255 * 100, abs=0.1)
    assert body["advice_bn"]


def test_advisor_upload_without_key_is_503(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    response = client.post(
        "/advisor/upload", files={"file": ("leaf.png", _png_bytes(), "image/png")}
    )
    assert response.status_code == 503
