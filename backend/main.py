"""Bangladesh Agriculture Intelligence API.

Days 4-5 slice: photo upload + Bangla advisor reply.
Advisor now runs on OpenRouter (Gemma 4 31B, free) instead of Gemini.
No database, no auth, no storage of uploaded files.
Uploaded images are processed in memory only.
"""

import base64
import io
import os
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI(title="Bangladesh Agriculture Intelligence")

# Allow the local React dev server to call the API from the browser.
# Without this, the browser blocks cross-origin requests from
# localhost:5173 to 127.0.0.1:8000 even though the server is up.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl:free"

NOT_CONFIGURED_BN = (
    "OpenRouter এখনো সেট করা হয়নি। "
    "দয়া করে OPENROUTER_API_KEY সেট করুন।"
)
NOT_CONFIGURED_EN = (
    "OpenRouter is not configured yet. Please set OPENROUTER_API_KEY."
)

PROMPT_BN = (
    "আপনি বাংলাদেশের ছোট চাষিদের জন্য একজন কৃষি উপদেষ্টা। "
    "ছবিটি দেখে সহজ বাংলায় বলুন: ফসলের স্বাস্থ্য কেমন মনে হচ্ছে, "
    "কী লক্ষণ দেখা যাচ্ছে, এবং পরবর্তী নিরাপদ পদক্ষেপ কী হতে পারে। "
    "কোনো রাসায়নিকের নাম বা মাত্রা নিজে থেকে বানাবেন না। "
    "নিশ্চিত না হলে তা পরিষ্কারভাবে বলুন এবং স্থানীয় কৃষি অফিসে "
    "যোগাযোগ করতে বলুন। উত্তর সংক্ষিপ্ত ও সহজ ভাষায় দিন।"
)


MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "rice_disease_mobilenetv2_int8.tflite"

# Class order must match the training label order of the TFLite model.
DISEASES = {
    "bacterial_leaf_blight": {
        "en": "Bacterial Leaf Blight",
        "bn": "ব্যাকটেরিয়াল পাতা ঝলসানো রোগ",
        "advice_bn": (
            "পাতা হলুদ হয়ে শুকিয়ে যাওয়ার লক্ষণ দেখা যাচ্ছে। "
            "আক্রান্ত পাতা তুলে ফেলুন, অতিরিক্ত ইউরিয়া সার কমান "
            "এবং জমির পানা নিষ্কাশন ঠিক রাখুন। "
            "নিশ্চিত হতে উপজেলা কৃষি অফিসে যোগাযোগ করুন।"
        ),
    },
    "brown_spot": {
        "en": "Brown Spot",
        "bn": "ব্রাউন স্পট (বাদামি দাগ) রোগ",
        "advice_bn": (
            "পাতায় গোলাকার বাদামি দাগ দেখা যাচ্ছে। "
            "ক্ষতিগ্রস্ত পাতা সরিয়ে ফেলুন এবং জমিতে পর্যাপ্ত পানি ও "
            "সঠিক সার নিশ্চিত করুন। প্রয়োজনে উপজেলা কৃষি অফিসের পরামর্শ নিন।"
        ),
    },
    "leaf_blast": {
        "en": "Leaf Blast",
        "bn": "পাতা ছড়া (লিফ ব্লাস্ট) রোগ",
        "advice_bn": (
            "পাতায় চোখের মতো লম্বা দাগ দেখা যাচ্ছে। "
            "জমি যেন বেশি শুকিয়ে না যায় সেটা দেখুন এবং সারের ভারসাম্য রাখুন। "
            "ওষুধ প্রয়োগের আগে উপজেলা কৃষি অফিসের পরামর্শ নিন।"
        ),
    },
    "healthy": {
        "en": "Healthy",
        "bn": "সুস্থ",
        "advice_bn": (
            "পাতা সুস্থ দেখাচ্ছে। নিয়মিত জমি পরিদর্শন করুন "
            "এবং সঠিক সময়ে পানি ও সার দিন।"
        ),
    },
}
# Model output index -> disease key (training label order).
CLASS_ORDER = list(DISEASES)

_interpreter = None
_model_unavailable = False


class ModelUnavailable(Exception):
    """The TFLite model file or its runtime is not available."""


def _get_interpreter():
    """Load the TFLite model on first use, once per process."""
    global _interpreter, _model_unavailable
    if _model_unavailable:
        raise ModelUnavailable()
    if _interpreter is not None:
        return _interpreter
    if not MODEL_PATH.is_file():
        _model_unavailable = True
        raise ModelUnavailable()
    try:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter
        _interpreter = Interpreter(model_path=str(MODEL_PATH))
        _interpreter.allocate_tensors()
    except Exception as exc:
        print(f"TFLite model could not be loaded: {exc}")
        _model_unavailable = True
        raise ModelUnavailable()
    return _interpreter


def _preprocess(image: Image.Image, input_details: dict) -> np.ndarray:
    """Resize the photo and scale/quantize it for the model input."""
    _, height, width, _ = input_details[0]["shape"]
    image = image.resize((int(width), int(height)))
    pixels = np.asarray(image, dtype=np.float32)

    input_dtype = input_details[0]["dtype"]
    if input_dtype == np.float32:
        return pixels / 255.0
    # Quantized input: undo the same normalization the model was trained
    # with. A scale of 1/127.5 with zero point -128 means [-1, 1] inputs;
    # anything near 1/255 means [0, 1] inputs.
    scale, zero_point = input_details[0]["quantization"]
    if scale > 0 and abs(scale * 127.5 - 1.0) < 0.01:
        normalized = pixels / 127.5 - 1.0
    else:
        normalized = pixels / 255.0
    quantized = normalized / scale + zero_point
    if input_dtype == np.int8:
        return np.clip(quantized, -128, 127).astype(np.int8)
    return np.clip(quantized, 0, 255).astype(np.uint8)


def _run_inference(image_bytes: bytes) -> tuple[dict, float]:
    """Run the TFLite model and return (disease, confidence_percent)."""
    interpreter = _get_interpreter()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    interpreter.set_tensor(
        input_details[0]["index"], _preprocess(image, input_details)
    )
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]["index"]).flatten()

    scale, zero_point = output_details[0]["quantization"]
    if scale > 0:
        scores = (scores.astype(np.float32) - zero_point) * scale
    if scores.max() > 1.5:  # raw logits or counts -> make it a probability
        scores = scores / scores.sum()
    best = int(np.argmax(scores))
    return DISEASES[CLASS_ORDER[best]], round(float(scores[best]) * 100.0, 1)


def _image_error(file: UploadFile) -> dict | None:
    """Return a shared 400 error body for non-images or files over 5 MB."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return {
            "message_bn": "শুধু ছবি আপলোড করুন (JPEG, PNG বা WebP)।",
            "message_en": "Please upload an image only (JPEG, PNG or WebP).",
        }
    return None


@app.get("/health")
def health() -> dict[str, str]:
    """Return a simple health status for uptime checks."""
    return {"status": "ok"}


def _openrouter_settings() -> tuple[str, str]:
    """Return the OpenRouter key and model, or empty strings."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL).strip()
    if api_key == "PASTE_YOUR_OPENROUTER_API_KEY_HERE":
        api_key = ""
    return api_key, model


@app.post("/advisor/upload")
async def advisor_upload(file: UploadFile = File(...)) -> JSONResponse:
    """Take one crop photo and return a Bangla advisor reply.

    Returns 400 for a non-image file or a file over 5 MB.
    Returns 503 when OpenRouter is not configured, with no diagnosis.
    """
    error = _image_error(file)
    if error is not None:
        return JSONResponse(status_code=400, content=error)

    image_bytes = await file.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return JSONResponse(
            status_code=400,
            content={
                "message_bn": "ছবিটি ৫ মেগাবাইটের বেশি বড়। ছোট ছবি আপলোড করুন।",
                "message_en": "The image is larger than 5 MB. Upload a smaller image.",
            },
        )

    api_key, model = _openrouter_settings()
    if not api_key or not model:
        return JSONResponse(
            status_code=503,
            content={
                "message_bn": NOT_CONFIGURED_BN,
                "message_en": NOT_CONFIGURED_EN,
            },
        )

    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    image_url = f"data:{file.content_type};base64,{image_b64}"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": PROMPT_BN},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url},
                                },
                            ],
                        }
                    ],
                },
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        # Log the real OpenRouter error to the terminal so a 502 can be
        # diagnosed. Never log the API key itself.
        print(f"OpenRouter call failed: {exc}")
        return JSONResponse(
            status_code=502,
            content={
                "message_bn": "উত্তর পাওয়া যায়নি। পরে আবার চেষ্টা করুন।",
                "message_en": "Could not get an answer. Please try again later.",
                "detail": str(exc)[:300],
            },
        )

    try:
        reply = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError, TypeError):
        reply = ""
    if not reply:
        return JSONResponse(
            status_code=502,
            content={
                "message_bn": "উত্তর পাওয়া যায়নি। পরে আবার চেষ্টা করুন।",
                "message_en": "Could not get an answer. Please try again later.",
            },
        )

    return JSONResponse(
        status_code=200,
        content={"reply_bn": reply, "model": model},
    )


MODEL_MISSING_BN = (
    "অফলাইন রোগ নির্ণয়ের মডেল পাওয়া যায়নি। "
    "ml/rice_disease_mobilenetv2_int8.tflite ফাইলটি যোগ করুন।"
)
MODEL_MISSING_EN = (
    "The offline diagnosis model was not found. "
    "Add ml/rice_disease_mobilenetv2_int8.tflite."
)


@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)) -> JSONResponse:
    """Run the on-device TFLite rice disease model on one leaf photo.

    Returns disease name in English and Bangla, a confidence percentage
    and short Bangla advice. Returns 400 for a non-image or oversized
    file, and 503 when the model file is missing — never a fake result.
    """
    error = _image_error(file)
    if error is not None:
        return JSONResponse(status_code=400, content=error)

    image_bytes = await file.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return JSONResponse(
            status_code=400,
            content={
                "message_bn": "ছবিটি ৫ মেগাবাইটের বেশি বড়। ছোট ছবি আপলোড করুন।",
                "message_en": "The image is larger than 5 MB. Upload a smaller image.",
            },
        )

    try:
        disease, confidence = _run_inference(image_bytes)
    except ModelUnavailable:
        return JSONResponse(
            status_code=503,
            content={
                "message_bn": MODEL_MISSING_BN,
                "message_en": MODEL_MISSING_EN,
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "disease_en": disease["en"],
            "disease_bn": disease["bn"],
            "confidence_percent": confidence,
            "advice_bn": disease["advice_bn"],
        },
    )
