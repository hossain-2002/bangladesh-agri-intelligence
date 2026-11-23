"""Bangladesh Agriculture Intelligence API.

Days 4-5 slice: photo upload + Gemini Bangla advisor reply.
No database, no auth, no storage of uploaded files.
Uploaded images are processed in memory only.
"""

import os

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="Bangladesh Agriculture Intelligence")

MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

NOT_CONFIGURED_BN = (
    "Gemini এখনো সেট করা হয়নি। "
    "দয়া করে GEMINI_API_KEY এবং GEMINI_MODEL_FLASH সেট করুন।"
)
NOT_CONFIGURED_EN = (
    "Gemini is not configured yet. "
    "Please set GEMINI_API_KEY and GEMINI_MODEL_FLASH."
)

PROMPT_BN = (
    "আপনি বাংলাদেশের ছোট চাষিদের জন্য একজন কৃষি উপদেষ্টা। "
    "ছবিটি দেখে সহজ বাংলায় বলুন: ফসলের স্বাস্থ্য কেমন মনে হচ্ছে, "
    "কী লক্ষণ দেখা যাচ্ছে, এবং পরবর্তী নিরাপদ পদক্ষেপ কী হতে পারে। "
    "কোনো রাসায়নিকের নাম বা মাত্রা নিজে থেকে বানাবেন না। "
    "নিশ্চিত না হলে তা পরিষ্কারভাবে বলুন এবং স্থানীয় কৃষি অফিসে "
    "যোগাযোগ করতে বলুন। উত্তর সংক্ষিপ্ত ও সহজ ভাষায় দিন।"
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a simple health status for uptime checks."""
    return {"status": "ok"}


def _gemini_settings() -> tuple[str, str]:
    """Return the Gemini key and Flash model, or empty strings."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("GEMINI_MODEL_FLASH", "").strip()
    if api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        api_key = ""
    if model == "YOUR_GEMINI_FLASH_MODEL_ID":
        model = ""
    return api_key, model


@app.post("/advisor/upload")
async def advisor_upload(file: UploadFile = File(...)) -> JSONResponse:
    """Take one crop photo and return a Bangla advisor reply.

    Returns 400 for a non-image file or a file over 5 MB.
    Returns 503 when Gemini is not configured, with no diagnosis.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return JSONResponse(
            status_code=400,
            content={
                "message_bn": "শুধু ছবি আপলোড করুন (JPEG, PNG বা WebP)।",
                "message_en": "Please upload an image only (JPEG, PNG or WebP).",
            },
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return JSONResponse(
            status_code=400,
            content={
                "message_bn": "ছবিটি ৫ মেগাবাইটের বেশি বড়। ছোট ছবি আপলোড করুন।",
                "message_en": "The image is larger than 5 MB. Upload a smaller image.",
            },
        )

    api_key, model = _gemini_settings()
    if not api_key or not model:
        return JSONResponse(
            status_code=503,
            content={
                "message_bn": NOT_CONFIGURED_BN,
                "message_en": NOT_CONFIGURED_EN,
            },
        )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=[
                PROMPT_BN,
                types.Part.from_bytes(
                    data=image_bytes, mime_type=file.content_type
                ),
            ],
        )
    except Exception:
        return JSONResponse(
            status_code=502,
            content={
                "message_bn": "উত্তর পাওয়া যায়নি। পরে আবার চেষ্টা করুন।",
                "message_en": "Could not get an answer. Please try again later.",
            },
        )

    reply = (response.text or "").strip()
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
