"""Bangladesh Agriculture Intelligence API.

Days 4-5 slice: photo upload + Bangla advisor reply.
Advisor now runs on OpenRouter (Gemma 4 31B, free) instead of Gemini.
No database, no auth, no storage of uploaded files.
Uploaded images are processed in memory only.
"""

import base64
import os

import httpx
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
DEFAULT_MODEL = "google/gemma-4-31b-it:free"

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
