# Backend

Days 4-5 slice: photo upload + Gemini Bangla advisor reply.
No database, no auth, no storage of uploaded files.

## Run

    python -m venv .venv
    .venv/Scripts/Activate.ps1
    pip install -r requirements.txt
    uvicorn main:app --reload

## Gemini setup

Copy the root `.env.example` to `.env` and fill in your own values:

    GEMINI_API_KEY=your_key_here
    GEMINI_MODEL_FLASH=your_flash_model_id_here

Use the exact model ID shown in Google AI Studio. Do not guess one.
Never commit `.env`.

## Check

Open http://127.0.0.1:8000/health

It returns `{"status": "ok"}`.

## Advisor endpoint

POST /advisor/upload with a multipart image file (field name `file`).

- Accepts JPEG, PNG or WebP only, up to 5 MB.
- Wrong file type or a file over 5 MB returns 400.
- With GEMINI_API_KEY and GEMINI_MODEL_FLASH set, it returns the
  model's Bangla reply as `reply_bn`.
- Without the key or model set, it returns 503 with a message in
  Bangla and English saying Gemini is not configured. It does not
  invent a diagnosis, chemical name or dosage in that case.
