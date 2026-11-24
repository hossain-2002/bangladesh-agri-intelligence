# Backend

Days 4-5 slice: photo upload + Bangla advisor reply.
Advisor runs on OpenRouter (Gemma 4 31B, free) instead of Gemini.
No database, no auth, no storage of uploaded files.

## Run

    python -m venv .venv
    .venv/Scripts/Activate.ps1
    pip install -r requirements.txt
    uvicorn main:app --reload

## OpenRouter setup

Get a free key at https://openrouter.ai/keys (sign in, Create API Key).
Set it in the terminal session before starting uvicorn:

    $env:OPENROUTER_API_KEY="your_key_here"

The model defaults to `google/gemma-4-31b-it:free`. To use a different
free OpenRouter vision model, set:

    $env:OPENROUTER_MODEL="google/gemma-4-31b-it:free"

Never share your key in chat or commit it. OpenRouter free models
allow 50 requests/day.

## Check

Open http://127.0.0.1:8000/health

It returns `{"status": "ok"}`.

## Advisor endpoint

POST /advisor/upload with a multipart image file (field name `file`).

- Accepts JPEG, PNG or WebP only, up to 5 MB.
- Wrong file type or a file over 5 MB returns 400.
- With OPENROUTER_API_KEY set, it returns the
  model's Bangla reply as `reply_bn`.
- Without the key set, it returns 503 with a message in
  Bangla and English saying OpenRouter is not configured. It does not
  invent a diagnosis, chemical name or dosage in that case.
