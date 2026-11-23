# Backend skeleton

Days 2-3 skeleton only. No application features yet.

## Run

    python -m venv .venv
    .venv/Scripts/Activate.ps1
    pip install -r requirements.txt
    uvicorn main:app --reload

## Check

Open http://127.0.0.1:8000/health

It returns `{"status": "ok"}`.

No diagnosis, advisor, auth, database or Gemini code yet.
