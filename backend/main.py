"""Bangladesh Agriculture Intelligence API skeleton.

Days 2-3 skeleton only. No diagnosis, advisor, auth,
database or Gemini code yet.
"""

from fastapi import FastAPI

app = FastAPI(title="Bangladesh Agriculture Intelligence")


@app.get("/health")
def health() -> dict[str, str]:
    """Return a simple health status for uptime checks."""
    return {"status": "ok"}
