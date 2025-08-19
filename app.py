import os
import urllib.parse
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from analyze import process_audio

APP_VERSION = os.getenv("ANALYZER_VERSION", "locker-analyzer@1.1.0")
app = FastAPI(title="Locker Analyzer", version=APP_VERSION)

# Optional: only allow signed URLs from your Supabase Storage endpoint(s)
# Provide a comma-separated list in ALLOWED_HOSTS, e.g.
# "jgbwsytvtnidtmwpfpdo.storage.supabase.co,dl.supabase.co"
ALLOWED_HOSTS = [
    h.strip() for h in os.getenv("ALLOWED_HOSTS", "").split(",") if h.strip()
]

class AnalyzeIn(BaseModel):
    url: str
    max_time: Optional[int] = 180  # hard cap so we don't load full albums by mistake

def _validate_url(url: str):
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid_url")

    if parsed.scheme not in ("https",):
        raise HTTPException(status_code=400, detail="only_https_allowed")

    if ALLOWED_HOSTS:
        host = parsed.netloc.lower()
        if host not in ALLOWED_HOSTS:
            raise HTTPException(status_code=400, detail=f"host_not_allowed:{host}")

@app.get("/health")
def health():
    return {"ok": True, "version": APP_VERSION}

@app.post("/analyze")
def analyze(inp: AnalyzeIn):
    _validate_url(inp.url)
    try:
        result = process_audio(inp.url, max_time=inp.max_time or 180)
        # Return exactly what Supabase Edge Function expects
        return result
    except HTTPException:
        raise
    except Exception as e:
        # surface a clear error yet not leak internals
        raise HTTPException(status_code=400, detail=f"analyze_failed:{type(e).__name__}")
