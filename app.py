# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging

# Import the analyzer entrypoint. This MUST exist in analyze.py.
# If your analyzer uses a different function name, either:
#   1) rename it to process_audio in analyze.py, OR
#   2) add a tiny wrapper in analyze.py:
#        def process_audio(url: str, max_time: int | None = 180) -> dict:
#            return analyze_track(url, max_time=max_time)
try:
    from analyze import process_audio
except Exception as e:
    # Fail fast with a clear message in Render logs if the function is missing/renamed.
    raise RuntimeError(
        "Failed to import 'process_audio' from analyze.py. "
        "Ensure analyze.py defines: process_audio(url: str, max_time: int | None = 180) -> dict"
    ) from e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("locker-analyzer")

APP_VERSION = os.getenv("ANALYZER_VERSION", "locker-analyzer@1.1.0")

app = FastAPI(title="Locker Analyzer", version=APP_VERSION)


class AnalyzeIn(BaseModel):
    url: str
    # Optional cap on how many seconds of audio to scan, defaults to 180 if omitted.
    max_time: int | None = 180


@app.get("/health")
def health():
    return {"ok": True, "version": APP_VERSION}


@app.post("/analyze")
def analyze(inp: AnalyzeIn):
    """
    Body:
    {
      "url": "<public or signed URL to audio>",
      "max_time": 180  // optional
    }
    """
    try:
        # Call the analyzer. It should return a plain dict (JSON-serializable).
        result = process_audio(inp.url, max_time=inp.max_time)
        # If you prefer a fixed envelope, uncomment next line and return that instead.
        # return {"ok": True, "result": result}
        return result
    except HTTPException:
        # Let explicit HTTP errors bubble up as-is.
        raise
    except Exception as e:
        # Surface a clean 400 to clients and Render logs while keeping the real error message.
        logger.exception("analyze_failed: %s", e)
        raise HTTPException(status_code=400, detail=f"analyze_failed: {e}")
