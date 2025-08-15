# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# This must match the function name defined in analyze.py
from analyze import process_audio

APP_VERSION = os.getenv("ANALYZER_VERSION", "locker-analyzer@1.0.0")

app = FastAPI(title="Locker Analyzer", version=APP_VERSION)


class AnalyzeIn(BaseModel):
    url: str
    max_time: int | None = 180  # seconds cap for analysis


@app.get("/health")
def health():
    return {"ok": True, "version": APP_VERSION}


@app.post("/analyze")
def analyze(inp: AnalyzeIn):
    try:
        result = process_audio(inp.url, max_time=inp.max_time or 180)
        # You can wrap the result if you want a standard envelope:
        # return {"ok": True, "result": result}
        return result
    except Exception as e:
        # Surface a clear error to clients & Render logs
        raise HTTPException(status_code=400, detail=f"analyze_failed: {e}")
