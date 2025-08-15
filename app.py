from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from analyze import process_audio
import os

from fastapi.middleware.cors import CORSMiddleware

class AnalyzeReq(BaseModel):
    url: HttpUrl

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten later to your app’s domain(s)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "version": os.getenv("ANALYZER_VERSION", "locker-analyzer@dev")}

@app.post("/analyze")
def analyze(body: AnalyzeReq):
    try:
        result = process_audio(str(body.url))
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"analyze_failed: {e}")
