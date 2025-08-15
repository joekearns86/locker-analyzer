# analyze.py
import os
import gc
import math
import tempfile
import subprocess
from typing import Optional, Dict, Any

import numpy as np
import requests
import soundfile as sf
import librosa

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from threading import BoundedSemaphore

ANALYZER_VERSION = os.getenv("ANALYZER_VERSION", "locker-analyzer@1.0.0")

# Tuning knobs (safe defaults)
TARGET_SR = int(os.getenv("TARGET_SAMPLE_RATE", "22050"))     # lower SR = lower RAM/CPU
MAX_SECONDS = float(os.getenv("MAX_ANALYZE_SECONDS", "90"))    # analyze only first N seconds
CONCURRENCY = int(os.getenv("ANALYZER_CONCURRENCY", "2"))      # max concurrent analyses

# Simple concurrency limiter (process files sequentially or with small parallelism)
_sema = BoundedSemaphore(CONCURRENCY)

app = FastAPI()

# -----------------------
# Request / Response models
# -----------------------
class AnalyzeRequest(BaseModel):
    url: str


# -----------------------
# Helpers
# -----------------------
def _download_to_tmp(url: str) -> str:
    """Stream the file from URL to a temp file (no large buffer in memory)."""
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            suffix = os.path.splitext(url.split("?")[0])[-1]
            fd, path = tempfile.mkstemp(suffix=suffix or ".audio")
            with os.fdopen(fd, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        return path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"download_failed: {e}")


def _safe_duration_seconds(path: str) -> Optional[float]:
    """
    Get duration cheaply. soundfile.info reads header/metadata.
    If that fails, we leave None (fallback to librosa duration after partial load).
    """
    try:
        info = sf.info(path)
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    return None


def _analyze_core(path: str) -> Dict[str, Any]:
    """
    Memory-friendly analysis:
    - loads mono float32 at TARGET_SR
    - caps to first MAX_SECONDS
    """
    y, sr = librosa.load(
        path,
        sr=TARGET_SR,
        mono=True,
        dtype=np.float32,
        duration=MAX_SECONDS if MAX_SECONDS > 0 else None,
    )

    # If empty or failed
    if y is None or y.size == 0:
        raise HTTPException(status_code=400, detail="empty_audio")

    result: Dict[str, Any] = {
        "analyzer_version": ANALYZER_VERSION,
        "duration_s": None,
        "sample_rate_hz": int(sr),
        "bpm": None,
        "bpm_confidence": None,
        "bpm_alt_half": None,
        "bpm_alt_double": None,
        "time_signature": None,
        "key_root": None,
        "key_mode": None,
        "key_confidence": None,
        "tuning_cents": None,
        "has_vocals": None,
        "is_instrumental": None,
        "energy": None,
        "valence": None,
        "language_iso639_1": None,
        "genre_primary": None,
        "genre_secondary": None,
        "fingerprint_id": None,
        "confidence": None,
        "alt_half": None,
        "alt_double": None,
    }

    # Duration: prefer header duration; otherwise from current buffer length
    dur_header = _safe_duration_seconds(path)
    if dur_header is not None:
        result["duration_s"] = float(dur_header)
    else:
        result["duration_s"] = float(len(y)) / float(sr)

    # --- BPM (tempo) ---
    # Use a modest hop_length to keep memory down
    hop_length = 512
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        if tempo is not None and not np.isnan(tempo):
            tempo = float(tempo)
            result["bpm"] = tempo
            # very naive confidence from beat coverage
            result["bpm_confidence"] = float(min(1.0, len(beat_frames) / max(1.0, (len(y) / hop_length) / 2.0)))
            # alt suggestions
            result["bpm_alt_half"] = float(max(1.0, tempo / 2.0))
            result["bpm_alt_double"] = float(tempo * 2.0)
    except Exception:
        pass

    # --- Key estimation (lightweight heuristic) ---
    # Use chroma_cqt on short window (we already limited duration)
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        # crude major/minor detection via profile comparison
        # (kept light; you can swap in better models later)
        # Krumhansl profiles (normalized)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                  2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                  2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)
        major_profile /= major_profile.sum()
        minor_profile /= minor_profile.sum()

        # rotate profiles and score against chroma mean
        scores = []
        for root in range(12):
            score_maj = float((np.roll(major_profile, root) * chroma_mean).sum())
            score_min = float((np.roll(minor_profile, root) * chroma_mean).sum())
            scores.append(("major", root, score_maj))
            scores.append(("minor", root, score_min))
        # pick best
        best = max(scores, key=lambda t: t[2])
        mode, root, conf = best

        note_names = ["C", "C#", "D", "D#", "E", "F",
                      "F#", "G", "G#", "A", "A#", "B"]
        result["key_root"] = note_names[root]
        result["key_mode"] = mode
        # clamp to [0,1]
        result["key_confidence"] = float(max(0.0, min(1.0, conf / (chroma_mean.sum() + 1e-9))))
    except Exception:
        pass

    # --- Energy (simple RMS) ---
    try:
        rms = librosa.feature.rms(y=y).mean()
        # scale to 0..1-ish
        result["energy"] = float(max(0.0, min(1.0, rms * 5.0)))
    except Exception:
        pass

    # cleanup large arrays early
    del y
    gc.collect()

    return result


# -----------------------
# Routes
# -----------------------
@app.get("/health")
def health():
    return {"ok": True, "version": ANALYZER_VERSION}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    _sema.acquire()   # limit concurrency
    tmp_path = None
    try:
        tmp_path = _download_to_tmp(req.url)
        analysis = _analyze_core(tmp_path)

        # (Optional) normalize some fields/aliases you already use in your app
        out = {
            "analyzer_version": analysis.get("analyzer_version", ANALYZER_VERSION),
            "duration_s": analysis.get("duration_s"),
            "sample_rate_hz": analysis.get("sample_rate_hz"),
            "bpm": analysis.get("bpm"),
            "bpm_confidence": analysis.get("bpm_confidence"),
            "bpm_alt_half": analysis.get("bpm_alt_half"),
            "bpm_alt_double": analysis.get("bpm_alt_double"),
            "time_signature": analysis.get("time_signature"),
            "key_root": analysis.get("key_root"),
            "key_mode": analysis.get("key_mode"),
            "key_confidence": analysis.get("key_confidence"),
            "tuning_cents": analysis.get("tuning_cents"),
            "has_vocals": analysis.get("has_vocals"),
            "is_instrumental": analysis.get("is_instrumental"),
            "energy": analysis.get("energy"),
            "valence": analysis.get("valence"),
            "language_iso639_1": analysis.get("language_iso639_1"),
            "genre_primary": analysis.get("genre_primary"),
            "genre_secondary": analysis.get("genre_secondary"),
            "fingerprint_id": analysis.get("fingerprint_id"),
            "confidence": analysis.get("confidence"),
            "alt_half": analysis.get("alt_half"),
            "alt_double": analysis.get("alt_double"),
        }
        return out

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"analyze_failed: {e}")
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        _sema.release()
        gc.collect()
