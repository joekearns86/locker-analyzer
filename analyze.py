# analyze.py
import os
import tempfile
import subprocess
import math
from typing import Optional, Dict, Any

import requests
import numpy as np
import librosa

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

app = FastAPI()

ANALYZER_VERSION = "locker-analyzer@1.0.0"


class AnalyzeRequest(BaseModel):
    url: HttpUrl


def _download_to_tmp(url: str) -> str:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    # save to /tmp with same extension if present
    suffix = os.path.splitext(url.split("?")[0])[-1] or ".audio"
    fd, path = tempfile.mkstemp(prefix="locker_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return path


def _ffprobe(path: str) -> Dict[str, Any]:
    """
    Return {'duration': float or None, 'sample_rate': int or None}
    """
    try:
        # duration
        dur_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=duration,sample_rate",
            "-of", "default=nw=1:nk=1", path
        ]
        out = subprocess.check_output(dur_cmd, stderr=subprocess.STDOUT).decode().strip().splitlines()
        # ffprobe sometimes prints sample_rate then duration (depends on codec). Parse robustly.
        duration = None
        sample_rate = None
        for line in out:
            if line.isdigit():
                sample_rate = int(line)
            else:
                try:
                    duration = float(line)
                except Exception:
                    pass
        return {"duration": duration, "sample_rate": sample_rate}
    except Exception:
        return {"duration": None, "sample_rate": None}


def _estimate_bpm_and_key(path: str, target_sr: Optional[int]) -> Dict[str, Any]:
    """
    Light-weight analysis with librosa. Returns dict with:
    bpm, bpm_confidence, key_root, key_mode, key_confidence, tuning_cents, time_signature
    """
    try:
        # Load a maximum of ~90 seconds to save CPU (mono)
        y, sr = librosa.load(path, sr=target_sr or 22050, mono=True, duration=90.0)

        # BPM
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="time")
        bpm = float(tempo) if np.isfinite(tempo) else None
        bpm_conf = float(min(1.0, len(beat_frames) / (0.5 * sr))) if bpm else None  # toy confidence

        # Key (very simple heuristic)
        # Compute chroma energy and correlate with major/minor templates.
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)  # shape (12,)

        # Major/minor templates (Krumhansl-Schmuckler key profiles)
        maj_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        min_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        # Try all 12 rotations and pick best correlation
        def best_key(chroma_vec, profile):
            scores = []
            for i in range(12):
                score = np.corrcoef(chroma_vec, np.roll(profile, i))[0, 1]
                scores.append(score)
            idx = int(np.argmax(scores))
            return idx, float(np.max(scores))

        root_maj, score_maj = best_key(chroma_mean, maj_profile)
        root_min, score_min = best_key(chroma_mean, min_profile)

        if score_maj >= score_min:
            key_root_idx = root_maj
            key_mode = "major"
            key_conf = score_maj
        else:
            key_root_idx = root_min
            key_mode = "minor"
            key_conf = score_min

        pitch_names = ["C", "C#", "D", "D#", "E", "F",
                       "F#", "G", "G#", "A", "A#", "B"]
        key_root = pitch_names[key_root_idx]

        # Tuning estimate (cents off A440); rough via librosa estimate_tuning
        try:
            tuning = float(librosa.pitch.estimate_tuning(y=y, sr=sr))
            tuning_cents = tuning * 100.0
        except Exception:
            tuning_cents = None

        return {
            "bpm": round(bpm, 1) if bpm else None,
            "bpm_confidence": round(bpm_conf, 2) if bpm_conf else None,
            "key_root": key_root,
            "key_mode": key_mode,
            "key_confidence": round(float(key_conf), 2) if np.isfinite(key_conf) else None,
            "tuning_cents": round(tuning_cents, 1) if tuning_cents is not None else None,
            "time_signature": "4/4",  # heuristic default
        }
    except Exception:
        return {
            "bpm": None, "bpm_confidence": None,
            "key_root": None, "key_mode": None,
            "key_confidence": None, "tuning_cents": None,
            "time_signature": None
        }


def _has_vocals_heuristic(path: str, target_sr: Optional[int]) -> Optional[bool]:
    """
    Extremely simple heuristic (do NOT rely on this for production):
    uses harmonic-percussive ratio and zero-crossing to guess vocals presence.
    """
    try:
        y, sr = librosa.load(path, sr=target_sr or 22050, mono=True, duration=60.0)
        y_h, y_p = librosa.effects.hpss(y)
        harm_ratio = np.mean(np.abs(y_h)) / (np.mean(np.abs(y_p)) + 1e-9)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        # vocals tend to have higher harmonic content and moderate ZCR
        return bool(harm_ratio > 1.2 and 0.03 < zcr < 0.2)
    except Exception:
        return None


@app.get("/health")
def health():
    return {"ok": True, "version": ANALYZER_VERSION}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    # 1) download
    path = None
    try:
        path = _download_to_tmp(str(req.url))
        meta = _ffprobe(path)
        duration = meta.get("duration")
        sample_rate = meta.get("sample_rate")

        # 2) features
        feat = _estimate_bpm_and_key(path, sample_rate)
        has_vocals = _has_vocals_heuristic(path, sample_rate)

        # 3) assemble payload using your table columns
        result = {
            "analyzer_version": ANALYZER_VERSION,
            "fingerprint_id": None,  # populate later when you add real fingerprinting
            "duration_s": round(duration, 2) if duration else None,
            "sample_rate_hz": sample_rate,

            "bpm": feat["bpm"],
            "bpm_confidence": feat["bpm_confidence"],
            "bpm_alt_half": round(feat["bpm"] / 2, 1) if feat["bpm"] else None,
            "bpm_alt_double": round(feat["bpm"] * 2, 1) if feat["bpm"] else None,

            "time_signature": feat["time_signature"],

            "key_root": feat["key_root"],
            "key_mode": feat["key_mode"],
            "key_confidence": feat["key_confidence"],
            "tuning_cents": feat["tuning_cents"],

            "has_vocals": has_vocals,
            "is_instrumental": (False if has_vocals is True else (True if has_vocals is False else None)),

            # leave these as None for now; you can add models to compute them later
            "energy": None,
            "valence": None,
        }

        return {"ok": True, "result": result}
    except requests.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"download_failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"analyze_failed: {e}")
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
