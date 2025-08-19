# analyze.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import tempfile, os, math
import requests
import numpy as np

import librosa
import soundfile as sf

APP_VERSION = "locker-analyzer@1.0.1"  # bump when you change logic

app = FastAPI()


class AnalyzeIn(BaseModel):
    url: HttpUrl


@app.get("/health")
def health():
    return {"ok": True, "version": APP_VERSION}


# --------------------------- helpers ---------------------------

def _download_to_tmp(url: str) -> str:
    """Stream download to a tmp file to avoid loading into RAM."""
    fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(url)[-1] or ".audio")
    os.close(fd)

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    return tmp_path


def _safe_load(path: str):
    """
    Load audio to mono. Keep native sample rate for better BPM/key.
    Returns (y, sr, duration_s).
    """
    # mono=True reduces RAM and simplifies downstream analysis
    y, sr = librosa.load(path, sr=None, mono=True)
    duration_s = float(len(y)) / float(sr) if len(y) and sr else 0.0
    return y, sr, duration_s


def _bpm_with_confidence(y, sr):
    """
    Estimate BPM and a lightweight confidence.
    Confidence here blends:
      - clarity of the tempo peak from tempogram
      - proportion of frames flagged as beats (beat density)
    Both are crude but useful as 0..1 indicators.
    """
    if y is None or len(y) < 1024:
        return None, None, None, None

    hop = 512
    # Onset envelope (energy flux) → robust to genre
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.median)

    # Primary tempo estimate (median of multiple temps is stabler)
    # aggregate=None returns an array of tempo candidates; we take the median
    tempos = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop, aggregate=None)
    if tempos is None or len(tempos) == 0:
        return None, None, None, None

    bpm = float(np.median(tempos))

    # Beat tracking for density
    tempo_bt, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, units="frames")
    beat_density = 0.0
    if onset_env.size > 0:
        beat_density = min(1.0, len(beat_frames) / (0.5 + onset_env.size))  # simple normalization

    # Tempogram peak sharpness ~ clarity
    tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop)
    acf = np.mean(tg, axis=1) if tg.size else np.array([0.0])
    # normalize autocorrelation and pick the max at T>=2 frames to avoid DC
    if acf.size > 3:
        peak = float(np.max(acf[2:]))  # simple proxy of periodicity
        clarity = float(peak / (1e-9 + np.sum(acf)))
        clarity = max(0.0, min(1.0, clarity * 8.0))  # re-scale into 0..1-ish
    else:
        clarity = 0.0

    bpm_conf = float(np.clip(0.6 * clarity + 0.4 * beat_density, 0.0, 1.0))

    # derive alternates
    bpm_alt_half = float(bpm / 2.0)
    bpm_alt_double = float(bpm * 2.0)

    return bpm, bpm_conf, bpm_alt_half, bpm_alt_double


def _key_with_confidence(y, sr):
    """
    Estimate key using Krumhansl-Schmuckler style template matching on chroma_cqt.
    Returns (key_root, key_mode, key_confidence) or (None, None, None) on failure.
    """
    if y is None or len(y) < 2048:
        return None, None, None

    # Chroma CQT is usually more stable for key than STFT-chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    if chroma is None or chroma.size == 0:
        return None, None, None

    chroma_norm = chroma / (1e-9 + np.sum(chroma, axis=0, keepdims=True))
    chroma_mean = np.mean(chroma_norm, axis=1)

    # Krumhansl major/minor profiles (normalized)
    # These are common reference vectors; scaled/normalized here
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                    2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
    min = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                    2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)
    maj = maj / np.sum(maj)
    min = min / np.sum(min)

    # test all 12 rotations for major and minor
    scores = []
    for pitch_class in range(12):
        maj_rot = np.roll(maj, pitch_class)
        min_rot = np.roll(min, pitch_class)
        s_maj = float(np.dot(chroma_mean, maj_rot))
        s_min = float(np.dot(chroma_mean, min_rot))
        scores.append(("major", pitch_class, s_maj))
        scores.append(("minor", pitch_class, s_min))

    # pick best and second-best to form a confidence gap
    scores.sort(key=lambda x: x[2], reverse=True)
    best_mode, best_pc, best_score = scores[0]
    second_score = scores[1][2] if len(scores) > 1 else 0.0

    # confidence = softmax-like gap
    gap = max(0.0, best_score - second_score)
    denom = max(1e-6, best_score + second_score)
    key_conf = float(np.clip(gap / denom, 0.0, 1.0))

    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]
    key_root = NOTE_NAMES[int(best_pc)]
    key_mode = best_mode

    return key_root, key_mode, key_conf


def _tuning_cents(y, sr):
    try:
        bins = librosa.estimate_tuning(y=y, sr=sr)  # fractional bins
        return float(bins * 100.0)  # convert to cents
    except Exception:
        return None


# --------------------------- routes ---------------------------

@app.post("/analyze")
def analyze(inp: AnalyzeIn):
    tmp = None
    try:
        tmp = _download_to_tmp(inp.url)

        # Load
        y, sr, duration_s = _safe_load(tmp)
        if duration_s <= 0:
            raise HTTPException(status_code=400, detail="Could not decode audio")

        # Metrics
        bpm, bpm_conf, bpm_half, bpm_double = _bpm_with_confidence(y, sr)
        key_root, key_mode, key_conf = _key_with_confidence(y, sr)
        tuning_cents = _tuning_cents(y, sr)

        result = {
            "analyzer_version": APP_VERSION,
            "duration_s": float(duration_s) if duration_s else None,
            "sample_rate_hz": int(sr) if sr else None,

            "bpm": float(bpm) if bpm else None,
            "bpm_confidence": float(bpm_conf) if bpm_conf is not None else None,
            "bpm_alt_half": float(bpm_half) if bpm_half else None,
            "bpm_alt_double": float(bpm_double) if bpm_double else None,

            "key_root": key_root,
            "key_mode": key_mode,
            "key_confidence": float(key_conf) if key_conf is not None else None,

            "tuning_cents": float(tuning_cents) if tuning_cents is not None else None,

            # You can add energy/valence later—leaving placeholders for schema compatibility
            "energy": None,
            "valence": None,
        }

        return {"ok": True, "result": result}

    except requests.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"download_failed: {e}")
    except Exception as e:
        # Don’t leak internals; give a stable tag for the edge function
        raise HTTPException(status_code=400, detail=f"analyze_failed: {str(e)[:300]}")
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
# --- REQUIRED EXPORT FOR app.py ---
# app.py imports: from analyze import process_audio
# Make sure THIS exact function exists.

def process_audio(url: str, max_time: int | None = 180) -> dict:
    """
    Required entrypoint. Call your real analyzer from here.
    Replace `analyze_track` with whatever your actual function is named.
    Keep imports inside to avoid top-level import crashes at app startup.
    """
    # Import heavy libs lazily to avoid init-time crashes preventing app boot.
    # from librosa import load  # example: do heavy imports inside your real function

    # >>> CHANGE THIS LINE to call your real implementation <<<
    # Example if your function is named `analyze_track` in this same file:
    return analyze_track(url=url, max_time=max_time)

def process_audio(url: str, max_time: int = 180) -> dict:
    """
    Download the audio from `url`, run analysis, and return structured metadata.
    """
    # 1. Load audio (downsample for speed)
    import librosa
    import numpy as np
    import tempfile
    import requests
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(tmp.name, "wb") as f:
            for chunk in r.iter_content(1024 * 64):
                f.write(chunk)

        # Load mono, 22.05kHz for speed
        y, sr = librosa.load(tmp.name, sr=22050, mono=True, duration=max_time)

    # 2. BPM estimation
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm_confidence = float(len(beat_frames) / (len(y) / sr)) if len(beat_frames) > 0 else 0.0
    bpm_alt_half = tempo / 2.0
    bpm_alt_double = tempo * 2.0

    # 3. Key estimation
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key_root = int(np.argmax(chroma_mean))
    key_mode = "major" if chroma_mean[key_root] > np.median(chroma_mean) else "minor"
    key_confidence = float(chroma_mean[key_root] / chroma_mean.sum())

    # 4. Tuning (cents offset from equal temperament)
    tuning_cents = float(librosa.pitch_tuning(librosa.yin(y, fmin=50, fmax=2000, sr=sr)))

    # 5. Return structured dict
    return {
        "bpm": float(tempo),
        "bpm_confidence": bpm_confidence,
        "bpm_alt_half": float(bpm_alt_half),
        "bpm_alt_double": float(bpm_alt_double),
        "key_root": str(key_root),
        "key_mode": key_mode,
        "key_confidence": key_confidence,
        "tuning_cents": tuning_cents,
    }
