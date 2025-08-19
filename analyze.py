"""
Library-only analyzer.
- Streams the remote file to a temp path (no in-memory copy)
- Downsamples to mono, 22050 Hz for robust tempo/key estimates
- Returns a dict with BPM, confidence, alt tempos, key, key confidence,
  tuning cents, energy, valence, basic instrumentation/vocal presence

No FastAPI here. app.py imports process_audio() from this file.
"""

import os
import tempfile
import math
import requests
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple, Dict

APP_VERSION = os.getenv("ANALYZER_VERSION", "locker-analyzer@1.1.0")

# -----------------------------
# Helpers (secure streaming)
# -----------------------------

def _download_to_tmp(url: str) -> str:
    """Stream the remote file to a tmp path to avoid RAM blowups."""
    fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(url.split("?")[0])[1] or ".audio")
    os.close(fd)

    with requests.get(url, stream=True, timeout=(10, 60)) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

    return tmp_path

def _load_mono_22k(path: str, max_time: int) -> Tuple[np.ndarray, float, float]:
    """
    Load to mono, sr=22050. Hard duration cap for speed / stability.
    Returns (y, sr, duration_s_file).
    """
    info = sf.info(path)
    duration_s = float(info.frames) / float(info.samplerate or 1.0)

    # Respect max cap but never exceed file duration
    cap = min(float(max_time or 180), duration_s)

    y, sr = librosa.load(path, sr=22050, mono=True, duration=cap)
    return y, float(sr), duration_s

# -----------------------------
# Tempo
# -----------------------------

def _estimate_tempo(y: np.ndarray, sr: float) -> Tuple[float, float, float, float]:
    """
    BPM with confidence and alt tempos.
    - uses beat tracker + onsets
    - confidence is heuristic (peak/second-peak)
    """
    # Onset strength envelope for beat tracking
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean)

    # Global tempo estimation
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    bpm = float(tempo)

    # Also consider tempogram peak ratio for a crude confidence
    # (high single peak -> high confidence)
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    ac_global = np.mean(tempogram, axis=1)  # autocorrelation-like
    peak_idx = np.argmax(ac_global)
    top_val = float(ac_global[peak_idx]) if ac_global.size > 0 else 0.0
    second_val = float(np.partition(ac_global, -2)[-2]) if ac_global.size > 1 else 0.0
    bpm_confidence = (top_val / (second_val + 1e-9)) if top_val > 0 else 0.0
    bpm_confidence = max(0.0, min(bpm_confidence, 5.0))  # cap range

    # alt tempos
    bpm_half = bpm / 2.0 if bpm > 0 else None
    bpm_double = bpm * 2.0 if bpm > 0 else None

    return bpm, bpm_confidence, bpm_half, bpm_double

# -----------------------------
# Key & tuning
# -----------------------------

_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def _estimate_key(y: np.ndarray, sr: float) -> Tuple[str, str, float, float]:
    """
    Returns (key_root, key_mode, key_confidence, tuning_cents).
    Uses Krumhansl-Schmuckler tonic profiles on chroma-cqt.
    Tuning estimated via librosa.estimate_tuning in cents.
    """
    # tuning
    try:
        tuning = librosa.estimate_tuning(y=y, sr=sr)
        tuning_cents = float(tuning * 100.0)
    except Exception:
        tuning_cents = 0.0

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma = np.mean(chroma, axis=1)  # average over time
    chroma = chroma / (np.linalg.norm(chroma) + 1e-9)

    scores = []
    for mode_name, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
        prof = profile / (np.linalg.norm(profile) + 1e-9)
        # correlate with all 12 rotations
        mode_scores = []
        for i in range(12):
            rotated = np.roll(prof, i)
            mode_scores.append(float(np.dot(chroma, rotated)))
        scores.append((mode_name, np.array(mode_scores)))

    # Get best across modes and rotations
    mode_best, vals = max(scores, key=lambda t: float(np.max(t[1])))
    idx = int(np.argmax(vals))
    best = float(vals[idx])

    # second best for confidence
    flat_vals = np.concatenate([s[1] for s in scores], axis=0)
    sorted_vals = np.sort(flat_vals)
    second = float(sorted_vals[-2]) if sorted_vals.size >= 2 else 1e-9
    key_confidence = best / (second + 1e-9)
    key_confidence = max(0.0, min(key_confidence, 5.0))

    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]
    key_root = NOTE_NAMES[idx]
    key_mode = mode_best
    return key_root, key_mode, key_confidence, tuning_cents

# -----------------------------
# Energy / Valence / Vocals / Instruments (light heuristics)
# -----------------------------

def _estimate_energy_valence(y: np.ndarray, sr: float) -> Tuple[float, float]:
    """
    Energy ~ RMS (normalized), Valence ~ spectral centroid proxy (normalized).
    """
    rms = librosa.feature.rms(y=y).mean()
    energy = float(np.clip(rms * 5.0, 0.0, 1.0))  # scale to 0..1-ish

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    max_centroid = sr / 2.0
    valence = float(np.clip(centroid / (max_centroid + 1e-9), 0.0, 1.0))
    return energy, valence

def _estimate_vocals_and_instruments(y: np.ndarray, sr: float) -> Dict[str, object]:
    """
    Very rough heuristics:
      - HPSS to separate harmonic/percussive
      - If harmonic fraction is high and ZCR matches typical vocal ranges -> has_vocals
      - Instruments guessed by spectral rolloff / percussive ratio
    """
    H, P = librosa.effects.hpss(y)
    harmonic_ratio = float(np.mean(np.abs(H)) / (np.mean(np.abs(y)) + 1e-9))
    percussive_ratio = float(np.mean(np.abs(P)) / (np.mean(np.abs(y)) + 1e-9))

    zcr = librosa.feature.zero_crossing_rate(y=y).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean() / (sr / 2.0)

    # crude vocal detection
    vocal_presence_conf = float(np.clip(harmonic_ratio * (0.6 + 0.8 * (0.05 < zcr < 0.2)), 0.0, 1.0))
    has_vocals = vocal_presence_conf > 0.45
    is_instrumental = not has_vocals

    # instruments (very rough)
    instruments_primary = "vocals, guitar, drums" if has_vocals else "guitar, drums"
    instruments_secondary = []
    if rolloff > 0.7:
        instruments_secondary.append("synth")
    if percussive_ratio > 0.45:
        instruments_secondary.append("bass")

    return {
        "has_vocals": bool(has_vocals),
        "is_instrumental": bool(is_instrumental),
        "vocal_presence_confidence": float(vocal_presence_conf),
        "instruments_primary": instruments_primary,
        "instruments_secondary": instruments_secondary or None,
    }

def _heuristic_genre_moods(bpm: float, energy: float, valence: float) -> Tuple[str, str, list]:
    """
    Simple rule-based guesses to give you *something* structured without an ML model.
    You can replace these later with a classifier if desired.
    """
    genre = "pop"
    if bpm and bpm > 140 and energy > 0.5:
        genre = "electronic"
    elif bpm and bpm < 90 and energy < 0.4:
        genre = "ambient"

    moods = []
    if energy > 0.65:
        moods.append("energetic")
    elif energy < 0.35:
        moods.append("chill")

    if valence > 0.6:
        moods.append("uplifting")
    elif valence < 0.4:
        moods.append("moody")

    secondary = None
    if genre == "pop" and "energetic" in moods:
        secondary = "dance"

    return genre, secondary, moods

# -----------------------------
# Main entry
# -----------------------------

def process_audio(url: str, max_time: int = 180) -> dict:
    """
    Main entry point called by app.py.
    Returns dict ready to PATCH into public.tracks.
    """
    tmp = None
    try:
        tmp = _download_to_tmp(url)
        y, sr, file_duration_s = _load_mono_22k(tmp, max_time=max_time)

        bpm, bpm_conf, bpm_half, bpm_double = _estimate_tempo(y, sr)
        key_root, key_mode, key_conf, tuning_cents = _estimate_key(y, sr)
        energy, valence = _estimate_energy_valence(y, sr)
        inst = _estimate_vocals_and_instruments(y, sr)
        genre, genre2, moods = _heuristic_genre_moods(bpm, energy, valence)

        result = {
            "analyzer_version": APP_VERSION,
            "duration_s": float(min(file_duration_s, float(max_time or 180))),
            "sample_rate_hz": int(sr),

            "bpm": float(bpm) if bpm else None,
            "bpm_confidence": float(bpm_conf) if bpm_conf is not None else None,
            "bpm_alt_half": float(bpm_half) if bpm_half else None,
            "bpm_alt_double": float(bpm_double) if bpm_double else None,

            "key_root": key_root,
            "key_mode": key_mode,
            "key_confidence": float(key_conf),
            "tuning_cents": float(tuning_cents),

            "energy": float(energy),
            "valence": float(valence),

            "genre_primary": genre,
            "genre_secondary": genre2,
            "moods": moods,

            "has_vocals": inst["has_vocals"],
            "is_instrumental": inst["is_instrumental"],
            "vocal_presence_confidence": inst["vocal_presence_confidence"],
            "instruments_primary": inst["instruments_primary"],
            "instruments_secondary": inst["instruments_secondary"],

            # Keep placeholders you already use in your app. Fill later if needed.
            "dominant_elements": None,
            "structure_highlight": None,
            "ar_summary": "",
            "tags_freeform": None,

            # bookkeeping (your Edge Function sets these final 2 fields)
            # "analyze_status": "ok",
            # "analyzed_at": "...",
        }

        return result
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
