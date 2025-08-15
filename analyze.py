# analyze.py
from __future__ import annotations

import os
import math
import tempfile
import contextlib
from typing import Dict, Any, Tuple

import numpy as np
import requests
import librosa


ANALYZER_VERSION = "locker-analyzer@1.0.0"


def _download_to_tmp(url: str) -> str:
    """
    Stream-download a remote file to a temporary path and return the path.
    Raises on HTTP errors.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as f:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1024 * 256):  # 256 KB
                if chunk:
                    f.write(chunk)
        return f.name


def _load_mono(path: str, max_time: int = 60, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load audio as mono at target_sr. Limit to max_time seconds to keep CPU/memory in check.
    """
    # librosa.load supports duration cap without decoding the entire file
    y, sr = librosa.load(path, sr=target_sr, mono=True, duration=max_time)
    if y is None or y.size == 0:
        raise ValueError("Empty audio after load")
    return y, sr


def _estimate_bpm(y: np.ndarray, sr: int) -> Tuple[float, float, float, float]:
    """
    Estimate BPM with a simple approach:
    - onset strength -> tempo
    Returns (bpm, conf, half, double)
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    if onset_env.size == 0 or np.all(onset_env == 0):
        return 0.0, 0.0, None, None

    # Use median-aggregated onset env for robustness
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=np.median)
    bpm = float(tempo[0]) if np.ndim(tempo) else float(tempo)

    # crude confidence proxy: normalized variance of onset strength
    var = float(np.var(onset_env))
    conf = 1.0 / (1.0 + math.exp(-5 * (var - 0.002)))  # squashed to (0,1)

    bpm_half = bpm / 2.0 if bpm > 0 else None
    bpm_double = bpm * 2.0 if bpm > 0 else None
    return bpm, conf, bpm_half, bpm_double


# Krumhansl-Schmuckler profiles for major/minor (normalized)
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float
)
_KS_MAJOR /= _KS_MAJOR.max()
_KS_MINOR /= _KS_MINOR.max()

_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _estimate_key(y: np.ndarray, sr: int) -> Tuple[str | None, str | None, float | None]:
    """
    Estimate (key_root, key_mode, confidence) from chroma using simple
    Krumhansl-Schmuckler key profiles.
    """
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    except Exception:
        # Fallback to stft-based chroma if cqt fails for some reason
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    if chroma is None or chroma.size == 0:
        return None, None, None

    chroma_mean = np.mean(chroma, axis=1)  # shape (12,)

    # best rotation for major/minor
    best_score = -1e9
    best_root = None
    best_mode = None

    for mode_name, ref in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
        for shift in range(12):
            ref_rot = np.roll(ref, shift)
            score = float(np.dot(chroma_mean, ref_rot))
            if score > best_score:
                best_score = score
                best_root = _PITCH_NAMES[shift]
                best_mode = mode_name

    # crude confidence: normalized dot vs its max
    # normalize chroma_mean for more sensible scaling
    cm = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-9)
    confs = []
    for mode_name, ref in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
        for shift in range(12):
            ref_rot = np.roll(ref, shift)
            ref_norm = ref_rot / (np.linalg.norm(ref_rot) + 1e-9)
            confs.append(float(np.dot(cm, ref_norm)))
    max_possible = max(confs) if confs else 1.0
    key_conf = float(max(0.0, min(1.0, best_score / (max_possible + 1e-9))))

    return best_root, best_mode, key_conf


def _energy_rms(y: np.ndarray) -> float:
    """
    Return a simple energy proxy in [0,1] by scaling RMS.
    """
    rms = librosa.feature.rms(y=y).mean()
    # Normalize RMS by a soft clip to [0,1] (heuristic)
    return float(np.clip(rms / 0.1, 0.0, 1.0))


def process_audio(url: str, max_time: int = 60) -> Dict[str, Any]:
    """
    Entry point expected by app.py:
      - downloads the file
      - runs lightweight analysis
      - returns analyzer_result dict

    The FastAPI layer constructs the final response and PATCHes to DB.
    """
    tmp_path = None
    try:
        tmp_path = _download_to_tmp(url)
        y, sr = _load_mono(tmp_path, max_time=max_time, target_sr=22050)

        duration_s = int(round(len(y) / float(sr)))
        bpm, bpm_conf, bpm_half, bpm_double = _estimate_bpm(y, sr)
        key_root, key_mode, key_conf = _estimate_key(y, sr)
        energy = _energy_rms(y)

        analyzer_result: Dict[str, Any] = {
            "analyzer_version": ANALYZER_VERSION,
            "duration_s": duration_s,
            "sample_rate_hz": sr,
            "bpm": float(bpm) if bpm else 0.0,
            "bpm_confidence": float(bpm_conf) if bpm_conf is not None else None,
            "bpm_alt_half": float(bpm_half) if bpm_half else None,
            "bpm_alt_double": float(bpm_double) if bpm_double else None,
            "time_signature": None,               # not estimated here
            "key_root": key_root,
            "key_mode": key_mode,
            "key_confidence": float(key_conf) if key_conf is not None else None,
            "tuning_cents": None,                  # not estimated here
            "has_vocals": None,                    # future: vocal/instrumental model
            "is_instrumental": None,               # future
            "energy": float(energy),
            "valence": None,                       # future: model-based
            "language_iso639_1": "en",             # unknown; keep 'en' to match earlier output
            "genre_primary": None,
            "genre_secondary": None,
            "fingerprint_id": None,
            "confidence": None,
            "alt_half": None,
            "alt_double": None,
        }
        return analyzer_result

    finally:
        if tmp_path and os.path.exists(tmp_path):
            with contextlib.suppress(Exception):
                os.remove(tmp_path)
