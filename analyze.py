# analyze.py
from __future__ import annotations

import os
import math
import re
import tempfile
import contextlib
from typing import Dict, Any, Tuple, Optional

import numpy as np
import requests
import librosa

ANALYZER_VERSION = os.getenv("ANALYZER_VERSION", "locker-analyzer@1.1.0")
TARGET_SR = int(os.getenv("TARGET_SAMPLE_RATE", "22050"))
MAX_TIME = int(os.getenv("MAX_ANALYZE_SECONDS", "90"))  # analyze first N seconds


# ---------- helpers ----------
def _download_to_tmp(url: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as f:
        with requests.get(url, stream=True, timeout=45) as r:
            r.raise_for_status()
            for chunk in r.iter_content(1024 * 256):
                if chunk:
                    f.write(chunk)
        return f.name


def _parse_filename_hints(url: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Parse hints like '118bpm' and 'Gbmaj' / 'Amin' from the URL/filename.
    """
    name = url.split("/")[-1].lower()

    bpm = None
    m = re.search(r'(\d{2,3})\s*bpm', name)
    if m:
        try:
            bpm = int(m.group(1))
        except Exception:
            pass

    key_root = None
    key_mode = None
    m = re.search(r'\b([a-g])([#b]?)(maj|major|min|minor|m)\b', name)
    if m:
        letter = m.group(1).upper()
        accidental = m.group(2) or ""
        mode = m.group(3)
        key_root = f"{letter}{accidental}".replace("b", "b")  # keep ASCII flats
        key_mode = "minor" if mode in ("min", "minor", "m") else "major"

    return bpm, key_root, key_mode


def _load_audio(path: str, max_time: int, sr: int) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sr, mono=True, duration=max_time)
    if y is None or y.size == 0:
        raise ValueError("empty_audio")
    # light high-pass to remove rumble
    y = librosa.effects.preemphasis(y, coef=0.0)  # set >0 if you want stronger HPF feel
    return y.astype(np.float32, copy=False), sr


# ---------- tempo ----------
def _score_tempo(onset_env: np.ndarray, sr: int, hop_length: int, bpm: float) -> float:
    """
    Score how well 'bpm' aligns to the onset envelope by summing energy at a beat grid.
    We search best phase by scanning one bar worth of offsets.
    """
    if not bpm or bpm <= 0:
        return 0.0
    # period in frames between beats
    period = (60.0 * sr) / (hop_length * bpm)
    if period < 2:
        return 0.0

    # try phases from 0..period-1 (rounded)
    period_i = int(round(period))
    if period_i < 2:
        return 0.0

    best = 0.0
    # step phase in coarse steps to save time (e.g., 8 steps)
    steps = max(1, min(period_i, 8))
    for p in np.linspace(0, period_i - 1, steps):
        idx = (np.arange(0, len(onset_env) - int(p), period_i) + int(p)).astype(int)
        s = float(onset_env[idx].sum())
        if s > best:
            best = s

    # normalize by total onset energy to get 0..1-ish
    total = float(onset_env.sum()) + 1e-9
    return float(np.clip(best / total, 0.0, 1.0))


def _estimate_bpm(y: np.ndarray, sr: int, hop_length: int = 512) -> Tuple[float, float, float, float]:
    # Percussive component stabilizes tempo
    y_h, y_p = librosa.effects.hpss(y)
    onset_env = librosa.onset.onset_strength(y=y_p, sr=sr, hop_length=hop_length, aggregate=np.median)
    if onset_env.size == 0 or np.all(onset_env == 0):
        return 0.0, 0.0, None, None

    # Candidate distribution
    cand = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length,
                              aggregate=None, start_bpm=120.0, max_tempo=220.0)
    # Keep unique, top-k
    cands = sorted({float(x) for x in cand if 40.0 <= float(x) <= 230.0}, key=lambda x: -x)[:5]
    if not cands:
        cands = [float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length))]

    # Score each candidate; also consider half/double
    best_bpm = 0.0
    best_score = -1.0
    for t in cands:
        for t2 in (t / 2.0, t, t * 2.0):
            s = _score_tempo(onset_env, sr, hop_length, t2)
            if s > best_score:
                best_score = s
                best_bpm = float(t2)

    # sanity clamp & snapping
    if best_bpm > 200.0:
        best_bpm /= 2.0
    if best_bpm < 60.0 and best_bpm > 0:
        best_bpm *= 2.0

    # better alts from the final bpm
    bpm_half = best_bpm / 2.0 if best_bpm > 0 else None
    bpm_double = best_bpm * 2.0 if best_bpm > 0 else None

    # round nicely
    best_bpm = float(np.round(best_bpm, 1))
    if abs(best_bpm - round(best_bpm)) < 0.15:
        best_bpm = float(round(best_bpm))

    return best_bpm, float(np.clip(best_score, 0.0, 1.0)), bpm_half, bpm_double


# ---------- key ----------
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)
_KS_MAJOR /= _KS_MAJOR.sum()
_KS_MINOR /= _KS_MINOR.sum()
_PITCH = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _estimate_key(y: np.ndarray, sr: int, hop_length: int = 512) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[float]]:
    """
    Harmonic CQT chroma + Krumhansl templates over time with voting.
    Returns (root, mode, confidence 0..1, tuning_cents).
    """
    y_h, _ = librosa.effects.hpss(y)

    # tuning correction
    try:
        tuning = float(librosa.pitch.estimate_tuning(y=y_h, sr=sr))
    except Exception:
        tuning = 0.0
    tuning_cents = float(tuning * 100.0)

    try:
        chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=hop_length, tuning=tuning)
    except Exception:
        chroma = librosa.feature.chroma_stft(y=y_h, sr=sr, hop_length=hop_length)

    if chroma is None or chroma.size == 0:
        return None, None, None, tuning_cents

    # frame-wise best key (root+mode), then vote
    frames = chroma.shape[1]
    votes = {}
    strengths = []

    for t in range(frames):
        v = chroma[:, t]
        if not np.any(v):
            continue
        v = v / (v.sum() + 1e-9)

        best_score = -1.0
        best = None

        for mode_name, prof in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
            for root in range(12):
                score = float(np.dot(v, np.roll(prof, root)))
                if score > best_score:
                    best_score = score
                    best = (root, mode_name)

        if best is not None:
            votes[best] = votes.get(best, 0) + 1
            strengths.append(best_score)

    if not votes:
        return None, None, None, tuning_cents

    # pick most-voted key; confidence: votes/frames weighted by strength
    best_key, best_votes = max(votes.items(), key=lambda kv: kv[1])
    root_idx, mode = best_key
    root = _PITCH[root_idx]
    vote_conf = best_votes / max(1, frames)
    str_conf = float(np.mean(strengths)) if strengths else 0.0

    # combine (bounded to 0..1)
    key_conf = float(np.clip(0.5 * vote_conf + 0.5 * str_conf, 0.0, 1.0))

    return root, mode, key_conf, tuning_cents


# ---------- main entry ----------
def process_audio(url: str, max_time: int = MAX_TIME) -> Dict[str, Any]:
    tmp = None
    try:
        hint_bpm, hint_key_root, hint_key_mode = _parse_filename_hints(url)

        tmp = _download_to_tmp(url)
        y, sr = _load_audio(tmp, max_time=max_time, sr=TARGET_SR)

        # Tempo
        bpm, bpm_conf, bpm_half, bpm_double = _estimate_bpm(y, sr)

        # If filename says a BPM and we’re at exact half/double, adopt the filename
        if hint_bpm and bpm:
            if abs(bpm - (hint_bpm * 2)) <= 2:
                bpm = float(hint_bpm)
                bpm_half = bpm / 2.0
                bpm_double = bpm * 2.0
            elif abs(bpm * 2 - hint_bpm) <= 2:
                bpm = float(hint_bpm)
                bpm_half = bpm / 2.0
                bpm_double = bpm * 2.0
            elif abs(bpm - hint_bpm) <= 2:
                bpm = float(hint_bpm)

        # Key
        key_root, key_mode, key_conf, tuning_cents = _estimate_key(y, sr)

        # If filename hints a key and our confidence is low, prefer the hint
        if hint_key_root and hint_key_mode:
            if (key_conf is None) or (key_conf < 0.45):
                key_root = hint_key_root.replace("b", "b").upper()
                key_mode = hint_key_mode

        # Energy proxy
        try:
            energy = float(np.clip(librosa.feature.rms(y=y).mean() * 5.0, 0.0, 1.0))
        except Exception:
            energy = None

        # Duration we analyzed (approx)
        duration_s = float(len(y)) / float(sr)

        return {
            "analyzer_version": ANALYZER_VERSION,
            "duration_s": float(np.round(duration_s, 2)),
            "sample_rate_hz": int(sr),

            "bpm": float(bpm) if bpm else None,
            "bpm_confidence": float(np.round(bpm_conf, 3)) if bpm_conf is not None else None,
            "bpm_alt_half": float(np.round(bpm_half, 1)) if bpm_half else None,
            "bpm_alt_double": float(np.round(bpm_double, 1)) if bpm_double else None,
            "time_signature": None,  # not estimated here

            "key_root": key_root,
            "key_mode": key_mode,
            "key_confidence": float(np.round(key_conf, 3)) if key_conf is not None else None,
            "tuning_cents": float(np.round(tuning_cents, 1)) if tuning_cents is not None else None,

            "has_vocals": None,         # future upgrade
            "is_instrumental": None,    # future upgrade
            "energy": energy,
            "valence": None,

            "language_iso639_1": None,
            "genre_primary": None,
            "genre_secondary": None,

            "fingerprint_id": None,
            "confidence": None,
            "alt_half": None,
            "alt_double": None,
        }

    finally:
        if tmp and os.path.exists(tmp):
            with contextlib.suppress(Exception):
                os.remove(tmp)
