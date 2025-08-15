# analyze.py
import os
import tempfile
import math
import requests
import numpy as np
import librosa

# Optionally use soundfile if available (librosa can write via soundfile)
try:
    import soundfile as sf  # noqa: F401
except Exception:
    sf = None  # not required just for reading through librosa


NOTE_NAMES = np.array(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])


def _est_key_and_mode(y, sr):
    """
    Extremely simple key estimate using chroma + peak.
    Returns (key_root_str, key_mode_str) or (None, None) if uncertain.
    """
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        # Average across time
        prof = chroma.mean(axis=1)
        root_idx = int(np.argmax(prof))
        key_root = NOTE_NAMES[root_idx].item()

        # Very rough "mode" heuristic using major triad vs minor triad energy
        # This is intentionally simple so it never crashes.
        major_triad = prof[[root_idx, (root_idx + 4) % 12, (root_idx + 7) % 12]].sum()
        minor_triad = prof[[root_idx, (root_idx + 3) % 12, (root_idx + 7) % 12]].sum()
        key_mode = "major" if major_triad >= minor_triad else "minor"

        return key_root, key_mode
    except Exception:
        return None, None


def _safe_float(x):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return None
        return float(x)
    except Exception:
        return None


def process_audio(url: str, max_time: int = 180) -> dict:
    """
    Downloads the file at `url`, analyzes the first `max_time` seconds, and returns a dict.
    Fields align with your Supabase `tracks` schema where possible.
    """
    if not url or not isinstance(url, str):
        raise ValueError("url is required")

    # Download to a temp file (librosa.load works on local paths)
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp.flush()

        # Load with librosa (mono for robust features)
        y, sr = librosa.load(tmp.name, sr=None, mono=True)

    if y.size == 0:
        raise ValueError("empty_audio")

    # Truncate to max_time if longer
    if max_time and max_time > 0:
        y = y[: int(sr * max_time)]

    duration_s = _safe_float(y.size / sr)

    # Tempo (BPM)
    try:
        # Use librosa.beat.tempo; returns array
        tempo_arr = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
        bpm = _safe_float(np.median(tempo_arr)) if tempo_arr is not None and tempo_arr.size else None
        bpm_conf = None  # we don't compute a real confidence here
    except Exception:
        bpm, bpm_conf = None, None

    # Key and mode
    key_root, key_mode = _est_key_and_mode(y, sr)

    # Very rough extras (safe to store nulls where you don't have fields)
    try:
        energy = _safe_float(np.mean(np.square(y)))
    except Exception:
        energy = None

    # We don't run a vocal detector—return None so your PATCH code can set nulls safely
    has_vocals = None
    is_instrumental = None
    valence = None
    language_iso639_1 = None
    genre_primary = None

    result = {
        # Required/expected fields (align with your DB columns)
        "analyzer_version": os.getenv("ANALYZER_VERSION", "locker-analyzer@1.0.0"),
        "duration_s": duration_s,
        "sample_rate_hz": sr,
        "bpm": bpm,
        "bpm_confidence": bpm_conf,
        "bpm_alt_half": None,
        "bpm_alt_double": None,
        "time_signature": None,

        "key_root": key_root,
        "key_mode": key_mode,
        "key_confidence": None,
        "tuning_cents": None,

        "has_vocals": has_vocals,
        "is_instrumental": is_instrumental,

        # Nice-to-haves (your table has energy & valence)
        "energy": energy,
        "valence": valence,

        # Other metadata your table already has columns for (safe as nulls)
        "language_iso639_1": language_iso639_1,
        "genre_primary": genre_primary,
        "genre_secondary": None,

        # These are not persisted by this function; your edge function handles DB PATCH
        "fingerprint_id": None,
        "confidence": None,
        "alt_half": None,
        "alt_double": None,
    }

    return result
