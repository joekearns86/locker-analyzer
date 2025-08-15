import tempfile, os, uuid, json, shutil, subprocess, math
import requests
import numpy as np
import librosa

ANALYZER_VERSION = os.getenv("ANALYZER_VERSION", "locker-analyzer@dev")

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        # Return full ffmpeg error to the API so we can see what's wrong
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\nSTDERR:\n{p.stderr.strip()[:4000]}")
    return p.stdout

def download_to_file(url: str, dest_path: str, max_mb: int = 200):
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        size = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                f.write(chunk)
                size += len(chunk)
                if size > max_mb * 1024 * 1024:
                    raise RuntimeError(f"file too large (> {max_mb} MB)")

def ffmpeg_to_wav(src_path: str, out_wav: str):
    # Convert any audio to 44.1kHz mono WAV for stable analysis
    run([
        "ffmpeg",
        "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y", "-i", src_path,
        "-vn", "-ac", "1", "-ar", "44100",
        out_wav
    ])

def acoustid_fingerprint(path: str) -> str | None:
    try:
        out = run(["fpcalc", path])
        for line in out.splitlines():
            if line.startswith("FINGERPRINT="):
                return "fp:" + line.split("=",1)[1][:64]
    except Exception:
        pass
    return None

# ---- Key detection helpers (simple major/minor template) ----
_MAJOR = np.array([6,2,3,2,4,3,2,5,2,4,2,3], dtype=float)
_MINOR = np.array([6,2,3,4,2,3,2,5,2,4,3,2], dtype=float)
_PITCHES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def estimate_key(chroma: np.ndarray):
    prof = chroma.mean(axis=1)
    best_score, best_key, best_mode = -1e9, "unknown", "other"
    for i in range(12):
        score_maj = float(np.dot(prof, np.roll(_MAJOR, i)))
        score_min = float(np.dot(prof, np.roll(_MINOR, i)))
        if score_maj > best_score:
            best_score, best_key, best_mode = score_maj, _PITCHES[i], "major"
        if score_min > best_score:
            best_score, best_key, best_mode = score_min, _PITCHES[i], "minor"
    conf = float((prof.max() - np.median(prof)) / (prof.sum() + 1e-9))
    conf = max(0.0, min(1.0, conf * 4))
    return best_key, best_mode, conf

def process_audio(url: str) -> dict:
    work = tempfile.mkdtemp(prefix="locker-")
    try:
        # 1) Download to local temp
        src = os.path.abspath(os.path.join(work, f"src-{uuid.uuid4()}"))
        wav = os.path.abspath(os.path.join(work, f"in-{uuid.uuid4()}.wav"))
        download_to_file(url, src)

        # 2) Convert with ffmpeg
        ffmpeg_to_wav(src, wav)

        # 3) Analysis
        y, sr = librosa.load(wav, sr=44100, mono=True)
        duration = float(len(y)) / sr

        tempo = float(librosa.beat.tempo(y=y, sr=sr, max_tempo=200, aggregate=np.median))
        bpm_alt_half = round(tempo/2, 1) if tempo else None
        bpm_alt_double = round(tempo*2, 1) if tempo else None

        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        flux = np.diff(S, axis=1)
        spectral_flux = float(np.mean(np.sqrt((flux**2).sum(axis=0))) / (S.shape[0] + 1e-9))
        centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
        energy = max(0.0, min(1.0, spectral_flux * 2.0))
        valence = max(0.0, min(1.0, centroid / 5000.0))

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_root, key_mode, key_conf = estimate_key(chroma)

        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        has_vocals = bool(zcr < 0.12 and centroid > 1500)  # quick heuristic
        is_instrumental = not has_vocals

        genre_primary = "dance_pop" if 110 <= tempo <= 130 and centroid > 1800 else None
        moods = [m for m in ["euphoric" if energy>0.6 else None, "dark" if valence<0.35 else None] if m]
        instruments_primary = "female_vocal" if has_vocals else None
        instruments_secondary = ["synths"] if centroid > 1500 else []

        fp = acoustid_fingerprint(wav)

        return {
            "analyzer_version": ANALYZER_VERSION,
            "file": {"url": url, "duration_s": round(duration,2), "sample_rate_hz": sr},
            "fingerprint_id": fp,
            "tempo": {
                "bpm": round(tempo,1) if tempo else None,
                "confidence": 0.8,
                "alt_half": bpm_alt_half,
                "alt_double": bpm_alt_double,
                "time_signature": "4/4"
            },
            "tonality": {
                "key_root": key_root,
                "mode": key_mode,
                "confidence": round(key_conf,2),
                "tuning_cents": 0
            },
            "vocals": {
                "has_vocals": has_vocals,
                "is_instrumental": is_instrumental,
                "confidence": 0.7,
                "voice_type": None,
                "language_iso639_1": None
            },
            "descriptors": {
                "energy": round(energy,2),
                "valence": round(valence,2),
                "genre_primary": genre_primary,
                "genre_secondary": None,
                "moods": moods,
                "instruments_primary": instruments_primary,
                "instruments_secondary": instruments_secondary,
                "dominant_elements": ["big_drums"] if energy>0.6 else [],
                "structure_highlight": None
            }
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)
