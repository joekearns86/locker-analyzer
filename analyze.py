import tempfile, os, uuid, json, shutil, subprocess, math
import numpy as np
import librosa

ANALYZER_VERSION = os.getenv("ANALYZER_VERSION", "locker-analyzer@dev")

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip()[:2000])
    return p.stdout

def ffmpeg_to_wav(src_url: str, out_wav: str):
    # Stable decode: 44.1kHz mono WAV to simplify analysis speed/consistency
    run(["ffmpeg", "-y", "-i", src_url, "-vn", "-ac", "1", "-ar", "44100", out_wav])

def acoustid_fingerprint(path: str) -> str | None:
    try:
        out = run(["fpcalc", path])
        for line in out.splitlines():
            if line.startswith("FINGERPRINT="):
                return "fp:" + line.split("=",1)[1][:64]
    except Exception:
        pass
    return None

# ---- Key detection helpers (simple major/minor template method) ----
_MAJOR = np.array([6,2,3,2,4,3,2,5,2,4,2,3], dtype=float)
_MINOR = np.array([6,2,3,4,2,3,2,5,2,4,3,2], dtype=float)
_PITCHES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def estimate_key(chroma: np.ndarray):
    # average chroma across time
    prof = chroma.mean(axis=1)
    best_score, best_key, best_mode = -1e9, "unknown", "other"
    for i in range(12):
        score_maj = np.dot(prof, np.roll(_MAJOR, i))
        score_min = np.dot(prof, np.roll(_MINOR, i))
        if score_maj > best_score:
            best_score = score_maj
            best_key, best_mode = _PITCHES[i], "major"
        if score_min > best_score:
            best_score = score_min
            best_key, best_mode = _PITCHES[i], "minor"
    # crude confidence: how peaked the profile is
    conf = float((prof.max() - np.median(prof)) / (prof.sum() + 1e-9))
    conf = max(0.0, min(1.0, conf * 4))
    return best_key, best_mode, conf

def process_audio(url: str) -> dict:
    work = tempfile.mkdtemp(prefix="locker-")
    wav = os.path.join(work, f"in-{uuid.uuid4()}.wav")
    try:
        ffmpeg_to_wav(url, wav)
        y, sr = librosa.load(wav, sr=44100, mono=True)

        # duration
        duration = float(len(y)) / sr

        # tempo (bpm) using librosa's beat tracker
        tempo = float(librosa.beat.tempo(y=y, sr=sr, max_tempo=200, aggregate=np.median))
        bpm_alt_half = round(tempo/2, 1) if tempo else None
        bpm_alt_double = round(tempo*2, 1) if tempo else None

        # spectral features → energy/valence heuristics
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        flux = np.diff(S, axis=1)
        spectral_flux = float(np.mean(np.sqrt((flux**2).sum(axis=0))) / (S.shape[0] + 1e-9))
        centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))

        energy = max(0.0, min(1.0, spectral_flux * 2.0))
        valence = max(0.0, min(1.0, centroid / 5000.0))

        # key estimation via chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_root, key_mode, key_conf = estimate_key(chroma)

        # simple vocal presence heuristic (voicedness proxy)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        has_vocals = bool(zcr < 0.12 and centroid > 1500)  # tuned for pop-ish material
        is_instrumental = not has_vocals

        # placeholders for descriptors (upgrade later with models)
        genre_primary = "dance_pop" if 110 <= tempo <= 130 and centroid > 1800 else None
        moods = [m for m in ["euphoric" if energy>0.6 else None, "dark" if valence<0.35 else None] if m]
        instruments_primary = "female_vocal" if has_vocals else None
        instruments_secondary = ["synths"] if centroid > 1500 else []

        # fingerprint
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
