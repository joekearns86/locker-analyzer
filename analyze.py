import tempfile, os, uuid, json, shutil, subprocess, math

ANALYZER_VERSION = os.getenv("ANALYZER_VERSION", "locker-analyzer@dev")

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip()[:2000])
    return p.stdout

def ffmpeg_to_wav(src_url: str, out_wav: str):
    # Convert any audio URL to a 44.1kHz stereo WAV for stable analysis
    run(["ffmpeg", "-y", "-i", src_url, "-vn", "-ac", "2", "-ar", "44100", out_wav])

def acoustid_fingerprint(path: str) -> str | None:
    # Best-effort local fingerprint using chromaprint (no external API)
    try:
        out = run(["fpcalc", path])
        for line in out.splitlines():
            if line.startswith("FINGERPRINT="):
                return "fp:" + line.split("=",1)[1][:64]
    except Exception:
        pass
    return None

def essentia_extract(in_wav: str, out_json: str):
    # Run the Essentia streaming music extractor
    # Binary name may be 'essentia_streaming_extractor_music' (as installed by the .deb)
    run(["essentia_streaming_extractor_music", in_wav, out_json])

def map_fields(rep: dict, file_url: str) -> dict:
    # Core audio props
    ap = rep.get("metadata", {}).get("audio_properties", {})
    duration = float(ap.get("length", 0.0))
    sr = int(ap.get("sample_rate", 44100))

    # Tempo
    rhythm = rep.get("rhythm", {})
    bpm = float(rhythm.get("bpm", 0.0))
    bpm_conf = float(rhythm.get("bpm_confidence", 0.8))

    # Key
    tonal = rep.get("tonal", {})
    key_root = str(tonal.get("chords_key", "unknown")).upper()
    key_mode = str(tonal.get("chords_scale", "other")).lower()
    key_conf = float(tonal.get("chords_strength", 0.5))
    tuning_freq = float(tonal.get("tuning_frequency", 440.0))
    # cents vs A440
    tuning_cents = round(1200 * math.log2(max(1e-6, tuning_freq/440.0)))

    # Low-level features (heuristics)
    low = rep.get("lowlevel", {})
    spectral_flux = float(low.get("spectral_flux", {}).get("mean", 0.3))
    centroid = float(low.get("spectral_centroid", {}).get("mean", 2000.0))

    # Heuristic estimates (0-1)
    energy = max(0.0, min(1.0, spectral_flux * 2.0))
    valence = max(0.0, min(1.0, centroid / 5000.0))

    # Simple vocal presence heuristic; upgrade later with model/stems
    chords_changes = float(tonal.get("chords_changes_rate", 0.01))
    has_vocals = bool(chords_changes > 0.02 and centroid > 1500.0)
    is_instrumental = not has_vocals

    # Basic genre/instrument/mood placeholders (replace with classifiers later)
    genre_primary = "dance_pop" if 110 <= bpm <= 130 and centroid > 1800 else None
    moods = [m for m in [
        "euphoric" if energy > 0.6 else None,
        "dark" if valence < 0.35 else None
    ] if m]
    instruments_primary = "female_vocal" if has_vocals else None
    instruments_secondary = ["synths"] if centroid > 1500 else []

    # Fixed for MVP
    time_signature = "4/4"

    return {
        "analyzer_version": ANALYZER_VERSION,
        "file": {
            "url": file_url,
            "duration_s": round(duration, 2),
            "sample_rate_hz": sr
        },
        "fingerprint_id": None,  # set after fpcalc
        "tempo": {
            "bpm": round(bpm, 1) if bpm else None,
            "confidence": round(bpm_conf, 2),
            "alt_half": round(bpm/2, 1) if bpm else None,
            "alt_double": round(bpm*2, 1) if bpm else None,
            "time_signature": time_signature
        },
        "tonality": {
            "key_root": key_root,
            "mode": key_mode,
            "confidence": round(key_conf, 2),
            "tuning_cents": tuning_cents
        },
        "vocals": {
            "has_vocals": has_vocals,
            "is_instrumental": is_instrumental,
            "confidence": 0.8,
            "voice_type": None,
            "language_iso639_1": None
        },
        "descriptors": {
            "energy": round(energy, 2),
            "valence": round(valence, 2),
            "genre_primary": genre_primary,
            "genre_secondary": None,
            "moods": moods,
            "instruments_primary": instruments_primary,
            "instruments_secondary": instruments_secondary,
            "dominant_elements": ["big_drums"] if energy > 0.6 else [],
            "structure_highlight": None
        }
    }

def process_audio(url: str) -> dict:
    work = tempfile.mkdtemp(prefix="locker-")
    wav = os.path.join(work, f"in-{uuid.uuid4()}.wav")
    rep_json = os.path.join(work, "rep.json")
    try:
        ffmpeg_to_wav(url, wav)
        fp = acoustid_fingerprint(wav)
        essentia_extract(wav, rep_json)
        with open(rep_json, "r") as f:
            rep = json.load(f)
        out = map_fields(rep, url)
        out["fingerprint_id"] = fp
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)
