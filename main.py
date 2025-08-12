from flask import Flask, request, jsonify
import librosa
import soundfile as sf  # ensure libsndfile is loaded
import numpy as np
import os, tempfile, requests

app = Flask(__name__)

# Simple CORS (safe for dev; restrict later to your domains)
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp

# ---------- lightweight helpers ----------
MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
PITCH_CLASS_NAMES = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]

def estimate_key(y, sr):
    # chroma from short clip; tiny hop to keep RAM low
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=1024)
    chroma_mean = chroma.mean(axis=1)
    scores_major = [np.corrcoef(np.roll(MAJOR_PROFILE,i), chroma_mean)[0,1] for i in range(12)]
    scores_minor = [np.corrcoef(np.roll(MINOR_PROFILE,i), chroma_mean)[0,1] for i in range(12)]
    maj_idx = int(np.argmax(scores_major)); min_idx = int(np.argmax(scores_minor))
    if scores_major[maj_idx] >= scores_minor[min_idx]:
        return PITCH_CLASS_NAMES[maj_idx], "major"
    else:
        return PITCH_CLASS_NAMES[min_idx], "minor"

def naive_vocal_and_instrument(y, sr):
    # use smaller STFT frame to reduce memory
    S, _ = librosa.magphase(librosa.stft(y, n_fft=1024, hop_length=512))
    flatness = librosa.feature.spectral_flatness(S=S).mean()
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr).mean()
    y_h, y_p = librosa.effects.hpss(y)
    harm_ratio = (np.mean(np.abs(y_h))+1e-9)/(np.mean(np.abs(y))+1e-9)
    vocal = bool(harm_ratio > 0.6 and 800 < centroid < 2500 and flatness < 0.4)
    main_inst = "vocals" if vocal else ("drums" if np.mean(np.abs(y_p)) > np.mean(np.abs(y_h)) else "harmonic")
    return vocal, main_inst

def analyze_file(path):
    # Downsample + cap duration to keep memory low on free tier
    y, sr = librosa.load(path, sr=11025, mono=True, duration=90.0)  # 90s max, 11.025 kHz
    # tempo via onset envelope (lighter than full beat pipeline)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=‘median’)  # ← string, not np.median
    bpm = int(float(np.squeeze(tempo))) if np.size(tempo) else 0
    key_pc, mode = estimate_key(y, sr)
    musical_key = f"{key_pc}{'m' if mode=='minor' else ''}"
    mood = "dark" if (mode=="minor" and bpm <= 130) else ("uplifting" if (mode=="major" and bpm >= 120) else "neutral")
    vocal, main_inst = naive_vocal_and_instrument(y, sr)
    return {"bpm": bpm, "key": musical_key, "mode": mode, "mood": mood, "vocal": vocal, "mainInstrument": main_inst}

# ---------- routes ----------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"})

@app.route("/analyze", methods=["POST","OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return ("", 204)
    tmp_path = None
    try:
        # JSON {url} (what Lovable sends)
        if request.is_json and isinstance(request.get_json(silent=True), dict) and "url" in request.get_json():
            url = request.get_json()["url"]
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
                r = requests.get(url, timeout=30, stream=True)
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
                tmp_path = tmp.name
            return jsonify(analyze_file(tmp_path))
        # Manual multipart upload
        if "file" in request.files:
            f = request.files["file"]
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
                f.save(tmp.name); tmp_path = tmp.name
            return jsonify(analyze_file(tmp_path))
        return jsonify({"error":"No audio provided. Send JSON {url} or form-data 'file'."}), 400
    except Exception as e:
        return jsonify({"error": f"analyze_failed: {e}"}), 400
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
