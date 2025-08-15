# Locker Analyzer v1.0 (CPU)

FastAPI service that analyzes audio and returns Locker-ready JSON.

## Endpoints
- GET /health → { ok: true, version }
- POST /analyze → body: { "url": "<signed-or-public-audio-url>" }

## Local run (requires ffmpeg + Essentia installed locally)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

## Docker (recommended)
docker build -t locker-analyzer:cpu .
docker run -p 8000:8000 locker-analyzer:cpu

## Render deploy
- Push this repo to GitHub.
- Render → New Web Service → use Docker, point to this repo.
- Env: PORT=8000, (optional) ANALYZER_VERSION=locker-analyzer@1.0.0
- Health Check Path: /health
- Deploy, then test:
  curl -s https://YOUR-RENDER-URL/health
  curl -s -X POST https://YOUR-RENDER-URL/analyze -H "Content-Type: application/json" -d '{"url":"<SIGNED_URL>"}'

## Notes
- This baseline uses Essentia CLI; it reports tempo (bpm), key/mode, duration, heuristics for vocals/energy/valence, and simple placeholders for genre/moods/instruments.
- Upgrade later with stems (Demucs) and Whisper without changing the response shape. 
