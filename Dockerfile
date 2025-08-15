FROM python:3.11-slim

# System deps: ffmpeg for audio decode, libs for soundfile, chromaprint for fpcalc
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libchromaprint-tools ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PORT=8000 ANALYZER_VERSION=locker-analyzer@1.0.0

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
