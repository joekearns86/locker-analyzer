FROM python:3.11-slim

# Helpful envs (optional but nice)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps: ffmpeg (decode), libsndfile (for soundfile), chromaprint optional, certs for TLS
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libchromaprint-tools \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Keep your analyzer version env if you use it
ENV ANALYZER_VERSION=locker-analyzer@1.0.0

# Let Render tell us which port to bind to
ENV PORT=8000
EXPOSE 8000

# IMPORTANT: use ${PORT} (Render sets it). Keep your module name app:app.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
