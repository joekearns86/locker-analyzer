FROM python:3.11-slim

# System deps: ffmpeg for audio decode, libsndfile for soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Security: run as non-root
RUN useradd -m appuser
USER appuser

ENV PORT=8000
ENV ANALYZER_VERSION=locker-analyzer@1.1.0

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
