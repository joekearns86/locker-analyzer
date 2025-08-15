FROM ubuntu:22.04

# System deps (Ubuntu repos include Essentia + the extractor binary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    ffmpeg ca-certificates wget \
    libchromaprint-tools \
    essentia essentia-examples \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
ENV PORT=8000 \
    ANALYZER_VERSION=locker-analyzer@1.0.0

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
