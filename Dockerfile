FROM python:3.11-slim

# System deps (ffmpeg, chromaprint, essentials for Essentia)
RUN apt-get update && apt-get install -y --no-install-recommends     ffmpeg wget ca-certificates libyaml-0-2 libfftw3-3 libsamplerate0     liblapack3 libblas3 libchromaprint-tools   && rm -rf /var/lib/apt/lists/*

# Install Essentia prebuilt (adjust version if needed)
RUN wget -O /tmp/essentia.deb https://github.com/MTG/essentia/releases/download/v2.1_beta5/essentia_2.1b5_amd64.deb   && apt-get update && apt-get install -y /tmp/essentia.deb   && rm /tmp/essentia.deb

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PORT=8000     ANALYZER_VERSION=locker-analyzer@1.0.0     ENABLE_STEMS=false     ENABLE_LANGUAGE=false

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
