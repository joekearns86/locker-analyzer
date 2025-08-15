FROM ubuntu:22.04

# OS deps: ffmpeg + libraries Essentia needs + chromaprint (fpcalc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    ffmpeg wget ca-certificates \
    libyaml-0-2 libfftw3-3 libsamplerate0 liblapack3 libblas3 \
    libchromaprint-tools \
  && rm -rf /var/lib/apt/lists/*

# Install Essentia (prebuilt .deb from MTG)
RUN wget -O /tmp/essentia.deb https://github.com/MTG/essentia/releases/download/v2.1_beta5/essentia_2.1b5_amd64.deb \
  && apt-get update && apt-get install -y /tmp/essentia.deb \
  && rm /tmp/essentia.deb

WORKDIR /app
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
ENV PORT=8000 \
    ANALYZER_VERSION=locker-analyzer@1.0.0

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
