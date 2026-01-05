# WhisperX Docker Image
# Fast automatic speech recognition with word-level timestamps and speaker diarization

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies and Python in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Install PyTorch with CUDA support first (smaller than full torch)
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 torchaudio==2.1.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install WhisperX and dependencies
RUN pip install --no-cache-dir \
    whisperx faster-whisper pyannote.audio \
    transformers nltk fastapi uvicorn python-multipart \
    && python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')" \
    && rm -rf ~/.cache/pip

# Create directories
RUN mkdir -p /app/models /app/data /app/output

# Copy application files
COPY entrypoint.sh /app/
COPY api/ /app/api/
RUN chmod +x /app/entrypoint.sh

# Environment
ENV WHISPER_MODEL=large-v3 \
    COMPUTE_TYPE=float16 \
    BATCH_SIZE=16 \
    DEVICE=cuda \
    HF_HOME=/app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import whisperx; print('ok')" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
