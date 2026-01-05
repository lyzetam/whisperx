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

# Install PyTorch 2.1.x with CUDA support (compatible with pyannote)
RUN pip install --no-cache-dir \
    torch==2.1.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install WhisperX and dependencies (pin versions for compatibility)
RUN pip install --no-cache-dir \
    faster-whisper==1.0.3 \
    pyannote.audio==3.1.1 \
    whisperx \
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
