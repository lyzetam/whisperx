# WhisperX Docker Image
# Fast automatic speech recognition with word-level timestamps and speaker diarization

# Use CUDA 12.4 with cuDNN 9 for PyTorch 2.8+ compatibility
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

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

# Install PyTorch with CUDA 12.4 support (matches base image)
RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Install WhisperX and dependencies (let pip resolve compatible versions)
RUN pip install --no-cache-dir \
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

# Environment - let PyTorch use its bundled cuDNN (9.8.0) instead of system cuDNN (9.1.0)
ENV WHISPER_MODEL=large-v3 \
    COMPUTE_TYPE=float16 \
    BATCH_SIZE=16 \
    DEVICE=cuda \
    HF_HOME=/app/models \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import whisperx; print('ok')" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
