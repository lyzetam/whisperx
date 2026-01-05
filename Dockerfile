# WhisperX Docker Image
# Fast automatic speech recognition with word-level timestamps and speaker diarization
# Supports GPU (CUDA 12.x) and CPU modes

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd -m -u 1000 whisperx
WORKDIR /app

# Create virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install WhisperX and dependencies
RUN pip install --no-cache-dir \
    whisperx \
    torch \
    torchaudio \
    faster-whisper \
    pyannote.audio \
    transformers \
    nltk

# Download NLTK data for sentence tokenization
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/output && \
    chown -R whisperx:whisperx /app

# Copy entrypoint script
COPY --chown=whisperx:whisperx entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Copy API server if present
COPY --chown=whisperx:whisperx api/ /app/api/

# Switch to non-root user
USER whisperx

# Default environment variables
ENV WHISPER_MODEL=large-v3
ENV COMPUTE_TYPE=float16
ENV BATCH_SIZE=16
ENV DEVICE=cuda
ENV HF_HOME=/app/models

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import whisperx; print('healthy')" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
