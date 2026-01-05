# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Dockerized deployment of WhisperX - fast automatic speech recognition with:
- 70x realtime transcription using batched whisper + faster-whisper
- Word-level timestamps via wav2vec2 alignment
- Speaker diarization via pyannote-audio
- REST API for HTTP-based transcription

## Commands

### Docker Development

```bash
# Build images
docker compose build

# Start GPU version (requires NVIDIA GPU + Container Toolkit)
docker compose up -d whisperx

# Start CPU version
docker compose --profile cpu up -d whisperx-cpu

# View logs
docker compose logs -f whisperx

# Stop services
docker compose down

# Rebuild and restart
docker compose up -d --build whisperx
```

### API Testing

```bash
# Health check
curl http://localhost:8000/health

# Transcribe audio
curl -X POST http://localhost:8000/transcribe \
  -F "file=@test.wav"

# With diarization
curl -X POST http://localhost:8000/transcribe \
  -F "file=@test.wav" \
  -F "diarize=true"
```

### CLI Transcription

```bash
# Direct transcription
docker run --rm --gpus all \
  -v $(pwd):/data \
  whisperx:latest /data/audio.wav

# With specific model
docker run --rm --gpus all \
  -e WHISPER_MODEL=medium \
  -v $(pwd):/data \
  whisperx:latest /data/audio.wav
```

## Architecture

```
whisperx/
├── Dockerfile           # CUDA-enabled GPU image
├── Dockerfile.cpu       # CPU-only image (ARM compatible)
├── docker-compose.yml   # Local development setup
├── entrypoint.sh        # Container entrypoint script
├── api/
│   ├── __init__.py
│   └── main.py          # FastAPI server
└── .github/workflows/
    └── build-push.yml   # GitHub Actions CI/CD
```

## Key Files

- `Dockerfile` - NVIDIA CUDA 12.4 base image with GPU support
- `Dockerfile.cpu` - Python 3.10 slim image for CPU-only systems
- `api/main.py` - FastAPI application with `/transcribe` endpoint
- `entrypoint.sh` - Handles CLI mode, API server mode, and worker mode

## Environment Variables

| Variable | GPU Default | CPU Default | Description |
|----------|-------------|-------------|-------------|
| WHISPER_MODEL | large-v3 | base | Model size |
| COMPUTE_TYPE | float16 | int8 | Compute precision |
| BATCH_SIZE | 16 | 8 | Inference batch size |
| DEVICE | cuda | cpu | Compute device |
| HF_TOKEN | - | - | Hugging Face token for diarization |

## Deployment

### GitHub Actions

The workflow at `.github/workflows/build-push.yml` automatically:
1. Builds both CUDA and CPU images
2. Pushes to Docker Hub with tags: `latest`, `cuda`, `cpu`, `{sha}`

Required secrets:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

### K8s Deployment (Optional)

For homelab K8s deployment, create a manifest in `fako-cluster/apps/whisperx/`.

## Common Issues

1. **CUDA out of memory**: Reduce `BATCH_SIZE` or use smaller model
2. **Diarization fails**: Ensure `HF_TOKEN` is set and model agreements accepted
3. **Slow first request**: Set `PRELOAD_MODEL=true` to load model on startup
