# WhisperX Docker Deployment

Dockerized deployment of [WhisperX](https://github.com/m-bain/whisperX) - fast automatic speech recognition with word-level timestamps and speaker diarization.

## Features

- **70x realtime transcription** with large-v2/v3 models
- **Word-level timestamps** using wav2vec2 alignment
- **Speaker diarization** with pyannote-audio
- **REST API** for HTTP-based transcription
- **GPU acceleration** with CUDA 12.x
- **CPU fallback** for systems without NVIDIA GPUs

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/whisperx.git
cd whisperx

# Copy environment file
cp .env.example .env
# Edit .env with your settings (especially HF_TOKEN for diarization)

# Start GPU version
docker compose up -d whisperx

# Or start CPU version
docker compose --profile cpu up -d whisperx-cpu
```

### Using Docker Run

```bash
# GPU version
docker run -d \
  --name whisperx \
  --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN=your_token \
  landryzetam/whisperx:latest api

# CPU version
docker run -d \
  --name whisperx \
  -p 8000:8000 \
  -e WHISPER_MODEL=base \
  landryzetam/whisperx:cpu api
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Transcribe Audio

```bash
# Basic transcription
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav"

# With speaker diarization
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "diarize=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4"

# Specify language
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=de"
```

### CLI Transcription

```bash
# Mount audio file and transcribe
docker run --rm --gpus all \
  -v /path/to/audio:/data \
  landryzetam/whisperx:latest \
  /data/audio.wav --output_dir /data

# With diarization
docker run --rm --gpus all \
  -e HF_TOKEN=your_token \
  -v /path/to/audio:/data \
  landryzetam/whisperx:latest \
  /data/audio.wav --diarize --output_dir /data
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `large-v3` | Model size (tiny, base, small, medium, large-v2, large-v3) |
| `COMPUTE_TYPE` | `float16` | Compute type (float16, int8, float32) |
| `BATCH_SIZE` | `16` | Batch size for inference |
| `DEVICE` | `cuda` | Device (cuda, cpu) |
| `HF_TOKEN` | - | Hugging Face token for speaker diarization |
| `PRELOAD_MODEL` | `false` | Preload model on startup |

### Speaker Diarization Setup

1. Create a Hugging Face account and get an access token from [here](https://huggingface.co/settings/tokens)

2. Accept the user agreements for:
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

3. Set the `HF_TOKEN` environment variable

## Model Sizes

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| tiny | ~1GB | Fastest | Basic |
| base | ~1GB | Fast | Good |
| small | ~2GB | Medium | Better |
| medium | ~5GB | Slower | Great |
| large-v2 | ~10GB | Slow | Excellent |
| large-v3 | ~10GB | Slow | Best |

## Development

```bash
# Build locally
docker compose build

# Run tests
docker compose run --rm whisperx python -m pytest

# View logs
docker compose logs -f whisperx
```

## GitHub Actions

This repository uses GitHub Actions to automatically build and push Docker images to Docker Hub on every push to `main`.

### Required Secrets

Set these in your GitHub repository settings:

- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Your Docker Hub access token

## License

This project wraps [WhisperX](https://github.com/m-bain/whisperX) which is licensed under BSD-4-Clause.
