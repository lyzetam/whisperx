#!/bin/bash
set -e

# WhisperX Docker Entrypoint
# Supports CLI mode and API server mode

if [ "$1" = "api" ]; then
    echo "Starting WhisperX API server..."
    exec python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
elif [ "$1" = "worker" ]; then
    echo "Starting WhisperX worker..."
    exec python3 -m api.worker
elif [ "$1" = "--help" ] || [ -z "$1" ]; then
    echo "WhisperX Docker Container"
    echo ""
    echo "Usage:"
    echo "  docker run whisperx api                    # Start API server"
    echo "  docker run whisperx worker                 # Start background worker"
    echo "  docker run whisperx <audio_file> [options] # Transcribe audio file"
    echo ""
    echo "Environment Variables:"
    echo "  WHISPER_MODEL   - Model size (tiny, base, small, medium, large-v2, large-v3)"
    echo "  COMPUTE_TYPE    - Compute type (float16, int8, float32)"
    echo "  BATCH_SIZE      - Batch size for inference"
    echo "  DEVICE          - Device (cuda, cpu)"
    echo "  HF_TOKEN        - Hugging Face token for speaker diarization"
    echo ""
    echo "Examples:"
    echo "  # Transcribe with default settings"
    echo "  docker run -v /path/to/audio:/data whisperx /data/audio.wav"
    echo ""
    echo "  # Transcribe with speaker diarization"
    echo "  docker run -e HF_TOKEN=xxx -v /path/to/audio:/data whisperx /data/audio.wav --diarize"
    echo ""
    echo "  # Start API server"
    echo "  docker run -p 8000:8000 whisperx api"
else
    # CLI transcription mode
    ARGS=""

    # Add model if specified
    if [ -n "$WHISPER_MODEL" ]; then
        ARGS="$ARGS --model $WHISPER_MODEL"
    fi

    # Add compute type if specified
    if [ -n "$COMPUTE_TYPE" ]; then
        ARGS="$ARGS --compute_type $COMPUTE_TYPE"
    fi

    # Add batch size if specified
    if [ -n "$BATCH_SIZE" ]; then
        ARGS="$ARGS --batch_size $BATCH_SIZE"
    fi

    # Add device if specified
    if [ -n "$DEVICE" ]; then
        ARGS="$ARGS --device $DEVICE"
    fi

    # Add HF token for diarization if specified
    if [ -n "$HF_TOKEN" ]; then
        ARGS="$ARGS --hf_token $HF_TOKEN"
    fi

    echo "Running: whisperx $@ $ARGS"
    exec whisperx "$@" $ARGS
fi
