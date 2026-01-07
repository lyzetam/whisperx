"""
WhisperX FastAPI Server

Provides HTTP endpoints for audio transcription with word-level timestamps
and optional speaker diarization.
"""

import os
import gc
import tempfile
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import torch

# PyTorch 2.6+ compatibility fix for pyannote VAD models
# These models use OmegaConf which isn't in torch's default safe globals
try:
    from omegaconf import DictConfig, ListConfig
    torch.serialization.add_safe_globals([DictConfig, ListConfig])
except ImportError:
    pass  # OmegaConf not installed, skip

import whisperx
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.getenv("HF_TOKEN")

# Global model cache
models = {}


def get_whisper_model():
    """Load or retrieve cached Whisper model."""
    if "whisper" not in models:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL} on {DEVICE}")
        models["whisper"] = whisperx.load_model(
            WHISPER_MODEL,
            DEVICE,
            compute_type=COMPUTE_TYPE
        )
    return models["whisper"]


def get_align_model(language_code: str):
    """Load or retrieve cached alignment model for language."""
    key = f"align_{language_code}"
    if key not in models:
        logger.info(f"Loading alignment model for: {language_code}")
        model_a, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=DEVICE
        )
        models[key] = (model_a, metadata)
    return models[key]


def get_diarize_model():
    """Load or retrieve cached diarization model."""
    if not HF_TOKEN:
        raise HTTPException(
            status_code=400,
            detail="HF_TOKEN required for speaker diarization"
        )
    if "diarize" not in models:
        logger.info("Loading diarization model")
        from whisperx.diarize import DiarizationPipeline
        models["diarize"] = DiarizationPipeline(
            use_auth_token=HF_TOKEN,
            device=DEVICE
        )
    return models["diarize"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload models on startup."""
    logger.info("WhisperX API starting...")
    logger.info(f"Device: {DEVICE}, Model: {WHISPER_MODEL}, Compute: {COMPUTE_TYPE}")

    # Optionally preload whisper model
    if os.getenv("PRELOAD_MODEL", "false").lower() == "true":
        get_whisper_model()

    yield

    # Cleanup on shutdown
    models.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="WhisperX API",
    description="Fast automatic speech recognition with word-level timestamps and speaker diarization",
    version="1.0.0",
    lifespan=lifespan
)


class TranscriptionResult(BaseModel):
    """Transcription response model."""
    segments: list = Field(description="Transcribed segments with timestamps")
    language: str = Field(description="Detected language code")
    word_segments: Optional[list] = Field(default=None, description="Word-level segments")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    device: str
    model: str
    cuda_available: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        device=DEVICE,
        model=WHISPER_MODEL,
        cuda_available=torch.cuda.is_available()
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "WhisperX API",
        "version": "1.0.0",
        "model": WHISPER_MODEL,
        "device": DEVICE,
        "endpoints": {
            "transcribe": "POST /transcribe",
            "health": "GET /health"
        }
    }


@app.post("/transcribe", response_model=TranscriptionResult)
async def transcribe(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(None, description="Language code (auto-detect if not specified)"),
    align: bool = Form(True, description="Enable word-level alignment"),
    diarize: bool = Form(False, description="Enable speaker diarization"),
    min_speakers: Optional[int] = Form(None, description="Minimum number of speakers"),
    max_speakers: Optional[int] = Form(None, description="Maximum number of speakers"),
    highlight_words: bool = Form(False, description="Include word-level timestamps")
):
    """
    Transcribe an audio file.

    Supports various audio formats (wav, mp3, m4a, etc.).

    - **file**: Audio file to transcribe
    - **language**: Force language (optional, auto-detected if not specified)
    - **align**: Enable word-level alignment (default: true)
    - **diarize**: Enable speaker diarization (requires HF_TOKEN)
    - **min_speakers/max_speakers**: Speaker count hints for diarization
    """
    # Save uploaded file temporarily
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load audio
        audio = whisperx.load_audio(tmp_path)

        # Transcribe
        model = get_whisper_model()
        result = model.transcribe(
            audio,
            batch_size=BATCH_SIZE,
            language=language
        )

        detected_language = result.get("language", language or "en")

        # Align if requested
        if align:
            try:
                model_a, metadata = get_align_model(detected_language)
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    DEVICE,
                    return_char_alignments=False
                )
            except Exception as e:
                logger.warning(f"Alignment failed: {e}")

        # Diarize if requested
        if diarize:
            try:
                diarize_model = get_diarize_model()
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                logger.warning(f"Diarization failed: {e}")

        return TranscriptionResult(
            segments=result.get("segments", []),
            language=detected_language,
            word_segments=result.get("word_segments") if highlight_words else None
        )

    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/transcribe/batch")
async def transcribe_batch(
    files: list[UploadFile] = File(..., description="Audio files to transcribe"),
    language: Optional[str] = Form(None),
    align: bool = Form(True),
):
    """
    Batch transcribe multiple audio files.

    Returns a list of transcription results.
    """
    results = []
    for file in files:
        try:
            result = await transcribe(
                file=file,
                language=language,
                align=align,
                diarize=False
            )
            results.append({
                "filename": file.filename,
                "success": True,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
