from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import os
import tempfile
import subprocess

import numpy as np
import torch
from speechbrain.pretrained import EncoderClassifier


app = FastAPI(
    title="Voice Embedding Service",
    version="2.1.0",
    description="Stateless audio-to-vector embedding. Trims silence and embeds a fixed 3-second window.",
)

classifier: Optional[EncoderClassifier] = None

TARGET_SR = 16000
TARGET_SECONDS = 3.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_SECONDS)


def resolve_device() -> str:
    requested = os.getenv("EMBED_DEVICE", "auto").lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = resolve_device()


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int
    norm: float
    model: str
    device: str
    target_seconds: float


class UrlRequest(BaseModel):
    url: HttpUrl


@app.on_event("startup")
def load_model():
    global classifier
    try:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": DEVICE},
        )
        print(f"[voice-embedder] Model loaded on device: {DEVICE}")
    except Exception as e:
        print(f"[voice-embedder] ERROR loading model: {e}")
        classifier = None


@app.get("/health")
def health():
    return {
        "status": "ok" if classifier is not None else "error",
        "model_loaded": classifier is not None,
        "device": DEVICE,
        "model": "speechbrain/spkrec-ecapa-voxceleb",
        "target_seconds": TARGET_SECONDS,
    }


def build_trim_filter() -> str:
    """
    Remove leading and trailing silence.
    """
    return (
        "silenceremove=start_periods=1:start_silence=0.10:start_threshold=-40dB,"
        "areverse,"
        "silenceremove=start_periods=1:start_silence=0.10:start_threshold=-40dB,"
        "areverse"
    )


def decode_with_ffmpeg_from_path(input_source: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    """
    Decode audio via ffmpeg, trim silence, force mono 16kHz, and return exactly 3s float32 PCM.

    Rules:
    - Trim leading/trailing silence
    - If shorter than 3s after trimming, repeat/loop the audio until it reaches 3s
    - If longer than 3s, trim to 3s
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_source,
        "-vn",
        "-ac", "1",
        "-ar", str(target_sr),
        "-af", build_trim_filter(),
        "-f", "f32le",
        "pipe:1",
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.returncode != 0:
        err_msg = proc.stderr.decode(errors="ignore")
        raise HTTPException(
            status_code=400,
            detail=f"ffmpeg failed to decode audio (code {proc.returncode}): {err_msg[:700]}",
        )

    audio_bytes = proc.stdout
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio samples decoded from ffmpeg.")

    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
    if audio_np.size == 0:
        raise HTTPException(status_code=400, detail="Decoded audio is empty.")

    # Force exact 3-second window
    # Short clips are looped, long clips are trimmed
    if audio_np.size < TARGET_SAMPLES:
        repeats = int(np.ceil(TARGET_SAMPLES / audio_np.size))
        audio_np = np.tile(audio_np, repeats)[:TARGET_SAMPLES]
    elif audio_np.size > TARGET_SAMPLES:
        audio_np = audio_np[:TARGET_SAMPLES]

    return torch.from_numpy(audio_np.copy()).unsqueeze(0)  # [1, TARGET_SAMPLES]


def decode_with_ffmpeg(file_bytes: bytes, original_filename: str | None = None, target_sr: int = TARGET_SR) -> torch.Tensor:
    """
    Decode arbitrary audio bytes to mono 16kHz float32 PCM via ffmpeg, trim silence,
    and return a fixed 3-second tensor.
    """
    suffix = ".bin"
    if original_filename:
        _, ext = os.path.splitext(original_filename)
        if ext:
            suffix = ext

    input_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
            tmp_in.write(file_bytes)
            tmp_in.flush()
            input_path = tmp_in.name

        return decode_with_ffmpeg_from_path(input_path, target_sr=target_sr)

    finally:
        if input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass


def embed_waveform(waveform: torch.Tensor) -> EmbeddingResponse:
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health.")

    waveform = waveform.to(DEVICE)

    try:
        with torch.no_grad():
            emb = classifier.encode_batch(waveform)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    emb = emb.squeeze()

    # Normalize for stable cosine comparisons
    emb = emb / torch.norm(emb, p=2)

    emb_list = emb.cpu().tolist()

    return EmbeddingResponse(
        embedding=emb_list,
        dimension=len(emb_list),
        norm=torch.norm(emb).item(),
        model="speechbrain/spkrec-ecapa-voxceleb",
        device=DEVICE,
        target_seconds=TARGET_SECONDS,
    )


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_audio(file: UploadFile = File(...)):
    """
    Multipart upload. Field name: file.
    Accepts any audio format ffmpeg can decode.
    Trims silence and embeds exactly 3 seconds.
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")
    waveform = decode_with_ffmpeg(file_bytes=file_bytes, original_filename=file.filename)
    return JSONResponse(content=embed_waveform(waveform).dict())


@app.post("/embed_raw", response_model=EmbeddingResponse)
async def embed_audio_raw(request: Request):
    """
    Raw binary body. No multipart wrapper.
    Ideal for n8n 'Send Binary Data' nodes.
    """
    file_bytes = await request.body()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty request body.")
    waveform = decode_with_ffmpeg(file_bytes=file_bytes, original_filename=None)
    return JSONResponse(content=embed_waveform(waveform).dict())


@app.post("/embed_url", response_model=EmbeddingResponse)
async def embed_audio_url(body: UrlRequest):
    """
    JSON body:
    {
      "url": "https://example.com/audio.mp3"
    }

    ffmpeg reads the remote URL directly, trims silence, and embeds exactly 3 seconds.
    """
    waveform = decode_with_ffmpeg_from_path(str(body.url), target_sr=TARGET_SR)
    return JSONResponse(content=embed_waveform(waveform).dict())