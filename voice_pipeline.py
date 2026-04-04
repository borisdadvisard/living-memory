"""
voice_pipeline.py — Living Memory voice loop

Stages:
  1. Record mic audio with energy-based VAD (sounddevice)
  2. Transcribe with whisper-cli subprocess (Metal GPU)
  3. Send transcript to Ollama qwen3:30b (localhost:11434, streaming)
  4. Synthesise each sentence with Kokoro ONNX (CPU)
  5. Play audio through speaker (sounddevice, blocking per sentence)

GPU timeline is strictly sequential: Whisper exits before Ollama starts;
Kokoro never touches Metal.
"""

import logging
import os
import re
import subprocess
import sys
import tempfile
import wave
import pathlib

import numpy as np
import sounddevice as sd
import ollama
from kokoro_onnx import Kokoro

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).parent
MODELS_DIR   = PROJECT_ROOT / "models"

# Whisper (Metal GPU via subprocess)
WHISPER_CLI        = "/opt/homebrew/Cellar/whisper-cpp/1.8.4/bin/whisper-cli"
WHISPER_MODEL      = str(MODELS_DIR / "ggml-base.en.bin")
WHISPER_THREADS    = 4          # sweet-spot for base.en on M-series

# Kokoro TTS (CPU only)
KOKORO_MODEL       = str(MODELS_DIR / "kokoro-v1.0.onnx")
KOKORO_VOICES      = str(MODELS_DIR / "voices-v1.0.bin")
KOKORO_VOICE       = "af_bella"
KOKORO_SPEED       = 1.0
KOKORO_LANG        = "en-us"
KOKORO_SR          = 24000      # hard-coded in kokoro_onnx/config.py

# Ollama LLM
OLLAMA_HOST        = "http://localhost:11434"
OLLAMA_MODEL       = "qwen3:30b"
OLLAMA_KEEP_ALIVE  = "10m"

# Microphone / VAD
MIC_SR             = 16000      # whisper-cli requires 16 kHz input
MIC_CHANNELS       = 1
MIC_DTYPE          = "int16"
BLOCK_FRAMES       = 1600       # 100 ms blocks at 16 kHz
VAD_THRESH         = int(os.getenv("LM_VAD_THRESHOLD", "300"))  # int16 RMS, 0–32768
VAD_SILENCE_BLOCKS = 15         # 1.5 s of silence ends the utterance
VAD_MAX_BLOCKS     = 300        # 30 s hard cap

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Respond concisely in plain spoken language. "
    "Do not use markdown, bullet points, or special characters."
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("voice_pipeline")

# ---------------------------------------------------------------------------
# PREREQUISITE CHECK
# ---------------------------------------------------------------------------

def check_prerequisites() -> None:
    """Fail fast with clear remediation messages before entering the loop."""
    missing = []

    if not pathlib.Path(WHISPER_CLI).exists():
        missing.append(
            f"whisper-cli not found at {WHISPER_CLI}\n"
            "  Install with: brew install whisper-cpp"
        )
    if not pathlib.Path(WHISPER_MODEL).exists():
        missing.append(
            f"Whisper model missing: {WHISPER_MODEL}\n"
            "  Download with:\n"
            "    mkdir -p models && curl -L -o models/ggml-base.en.bin \\\n"
            "      https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
        )
    if not pathlib.Path(KOKORO_MODEL).exists():
        missing.append(
            f"Kokoro model missing: {KOKORO_MODEL}\n"
            "  Download with:\n"
            "    curl -L -o models/kokoro-v1.0.onnx \\\n"
            "      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
        )
    if not pathlib.Path(KOKORO_VOICES).exists():
        missing.append(
            f"Kokoro voices missing: {KOKORO_VOICES}\n"
            "  Download with:\n"
            "    curl -L -o models/voices-v1.0.bin \\\n"
            "      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
        )

    if missing:
        for msg in missing:
            log.error(msg)
        sys.exit(1)


# ---------------------------------------------------------------------------
# STAGE 1 — RECORD WITH ENERGY-BASED VAD
# ---------------------------------------------------------------------------

def record_until_silence() -> np.ndarray:
    """
    Open the default mic at 16 kHz / mono / int16.
    Wait for speech onset (RMS >= VAD_THRESH), then record until
    VAD_SILENCE_BLOCKS consecutive silent blocks have elapsed.

    Returns a 1-D int16 ndarray, or empty array if nothing was captured.
    """
    log.info("Listening… (speak now)")
    frames: list[np.ndarray] = []
    silence_count = 0
    speech_started = False

    with sd.InputStream(
        samplerate=MIC_SR,
        channels=MIC_CHANNELS,
        dtype=MIC_DTYPE,
        blocksize=BLOCK_FRAMES,
    ) as stream:
        while True:
            block, overflowed = stream.read(BLOCK_FRAMES)
            if overflowed:
                log.warning("Audio input buffer overflowed — some samples dropped")

            rms = int(np.sqrt(np.mean(block.astype(np.float32) ** 2)))

            if not speech_started:
                if rms >= VAD_THRESH:
                    speech_started = True
                    frames.append(block.copy())
                continue

            frames.append(block.copy())

            if rms < VAD_THRESH:
                silence_count += 1
                if silence_count >= VAD_SILENCE_BLOCKS:
                    log.info("Silence detected — stopping recording")
                    break
            else:
                silence_count = 0

            if len(frames) >= VAD_MAX_BLOCKS:
                log.warning("Max recording duration reached (30 s)")
                break

    if not frames:
        return np.array([], dtype=np.int16)

    return np.concatenate([f.flatten() for f in frames])


# ---------------------------------------------------------------------------
# STAGE 2 — TRANSCRIBE WITH WHISPER-CLI (Metal GPU)
# ---------------------------------------------------------------------------

def transcribe(audio_int16: np.ndarray) -> str:
    """
    Write audio to a temp WAV (16 kHz / int16 / mono), call whisper-cli,
    return the transcript from stdout. Cleans up the temp file on exit.

    --no-prints  (-np): suppresses all stdout except the transcript
    --no-timestamps (-nt): raw text only, no [00:00:00] prefixes
    Metal init noise goes to stderr and is captured separately.
    """
    if audio_int16.size == 0:
        return ""

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(MIC_CHANNELS)
            wf.setsampwidth(2)          # int16 = 2 bytes
            wf.setframerate(MIC_SR)
            wf.writeframes(audio_int16.tobytes())

        result = subprocess.run(
            [
                WHISPER_CLI,
                "--model",        WHISPER_MODEL,
                "--file",         tmp_path,
                "--language",     "en",
                "--threads",      str(WHISPER_THREADS),
                "--no-prints",
                "--no-timestamps",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        transcript = result.stdout.strip()
        if result.returncode != 0 and not transcript:
            log.error("whisper-cli exited %d: %s", result.returncode,
                      result.stderr[:300])
        return transcript

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# STAGE 3 + 4 + 5 — LLM (stream) → TTS → PLAY  (sentence-by-sentence)
# ---------------------------------------------------------------------------

def _flush_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Extract complete sentences from buffer using lookbehind on [.!?].
    Returns (complete_sentences, remaining_fragment).
    """
    parts = re.split(r"(?<=[.!?])\s+", buffer)
    if len(parts) > 1:
        return parts[:-1], parts[-1]
    return [], buffer


def _synthesise_and_play(tts: Kokoro, text: str) -> None:
    """Synthesise one sentence on CPU and play it blocking."""
    if not text.strip():
        return
    audio_np, sr = tts.create(
        text,
        voice=KOKORO_VOICE,
        speed=KOKORO_SPEED,
        lang=KOKORO_LANG,
    )
    sd.play(audio_np, samplerate=sr, blocking=True)


def respond_and_speak(tts: Kokoro, messages: list[dict]) -> str:
    """
    Stream the Ollama response, synthesise and play each complete sentence
    as soon as it arrives. Returns the full response text.
    """
    client = ollama.Client(host=OLLAMA_HOST)
    full_response = ""
    sentence_buffer = ""

    stream = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=True,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )

    for chunk in stream:
        fragment = chunk.message.content or ""
        full_response += fragment
        sentence_buffer += fragment

        sentences, sentence_buffer = _flush_sentences(sentence_buffer)
        for sentence in sentences:
            _synthesise_and_play(tts, sentence)

        if chunk.done:
            break

    # Flush any trailing fragment (response may not end with punctuation)
    if sentence_buffer.strip():
        _synthesise_and_play(tts, sentence_buffer)

    return full_response


# ---------------------------------------------------------------------------
# INITIALISATION
# ---------------------------------------------------------------------------

def _build_tts() -> Kokoro:
    """
    Load Kokoro with explicit CPUExecutionProvider.
    onnxruntime 1.24.4 (non-GPU build) defaults to CPU already;
    the env var guards against any CoreML fallback on macOS.
    """
    os.environ.setdefault("ONNX_PROVIDER", "CPUExecutionProvider")
    return Kokoro(KOKORO_MODEL, KOKORO_VOICES)


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def main() -> None:
    check_prerequisites()

    log.info("Loading Kokoro TTS (CPU)…")
    tts = _build_tts()
    log.info("Kokoro ready. Voice: %s", KOKORO_VOICE)

    conversation: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    log.info("Voice pipeline ready. Press Ctrl-C to quit.")

    while True:
        try:
            # Stage 1 — Record
            audio = record_until_silence()
            if audio.size == 0:
                log.info("No audio captured — skipping turn")
                continue

            # Stage 2 — Transcribe (Whisper on Metal)
            transcript = transcribe(audio)
            if not transcript:
                log.info("Empty transcript — skipping turn")
                continue
            log.info("User: %s", transcript)

            # Stages 3+4+5 — LLM stream + TTS + play
            conversation.append({"role": "user", "content": transcript})
            response = respond_and_speak(tts, conversation)
            log.info("Assistant: %s", response)
            conversation.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            log.info("Shutting down.")
            break
        except ollama.ResponseError as exc:
            log.error("Ollama error: %s", exc)
            # Remove the poisoned user message so history stays clean
            if conversation and conversation[-1]["role"] == "user":
                conversation.pop()
        except subprocess.TimeoutExpired:
            log.error("whisper-cli timed out — skipping turn")
        except Exception as exc:  # noqa: BLE001
            log.exception("Unexpected error: %s", exc)


if __name__ == "__main__":
    main()
