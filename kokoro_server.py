"""
kokoro_server.py — Lightweight HTTP TTS server for Living Memory

Loads Kokoro ONNX once at startup and serves synthesis requests over HTTP so
that main.py can stream LLM chunks and send each sentence for synthesis
without re-initialising the ONNX runtime per turn.

Endpoints:
  GET  /health       → 200 "ok"
  POST /synthesise   → JSON {"text": str, "voice": str, "speed": float}
                      → audio/wav  (16-bit PCM, mono, 24 kHz)

Run:
  venv/bin/python kokoro_server.py           # binds 127.0.0.1:8181
  LM_KOKORO_PORT=8182 venv/bin/python kokoro_server.py
"""

import io
import json
import logging
import os
import pathlib
import struct
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import numpy as np
from kokoro_onnx import Kokoro

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PROJECT_ROOT  = pathlib.Path(__file__).parent
MODELS_DIR    = PROJECT_ROOT / "models"
KOKORO_MODEL  = str(MODELS_DIR / "kokoro-v1.0.onnx")
KOKORO_VOICES = str(MODELS_DIR / "voices-v1.0.bin")
KOKORO_SR     = 24000   # hard-coded in kokoro_onnx/config.py

HOST = "127.0.0.1"
PORT = int(os.getenv("LM_KOKORO_PORT", "8181"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s [kokoro_server] %(message)s")
log = logging.getLogger("kokoro_server")

# ---------------------------------------------------------------------------
# MODEL — loaded once at module import
# ---------------------------------------------------------------------------

os.environ.setdefault("ONNX_PROVIDER", "CPUExecutionProvider")
log.info("Loading Kokoro ONNX model…")
_TTS = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
log.info("Kokoro ready.")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _float32_to_wav_bytes(audio_np: np.ndarray, sr: int) -> bytes:
    """Convert float32 ndarray → int16 WAV bytes in memory."""
    pcm = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # int16 = 2 bytes
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# HTTP HANDLER
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):

    def log_message(self, *_):
        pass  # suppress per-request access log noise

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, b"ok", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/synthesise":
            self._send(404, b"not found", "text/plain")
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send(400, f"bad json: {exc}".encode(), "text/plain")
            return

        text  = payload.get("text", "").strip()
        voice = payload.get("voice", "af_bella")
        speed = float(payload.get("speed", 1.0))
        lang  = payload.get("lang", "en-us")

        if not text:
            self._send(400, b"text is empty", "text/plain")
            return

        try:
            audio_np, sr = _TTS.create(text, voice=voice, speed=speed, lang=lang)
            wav_bytes = _float32_to_wav_bytes(audio_np, sr)
            self._send(200, wav_bytes, "audio/wav")
        except Exception as exc:  # noqa: BLE001
            log.exception("synthesis error: %s", exc)
            self._send(500, str(exc).encode(), "text/plain")


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a thread (Kokoro is CPU-only and thread-safe)."""
    daemon_threads = True


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    server = _ThreadingHTTPServer((HOST, PORT), _Handler)
    log.info("Kokoro server listening on %s:%d", HOST, PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down.")
