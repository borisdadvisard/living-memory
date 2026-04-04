"""
main.py — Living Memory orchestration

Start-up sequence:
  1. Ensure Ollama is running (launchctl load if not)
  2. Ensure nomic-embed-text is pulled
  3. Ensure kokoro_server.py is running on :8181

Voice loop (per turn):
  a. Record mic → Whisper STT → transcript
  b. Build graph context (HNSW semantic search + name search + neighbors)
  c. Call qwen3:30b with tool schemas (stream, think=False)
  d. Parse tool calls → async graph writes + background embedding jobs
  e. Print live graph growth counter

GPU timeline:
  whisper-cli subprocess exits → Ollama chat() starts → Kokoro on CPU
  (no Metal contention; nomic-embed-text only fires after audio plays)
"""

import io
import json
import logging
import os
import pathlib
import subprocess
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ollama
import requests
import sounddevice as sd

from graph_store import GraphStore, ALLOWED_RELATIONSHIPS
from voice_pipeline import (
    check_prerequisites,
    record_until_silence,
    transcribe,
    _flush_sentences,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_KEEP_ALIVE,
    KOKORO_VOICE,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ROOT  = pathlib.Path(__file__).parent
PYTHON_EXE    = str(PROJECT_ROOT / "venv" / "bin" / "python")
KOKORO_SCRIPT = str(PROJECT_ROOT / "kokoro_server.py")
KOKORO_URL    = "http://127.0.0.1:8181"
OLLAMA_PLIST  = pathlib.Path.home() / "Library/LaunchAgents/homebrew.mxcl.ollama.plist"

EMBED_MODEL   = "nomic-embed-text"
EMBED_DIM     = int(os.getenv("LM_EMBEDDING_DIM", "768"))

# How many graph context lines to inject into the system prompt
CONTEXT_LINES = 20

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("main")

# ---------------------------------------------------------------------------
# TOOL SCHEMAS
# ---------------------------------------------------------------------------

UPSERT_NODE_TOOL = {
    "type": "function",
    "function": {
        "name": "upsert_node",
        "description": (
            "Store or update a named entity in the knowledge graph. "
            "Call once per distinct entity mentioned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name":  {"type": "string",  "description": "Canonical entity name"},
                "label": {"type": "string",  "description": "Entity type, e.g. Person, Place, Concept, Event, Organisation"},
                "props": {"type": "object",  "description": "Optional key/value attributes"},
            },
            "required": ["name", "label"],
        },
    },
}

ADD_EDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "add_edge",
        "description": (
            "Record a relationship between two entities. "
            f"relationship must be one of: {', '.join(sorted(ALLOWED_RELATIONSHIPS))}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "from_name":     {"type": "string"},
                "to_name":       {"type": "string"},
                "relationship":  {"type": "string", "enum": sorted(ALLOWED_RELATIONSHIPS)},
                "weight":        {"type": "number", "default": 1.0},
            },
            "required": ["from_name", "to_name", "relationship"],
        },
    },
}

TOOLS = [UPSERT_NODE_TOOL, ADD_EDGE_TOOL]

SYSTEM_PROMPT_BASE = (
    "You are a helpful voice assistant with a persistent memory.\n"
    "Respond concisely in plain spoken language. "
    "Do not use markdown, bullet points, or special characters.\n"
    "After your spoken reply, call upsert_node and add_edge for every "
    "entity and relationship you learned this turn.\n"
)

# ---------------------------------------------------------------------------
# START-UP HELPERS
# ---------------------------------------------------------------------------

def ensure_ollama_running() -> ollama.Client:
    """Return an Ollama client, starting the service via launchctl if needed."""
    client = ollama.Client(host=OLLAMA_HOST)
    try:
        client.list()
        log.info("Ollama already running.")
        return client
    except Exception:
        pass

    log.info("Ollama not responding — loading launchd service…")
    subprocess.run(
        ["launchctl", "load", "-w", str(OLLAMA_PLIST)],
        check=False,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            client.list()
            log.info("Ollama started.")
            return client
        except Exception:
            time.sleep(1)

    log.error("Ollama did not start within 20 s. Check %s", OLLAMA_PLIST)
    sys.exit(1)


def ensure_embed_model(client: ollama.Client) -> bool:
    """Pull nomic-embed-text if absent. Returns True when available."""
    try:
        names = [m.model for m in client.list().models]
    except Exception:
        return False

    if any(EMBED_MODEL in n for n in names):
        return True

    log.info("Pulling %s…", EMBED_MODEL)
    try:
        client.pull(EMBED_MODEL)
        log.info("%s ready.", EMBED_MODEL)
        return True
    except Exception as exc:
        log.warning("Could not pull %s: %s — semantic context disabled", EMBED_MODEL, exc)
        return False


def ensure_kokoro_server() -> None:
    """Start kokoro_server.py if /health does not respond."""
    try:
        r = requests.get(f"{KOKORO_URL}/health", timeout=2)
        if r.status_code == 200:
            log.info("Kokoro server already running.")
            return
    except requests.ConnectionError:
        pass

    log.info("Starting Kokoro server…")
    subprocess.Popen(
        [PYTHON_EXE, KOKORO_SCRIPT],
        stdout=subprocess.DEVNULL,
        stderr=sys.stderr,   # surface model-load errors
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{KOKORO_URL}/health", timeout=1)
            if r.status_code == 200:
                log.info("Kokoro server ready.")
                return
        except requests.ConnectionError:
            time.sleep(0.5)

    log.error("Kokoro server did not start within 15 s.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# EMBEDDING + CONTEXT
# ---------------------------------------------------------------------------

def _get_embedding(client: ollama.Client, text: str) -> list[float] | None:
    try:
        resp = client.embed(model=EMBED_MODEL, input=text)
        return resp.embeddings[0]
    except Exception as exc:
        log.debug("embed failed: %s", exc)
        return None


def _build_context(
    gs: GraphStore,
    client: ollama.Client,
    transcript: str,
    embed_available: bool,
) -> str:
    """
    Build up to CONTEXT_LINES lines of graph context to inject into the prompt.
    Strategy:
      1. HNSW semantic search on transcript embedding (if available)
      2. Text keyword search on first word of transcript
      3. 1-hop neighbors of top hit
    Deduplicates by name; truncates at CONTEXT_LINES.
    """
    seen: set[str] = set()
    lines: list[str] = []

    def _add(name: str, label: str, extra: str = "") -> None:
        if name in seen or len(lines) >= CONTEXT_LINES:
            return
        seen.add(name)
        entry = f"[{label}] {name}"
        if extra:
            entry += f" — {extra}"
        lines.append(entry)

    # 1. Semantic search
    if embed_available:
        emb = _get_embedding(client, transcript)
        if emb:
            for hit in gs.semantic_search(emb, top_n=8):
                _add(hit["name"], hit["label"])

    # 2. Name keyword search using first meaningful word
    words = [w for w in transcript.split() if len(w) > 3]
    if words:
        for hit in gs.search_nodes_by_name(words[0], limit=5):
            _add(hit["name"], hit["label"])

    # 3. Expand neighbors of first match
    if seen:
        anchor = next(iter(seen))
        for nb in gs.query_neighbors(anchor, hops=1):
            _add(nb["name"], nb["label"], f"{nb['hops']}-hop")

    if not lines:
        return ""

    return "Relevant knowledge graph context:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# TTS PLAYBACK
# ---------------------------------------------------------------------------

def _tts_play(sentence: str) -> None:
    """Send one sentence to kokoro_server, decode WAV, play blocking."""
    if not sentence.strip():
        return
    try:
        resp = requests.post(
            f"{KOKORO_URL}/synthesise",
            json={"text": sentence, "voice": KOKORO_VOICE, "speed": 1.0},
            timeout=30,
        )
        resp.raise_for_status()
        wav_bytes = resp.content
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        sd.play(audio, samplerate=sr, blocking=True)
    except Exception as exc:
        log.warning("TTS playback failed: %s", exc)


# ---------------------------------------------------------------------------
# LLM STREAMING + TOOL COLLECTION
# ---------------------------------------------------------------------------

def _stream_response_and_collect_tools(
    client: ollama.Client,
    messages: list[dict],
) -> tuple[str, list[dict]]:
    """
    Stream qwen3:30b with tools and think=False.
    Synthesise and play each complete sentence as it arrives.
    Returns (full_response_text, list_of_tool_call_dicts).
    """
    full_response   = ""
    sentence_buffer = ""
    tool_calls: list[dict] = []

    stream = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        tools=TOOLS,
        stream=True,
        options={"think": False},
        keep_alive=OLLAMA_KEEP_ALIVE,
    )

    for chunk in stream:
        msg = chunk.message

        # Accumulate tool calls (delivered in the final chunk)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "name":      tc.function.name,
                    "arguments": tc.function.arguments or {},
                })

        fragment = msg.content or ""
        full_response   += fragment
        sentence_buffer += fragment

        sentences, sentence_buffer = _flush_sentences(sentence_buffer)
        for s in sentences:
            _tts_play(s)

        if chunk.done:
            break

    # Flush trailing fragment
    if sentence_buffer.strip():
        _tts_play(sentence_buffer)

    return full_response, tool_calls


# ---------------------------------------------------------------------------
# TOOL CALL APPLICATION
# ---------------------------------------------------------------------------

def _apply_tool_calls(
    gs: GraphStore,
    tool_calls: list[dict],
    embed_executor: ThreadPoolExecutor,
    client: ollama.Client,
    embed_available: bool,
) -> None:
    """
    Execute graph writes fire-and-forget; schedule background embedding jobs
    for any newly upserted nodes.
    """
    upserted_names: list[str] = []

    for tc in tool_calls:
        name = tc["name"]
        args = tc["arguments"]

        if name == "upsert_node":
            node_name  = args.get("name", "").strip()
            node_label = args.get("label", "Unknown").strip()
            props      = args.get("props") or {}
            if node_name:
                gs.submit_upsert_node(node_name, node_label, props)
                upserted_names.append(node_name)

        elif name == "add_edge":
            from_n = args.get("from_name", "").strip()
            to_n   = args.get("to_name", "").strip()
            rel    = args.get("relationship", "").strip()
            weight = float(args.get("weight", 1.0))
            if from_n and to_n and rel:
                gs.submit_add_edge(from_n, to_n, rel, weight)

    # Background embedding: runs after audio has played, no voice impact
    if embed_available and upserted_names:
        def _embed_and_store(node_name: str) -> None:
            emb = _get_embedding(client, node_name)
            if emb:
                gs.update_node_embedding(node_name, emb)

        for nm in upserted_names:
            embed_executor.submit(_embed_and_store, nm)


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def main() -> None:
    check_prerequisites()

    log.info("=== Living Memory starting up ===")

    # 1. Ollama
    client = ensure_ollama_running()

    # 2. Embedding model
    embed_available = ensure_embed_model(client)
    if not embed_available:
        log.warning("Semantic context disabled (nomic-embed-text unavailable).")

    # 3. Kokoro TTS server
    ensure_kokoro_server()

    # Graph store
    gs = GraphStore()
    nodes0, edges0 = gs.stats()
    log.info("Graph loaded: %d nodes, %d edges", nodes0, edges0)

    # Background thread for embedding jobs (single worker, non-blocking)
    embed_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedder")

    conversation: list[dict] = []

    log.info("=== Living Memory ready. Press Ctrl-C to quit. ===")

    try:
        while True:
            # --- Stage 1: record ---
            audio = record_until_silence()
            if audio.size == 0:
                log.info("No audio captured — skipping turn")
                continue

            # --- Stage 2: transcribe (Whisper exits before Ollama loads) ---
            transcript = transcribe(audio)
            if not transcript:
                log.info("Empty transcript — skipping turn")
                continue
            log.info("User: %s", transcript)

            # --- Stage 3: build graph context ---
            context = _build_context(gs, client, transcript, embed_available)

            # Build messages
            system_content = SYSTEM_PROMPT_BASE
            if context:
                system_content += "\n\n" + context

            messages = [{"role": "system", "content": system_content}]
            messages.extend(conversation)
            messages.append({"role": "user", "content": transcript})

            # --- Stage 4: LLM stream + TTS ---
            response_text, tool_calls = _stream_response_and_collect_tools(client, messages)
            log.info("Assistant: %s", response_text)

            # Maintain rolling conversation history (system prompt rebuilt each turn)
            conversation.append({"role": "user",      "content": transcript})
            conversation.append({"role": "assistant", "content": response_text})

            # --- Stage 5: async graph writes + background embeddings ---
            if tool_calls:
                _apply_tool_calls(gs, tool_calls, embed_executor, client, embed_available)

            # --- Stage 6: graph growth counter ---
            nodes_now, edges_now = gs.stats()
            delta_n = nodes_now - nodes0
            delta_e = edges_now - edges0
            print(
                f"  Graph: {nodes_now} nodes (+{delta_n}), "
                f"{edges_now} edges (+{delta_e})"
            )

    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        embed_executor.shutdown(wait=False)
        gs.close()


if __name__ == "__main__":
    main()
