"""
main.py — Living Memory orchestration

Start-up sequence:
  1. Ensure mlx-lm server is running on :8080 (started as subprocess if needed)
  2. Load mlx-embeddings model in-process (mxbai-embed-large-v1, 1024-dim)
  3. Ensure kokoro_server.py is running on :8181

Two-pass LLM architecture (per turn):
  Pass 1 — silent, full tool schema:
    Build graph context → call LLM with TOOLS → collect + apply tool calls
    Text output discarded; no TTS; graph writes fire synchronously.
  Pass 2 — speak-only, no tools, constrained:
    Short SPEAK_PROMPT + tool_summary injected as context → stream to browser
    For voice turns: each sentence streamed to Kokoro TTS as it arrives.
    max_tokens=80 (voice) / 200 (text) → TTFT ~0.4s (no tool schema overhead).

Background jobs:
  Embedding generation fires after Pass 2 audio plays (no voice latency impact).

GPU timeline:
  whisper-cli subprocess exits → mlx-lm Pass 1 → mlx-lm Pass 2 → Kokoro on CPU
  (embeddings fire in background thread after audio completes)
"""

import io
import json
import logging
import os
import pathlib
import subprocess
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
import requests
import sounddevice as sd
from mlx_embeddings.utils import load as mlx_embed_load, generate as mlx_embed_generate
from openai import OpenAI

import viz_server

from graph_store import (
    GraphStore,
    ALLOWED_RELATION_TYPES,
    ALLOWED_RELATIONSHIPS,
    ENTITY_TYPES,
    EVENT_TYPES,
    COMMITMENT_STATUSES,
    COMMITMENT_PRIORITIES,
)
from spotify_tool import build_spotify_tools
from voice_pipeline import (
    check_prerequisites,
    record_until_silence,
    transcribe,
    _flush_sentences,
    MLX_LM_HOST,
    MLX_LM_MODEL,
    KOKORO_VOICE,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ROOT   = pathlib.Path(__file__).parent
PYTHON_EXE     = str(PROJECT_ROOT / "venv" / "bin" / "python")
KOKORO_SCRIPT  = str(PROJECT_ROOT / "kokoro_server.py")
KOKORO_URL     = "http://127.0.0.1:8181"
MLX_LM_SCRIPT  = "-m"   # launched as: python -m mlx_lm server ...

EMBED_MODEL    = "mlx-community/mxbai-embed-large-v1"
EMBED_DIM      = int(os.getenv("LM_EMBEDDING_DIM", "1024"))

# Module-level embedding model (loaded once at startup, reused across turns)
_embed_model      = None
_embed_tokenizer  = None

# How many graph context lines to inject into the system prompt
CONTEXT_LINES = 24

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("main")

# ---------------------------------------------------------------------------
# TOOL SCHEMAS
# ---------------------------------------------------------------------------

UPSERT_ENTITY_TOOL = {
    "type": "function",
    "function": {
        "name": "upsert_entity",
        "description": "Store or update an entity. Call for each person, place, org, object, or concept mentioned.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Entity name",
                },
                "type": {
                    "type": "string",
                    "enum": sorted(ENTITY_TYPES),
                    "description": "Entity type",
                },
                "attributes": {
                    "type": "object",
                    "description": "Key-value attributes",
                },
                "aliases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Aliases or abbreviations",
                },
            },
            "required": ["name", "type"],
        },
    },
}

UPSERT_EVENT_TOOL = {
    "type": "function",
    "function": {
        "name": "upsert_event",
        "description": "Record an event: delivery, meeting, conversation, transaction, or state change.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title",
                },
                "type": {
                    "type": "string",
                    "enum": sorted(EVENT_TYPES),
                    "description": "Event type",
                },
                "description": {
                    "type": "string",
                    "description": "Longer description",
                },
                "start_time": {
                    "type": "string",
                    "description": "ISO-8601 start time",
                },
                "end_time": {
                    "type": "string",
                    "description": "ISO-8601 end time",
                },
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Participant entity names",
                },
                "location": {
                    "type": "string",
                    "description": "Location entity name",
                },
            },
            "required": ["title", "type"],
        },
    },
}

UPSERT_COMMITMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "upsert_commitment",
        "description": "Record a commitment or task with an owner and deadline.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Obligation title",
                },
                "description": {
                    "type": "string",
                    "description": "Longer explanation",
                },
                "owner": {
                    "type": "string",
                    "description": "Responsible entity name",
                },
                "due_time": {
                    "type": "string",
                    "description": "ISO-8601 deadline",
                },
                "status": {
                    "type": "string",
                    "enum": sorted(COMMITMENT_STATUSES),
                    "description": "Current status",
                },
                "priority": {
                    "type": "string",
                    "enum": sorted(COMMITMENT_PRIORITIES),
                    "description": "Importance level",
                },
                "related_event_title": {
                    "type": "string",
                    "description": "Related event title",
                },
            },
            "required": ["title"],
        },
    },
}

ADD_RELATION_TOOL = {
    "type": "function",
    "function": {
        "name": "add_relation",
        "description": "Record a typed relationship between two objects.",
        "parameters": {
            "type": "object",
            "properties": {
                "from_name": {
                    "type": "string",
                    "description": "Source name or title",
                },
                "from_object_type": {
                    "type": "string",
                    "enum": ["entity", "event", "commitment"],
                    "description": "Source object type",
                },
                "to_name": {
                    "type": "string",
                    "description": "Target name or title",
                },
                "to_object_type": {
                    "type": "string",
                    "enum": ["entity", "event", "commitment"],
                    "description": "Target object type",
                },
                "type": {
                    "type": "string",
                    "enum": sorted(ALLOWED_RELATION_TYPES),
                    "description": "Relation type",
                },
                "valid_from": {
                    "type": "string",
                    "description": "ISO-8601 start",
                },
                "valid_to": {
                    "type": "string",
                    "description": "ISO-8601 end",
                },
            },
            "required": ["from_name", "from_object_type", "to_name", "to_object_type", "type"],
        },
    },
}

DELETE_ENTITY_TOOL = {
    "type": "function",
    "function": {
        "name": "delete_entity",
        "description": "Permanently delete an entity and all its relations. Only use when explicitly asked to forget something.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Entity name",
                },
            },
            "required": ["name"],
        },
    },
}

TOOLS = [UPSERT_ENTITY_TOOL, UPSERT_EVENT_TOOL, UPSERT_COMMITMENT_TOOL,
         ADD_RELATION_TOOL, DELETE_ENTITY_TOOL]

# Populated by main() if Spotify credentials are available
spotify_client = None

# ---------------------------------------------------------------------------
# BROWSER CHAT HANDLER
# ---------------------------------------------------------------------------

_chat_lock = threading.Lock()


def make_chat_handler(
    client: OpenAI,
    gs: GraphStore,
    conversation: list[dict],
    embed_available: bool,
    embed_executor: ThreadPoolExecutor,
):
    """
    Return a generator function: handle(message) → yields SSE-formatted strings.
    Called by viz_server for each POST /chat request.

    Two-pass architecture:
      Pass 1  silent, full TOOLS schema → collect + apply graph writes
      Pass 2  no tools, SPEAK_PROMPT + tool_summary → stream spoken reply
              max_tokens=80 (voice) or 200 (text) for sub-second TTFT
    """

    def handle(message: str):
        actual   = message.strip()
        is_voice = (actual == "/v")
        context  = ""  # populated in Pass 1, forwarded to Pass 2

        # ── Voice input: record + transcribe ─────────────────────────────────
        if is_voice:
            yield f"data: {json.dumps({'type': 'status', 'text': 'Listening…'})}\n\n"
            try:
                audio = record_until_silence()
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'text': f'Mic error: {exc}'})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
            if audio.size == 0:
                yield f"data: {json.dumps({'type': 'status', 'text': 'No audio captured.'})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
            yield f"data: {json.dumps({'type': 'status', 'text': 'Transcribing…'})}\n\n"
            actual = transcribe(audio)
            if not actual:
                yield f"data: {json.dumps({'type': 'status', 'text': 'Nothing transcribed.'})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
            log.info("[voice] %s", actual)
            yield f"data: {json.dumps({'type': 'transcript', 'text': actual})}\n\n"

        with _chat_lock:
            # ── Pass 1: silent — graph writes only ────────────────────────────
            context = _build_context(gs, actual, embed_available)
            system_p1 = SYSTEM_PROMPT_BASE + ("\n\n" + context if context else "")

            p1_msgs = [{"role": "system", "content": system_p1}]
            p1_msgs.extend(conversation)
            p1_msgs.append({"role": "user", "content": actual})

            collected_tools: list[dict] = []
            tool_results:    list[str]  = []

            try:
                _, collected_tools = _run_pass1(client, p1_msgs)
            except Exception as exc:
                log.warning("Pass 1 error: %s", exc)
                yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            if collected_tools:
                tool_results = _apply_tool_calls(
                    gs, collected_tools, embed_executor, embed_available
                )
                for tc in collected_tools:
                    args  = tc["arguments"]
                    label = (
                        args.get("name") or args.get("title")
                        or args.get("query") or tc["name"]
                    ).strip()
                    yield f"data: {json.dumps({'type': 'tool', 'fn': tc['name'], 'label': label})}\n\n"

            # ── Pass 2: speak-only — fast spoken reply ────────────────────────
            # Inject what was just written so the reply can reference it.
            tool_summary = _build_tool_summary(collected_tools)
            user_p2 = actual
            if tool_summary:
                user_p2 += f"\n\n[{tool_summary}]"
            if tool_results:
                # Surface Spotify / query-tool results as live context
                user_p2 += "\n\n[Live data: " + "; ".join(r for r in tool_results if r) + "]"
            if context:
                # Forward existing graph context so Pass 2 can answer retrieval questions
                graph_lines = context.replace("Relevant knowledge graph context:\n", "")
                user_p2 += f"\n\n[Memory: {graph_lines}]"

            # Prevent Pass 2 from hallucinating Spotify actions when no tool fired
            _play_words = {"play", "pause", "skip", "stop", "resume", "queue"}
            if not tool_results and any(w in actual.lower() for w in _play_words):
                user_p2 += (
                    "\n\n[No Spotify action was executed this turn. "
                    "Do NOT claim you played, paused, or queued anything. "
                    "If the user asked to play something, say you could not do it and ask them to try again.]"
                )

            p2_msgs = [{"role": "system", "content": SPEAK_PROMPT}]
            p2_msgs.extend(conversation)
            p2_msgs.append({"role": "user", "content": user_p2})

            # Voice turns are token-budget constrained for under-second TTFT;
            # text turns allow a fuller reply.
            max_tok = 80 if is_voice else 200

            full_response   = ""
            sentence_buffer = ""

            try:
                stream = client.chat.completions.create(
                    model=MLX_LM_MODEL,
                    messages=p2_msgs,
                    stream=True,
                    max_tokens=max_tok,
                )

                for chunk in stream:
                    choice = chunk.choices[0] if chunk.choices else None
                    if choice is None:
                        continue
                    fragment = choice.delta.content or ""
                    full_response += fragment
                    if fragment:
                        yield f"data: {json.dumps({'type': 'token', 'text': fragment})}\n\n"

                    if is_voice:
                        sentence_buffer += fragment
                        sentences, sentence_buffer = _flush_sentences(sentence_buffer)
                        for s in sentences:
                            _tts_play(s)

                    if choice.finish_reason:
                        break

                # Flush any trailing fragment
                if is_voice and sentence_buffer.strip():
                    _tts_play(sentence_buffer)

            except Exception as exc:
                log.warning("Pass 2 error: %s", exc)
                yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"

            # Append the Pass 2 response to history (what the user actually heard/saw)
            conversation.append({"role": "user",      "content": actual})
            conversation.append({"role": "assistant", "content": full_response})
            log.info(
                "2-pass turn: %d tool(s) | %s → %s…",
                len(collected_tools), actual[:50], full_response[:80],
            )

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return handle


SYSTEM_PROMPT_BASE = (
    "You are a helpful voice assistant with a persistent memory.\n"
    "Respond concisely in plain spoken language. "
    "Do not use markdown, bullet points, or special characters.\n"
    "After your spoken reply, call upsert_entity for every person, place, or "
    "organisation mentioned — one call per entity, never combined. "
    "Call upsert_event for every meeting, delivery, or state change. "
    "Call upsert_commitment for every task or obligation. "
    "Call add_relation for every relationship you learned. "
    "Use ISO-8601 dates wherever times are mentioned.\n"
    "When a task or commitment is mentioned as completed, finished, cancelled, or "
    "done, call upsert_commitment with the matching title and status set to the "
    "appropriate value. Only call delete_entity when the user explicitly asks to "
    "forget or remove something.\n"
    "Do NOT log Spotify actions to the knowledge graph. "
    "When the user asks to play or hear a song: call spotify_play immediately with "
    "the song name as the query — never call spotify_now_playing first. "
    "If the song name comes from graph memory context (e.g. 'play the song in my memory'), "
    "extract the entity name from the context and use it as the spotify_play query. "
    "Use spotify_control for pause/resume/skip/volume. "
    "Use spotify_now_playing ONLY when asked what is currently playing. "
    "Use spotify_recently_played for listening history. "
    "Use spotify_queue for what is coming up next.\n"
)

# Pass 2 spoken-reply prompt — kept deliberately short (~25 tokens) so
# TTFT on Pass 2 approaches the bare-minimum baseline (~0.4 s on 7B-4bit).
SPEAK_PROMPT = (
    "You are a helpful voice assistant. "
    "Reply in 1–2 spoken sentences. "
    "No markdown, no bullet points, no filler phrases."
)

# ---------------------------------------------------------------------------
# START-UP HELPERS
# ---------------------------------------------------------------------------

def ensure_mlx_server() -> OpenAI:
    """
    Return an OpenAI client pointed at the mlx-lm server.
    Starts the server as a subprocess if it is not already responding.
    The first startup can be slow if the model weights need downloading.
    """
    client = OpenAI(base_url=f"{MLX_LM_HOST}/v1", api_key="mlx")
    try:
        client.models.list()
        log.info("mlx-lm server already running.")
        return client
    except Exception:
        pass

    log.info("Starting mlx-lm server (model: %s)…", MLX_LM_MODEL)
    subprocess.Popen(
        [PYTHON_EXE, MLX_LM_SCRIPT, "mlx_lm", "server",
         "--model", MLX_LM_MODEL,
         "--host", "127.0.0.1",
         "--port", "8080"],
        stdout=subprocess.DEVNULL,
        stderr=sys.stderr,
    )

    # Allow up to 120 s — first run downloads weights (~4 GB)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            client.models.list()
            log.info("mlx-lm server ready.")
            return client
        except Exception:
            time.sleep(2)

    log.error("mlx-lm server did not start within 120 s.")
    sys.exit(1)


def ensure_embed_model() -> bool:
    """
    Load the mlx-embeddings model into the module-level globals.
    Returns True on success, False if unavailable (semantic context disabled).
    """
    global _embed_model, _embed_tokenizer
    try:
        log.info("Loading embedding model %s…", EMBED_MODEL)
        _embed_model, _embed_tokenizer = mlx_embed_load(EMBED_MODEL)
        log.info("Embedding model ready (%d-dim).", EMBED_DIM)
        return True
    except Exception as exc:
        log.warning("Could not load embedding model: %s — semantic context disabled", exc)
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
        stderr=sys.stderr,
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

def _get_embedding(text: str) -> list[float] | None:
    """Embed text with the in-process mlx-embeddings model; L2-normalise the vector."""
    if _embed_model is None:
        return None
    try:
        out = mlx_embed_generate(_embed_model, _embed_tokenizer, [text])
        v   = out.text_embeds[0]              # shape (EMBED_DIM,)
        v   = v / mx.sqrt(mx.sum(v * v))      # L2 normalise
        return v.tolist()
    except Exception as exc:
        log.debug("embed failed: %s", exc)
        return None


def _build_context(
    gs: GraphStore,
    transcript: str,
    embed_available: bool,
) -> str:
    """
    Build up to CONTEXT_LINES lines of graph context to inject into the prompt.
    Strategy:
      1. HNSW semantic search across all object types (entities, events, commitments)
      2. Cross-type name keyword search on first meaningful word
      3. 1-hop neighbors of top hit
      4. Active commitments (always included, up to 3)
    Deduplicates by object_id; truncates at CONTEXT_LINES.
    """
    seen_ids: set[str] = set()
    lines: list[str] = []

    def _add(object_id: str, object_type: str, subtype: str, display_name: str,
             extra: str = "") -> None:
        if object_id in seen_ids or len(lines) >= CONTEXT_LINES:
            return
        seen_ids.add(object_id)
        entry = f"[{object_type}:{subtype}] {display_name}"
        if extra:
            entry += f" — {extra}"
        lines.append(entry)

    # 1. Semantic search across all types
    if embed_available:
        emb = _get_embedding(transcript)
        if emb:
            for hit in gs.semantic_search(emb, top_n=8):
                _add(hit["object_id"], hit["object_type"],
                     hit["subtype"] or "", hit["display_name"])

    # 2. Cross-type name search using all meaningful words
    words = [w for w in transcript.split() if len(w) > 3]
    seen_keywords: set[str] = set()
    for word in words:
        if word.lower() in seen_keywords:
            continue
        seen_keywords.add(word.lower())
        for hit in gs.search_by_name(word, limit=3):
            _add(hit["object_id"], hit["object_type"],
                 hit["subtype"] or "", hit["display_name"])
        if len(lines) >= CONTEXT_LINES:
            break

    # 3. Expand neighbors of each matched hit with relation types
    for hit_id in list(seen_ids):
        # Determine type of this hit
        hit_type = "entity"
        with gs._lock:
            if gs._conn.execute(
                "SELECT 1 FROM events WHERE id::TEXT = ?", [hit_id]
            ).fetchone():
                hit_type = "event"
            elif gs._conn.execute(
                "SELECT 1 FROM commitments WHERE id::TEXT = ?", [hit_id]
            ).fetchone():
                hit_type = "commitment"
        for rel in gs.get_direct_relations(hit_id, hit_type):
            neighbor_name = rel["neighbor_name"]
            rel_type      = rel["rel_type"]
            direction     = rel["direction"]
            extra = (
                f"—[{rel_type}]→ {neighbor_name}"
                if direction == "outgoing"
                else f"←[{rel_type}]— {neighbor_name}"
            )
            _add(rel["neighbor_id"], rel["neighbor_type"], "", neighbor_name, extra)
        if len(lines) >= CONTEXT_LINES:
            break

    # 4. Active commitments — always inject up to 3
    if len(lines) < CONTEXT_LINES:
        for c in gs.get_active_commitments()[:3]:
            due = f"due {c['due_time']}" if c["due_time"] else "no deadline"
            _add(c["id"], "commitment", c["status"], c["title"],
                 f"{c['priority']} priority, {due}")

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
    client: OpenAI,
    messages: list[dict],
    speak: bool = True,
) -> tuple[str, list[dict]]:
    """
    Stream the mlx-lm response with tools (OpenAI-compatible streaming).
    When speak=True: synthesise and play each complete sentence via TTS.
    When speak=False: stream tokens live to stdout.
    Returns (full_response_text, list_of_tool_call_dicts).
    """
    full_response   = ""
    sentence_buffer = ""
    tc_accum: dict[int, dict] = {}   # index → {name, arguments_str}

    stream = client.chat.completions.create(
        model=MLX_LM_MODEL,
        messages=messages,
        tools=TOOLS,
        stream=True,
    )

    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if choice is None:
            continue
        delta = choice.delta

        # Accumulate tool-call deltas (name + arguments arrive in pieces)
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tc_accum:
                    tc_accum[idx] = {"name": "", "arguments_str": ""}
                if tc_delta.function and tc_delta.function.name:
                    tc_accum[idx]["name"] += tc_delta.function.name
                if tc_delta.function and tc_delta.function.arguments:
                    tc_accum[idx]["arguments_str"] += tc_delta.function.arguments

        fragment = delta.content or ""
        full_response   += fragment
        sentence_buffer += fragment

        if speak:
            sentences, sentence_buffer = _flush_sentences(sentence_buffer)
            for s in sentences:
                _tts_play(s)
        else:
            if fragment:
                print(fragment, end="", flush=True)

        if choice.finish_reason:
            break

    if speak:
        if sentence_buffer.strip():
            _tts_play(sentence_buffer)
    else:
        print()

    # Parse accumulated tool calls
    tool_calls: list[dict] = []
    for idx in sorted(tc_accum.keys()):
        tc = tc_accum[idx]
        try:
            args = json.loads(tc["arguments_str"]) if tc["arguments_str"] else {}
        except json.JSONDecodeError:
            args = {}
        tool_calls.append({"name": tc["name"], "arguments": args})

    return full_response, tool_calls


# ---------------------------------------------------------------------------
# PASS 1 + PASS 2 HELPERS
# ---------------------------------------------------------------------------

def _run_pass1(
    client: OpenAI,
    messages: list[dict],
) -> tuple[str, list[dict]]:
    """
    Pass 1 — silent tool-collection stream.
    Sends the full TOOLS schema; accumulates all tool-call deltas.
    Text output is discarded (not spoken, not shown).
    Returns (discarded_text, list_of_{name,arguments}_dicts).
    """
    full_text = ""
    tc_accum: dict[int, dict] = {}

    stream = client.chat.completions.create(
        model=MLX_LM_MODEL,
        messages=messages,
        tools=TOOLS,
        stream=True,
    )

    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if choice is None:
            continue
        delta = choice.delta

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tc_accum:
                    tc_accum[idx] = {"name": "", "arguments_str": ""}
                if tc_delta.function and tc_delta.function.name:
                    tc_accum[idx]["name"] += tc_delta.function.name
                if tc_delta.function and tc_delta.function.arguments:
                    tc_accum[idx]["arguments_str"] += tc_delta.function.arguments

        full_text += delta.content or ""

        if choice.finish_reason:
            break

    tool_calls: list[dict] = []
    for idx in sorted(tc_accum.keys()):
        tc = tc_accum[idx]
        try:
            args = json.loads(tc["arguments_str"]) if tc["arguments_str"] else {}
        except json.JSONDecodeError:
            args = {}
        tool_calls.append({"name": tc["name"], "arguments": args})

    return full_text, tool_calls


def _build_tool_summary(tool_calls: list[dict]) -> str:
    """
    Build a compact human-readable summary of graph writes to inject into
    Pass 2's user message, so the spoken reply can reference what was noted.

    Example: "Noted: Sarah (person), meeting 'Q3 planning' at 2026-05-20T14:00,
              task 'Review roadmap' [planned], Sarah works_at Anthropic"
    """
    if not tool_calls:
        return ""

    parts: list[str] = []
    for tc in tool_calls:
        name = tc["name"]
        args = tc["arguments"]

        if name == "upsert_entity":
            ename = args.get("name", "")
            etype = args.get("type", "entity")
            if ename:
                parts.append(f"{ename} ({etype})")

        elif name == "upsert_event":
            title = args.get("title", "")
            t     = args.get("start_time", "")
            if title:
                parts.append(f"event '{title}'" + (f" at {t}" if t else ""))

        elif name == "upsert_commitment":
            title  = args.get("title", "")
            status = args.get("status", "planned")
            if title:
                parts.append(f"task '{title}' [{status}]")

        elif name == "add_relation":
            fn  = args.get("from_name", "")
            rel = args.get("type", "")
            tn  = args.get("to_name", "")
            if fn and rel and tn:
                parts.append(f"{fn} {rel} {tn}")

        elif name == "delete_entity":
            ename = args.get("name", "")
            if ename:
                parts.append(f"removed '{ename}'")

        # Spotify write tools are silent — their results come back via tool_results

    return ("Noted: " + ", ".join(parts)) if parts else ""


# ---------------------------------------------------------------------------
# TOOL CALL APPLICATION
# ---------------------------------------------------------------------------

def _resolve_object_id(
    gs: GraphStore,
    name: str,
    object_type: str,
) -> str | None:
    """
    Resolve a name/title to a UUID string for the given object type.
    Returns None if not found.
    """
    with gs._lock:
        if object_type == "entity":
            row = gs._conn.execute(
                "SELECT id FROM entities WHERE name = ?", [name.strip()]
            ).fetchone()
        elif object_type == "event":
            row = gs._conn.execute(
                "SELECT id FROM events WHERE title = ?", [name.strip()]
            ).fetchone()
        elif object_type == "commitment":
            row = gs._conn.execute(
                "SELECT id FROM commitments WHERE title = ?", [name.strip()]
            ).fetchone()
        else:
            return None
    return str(row[0]) if row else None



def _apply_tool_calls(
    gs: GraphStore,
    tool_calls: list[dict],
    embed_executor: ThreadPoolExecutor,
    embed_available: bool,
) -> list[str]:
    """
    Execute graph writes fire-and-forget.
    Schedule background embedding jobs for newly upserted objects.
    Handles all 4 tool names (upsert_entity, upsert_event, upsert_commitment,
    add_relation) plus legacy upsert_node / add_edge for robustness.
    Returns a list of result strings from query tools (e.g. spotify_now_playing).
    """
    upserted: list[tuple[str, str]] = []  # (object_id, object_type)
    tool_results: list[str] = []

    for tc in tool_calls:
        name = tc["name"]
        args = tc["arguments"]

        # ── upsert_entity ──────────────────────────────────────────────
        if name == "upsert_entity":
            ename = args.get("name", "").strip()
            etype = args.get("type", "concept").strip()
            if ename:
                f = gs.submit_upsert_entity(
                    ename, etype,
                    aliases=args.get("aliases"),
                    attributes=args.get("attributes"),
                )
                try:
                    eid = f.result()
                    upserted.append((eid, "entity"))
                except Exception as exc:
                    log.warning("upsert_entity failed: %s", exc)

        # ── upsert_event ──────────────────────────────────────────────
        elif name == "upsert_event":
            etitle = args.get("title", "").strip()
            etype  = args.get("type", "state_change").strip()
            if etitle:
                # Resolve participant names → entity UUIDs
                participant_ids: list[str] = []
                for pname in (args.get("participants") or []):
                    pid = _resolve_object_id(gs, pname, "entity")
                    if pid:
                        participant_ids.append(pid)

                # Resolve location name → entity UUID
                loc_id: str | None = None
                if args.get("location"):
                    loc_id = _resolve_object_id(gs, args["location"], "entity")

                f = gs.submit_upsert_event(
                    etitle, etype,
                    description=args.get("description"),
                    start_time=args.get("start_time"),
                    end_time=args.get("end_time"),
                    participants=participant_ids,
                    location_id=loc_id,
                )
                try:
                    eid = f.result()
                    upserted.append((eid, "event"))
                except Exception as exc:
                    log.warning("upsert_event failed: %s", exc)

        # ── upsert_commitment ─────────────────────────────────────────
        elif name == "upsert_commitment":
            ctitle = args.get("title", "").strip()
            if ctitle:
                owner_id: str | None = None
                if args.get("owner"):
                    owner_id = _resolve_object_id(gs, args["owner"], "entity")

                event_id: str | None = None
                if args.get("related_event_title"):
                    event_id = _resolve_object_id(
                        gs, args["related_event_title"], "event"
                    )

                f = gs.submit_upsert_commitment(
                    ctitle,
                    description=args.get("description"),
                    owner_id=owner_id,
                    related_event_id=event_id,
                    due_time=args.get("due_time"),
                    status=args.get("status", "planned"),
                    priority=args.get("priority", "medium"),
                )
                try:
                    cid = f.result()
                    upserted.append((cid, "commitment"))
                except Exception as exc:
                    log.warning("upsert_commitment failed: %s", exc)

        # ── add_relation ──────────────────────────────────────────────
        elif name == "add_relation":
            from_name = args.get("from_name", "").strip()
            from_otype = args.get("from_object_type", "entity").strip()
            to_name   = args.get("to_name", "").strip()
            to_otype  = args.get("to_object_type", "entity").strip()
            rel_type  = args.get("type", "").strip()

            if from_name and to_name and rel_type:
                from_id = _resolve_object_id(gs, from_name, from_otype)
                to_id   = _resolve_object_id(gs, to_name, to_otype)
                if from_id and to_id:
                    gs.submit_add_relation(
                        from_id, from_otype,
                        to_id,   to_otype,
                        rel_type,
                        valid_from=args.get("valid_from"),
                        valid_to=args.get("valid_to"),
                    )
                else:
                    log.warning(
                        "add_relation: could not resolve '%s' (%s) or '%s' (%s)",
                        from_name, from_otype, to_name, to_otype,
                    )

        # ── delete_entity ─────────────────────────────────────────────
        elif name == "delete_entity":
            ename = args.get("name", "").strip()
            if ename:
                deleted = gs.delete_entity(ename)
                if deleted:
                    log.info("Deleted entity: %s", ename)
                else:
                    log.warning("delete_entity: '%s' not found", ename)

        # ── legacy upsert_node ────────────────────────────────────────
        elif name == "upsert_node":
            node_name  = args.get("name", "").strip()
            node_label = args.get("label", "Unknown").strip()
            props      = args.get("props") or {}
            if node_name:
                f = gs.submit_upsert_node(node_name, node_label, props)
                try:
                    eid = f.result()
                    upserted.append((eid, "entity"))
                except Exception as exc:
                    log.warning("upsert_node (legacy) failed: %s", exc)

        # ── legacy add_edge ───────────────────────────────────────────
        elif name == "add_edge":
            from_n = args.get("from_name", "").strip()
            to_n   = args.get("to_name", "").strip()
            rel    = args.get("relationship", "").strip()
            weight = float(args.get("weight", 1.0))
            if from_n and to_n and rel:
                gs.submit_add_edge(from_n, to_n, rel, weight)

        # ── spotify_play ──────────────────────────────────────────────
        elif name == "spotify_play" and spotify_client:
            try:
                result = spotify_client.play(
                    args.get("query", ""),
                    args.get("queue", False),
                )
                tool_results.append(result)
                log.info("spotify_play: %s", result)
            except Exception as exc:
                msg = f"Spotify error: {exc}"
                tool_results.append(msg)
                log.warning(msg)

        # ── spotify_control ───────────────────────────────────────────
        elif name == "spotify_control" and spotify_client:
            try:
                result = spotify_client.control(
                    args.get("action", ""),
                    args.get("volume"),
                )
                tool_results.append(result)
                log.info("spotify_control: %s", result)
            except Exception as exc:
                msg = f"Spotify error: {exc}"
                tool_results.append(msg)
                log.warning(msg)

        # ── spotify_now_playing ───────────────────────────────────────
        elif name == "spotify_now_playing" and spotify_client:
            try:
                result = spotify_client.now_playing()
                tool_results.append(result)
                log.info("spotify_now_playing: %s", result)
            except Exception as exc:
                msg = f"Spotify error: {exc}"
                tool_results.append(msg)
                log.warning(msg)

        # ── spotify_recently_played ───────────────────────────────────
        elif name == "spotify_recently_played" and spotify_client:
            try:
                limit  = int(args.get("limit", 5))
                result = spotify_client.recently_played(limit=limit)
                tool_results.append(result)
                log.info("spotify_recently_played: %s", result)
            except Exception as exc:
                msg = f"Spotify error: {exc}"
                tool_results.append(msg)
                log.warning(msg)

        # ── spotify_queue ─────────────────────────────────────────────
        elif name == "spotify_queue" and spotify_client:
            try:
                result = spotify_client.queue()
                tool_results.append(result)
                log.info("spotify_queue: %s", result)
            except Exception as exc:
                msg = f"Spotify error: {exc}"
                tool_results.append(msg)
                log.warning(msg)

    # Background embedding: fires after audio has played, no voice impact
    if embed_available and upserted:
        def _embed_and_store(object_id: str, object_type: str, display_name: str) -> None:
            emb = _get_embedding(display_name)
            if emb:
                gs.update_object_embedding(object_id, object_type, emb)

        for oid, otype in upserted:
            # Resolve display name for embedding text
            with gs._lock:
                if otype == "entity":
                    row = gs._conn.execute(
                        "SELECT name FROM entities WHERE id::TEXT = ?", [oid]
                    ).fetchone()
                elif otype == "event":
                    row = gs._conn.execute(
                        "SELECT title FROM events WHERE id::TEXT = ?", [oid]
                    ).fetchone()
                elif otype == "commitment":
                    row = gs._conn.execute(
                        "SELECT title FROM commitments WHERE id::TEXT = ?", [oid]
                    ).fetchone()
                else:
                    row = None
            if row:
                embed_executor.submit(_embed_and_store, oid, otype, row[0])

    return tool_results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    check_prerequisites()

    log.info("=== Living Memory starting up ===")

    # 1. mlx-lm inference server
    client = ensure_mlx_server()

    # 2. Embedding model (in-process)
    embed_available = ensure_embed_model()
    if not embed_available:
        log.warning("Semantic context disabled (embedding model unavailable).")

    # 3. Kokoro TTS server
    ensure_kokoro_server()

    # Graph store (auto-migrates legacy schema on first run)
    gs = GraphStore()
    s0 = gs.stats()
    log.info(
        "Graph loaded: %d entities, %d events, %d commitments, %d relations",
        s0["entities"], s0["events"], s0["commitments"], s0["relations"],
    )

    # 4. Spotify (optional — graceful if credentials absent or spotipy not installed)
    global spotify_client
    spotify_tools, spotify_client = build_spotify_tools()
    if spotify_tools:
        TOOLS.extend(spotify_tools)
        log.info(
            "Spotify tools enabled: %s",
            [t["function"]["name"] for t in spotify_tools],
        )
    else:
        log.info("Spotify tools not available.")

    # Background thread for embedding jobs (single worker, non-blocking)
    embed_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedder")

    conversation: list[dict] = []

    # Wire the browser chat handler, then start the viz server
    chat_handler = make_chat_handler(client, gs, conversation, embed_available, embed_executor)
    viz_server.start(gs, handle_chat_turn=chat_handler)

    log.info("=== Living Memory ready. Open http://localhost:5050 to chat. Ctrl-C to quit. ===")

    try:
        threading.Event().wait()   # block until KeyboardInterrupt
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        embed_executor.shutdown(wait=False)
        gs.close()


if __name__ == "__main__":
    main()
