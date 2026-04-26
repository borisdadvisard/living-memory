"""
main.py — Living Memory orchestration

Start-up sequence:
  1. Ensure Ollama is running (launchctl load if not)
  2. Ensure nomic-embed-text is pulled
  3. Ensure kokoro_server.py is running on :8181

Voice loop (per turn):
  a. Record mic → Whisper STT → transcript
  b. Build graph context (semantic search + name search + neighbors)
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
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ollama
import requests
import sounddevice as sd

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
        "description": (
            "Store or update a named entity in the knowledge graph. "
            "Call once per distinct person, place, org, object, or concept mentioned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Canonical entity name, e.g. 'Boris', 'Lyon', 'IKEA'",
                },
                "type": {
                    "type": "string",
                    "enum": sorted(ENTITY_TYPES),
                    "description": "Entity type",
                },
                "attributes": {
                    "type": "object",
                    "description": "Optional key/value attributes, e.g. {\"occupation\": \"architect\"}",
                },
                "aliases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Alternative names or abbreviations",
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
        "description": (
            "Record an event that happened or is planned to happen. "
            "Use for deliveries, meetings, conversations, transactions, or state changes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short descriptive title, e.g. 'Desk delivery Mar 23'",
                },
                "type": {
                    "type": "string",
                    "enum": sorted(EVENT_TYPES),
                    "description": "Event type",
                },
                "description": {
                    "type": "string",
                    "description": "Optional longer description",
                },
                "start_time": {
                    "type": "string",
                    "description": "ISO-8601 datetime, e.g. '2025-03-23T09:00:00'. Null if unknown.",
                },
                "end_time": {
                    "type": "string",
                    "description": "ISO-8601 datetime. Null if unknown or open-ended.",
                },
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names of entities involved",
                },
                "location": {
                    "type": "string",
                    "description": "Name of the location entity",
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
        "description": (
            "Record a commitment, obligation, or task. "
            "Use when someone has to do something by a certain time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short description of the obligation, e.g. 'Be present for desk delivery'",
                },
                "description": {
                    "type": "string",
                    "description": "Optional longer explanation",
                },
                "owner": {
                    "type": "string",
                    "description": "Name of the entity responsible",
                },
                "due_time": {
                    "type": "string",
                    "description": "ISO-8601 deadline. Null if unspecified.",
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
                    "description": "Title of the related event, if any",
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
        "description": (
            "Record a typed relationship between two graph objects. "
            f"type must be one of: {', '.join(sorted(ALLOWED_RELATION_TYPES))}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "from_name": {
                    "type": "string",
                    "description": "Name or title of the source object",
                },
                "from_object_type": {
                    "type": "string",
                    "enum": ["entity", "event", "commitment"],
                    "description": "Graph type of the source object",
                },
                "to_name": {
                    "type": "string",
                    "description": "Name or title of the target object",
                },
                "to_object_type": {
                    "type": "string",
                    "enum": ["entity", "event", "commitment"],
                    "description": "Graph type of the target object",
                },
                "type": {
                    "type": "string",
                    "enum": sorted(ALLOWED_RELATION_TYPES),
                    "description": "Relationship type",
                },
                "valid_from": {
                    "type": "string",
                    "description": "ISO-8601 datetime when this relation became true. Null if unknown.",
                },
                "valid_to": {
                    "type": "string",
                    "description": "ISO-8601 datetime when this relation ceased to be true. Null if ongoing.",
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
        "description": (
            "Permanently delete a named entity from the knowledge graph, "
            "removing all its relations and embeddings. "
            "Use only when explicitly asked to forget or remove an entity."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact name of the entity to delete, e.g. 'Boris', 'Lyon'",
                },
            },
            "required": ["name"],
        },
    },
}

TOOLS = [UPSERT_ENTITY_TOOL, UPSERT_EVENT_TOOL, UPSERT_COMMITMENT_TOOL,
         ADD_RELATION_TOOL, DELETE_ENTITY_TOOL]

# ---------------------------------------------------------------------------
# BROWSER CHAT HANDLER
# ---------------------------------------------------------------------------

_chat_lock = threading.Lock()


def make_chat_handler(
    client: ollama.Client,
    gs: GraphStore,
    conversation: list[dict],
    embed_available: bool,
    embed_executor: ThreadPoolExecutor,
):
    """
    Return a generator function: handle(message) → yields SSE-formatted strings.
    Called by viz_server for each POST /chat request.
    """

    def handle(message: str):
        actual = message.strip()

        # ── Voice mode ("/v") ─────────────────────────────────────────────────
        if actual == "/v":
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

        # ── LLM turn ──────────────────────────────────────────────────────────
        with _chat_lock:
            context = _build_context(gs, client, actual, embed_available)
            system  = SYSTEM_PROMPT_BASE
            if context:
                system += "\n\n" + context

            msgs = [{"role": "system", "content": system}]
            msgs.extend(conversation)
            msgs.append({"role": "user", "content": actual})

            full_response: str = ""
            collected_tools: list[dict] = []

            try:
                stream = client.chat(
                    model=OLLAMA_MODEL,
                    messages=msgs,
                    tools=TOOLS,
                    stream=True,
                    options={"think": False},
                    keep_alive=OLLAMA_KEEP_ALIVE,
                )

                for chunk in stream:
                    msg = chunk.message
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            collected_tools.append({
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments or {},
                            })

                    fragment = msg.content or ""
                    full_response += fragment
                    if fragment:
                        yield f"data: {json.dumps({'type': 'token', 'text': fragment})}\n\n"

                    if chunk.done:
                        break

                conversation.append({"role": "user",      "content": actual})
                conversation.append({"role": "assistant", "content": full_response})
                log.info("Chat: %s → %s…", actual[:60], full_response[:80])

                if collected_tools:
                    _apply_tool_calls(
                        gs, collected_tools, embed_executor, client, embed_available
                    )
                    for tc in collected_tools:
                        args  = tc["arguments"]
                        label = (
                            args.get("name") or args.get("title") or tc["name"]
                        ).strip()
                        yield f"data: {json.dumps({'type': 'tool', 'fn': tc['name'], 'label': label})}\n\n"

            except Exception as exc:
                log.warning("Chat error: %s", exc)
                yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return handle


SYSTEM_PROMPT_BASE = (
    "You are a helpful voice assistant with a persistent memory.\n"
    "Respond concisely in plain spoken language. "
    "Do not use markdown, bullet points, or special characters.\n"
    "After your spoken reply, call upsert_entity, upsert_event, upsert_commitment, "
    "and add_relation for every entity, event, commitment, and relationship you "
    "learned this turn. Use ISO-8601 dates wherever times are mentioned.\n"
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
        emb = _get_embedding(client, transcript)
        if emb:
            for hit in gs.semantic_search(emb, top_n=8):
                _add(hit["object_id"], hit["object_type"],
                     hit["subtype"] or "", hit["display_name"])

    # 2. Cross-type name search using first meaningful word
    words = [w for w in transcript.split() if len(w) > 3]
    if words:
        for hit in gs.search_by_name(words[0], limit=5):
            _add(hit["object_id"], hit["object_type"],
                 hit["subtype"] or "", hit["display_name"])

    # 3. Expand neighbors of first semantic/name hit
    if seen_ids:
        first_id = next(iter(seen_ids))
        # Determine type of first hit by checking which table it's in
        first_type = "entity"
        with gs._lock:
            if gs._conn.execute(
                "SELECT 1 FROM events WHERE id::TEXT = ?", [first_id]
            ).fetchone():
                first_type = "event"
            elif gs._conn.execute(
                "SELECT 1 FROM commitments WHERE id::TEXT = ?", [first_id]
            ).fetchone():
                first_type = "commitment"
        for nb in gs._query_neighbors_by_id(first_id, first_type, hops=1):
            _add(nb["object_id"], nb["object_type"],
                 nb["subtype"] or "", nb["display_name"],
                 f"{nb['hops']}-hop")

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
    client: ollama.Client,
    messages: list[dict],
    speak: bool = True,
) -> tuple[str, list[dict]]:
    """
    Stream qwen3:30b with tools and think=False.
    When speak=True: synthesise and play each complete sentence via TTS.
    When speak=False: stream tokens live to stdout.
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

        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "name":      tc.function.name,
                    "arguments": tc.function.arguments or {},
                })

        fragment = msg.content or ""
        full_response   += fragment
        sentence_buffer += fragment

        if speak:
            sentences, sentence_buffer = _flush_sentences(sentence_buffer)
            for s in sentences:
                _tts_play(s)
        else:
            if fragment:
                print(fragment, end="", flush=True)

        if chunk.done:
            break

    if speak:
        if sentence_buffer.strip():
            _tts_play(sentence_buffer)
    else:
        print()  # newline to close the streamed response

    return full_response, tool_calls


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
    client: ollama.Client,
    embed_available: bool,
) -> None:
    """
    Execute graph writes fire-and-forget.
    Schedule background embedding jobs for newly upserted objects.
    Handles all 4 tool names (upsert_entity, upsert_event, upsert_commitment,
    add_relation) plus legacy upsert_node / add_edge for robustness.
    """
    upserted: list[tuple[str, str]] = []  # (object_id, object_type)

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

    # Background embedding: fires after audio has played, no voice impact
    if embed_available and upserted:
        def _embed_and_store(object_id: str, object_type: str, display_name: str) -> None:
            emb = _get_embedding(client, display_name)
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


# ---------------------------------------------------------------------------
# MAIN
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

    # Graph store (auto-migrates legacy schema on first run)
    gs = GraphStore()
    s0 = gs.stats()
    log.info(
        "Graph loaded: %d entities, %d events, %d commitments, %d relations",
        s0["entities"], s0["events"], s0["commitments"], s0["relations"],
    )

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
