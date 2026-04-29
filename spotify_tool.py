"""
spotify_tool.py — Spotify playback tools for Living Memory

Provides five LLM-callable tools:
  - spotify_play            : search and play (or queue) a track, artist, or playlist
  - spotify_control         : pause, resume, skip, set volume
  - spotify_now_playing     : return the currently playing track as a speakable string
  - spotify_recently_played : list the most recently played tracks
  - spotify_queue           : list the upcoming tracks in the playback queue

No graph writes are performed — Spotify actions are pure side effects / queries.

Credentials are read from (in priority order):
  1. Environment variables: SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI
  2. spotify_config.json in the project root

OAuth tokens are cached to .spotify_token_cache (gitignored).
On first run the system browser opens for the Spotify authorization page;
paste the redirect URL into the terminal when prompted. Subsequent runs are silent.
"""

import http.server
import json
import logging
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlparse, parse_qs

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    import spotipy
    import spotipy.oauth2
    import spotipy.cache_handler
    import spotipy.exceptions
    _SPOTIPY_AVAILABLE = True
except ImportError:
    _SPOTIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------

def _load_credentials() -> tuple[str, str, str]:
    """
    Return (client_id, client_secret, redirect_uri).

    Priority:
      1. Environment variables SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET / SPOTIFY_REDIRECT_URI
      2. spotify_config.json in project root

    Raises ValueError if client_id or client_secret are missing.
    """
    client_id     = os.environ.get("SPOTIFY_CLIENT_ID", "")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
    redirect_uri  = os.environ.get("SPOTIFY_REDIRECT_URI", "")

    if not (client_id and client_secret):
        config_path = PROJECT_ROOT / "spotify_config.json"
        if config_path.exists():
            try:
                with open(config_path) as fh:
                    cfg = json.load(fh)
                client_id     = client_id     or cfg.get("client_id", "")
                client_secret = client_secret or cfg.get("client_secret", "")
                redirect_uri  = redirect_uri  or cfg.get("redirect_uri", "")
            except Exception as exc:
                raise ValueError(f"Failed to read spotify_config.json: {exc}") from exc

    if not client_id:
        raise ValueError(
            "Spotify client_id missing. Set SPOTIFY_CLIENT_ID or add it to spotify_config.json."
        )
    if not client_secret:
        raise ValueError(
            "Spotify client_secret missing. Set SPOTIFY_CLIENT_SECRET or add it to spotify_config.json."
        )

    redirect_uri = redirect_uri or "http://127.0.0.1:8888/callback"
    return client_id, client_secret, redirect_uri


# ---------------------------------------------------------------------------
# Local OAuth callback server
# ---------------------------------------------------------------------------

_CALLBACK_HTML_OK = b"""<!DOCTYPE html>
<html><head><title>Spotify authorised</title>
<style>body{font-family:sans-serif;display:flex;align-items:center;
justify-content:center;height:100vh;margin:0;background:#121212;color:#1db954;}
h1{font-size:2rem;}p{color:#b3b3b3;}</style></head>
<body><div><h1>&#10003; Authorised</h1>
<p>Living Memory is connected to Spotify. You can close this tab.</p>
</div></body></html>"""

_CALLBACK_HTML_ERR = b"""<!DOCTYPE html>
<html><head><title>Spotify error</title></head>
<body><h1>Authorization failed</h1><p>Check the terminal for details.</p>
</body></html>"""


class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """Single-use HTTP handler that captures the Spotify redirect."""

    def do_GET(self) -> None:
        self.server.callback_path = self.path          # store for caller
        if "code=" in self.path:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_CALLBACK_HTML_OK)
        else:
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_CALLBACK_HTML_ERR)

    def log_message(self, fmt: str, *args) -> None:  # suppress access log
        pass


def _await_oauth_callback(auth_manager, port: int = 8888) -> None:
    """
    Open the Spotify authorisation URL in the browser, start a one-shot
    HTTP server on 127.0.0.1:<port>, and exchange the returned code for tokens.
    Blocks until the user completes the browser flow (or 120 s timeout).
    """
    auth_url = auth_manager.get_authorize_url()
    log.info("Opening browser for Spotify authorisation…")
    print(
        f"\n[Spotify] Opening browser for authorisation.\n"
        f"If the browser does not open automatically, visit:\n  {auth_url}\n"
    )
    webbrowser.open(auth_url)

    server = http.server.HTTPServer(("127.0.0.1", port), _OAuthCallbackHandler)
    server.callback_path = None
    server.timeout = 120   # seconds to wait before giving up

    server.handle_request()   # blocks until one GET arrives (or timeout)
    server.server_close()

    if not server.callback_path:
        raise RuntimeError(
            "Spotify OAuth timed out waiting for the browser callback. "
            "Try running again."
        )

    # Reconstruct the full redirect URL and let spotipy exchange the code
    redirect_url = f"http://127.0.0.1:{port}{server.callback_path}"
    code = auth_manager.parse_response_code(redirect_url)
    auth_manager.get_access_token(code, as_dict=False)
    log.info("Spotify OAuth complete — token cached.")


# ---------------------------------------------------------------------------
# SpotifyClient
# ---------------------------------------------------------------------------

class SpotifyClient:
    """Thin wrapper around spotipy.Spotify with speakable return values."""

    SCOPES = (
        "user-read-playback-state "
        "user-modify-playback-state "
        "user-read-currently-playing "
        "user-read-recently-played"
    )

    def __init__(self) -> None:
        client_id, client_secret, redirect_uri = _load_credentials()

        cache_handler = spotipy.cache_handler.CacheFileHandler(
            cache_path=str(PROJECT_ROOT / ".spotify_token_cache")
        )
        auth_manager = spotipy.oauth2.SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=self.SCOPES,
            cache_handler=cache_handler,
            open_browser=False,   # we handle the browser ourselves
        )

        # If no valid cached token exists, run the local-server auth flow
        cached = cache_handler.get_cached_token()
        if not cached or auth_manager.is_token_expired(cached):
            port = int(urlparse(redirect_uri).port or 8888)
            _await_oauth_callback(auth_manager, port=port)

        self.sp = spotipy.Spotify(auth_manager=auth_manager)
        log.info("SpotifyClient initialised (redirect_uri=%s).", redirect_uri)

    # ── app launch ────────────────────────────────────────────────────────

    def _ensure_spotify_open(self, timeout: float = 8.0) -> bool:
        """
        Open the Spotify desktop app if no device is currently visible.
        On macOS uses `open -a Spotify`. On other platforms no-ops.
        Polls up to `timeout` seconds for a device to appear.
        Returns True if a device became available, False if timed out.
        """
        devices = self.sp.devices().get("devices", [])
        if devices:
            return True  # already available, nothing to do

        if sys.platform != "darwin":
            log.info("Non-macOS platform — cannot auto-open Spotify.")
            return False

        log.info("No Spotify device found — launching Spotify app…")
        try:
            subprocess.Popen(["open", "-a", "Spotify"])
        except FileNotFoundError:
            log.warning("'open' command not found — cannot launch Spotify.")
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(1.5)
            devices = self.sp.devices().get("devices", [])
            if devices:
                log.info("Spotify device ready: %s", devices[0]["name"])
                return True

        log.warning("Spotify did not register a device within %.0fs.", timeout)
        return False

    # ── device resolution ─────────────────────────────────────────────────

    def _get_device_id(self) -> str:
        """
        Return the ID of the active device.
        Falls back to the first available device if none is marked active.
        Raises RuntimeError with a user-friendly message if no device is found.
        """
        devices = self.sp.devices().get("devices", [])
        active = [d for d in devices if d.get("is_active")]
        if active:
            return active[0]["id"]
        if devices:
            log.info("No active Spotify device; using first available: %s", devices[0]["name"])
            return devices[0]["id"]
        raise RuntimeError(
            "No Spotify device found. Open Spotify on any device and try again."
        )

    # ── play ──────────────────────────────────────────────────────────────

    def play(self, query: str, queue: bool = False) -> str:
        """
        Search for a track, playlist, or artist and play or queue it.
        Returns a speakable confirmation string.
        """
        if not query.strip():
            return "Please provide something to search for."

        # Ensure Spotify is running before we attempt playback
        self._ensure_spotify_open()

        # Prefer track-only search for precision; fall back to broader search only
        # for clearly non-track queries (playlists, moods, artists without a title).
        _broad_hints = ("playlist", "mix", "radio", "vibes", "mood", "music")
        search_type = (
            "track,playlist,artist"
            if any(h in query.lower() for h in _broad_hints)
            else "track"
        )
        try:
            results = self.sp.search(q=query, limit=3, type=search_type)
        except spotipy.exceptions.SpotifyException as exc:
            return f"Spotify search failed: {exc}"

        uri: str | None   = None
        label: str        = query
        is_single_track   = False

        tracks    = (results.get("tracks")    or {}).get("items", [])
        playlists = (results.get("playlists") or {}).get("items", [])
        artists   = (results.get("artists")   or {}).get("items", [])

        if tracks:
            # Log all candidates so search quality is visible in the terminal
            for i, t in enumerate(tracks):
                log.info(
                    "spotify_play candidate[%d]: %s by %s",
                    i, t["name"], t["artists"][0]["name"],
                )
            uri   = tracks[0]["uri"]
            label = f"{tracks[0]['name']} by {tracks[0]['artists'][0]['name']}"
            is_single_track = True
        elif playlists:
            uri   = playlists[0]["uri"]
            label = playlists[0]["name"]
        elif artists:
            uri   = artists[0]["uri"]
            label = artists[0]["name"]
        else:
            return f"Nothing found on Spotify for: {query}"

        log.info("spotify_play search: query=%r → %r (uri=%s)", query, label, uri)

        try:
            device_id = self._get_device_id()
        except RuntimeError as exc:
            return str(exc)

        try:
            if queue:
                self.sp.add_to_queue(uri, device_id=device_id)
                return f"Added to queue: {label}"
            elif is_single_track:
                self.sp.start_playback(device_id=device_id, uris=[uri])
            else:
                self.sp.start_playback(device_id=device_id, context_uri=uri)
            log.info("spotify_play started: %s (device=%s)", label, device_id)
            return f"Now playing: {label}"
        except spotipy.exceptions.SpotifyException as exc:
            if exc.http_status == 403:
                return "Spotify Premium is required for playback control."
            log.warning("spotify_play start_playback failed: %s", exc)
            return f"Spotify error: {exc}"
        except Exception as exc:
            log.warning("spotify_play unexpected error: %s", exc)
            return f"Spotify error: {exc}"

    # ── control ───────────────────────────────────────────────────────────

    def control(self, action: str, volume: int | None = None) -> str:
        """
        Control playback: pause, resume, next, previous, or set_volume.
        Returns a speakable confirmation string.
        """
        try:
            device_id = self._get_device_id()
        except RuntimeError as exc:
            return str(exc)

        try:
            if action == "pause":
                self.sp.pause_playback(device_id=device_id)
                return "Paused."
            elif action == "resume":
                self.sp.start_playback(device_id=device_id)
                return "Resumed."
            elif action == "next":
                self.sp.next_track(device_id=device_id)
                return "Skipped to next track."
            elif action == "previous":
                self.sp.previous_track(device_id=device_id)
                return "Playing previous track."
            elif action == "set_volume":
                if volume is None:
                    return "Please specify a volume level between 0 and 100."
                vol = max(0, min(100, int(volume)))
                self.sp.volume(vol, device_id=device_id)
                return f"Volume set to {vol}%."
            else:
                return f"Unknown action: {action}. Use pause, resume, next, previous, or set_volume."
        except spotipy.exceptions.SpotifyException as exc:
            if exc.http_status == 403:
                return "Spotify Premium is required for playback control."
            return f"Spotify error: {exc}"

    # ── now_playing ───────────────────────────────────────────────────────

    def now_playing(self) -> str:
        """
        Return a speakable description of the currently playing track.
        """
        try:
            current = self.sp.current_playback()
        except spotipy.exceptions.SpotifyException as exc:
            return f"Spotify error: {exc}"

        if not current or not current.get("is_playing"):
            return "Nothing is currently playing on Spotify."

        item = current.get("item")
        if not item:
            return "Something is playing but track info is unavailable."

        track_name   = item["name"]
        artists      = ", ".join(a["name"] for a in item.get("artists", []))
        album        = (item.get("album") or {}).get("name", "")
        progress_ms  = current.get("progress_ms") or 0
        duration_ms  = item.get("duration_ms") or 1
        progress_pct = int(progress_ms / duration_ms * 100)

        parts = [f"{track_name} by {artists}"]
        if album:
            parts.append(f"from {album}")
        parts.append(f"{progress_pct}% through.")
        return " — ".join(parts)

    # ── recently_played ───────────────────────────────────────────────────

    def recently_played(self, limit: int = 5) -> str:
        """
        Return a speakable list of the most recently played tracks.
        """
        limit = max(1, min(limit, 10))
        try:
            result = self.sp.current_user_recently_played(limit=limit)
        except spotipy.exceptions.SpotifyException as exc:
            return f"Spotify error: {exc}"

        items = (result or {}).get("items", [])
        if not items:
            return "No recently played tracks found."

        # De-duplicate: the API can return the same track multiple times
        # if it was played in a loop; keep first occurrence only.
        seen: set[str] = set()
        lines: list[str] = []
        for i, item in enumerate(items, start=1):
            track = item.get("track") or {}
            uri = track.get("uri", "")
            if uri in seen:
                continue
            seen.add(uri)
            name    = track.get("name", "Unknown")
            artists = ", ".join(a["name"] for a in track.get("artists", []))
            lines.append(f"{len(lines)+1}. {name} by {artists}")
            if len(lines) >= limit:
                break

        return "Recently played: " + "; ".join(lines) + "."

    # ── queue ─────────────────────────────────────────────────────────────

    def queue(self) -> str:
        """
        Return a speakable description of the upcoming playback queue.
        """
        try:
            result = self.sp.queue()
        except spotipy.exceptions.SpotifyException as exc:
            return f"Spotify error: {exc}"

        if not result:
            return "Could not retrieve the queue."

        currently = result.get("currently_playing")
        upcoming  = result.get("queue") or []

        parts: list[str] = []

        if currently:
            name    = currently.get("name", "Unknown")
            artists = ", ".join(a["name"] for a in currently.get("artists", []))
            parts.append(f"Now playing: {name} by {artists}.")

        if not upcoming:
            parts.append("Nothing else in the queue.")
        else:
            next_tracks = upcoming[:5]
            track_strs  = [
                f"{t.get('name', 'Unknown')} by {', '.join(a['name'] for a in t.get('artists', []))}"
                for t in next_tracks
            ]
            parts.append("Up next: " + "; ".join(track_strs) + ".")
            if len(upcoming) > 5:
                parts.append(f"Plus {len(upcoming) - 5} more.")

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Tool JSON schemas
# ---------------------------------------------------------------------------

SPOTIFY_PLAY_TOOL = {
    "type": "function",
    "function": {
        "name": "spotify_play",
        "description": (
            "Search Spotify and immediately play or queue a track, artist, or playlist. "
            "Use ANY time the user wants to hear something — 'play X', 'put on X', "
            "'play the song in memory', 'play the song linked to Spotify'. "
            "If the song name comes from graph memory context, use that name as the query. "
            "Do NOT call spotify_now_playing before calling this — just play directly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query for Spotify. "
                        "For a specific known song, use field filter syntax for precision: "
                        "track:\"Song Title\" artist:\"Artist Name\" — "
                        "e.g. track:\"Billie Jean\" artist:\"Michael Jackson\". "
                        "For casual requests use plain text: 'Daft Punk', 'chill lo-fi playlist'."
                    ),
                },
                "queue": {
                    "type": "boolean",
                    "description": (
                        "If true, add to the queue instead of playing immediately. "
                        "Default false."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

SPOTIFY_CONTROL_TOOL = {
    "type": "function",
    "function": {
        "name": "spotify_control",
        "description": (
            "Control Spotify playback. "
            "Use for requests like 'pause', 'resume', 'skip', 'go back', "
            "or 'turn the volume up to 70'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["pause", "resume", "next", "previous", "set_volume"],
                    "description": "Playback action to perform.",
                },
                "volume": {
                    "type": "integer",
                    "description": (
                        "Volume level 0–100. Required when action is set_volume."
                    ),
                },
            },
            "required": ["action"],
        },
    },
}

SPOTIFY_NOW_PLAYING_TOOL = {
    "type": "function",
    "function": {
        "name": "spotify_now_playing",
        "description": (
            "Get the currently playing track on Spotify. "
            "ONLY use when the user asks what is playing RIGHT NOW — "
            "'what's playing?', 'what song is this?', 'who sings this?', 'what album is this from?'. "
            "Do NOT use this when the user wants to play a song — use spotify_play instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

SPOTIFY_RECENTLY_PLAYED_TOOL = {
    "type": "function",
    "function": {
        "name": "spotify_recently_played",
        "description": (
            "Get the user's recently played tracks on Spotify. "
            "Use when the user asks 'what was the last song?', 'what have I been "
            "listening to?', 'what did I play earlier?', or 'show my listening history'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent tracks to return (1–10, default 5).",
                },
            },
            "required": [],
        },
    },
}

SPOTIFY_QUEUE_TOOL = {
    "type": "function",
    "function": {
        "name": "spotify_queue",
        "description": (
            "Get the current Spotify playback queue. "
            "Use when the user asks 'what's coming up?', 'what's next?', "
            "'what's in my queue?', or 'what songs are queued?'."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

# ---------------------------------------------------------------------------
# Public factory — called from main() during startup
# ---------------------------------------------------------------------------

def build_spotify_tools() -> tuple[list[dict], "SpotifyClient | None"]:
    """
    Attempt to initialise the Spotify client.

    Returns:
        (tool_schemas, client)  on success
        ([], None)              if spotipy is not installed or credentials are missing
    """
    if not _SPOTIPY_AVAILABLE:
        log.warning(
            "spotipy not installed — Spotify tools disabled. "
            "Run: venv/bin/pip install spotipy"
        )
        return [], None

    try:
        client = SpotifyClient()
        schemas = [
            SPOTIFY_PLAY_TOOL,
            SPOTIFY_CONTROL_TOOL,
            SPOTIFY_NOW_PLAYING_TOOL,
            SPOTIFY_RECENTLY_PLAYED_TOOL,
            SPOTIFY_QUEUE_TOOL,
        ]
        return schemas, client
    except ValueError as exc:
        log.warning("Spotify credentials missing — tools disabled: %s", exc)
        return [], None
    except Exception as exc:
        log.warning("Spotify init failed — tools disabled: %s", exc)
        return [], None
