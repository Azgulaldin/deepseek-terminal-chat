from __future__ import annotations

# client.py - Local web server + browser relay for the DeepSeek roleplay app
# ---------------------------------------------------------------------------
# Behavior:
#   • Auto-installs missing dependencies (aiohttp, websockets)
#   • Starts a local HTTP server and opens the browser
#   • Serves the frontend from ./static/
#   • Relays browser WebSocket traffic to the remote FastAPI server
#   • Preserves local session logs
#   • Supports local session upload flows
#   • Reattaches after browser reloads, because browsers enjoy losing sockets
# ---------------------------------------------------------------------------

import asyncio
import json
import socket
import subprocess
import sys
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

REQUIRED_PACKAGES = ["aiohttp", "websockets"]


def install_packages() -> None:
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


install_packages()

import aiohttp
from aiohttp import web
import websockets
from websockets.client import WebSocketClientProtocol


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVER_URL = "wss://deepseek-terminal-chat-production-78d3.up.railway.app/chat"

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SESSION_ROOT = BASE_DIR / "sessions"
PERSONA_FILE = BASE_DIR / "user_config.json"

STATIC_DIR.mkdir(exist_ok=True)
SESSION_ROOT.mkdir(exist_ok=True)


def derive_http_base(ws_url: str) -> str:
    parsed = urlparse(ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    return f"{scheme}://{parsed.netloc}"


SERVER_HTTP_BASE = derive_http_base(SERVER_URL)


# ---------------------------------------------------------------------------
# Local session logging
# ---------------------------------------------------------------------------
class SessionLogger:
    """Writes a flat transcript into sessions/ with a date-based filename."""

    def __init__(self, username: str, persona: str) -> None:
        self.username = username
        self.persona = persona
        self.file_path: Optional[Path] = None

    def start(self) -> Path:
        today = datetime.now().strftime("%Y-%m-%d")
        existing_numbers = []
        for p in SESSION_ROOT.glob(f"{today}_*.txt"):
            try:
                existing_numbers.append(int(p.stem.split("_")[-1]))
            except Exception:
                pass
        next_number = max(existing_numbers, default=0) + 1
        self.file_path = SESSION_ROOT / f"{today}_{next_number}.txt"
        with self.file_path.open("w", encoding="utf-8") as f:
            f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Username: {self.username}\n")
            f.write(f"Persona: {self.persona}\n\n")
        return self.file_path

    def write(self, entry: str) -> None:
        if self.file_path is None:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {entry}\n")

    def log_system(self, message: str) -> None:
        self.write(f"SYSTEM: {message}")

    def log_chat(self, sender: str, message: str) -> None:
        self.write(f"{sender}: {message}")

    def log_reasoning(self, sender: str, reasoning: str) -> None:
        self.write(f"[{sender} REASONING]\n{reasoning}")


# ---------------------------------------------------------------------------
# Local config helpers
# ---------------------------------------------------------------------------

def load_user_config() -> dict:
    if PERSONA_FILE.exists():
        try:
            return json.loads(PERSONA_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_user_config(config: dict) -> None:
    PERSONA_FILE.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Shared relay state
# ---------------------------------------------------------------------------
@dataclass
class RelayState:
    local_clients: set[aiohttp.web.WebSocketResponse] = field(default_factory=set)
    remote_ws: Optional[WebSocketClientProtocol] = None
    receiver_task: Optional[asyncio.Task] = None
    reconnect_task: Optional[asyncio.Task] = None
    logger: Optional[SessionLogger] = None
    username: str = ""
    password: str = ""
    remember_password: bool = False
    role: str = "user"
    authenticated: bool = False
    connecting: bool = False
    last_session_state: Optional[dict] = None
    last_server_info: Optional[dict] = None
    pending_upload: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    shutdown: bool = False


state = RelayState()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


async def safe_send(ws: aiohttp.web.WebSocketResponse, payload: dict) -> None:
    try:
        await ws.send_json(payload)
    except Exception:
        pass


async def broadcast_local(payload: dict) -> None:
    dead = []
    for ws in list(state.local_clients):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.local_clients.discard(ws)


async def send_initial_state(ws: aiohttp.web.WebSocketResponse) -> None:
    if state.last_server_info:
        await safe_send(ws, state.last_server_info)
    if state.authenticated and state.username:
        await safe_send(ws, {
            "type": "login_accepted",
            "username": state.username,
            "role": state.role,
        })
        if state.last_session_state:
            await safe_send(ws, {"type": "session_state", **state.last_session_state})
        await safe_send(ws, {
            "type": "system",
            "message": f"Reattached to {state.username}.",
        })


async def close_remote() -> None:
    if state.remote_ws is not None:
        try:
            await state.remote_ws.close()
        except Exception:
            pass
    state.remote_ws = None
    if state.receiver_task and not state.receiver_task.done():
        state.receiver_task.cancel()
        try:
            await state.receiver_task
        except Exception:
            pass
    state.receiver_task = None


# ---------------------------------------------------------------------------
# Remote connection handling
# ---------------------------------------------------------------------------
async def connect_remote(username: str, password: str) -> bool:
    if state.connecting:
        return False
    state.connecting = True
    try:
        ws = await websockets.connect(SERVER_URL)
        await ws.send(json.dumps({
            "type": "join_request",
            "username": username,
            "password": password,
        }))
        raw = await ws.recv()
        resp = json.loads(raw)

        if resp.get("type") != "join_response" or not resp.get("accepted"):
            reason = resp.get("reason", "Unknown reason")
            await broadcast_local({"type": "login_failed", "reason": reason})
            try:
                await ws.close()
            except Exception:
                pass
            return False

        state.remote_ws = ws
        state.role = resp.get("role", "user")
        state.authenticated = True

        server_info = {
            "type": "server_info",
            "upload_url": SERVER_HTTP_BASE,
        }
        state.last_server_info = server_info
        await broadcast_local(server_info)
        await broadcast_local({
            "type": "login_accepted",
            "username": username,
            "role": state.role,
        })
        if state.logger is None:
            persona = load_user_config().get("persona", "")
            state.logger = SessionLogger(username, persona)
            state.logger.start()
        await broadcast_local({
            "type": "system",
            "message": f"Connected to server as {username}.",
        })

        if state.receiver_task and not state.receiver_task.done():
            state.receiver_task.cancel()
            try:
                await state.receiver_task
            except Exception:
                pass
        state.receiver_task = asyncio.create_task(remote_receiver())
        return True
    except Exception as e:
        await broadcast_local({"type": "system", "message": f"Remote connection failed: {e}"})
        return False
    finally:
        state.connecting = False


async def reconnect_remote() -> bool:
    if not state.username:
        return False
    if state.reconnect_task and not state.reconnect_task.done():
        return True

    async def _runner() -> None:
        await asyncio.sleep(3)
        if state.shutdown or not state.username:
            return
        ok = await connect_remote(state.username, state.password)
        if not ok and not state.shutdown:
            await broadcast_local({"type": "system", "message": "Still trying to reconnect..."})

    state.reconnect_task = asyncio.create_task(_runner())
    return True


async def remote_receiver() -> None:
    while not state.shutdown and state.remote_ws is not None:
        try:
            raw = await state.remote_ws.recv()
        except websockets.ConnectionClosed:
            await broadcast_local({"type": "system", "message": "Disconnected from server. Reconnecting..."})
            state.authenticated = False
            state.remote_ws = None
            await reconnect_remote()
            return
        except asyncio.CancelledError:
            return
        except Exception as e:
            await broadcast_local({"type": "system", "message": f"Remote receive error: {e}"})
            state.authenticated = False
            state.remote_ws = None
            await reconnect_remote()
            return

        try:
            data = json.loads(raw)
        except Exception:
            continue

        msg_type = data.get("type")

        if msg_type == "chat":
            sender = data.get("sender", "")
            message = data.get("message", "")
            if state.logger:
                state.logger.log_chat(sender, message)
        elif msg_type == "system":
            if state.logger:
                state.logger.log_system(data.get("message", ""))
        elif msg_type == "reasoning":
            sender = data.get("sender", "")
            reasoning = data.get("message", "")
            if state.logger:
                state.logger.log_reasoning(sender, reasoning)
        elif msg_type == "session_state":
            state.last_session_state = dict(data)
        elif msg_type == "new_session":
            if state.logger:
                state.logger.log_system(data.get("message", ""))
                state.logger.start()
        elif msg_type == "ai_config_changed":
            if state.logger:
                state.logger.log_system("AI configuration changed.")
        elif msg_type == "background_update":
            pass
        elif msg_type == "music_state":
            pass

        await broadcast_local(data)


async def send_to_remote(message: str) -> None:
    if state.remote_ws is None:
        await broadcast_local({"type": "system", "message": "Not connected to the server."})
        return
    try:
        await state.remote_ws.send(message)
    except websockets.ConnectionClosed:
        state.authenticated = False
        state.remote_ws = None
        await broadcast_local({"type": "system", "message": "Remote connection closed. Reconnecting..."})
        await reconnect_remote()
    except Exception as e:
        await broadcast_local({"type": "system", "message": f"Failed to send: {e}"})


# ---------------------------------------------------------------------------
# Local upload-from-log flow
# ---------------------------------------------------------------------------
async def handle_upload_session_file(ws: aiohttp.web.WebSocketResponse, filename: str) -> None:
    if not filename:
        await safe_send(ws, {"type": "system", "message": "Usage: /upload <filename>"})
        return

    filepath = Path(filename)
    if not filepath.is_absolute():
        filepath = SESSION_ROOT / filepath
    if not filepath.exists():
        await safe_send(ws, {"type": "system", "message": f"File not found: {filepath}"})
        return

    upload_id = f"{id(ws)}_{datetime.now().isoformat()}"
    state.pending_upload[upload_id] = {
        "ws": ws,
        "filepath": filepath,
        "username": state.username or "unknown",
    }
    await safe_send(ws, {"type": "ask_ai_name", "upload_id": upload_id, "filename": filepath.name})


async def complete_upload_session(upload_id: str, ai_name: str) -> None:
    info = state.pending_upload.pop(upload_id, None)
    if not info:
        return

    ws = info["ws"]
    filepath: Path = info["filepath"]
    username = info["username"]

    def parse_log_file(path: Path, ai_name_value: str, username_value: str) -> list[dict]:
        history = []
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if not line.startswith("[") or "] " not in line:
                    continue
                idx = line.index("] ")
                content = line[idx + 2 :]
                if content.startswith("SYSTEM:"):
                    continue
                if content.startswith("[") and content.endswith("]") and " REASONING]" in content:
                    continue
                if ": " not in content:
                    continue
                sender, message = content.split(": ", 1)
                if sender == "You":
                    history.append({"role": "user", "sender": username_value, "content": message})
                elif sender.lower() == ai_name_value.lower():
                    history.append({"role": "assistant", "sender": ai_name_value, "content": message})
        return history

    try:
        history = parse_log_file(filepath, ai_name, username)
    except Exception as e:
        await safe_send(ws, {"type": "system", "message": f"Error reading log file: {e}"})
        return

    if not history:
        await safe_send(ws, {"type": "system", "message": "No valid chat messages found."})
        return

    payload = {"type": "upload_session", "history": history}
    await send_to_remote(json.dumps(payload))
    await safe_send(ws, {"type": "system", "message": f"Uploading session from {filepath.name} with {len(history)} messages."})
    if state.logger:
        state.logger.log_system(f"Uploaded session from {filepath.name}")


# ---------------------------------------------------------------------------
# Browser websocket handling
# ---------------------------------------------------------------------------
async def handle_local_message(ws: aiohttp.web.WebSocketResponse, raw: str) -> None:
    try:
        data = json.loads(raw)
    except Exception:
        return

    msg_type = data.get("type")

    if msg_type == "login":
        username = str(data.get("username", "")).strip()
        password = str(data.get("password", ""))
        remember_password = bool(data.get("remember_password", False))
        if not username:
            await safe_send(ws, {"type": "login_failed", "reason": "Username required"})
            return

        config = load_user_config()
        config["username"] = username
        if remember_password:
            config["password"] = password
        else:
            config.pop("password", None)
        save_user_config(config)

        state.username = username
        state.password = password if remember_password else password
        state.remember_password = remember_password
        state.logger = SessionLogger(username, config.get("persona", ""))
        state.logger.start()

        if state.remote_ws is not None:
            await close_remote()

        ok = await connect_remote(username, password)
        if not ok:
            return
        return

    if msg_type == "chat_message":
        message = str(data.get("message", "")).strip()
        if not message:
            return
        if state.logger:
            state.logger.log_chat(state.username or "You", message)
        await send_to_remote(message)
        return

    if msg_type == "music_control":
        await send_to_remote(json.dumps(data))
        return

    if msg_type == "ai_config_update":
        await send_to_remote(json.dumps(data))
        return

    if msg_type in {"upload_session_file", "/upload"}:
        filename = str(data.get("filename", "")).strip()
        await handle_upload_session_file(ws, filename)
        return

    if msg_type == "ai_name_response":
        upload_id = data.get("upload_id")
        ai_name = str(data.get("ai_name", "")).strip()
        if upload_id and ai_name:
            await complete_upload_session(str(upload_id), ai_name)
        return

    await send_to_remote(json.dumps(data))


async def local_ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    state.local_clients.add(ws)

    try:
        await send_initial_state(ws)
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await handle_local_message(ws, msg.data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f"Local WebSocket error: {ws.exception()}")
    finally:
        state.local_clients.discard(ws)
        if not state.local_clients:
            await close_remote()
            state.authenticated = False
            state.username = ""
            state.password = ""
            state.logger = None
            state.last_session_state = None
            state.last_server_info = None

    return ws


# ---------------------------------------------------------------------------
# HTTP serving
# ---------------------------------------------------------------------------
async def index_handler(request: web.Request):
    return web.FileResponse(STATIC_DIR / "index.html")


async def start_local_server():
    app = web.Application()
    app.router.add_get("/ws", local_ws_handler)
    app.router.add_static("/static/", STATIC_DIR, show_index=False)
    app.router.add_get("/", index_handler)

    port = find_free_port()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", port)
    await site.start()
    print(f"Local server started at http://localhost:{port}")
    webbrowser.open(f"http://localhost:{port}")
    return runner, port


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    runner, _port = await start_local_server()
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        state.shutdown = True
        await close_remote()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
