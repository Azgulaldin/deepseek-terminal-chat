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


def normalize_inventory(value: Any) -> list[str]:
    items: list[str] = []
    if value is None:
        return items
    if isinstance(value, list):
        raw_parts = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return items
        if "\n" in text:
            raw_parts = text.splitlines()
        else:
            raw_parts = re.split(r"[;,]", text)
    else:
        raw_parts = [str(value)]
    for part in raw_parts:
        item = str(part).strip()
        if not item:
            continue
        item = re.sub(r"^[\-\*•\d\.\)\(\s]+", "", item).strip()
        if item and item not in items:
            items.append(item)
    return items


def inventory_to_text(inventory: Any) -> str:
    items = normalize_inventory(inventory)
    return "\n".join(items) if items else "[none]"


def parse_session_log_text(text: str, ai_name: str, username: str) -> list[dict]:
    """Parse a session transcript and preserve multi-paragraph entries."""
    history: list[dict] = []
    ai_sender = str(ai_name or "").strip().lower()
    current_header: str = ""
    current_lines: list[str] = []
    current_reasoning = False

    def flush_entry() -> None:
        nonlocal current_header, current_lines, current_reasoning
        if not current_header:
            current_lines = []
            current_reasoning = False
            return
        if current_reasoning:
            current_header = ""
            current_lines = []
            current_reasoning = False
            return

        header = current_header.strip()
        if not header or header.lower().startswith("system:"):
            current_header = ""
            current_lines = []
            return
        if ": " not in header:
            current_header = ""
            current_lines = []
            return

        sender, first_line = header.split(": ", 1)
        sender = sender.strip()
        message_lines = [first_line] if first_line else []
        message_lines.extend(current_lines)
        message = "\n".join(message_lines).rstrip()
        if not message:
            current_header = ""
            current_lines = []
            return

        sender_lower = sender.lower()
        role = "assistant" if ai_sender and sender_lower == ai_sender else "user"
        if sender_lower == "you":
            sender = username or sender
            role = "user"
        history.append({"role": role, "sender": sender, "content": message})
        current_header = ""
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")

        if line.startswith("Session started:") or line.startswith("Username:") or line.startswith("Persona:") or line.startswith("Inventory:"):
            continue

        timestamp_match = re.match(r"^\[(\d{2}:\d{2}:\d{2})\]\s*(.*)$", line)
        if timestamp_match:
            flush_entry()
            content = timestamp_match.group(2)
            if content.startswith("[") and content.endswith("]") and " REASONING]" in content:
                current_header = content
                current_reasoning = True
                current_lines = []
                continue
            current_header = content
            current_lines = []
            current_reasoning = False
            continue

        if current_reasoning:
            continue

        if not current_header:
            continue

        current_lines.append(line)

    flush_entry()
    return history


# ---------------------------------------------------------------------------
# Local session logging
# ---------------------------------------------------------------------------
class SessionLogger:
    """Writes a flat transcript into sessions/ with a date-based filename."""

    def __init__(self, username: str, persona: str, inventory: Optional[list[str]] = None) -> None:
        self.username = username
        self.persona = persona
        self.inventory = normalize_inventory(inventory or [])
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
            f.write(f"Persona: {self.persona}\n")
            f.write(f"Inventory: {inventory_to_text(self.inventory)}\n\n")
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
            data = json.loads(PERSONA_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("inventory", [])
                return data
            return {}
        except Exception:
            return {}
    return {}


def save_user_config(config: dict) -> None:
    payload = dict(config)
    payload["inventory"] = normalize_inventory(payload.get("inventory", []))
    PERSONA_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def list_local_session_files() -> list[dict]:
    """Return all local session logs under sessions/ as relative paths."""
    results: list[dict] = []
    for path in sorted(SESSION_ROOT.rglob("*.txt")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(SESSION_ROOT).as_posix()
        except Exception:
            continue
        stat = path.stat()
        results.append({
            "name": path.name,
            "path": rel,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        })
    results.sort(key=lambda item: (item["modified"], item["path"]), reverse=True)
    return results


def resolve_session_path(filename: str) -> Path:
    """Resolve a session file relative to SESSION_ROOT and keep it inside that tree."""
    filepath = Path(filename).expanduser()
    if not filepath.is_absolute():
        filepath = SESSION_ROOT / filepath
    filepath = filepath.resolve()
    root = SESSION_ROOT.resolve()
    if root not in filepath.parents and filepath != root:
        raise ValueError("Invalid session path")
    return filepath


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
    persona: str = ""
    inventory: list[str] = field(default_factory=list)
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
            "persona": state.persona,
            "inventory": state.inventory,
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
async def connect_remote(username: str, password: str, persona: str = "") -> bool:
    if state.connecting:
        return False
    state.connecting = True
    try:
        ws = await websockets.connect(SERVER_URL)
        await ws.send(json.dumps({
            "type": "join_request",
            "username": username,
            "password": password,
            "persona": persona,
            "inventory": state.inventory,
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
        state.persona = persona
        state.inventory = normalize_inventory(resp.get("inventory", state.inventory))
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
            "persona": state.persona,
            "inventory": state.inventory,
        })
        save_user_config({
            "username": username,
            "persona": state.persona,
            "inventory": state.inventory,
            **({"password": password} if state.remember_password else {}),
        })
        if state.logger is None:
            state.logger = SessionLogger(username, state.persona, state.inventory)
            state.logger.start()
        else:
            state.logger.inventory = list(state.inventory)
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
        ok = await connect_remote(state.username, state.password, state.persona)
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
            inv_map = data.get("online_user_inventories") or data.get("inventories") or {}
            if isinstance(inv_map, dict):
                my_inv = inv_map.get(state.username)
                if my_inv is not None:
                    state.inventory = normalize_inventory(my_inv)
                    save_user_config({
                        "username": state.username,
                        "persona": state.persona,
                        "inventory": state.inventory,
                        **({"password": state.password} if state.remember_password else {}),
                    })
                    if state.logger:
                        state.logger.inventory = list(state.inventory)
        elif msg_type == "inventory_state":
            target = str(data.get("username", "")).strip()
            inv = normalize_inventory(data.get("inventory", []))
            if target == state.username:
                state.inventory = inv
                save_user_config({
                    "username": state.username,
                    "persona": state.persona,
                    "inventory": state.inventory,
                    **({"password": state.password} if state.remember_password else {}),
                })
                if state.logger:
                    state.logger.inventory = list(state.inventory)
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

    try:
        filepath = resolve_session_path(filename)
    except Exception as e:
        await safe_send(ws, {"type": "system", "message": str(e)})
        return

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
        return parse_session_log_text(path.read_text(encoding="utf-8"), ai_name_value, username_value)

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
        persona = str(data.get("persona", "")).strip()
        inventory = normalize_inventory(data.get("inventory", []))
        remember_password = bool(data.get("remember_password", False))
        if not username:
            await safe_send(ws, {"type": "login_failed", "reason": "Username required"})
            return

        config = load_user_config()
        if not persona:
            persona = str(config.get("persona", "")).strip()
        if not inventory:
            inventory = normalize_inventory(config.get("inventory", []))
        config["username"] = username
        config["persona"] = persona
        config["inventory"] = inventory
        if remember_password:
            config["password"] = password
        else:
            config.pop("password", None)
        save_user_config(config)

        state.username = username
        state.password = password if remember_password else password
        state.persona = persona
        state.inventory = inventory
        state.remember_password = remember_password
        state.logger = None

        if state.remote_ws is not None:
            await close_remote()

        ok = await connect_remote(username, password, persona)
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
            state.persona = ""
            state.logger = None
            state.last_session_state = None
            state.last_server_info = None

    return ws


# ---------------------------------------------------------------------------
# HTTP serving
# ---------------------------------------------------------------------------
async def index_handler(request: web.Request):
    return web.FileResponse(STATIC_DIR / "index.html")


async def sessions_handler(request: web.Request):
    return web.json_response({
        "sessions": list_local_session_files(),
        "count": len(list_local_session_files()),
    })


async def start_local_server():
    app = web.Application()
    app.router.add_get("/ws", local_ws_handler)
    app.router.add_get("/sessions", sessions_handler)
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
