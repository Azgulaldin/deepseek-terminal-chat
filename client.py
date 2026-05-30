# client.py – Local web server + WebSocket relay for the DeepSeek roleplay app
# ---------------------------------------------------------------------------
# Double‑click to start. It will:
#   • Auto‑install missing Python packages (aiohttp, websockets)
#   • Start a local HTTP server on an available port
#   • Open your browser to the UI
#   • Relay all messages between the browser and the remote server
#   • Log all chats to local session files
#   • Support /upload from the UI
# ---------------------------------------------------------------------------

import asyncio
import json
import os
import sys
import subprocess
import webbrowser
import socket
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# -------------------------------------------------------------------
# Auto‑install required packages
# -------------------------------------------------------------------
REQUIRED_PACKAGES = ["aiohttp", "websockets"]

def install_packages():
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        # Use the same Python interpreter that's running this script
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

install_packages()

import aiohttp
from aiohttp import web
import aiohttp.web_runner
import websockets

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
SERVER_URL = "wss://deepseek-terminal-chat-production-78d3.up.railway.app/chat"

# Local directories (everything inside the folder where client.py lives)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"            # HTML/CSS/JS frontend
SESSION_ROOT = BASE_DIR / "sessions"        # local chat logs
PERSONA_FILE = BASE_DIR / "user_config.json"  # stores username, persona, password (if remembered)

# Create essential directories
STATIC_DIR.mkdir(exist_ok=True)
SESSION_ROOT.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# Session logging (identical to the old terminal client)
# -------------------------------------------------------------------
class SessionLogger:
    """Writes a transcript of the session to a flat file inside sessions/."""

    def __init__(self, username: str, persona: str) -> None:
        self.username = username
        self.persona = persona
        self.file_path: Optional[Path] = None

    def start(self) -> Path:
        session_dir = SESSION_ROOT
        today = datetime.now().strftime("%Y-%m-%d")
        # Find next session number for today
        existing_numbers = []
        for p in session_dir.glob(f"{today}_*.txt"):
            stem = p.stem
            try:
                num_part = stem.split("_")[-1]
                existing_numbers.append(int(num_part))
            except ValueError:
                pass
        next_number = max(existing_numbers, default=0) + 1
        filename = f"{today}_{next_number}.txt"
        self.file_path = session_dir / filename
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Username: {self.username}\n")
            f.write(f"Persona: {self.persona}\n\n")
        return self.file_path

    def write(self, entry: str) -> None:
        if self.file_path is None:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {entry}\n")

    def log_system(self, message: str) -> None:
        self.write(f"SYSTEM: {message}")

    def log_chat(self, sender: str, message: str) -> None:
        self.write(f"{sender}: {message}")

    def log_reasoning(self, sender: str, reasoning: str) -> None:
        self.write(f"[{sender} REASONING]\n{reasoning}")

# -------------------------------------------------------------------
# Load / save local user config (username, persona)
# -------------------------------------------------------------------
def load_user_config() -> dict:
    if PERSONA_FILE.exists():
        try:
            return json.loads(PERSONA_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_user_config(config: dict) -> None:
    PERSONA_FILE.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

# -------------------------------------------------------------------
# Global state for the relay
# -------------------------------------------------------------------
class RelayState:
    def __init__(self):
        self.remote_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.local_clients: Dict[aiohttp.web.WebSocketResponse, str] = {}  # local ws -> username
        self.logger: Optional[SessionLogger] = None
        self.loop = asyncio.get_event_loop()
        self.receiver_task: Optional[asyncio.Task] = None
        self.pending_upload: Dict[str, Any] = {}

state = RelayState()

# -------------------------------------------------------------------
# Helper: find a free TCP port
# -------------------------------------------------------------------
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# -------------------------------------------------------------------
# Remote WebSocket receiver (from Railway to all local browsers)
# -------------------------------------------------------------------
async def remote_receiver():
    """Receive messages from the remote server and broadcast to all local browsers."""
    while True:
        try:
            raw = await state.remote_ws.recv()
        except websockets.ConnectionClosed as e:
            print(f"Remote connection closed: {e}")
            # Notify local browsers and attempt reconnect
            for ws in list(state.local_clients.keys()):
                try:
                    await ws.send_json({"type": "system", "message": "Disconnected from server. Reconnecting..."})
                except:
                    pass
            # Try to reconnect after a delay
            await asyncio.sleep(5)
            if not await reconnect_remote():
                break
            continue
        except Exception as e:
            print(f"Unexpected remote receive error: {e}")
            break

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        # Log locally where appropriate
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
        elif msg_type == "new_session":
            if state.logger:
                state.logger.log_system(data.get("message", ""))
                state.logger.start()   # start a fresh log file

        # Forward to all local browser clients
        dead = []
        for ws in list(state.local_clients.keys()):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            state.local_clients.pop(ws, None)

# -------------------------------------------------------------------
# Send a message to the remote server
# -------------------------------------------------------------------
async def send_to_remote(message: str) -> None:
    if state.remote_ws and state.remote_ws.open:
        await state.remote_ws.send(message)

# -------------------------------------------------------------------
# Connect to the remote server (called after login)
# -------------------------------------------------------------------
async def connect_remote(username: str, password: str) -> bool:
    try:
        ws = await websockets.connect(SERVER_URL)
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        return False

    # Send join_request
    join_msg = json.dumps({
        "type": "join_request",
        "username": username,
        "password": password,
    })
    await ws.send(join_msg)

    # Wait for join_response
    try:
        raw = await ws.recv()
        resp = json.loads(raw)
        if resp.get("type") == "join_response" and resp.get("accepted"):
            state.remote_ws = ws
            role = resp.get("role", "user")
            print(f"Connected as {username} (role: {role})")

            # --- NEW: inform browsers of the server base URL for uploads ---
            # Convert wss:// URL to https:// and strip /chat
            remote_base_url = SERVER_URL.replace("wss://", "https://").rsplit("/", 1)[0]
            server_info = {
                "type": "server_info",
                "upload_url": remote_base_url,
            }
            for ws_local in list(state.local_clients.values()):
                try:
                    await ws_local.send_json(server_info)
                except:
                    pass
            # Also send a system message
            for ws_local in state.local_clients.values():
                try:
                    await ws_local.send_json({"type": "system", "message": f"Connected to server as {username}."})
                except:
                    pass
            return True
        else:
            reason = resp.get("reason", "Unknown reason")
            print(f"Join rejected: {reason}")
            for ws_local in state.local_clients.values():
                try:
                    await ws_local.send_json({"type": "login_failed", "reason": reason})
                except:
                    pass
            await ws.close()
            return False
    except Exception as e:
        print(f"Error during join handshake: {e}")
        return False

async def reconnect_remote() -> bool:
    config = load_user_config()
    username = config.get("username", "")
    password = config.get("password", "")
    if not username:
        return False
    return await connect_remote(username, password)

# -------------------------------------------------------------------
# Local WebSocket handler (browser <-> client.py)
# -------------------------------------------------------------------
async def local_ws_handler(request: aiohttp.web.Request) -> aiohttp.web.WebSocketResponse:
    ws = aiohttp.web.WebSocketResponse()
    await ws.prepare(request)
    state.local_clients[ws] = ""   # username assigned later
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await handle_local_message(ws, msg.data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f"Local WebSocket error: {ws.exception()}")
    finally:
        state.local_clients.pop(ws, None)
    return ws

# -------------------------------------------------------------------
# Handle messages from the browser
# -------------------------------------------------------------------
async def handle_local_message(ws: aiohttp.web.WebSocketResponse, raw: str):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return

    msg_type = data.get("type")

    if msg_type == "login":
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()
        if not username:
            await ws.send_json({"type": "login_failed", "reason": "Username required"})
            return

        config = load_user_config()
        config["username"] = username
        if data.get("remember_password"):
            config["password"] = password
        else:
            config.pop("password", None)
        save_user_config(config)

        # Start session logger
        persona = config.get("persona", "")
        state.logger = SessionLogger(username, persona)
        state.logger.start()
        await ws.send_json({"type": "login_accepted", "username": username})

        # Connect to remote
        success = await connect_remote(username, password)
        if not success:
            return

        # Start receiver if not already running
        if not state.receiver_task or state.receiver_task.done():
            state.receiver_task = asyncio.create_task(remote_receiver())

    elif msg_type == "chat_message":
        message = data.get("message", "")
        if not message:
            return
        # Log locally
        if state.logger:
            username = state.local_clients.get(ws, "You")
            state.logger.log_chat(username, message)
        # Send as plain text (the server expects non-JSON for chat)
        await send_to_remote(message)

    elif msg_type == "music_control":
        await send_to_remote(json.dumps(data))

    elif msg_type == "ai_config_update":
        await send_to_remote(json.dumps(data))

    elif msg_type == "upload_session_file" or msg_type == "/upload":
        filename = data.get("filename", "").strip()
        await handle_upload_session_file(ws, filename)

    elif msg_type == "ai_name_response":
        upload_id = data.get("upload_id")
        ai_name = data.get("ai_name", "").strip()
        if upload_id and ai_name:
            await complete_upload_session(upload_id, ai_name)

    else:
        # Forward any other JSON message directly to remote
        await send_to_remote(json.dumps(data))

# -------------------------------------------------------------------
# Session upload from local log file
# -------------------------------------------------------------------
async def handle_upload_session_file(ws: aiohttp.web.WebSocketResponse, filename: str):
    if not filename:
        await ws.send_json({"type": "system", "message": "Usage: /upload <filename>"})
        return

    filepath = Path(filename)
    if not filepath.is_absolute():
        filepath = SESSION_ROOT / filepath
    if not filepath.exists():
        await ws.send_json({"type": "system", "message": f"File not found: {filepath}"})
        return

    upload_id = str(id(ws)) + "_" + datetime.now().isoformat()
    state.pending_upload[upload_id] = {
        "ws": ws,
        "filepath": filepath,
        "username": state.local_clients.get(ws, "unknown"),
    }
    await ws.send_json({"type": "ask_ai_name", "upload_id": upload_id, "filename": filename})

async def complete_upload_session(upload_id: str, ai_name: str):
    info = state.pending_upload.pop(upload_id, None)
    if not info:
        return
    ws = info["ws"]
    filepath = info["filepath"]
    username = info["username"]

    def parse_log_file(filepath: Path, ai_name: str, username: str) -> list:
        history = []
        skip_next_reasoning = False
        with open(filepath, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if skip_next_reasoning:
                    skip_next_reasoning = False
                    continue
                if not line.startswith("[") or "] " not in line:
                    continue
                idx = line.index("] ")
                after_ts = line[idx + 2:]
                if after_ts.startswith("[") and after_ts.endswith("]") and " REASONING]" in after_ts:
                    skip_next_reasoning = True
                    continue
                if after_ts.startswith("SYSTEM:"):
                    continue
                if ": " not in after_ts:
                    continue
                sender, message = after_ts.split(": ", 1)
                if sender == "You":
                    history.append({"role": "user", "sender": username, "content": message})
                elif sender.lower() == ai_name.lower():
                    history.append({"role": "assistant", "sender": ai_name, "content": message})
        return history

    try:
        history = parse_log_file(filepath, ai_name, username)
    except Exception as e:
        await ws.send_json({"type": "system", "message": f"Error reading log file: {e}"})
        return

    if not history:
        await ws.send_json({"type": "system", "message": "No valid chat messages found."})
        return

    upload_msg = {
        "type": "upload_session",
        "history": history,
    }
    await send_to_remote(json.dumps(upload_msg))
    await ws.send_json({"type": "system", "message": f"Uploading session from {filepath.name} with {len(history)} messages."})
    if state.logger:
        state.logger.log_system(f"Uploaded session from {filepath.name}")

# -------------------------------------------------------------------
# Serve static files
# -------------------------------------------------------------------
async def index_handler(request):
    return web.FileResponse(STATIC_DIR / 'index.html')

# -------------------------------------------------------------------
# Main startup
# -------------------------------------------------------------------
async def start_local_server():
    app = web.Application()
    app.router.add_get('/ws', local_ws_handler)
    app.router.add_static('/static/', STATIC_DIR, show_index=False)
    app.router.add_get('/', index_handler)

    port = find_free_port()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', port)
    await site.start()
    print(f"Local server started at http://localhost:{port}")
    webbrowser.open(f"http://localhost:{port}")
    return runner, port

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
async def main():
    runner, port = await start_local_server()
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())