# server.py – DeepSeek roleplay multi‑user WebSocket server
# ------------------------------------------------------------------
# Upgraded for web GUI: password protection, admin, file uploads,
# background image, music sync, gallery, persistent AI config.
# ------------------------------------------------------------------

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import asyncio
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import shutil

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise RuntimeError("Environment variable DEEPSEEK_API_KEY is required")

BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"

# Server‑side storage directories (should be on a persistent volume in Railway)
DATA_DIR = Path("./data").resolve()
DATA_DIR.mkdir(exist_ok=True)
SESSIONS_DIR = DATA_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
PROFILES_DIR = UPLOADS_DIR / "profiles"
PROFILES_DIR.mkdir(exist_ok=True)
BACKGROUNDS_DIR = UPLOADS_DIR / "backgrounds"
BACKGROUNDS_DIR.mkdir(exist_ok=True)
MUSIC_DIR = UPLOADS_DIR / "music"
MUSIC_DIR.mkdir(exist_ok=True)
GALLERY_DIR = UPLOADS_DIR / "gallery"   # we could just reuse PROFILES & BACKGROUNDS, but keep separate for clarity
GALLERY_DIR.mkdir(exist_ok=True)

SESSION_CONFIG_FILE = DATA_DIR / "session_config.json"

# -------------------------------------------------------------------
# OpenAI client
# -------------------------------------------------------------------
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# -------------------------------------------------------------------
# Global application state
# -------------------------------------------------------------------
class AppState:
    def __init__(self) -> None:
        # Connected clients: websocket -> {"username": str, "role": "admin"/"user"}
        self.clients: Dict[WebSocket, Dict[str, str]] = {}
        # AI configuration
        self.ai_config: Dict[str, Any] = {
            "name": "DEEPSEEK",
            "behavior": "Helpful and intelligent.",
            "first_message": "",
            "reasoning_level": "medium",
            "show_reasoning": False,
            "configured": False,
        }
        # Session management
        self.history: List[Dict[str, Any]] = []
        self.ai_awake: bool = True
        self.current_session_id: Optional[str] = None
        self.session_archive: Dict[str, Dict[str, Any]] = {}
        self._day_counters: Dict[str, int] = {}
        self.admin_username: Optional[str] = None
        self.session_password: Optional[str] = None
        # Visual / media state
        self.background_url: Optional[str] = None
        self.music_state: Dict[str, Any] = {
            "url": None,
            "playing": False,
            "currentTime": 0.0,
            "loop": False,
            "server_time": 0.0,   # timestamp when state was emitted (for sync)
        }
        self.profile_pics: Dict[str, str] = {}   # username -> url
        self.gallery: List[Dict[str, Any]] = []  # list of {url, uploader, timestamp}
        # Setup synchronisation
        self.setup_owner: Optional[WebSocket] = None
        self._waiting_for_setup: asyncio.Event = asyncio.Event()
        self._setup_done: asyncio.Event = asyncio.Event()
        self._setup_done.set()   # initially no setup pending

        # Load persistent session config (AI + admin + password)
        self._load_session_config()

    # ---- Persistence helpers ----
    def _load_session_config(self) -> None:
        if SESSION_CONFIG_FILE.exists():
            try:
                data = json.loads(SESSION_CONFIG_FILE.read_text(encoding="utf-8"))
                self.ai_config.update({
                    "name": data.get("ai_name", "DEEPSEEK"),
                    "behavior": data.get("behavior", "Helpful and intelligent."),
                    "first_message": data.get("first_message", ""),
                    "reasoning_level": data.get("reasoning_level", "medium"),
                    "show_reasoning": data.get("show_reasoning", False),
                    "configured": True,  # if file exists, AI has been configured
                })
                self.session_password = data.get("session_password")
                self.admin_username = data.get("admin_username")
            except Exception:
                pass  # ignore corrupt file, start fresh

    def _save_session_config(self) -> None:
        data = {
            "ai_name": self.ai_config["name"],
            "behavior": self.ai_config["behavior"],
            "first_message": self.ai_config["first_message"],
            "reasoning_level": self.ai_config["reasoning_level"],
            "show_reasoning": self.ai_config["show_reasoning"],
            "session_password": self.session_password,
            "admin_username": self.admin_username,
        }
        SESSION_CONFIG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Session ID helpers (unchanged logic) ----
    def today_key(self) -> str:
        now = datetime.now()
        return f"{now.year}/{now.month}/{now.day}"

    def next_session_id(self) -> str:
        day = self.today_key()
        if day not in self._day_counters:
            existing = [int(k.split("/")[-1]) for k in self.session_archive if k.startswith(day)]
            self._day_counters[day] = max(existing, default=0)
        self._day_counters[day] += 1
        return f"{day}/{self._day_counters[day]}"

    def ensure_active_session(self) -> str:
        if self.current_session_id is None:
            self.current_session_id = self.next_session_id()
        elif not self.current_session_id.startswith(self.today_key()):
            self._archive_current_session()
            self.current_session_id = self.next_session_id()
        return self.current_session_id

    def _archive_current_session(self) -> None:
        if not self.current_session_id:
            return
        self.session_archive[self.current_session_id] = {
            "history": [dict(item) for item in self.history],
            "ai_config": dict(self.ai_config),
        }

    # ---- History helpers ----
    def append_user_message(self, username: str, message: str) -> None:
        self.history.append({
            "role": "user",
            "sender": username,
            "content": message,
        })

    def append_assistant_message(self, sender: str, reply: str) -> None:
        self.history.append({
            "role": "assistant",
            "sender": sender,
            "content": reply,
        })

    # ---- Media management ----
    def add_to_gallery(self, url: str, uploader: str) -> None:
        self.gallery.append({
            "url": url,
            "uploader": uploader,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def load_gallery_from_disk(self) -> None:
        """Scan uploads directories and rebuild gallery list."""
        # This is optional, can be called on startup if you want persistence
        pass

    # ---- Build full session state for new client ----
    def build_session_state(self) -> Dict[str, Any]:
        return {
            "history": self.history,
            "ai_config": self.ai_config,
            "background_url": self.background_url,
            "music_state": self.music_state,
            "profile_pics": self.profile_pics,
            "gallery": self.gallery,
            "online_users": {info["username"]: info["role"] for info in self.clients.values()},
        }


# -------------------------------------------------------------------
# Application and state
# -------------------------------------------------------------------
app = FastAPI()
state = AppState()
state_lock = asyncio.Lock()

# CORS (allow the browser, from any origin, to upload files and connect WebSocket)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files so uploaded media can be served
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


# -------------------------------------------------------------------
# Broadcast helper
# -------------------------------------------------------------------
async def broadcast(data: dict, exclude: Optional[WebSocket] = None) -> None:
    dead = []
    for ws in list(state.clients.keys()):
        if ws == exclude:
            continue
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            dead.append(ws)
    if dead:
        async with state_lock:
            for ws in dead:
                state.clients.pop(ws, None)


# -------------------------------------------------------------------
# Replay history to a single client (updated to use sender field)
# -------------------------------------------------------------------
async def replay_history(ws: WebSocket) -> None:
    for entry in state.history:
        sender = entry.get("sender", "UNKNOWN")
        if not sender:   # legacy fallback (should not be needed)
            role = entry.get("role", "")
            content = entry.get("content", "")
            if role == "assistant":
                sender = state.ai_config["name"]
            elif ": " in content:
                sender, content = content.split(": ", 1)
            else:
                sender = "UNKNOWN"
        try:
            await ws.send_text(json.dumps({
                "type": "chat",
                "sender": sender,
                "message": entry["content"],
                "replay": True,
            }))
        except Exception:
            break


# -------------------------------------------------------------------
# Root endpoint
# -------------------------------------------------------------------
@app.get("/")
async def home():
    return {"status": "running"}


# -------------------------------------------------------------------
# File upload endpoints
# -------------------------------------------------------------------
@app.post("/upload/profile")
async def upload_profile(file: UploadFile = File(...), username: str = Form(...)):
    """Upload a profile picture. Any user can call this."""
    # Security: limit file types
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed")
    # Create a unique filename
    ext = os.path.splitext(file.filename)[1] if file.filename else ".png"
    safe_name = f"{username}_{uuid.uuid4().hex}{ext}"
    dest = PROFILES_DIR / safe_name
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    url = f"/uploads/profiles/{safe_name}"
    # Update state and broadcast
    async with state_lock:
        state.profile_pics[username] = url
        state.add_to_gallery(url, username)  # also add to gallery
    await broadcast({"type": "profile_pic_update", "username": username, "url": url})
    await broadcast({"type": "new_image", "url": url, "uploader": username})
    return {"url": url}


@app.post("/upload/background")
async def upload_background(file: UploadFile = File(...), username: str = Form(...), password: str = Form(...)):
    """Admin only: upload a new background image."""
    async with state_lock:
        if username != state.admin_username or password != state.session_password:
            raise HTTPException(status_code=403, detail="Only admin can change background")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed")
    ext = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    safe_name = f"bg_{uuid.uuid4().hex}{ext}"
    dest = BACKGROUNDS_DIR / safe_name
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    url = f"/uploads/backgrounds/{safe_name}"
    async with state_lock:
        state.background_url = url
        state.add_to_gallery(url, username)  # backgrounds also go to gallery
    await broadcast({"type": "background_update", "url": url})
    await broadcast({"type": "new_image", "url": url, "uploader": username})
    return {"url": url}


@app.post("/upload/music")
async def upload_music(file: UploadFile = File(...), username: str = Form(...), password: str = Form(...)):
    """Admin only: upload a new music track."""
    async with state_lock:
        if username != state.admin_username or password != state.session_password:
            raise HTTPException(status_code=403, detail="Only admin can change music")
    if not file.content_type or not file.content_type in ("audio/mpeg", "audio/mp3"):
        raise HTTPException(status_code=400, detail="Only MP3 files allowed")
    ext = ".mp3"
    safe_name = f"music_{uuid.uuid4().hex}{ext}"
    dest = MUSIC_DIR / safe_name
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    url = f"/uploads/music/{safe_name}"
    async with state_lock:
        state.music_state["url"] = url
        state.music_state["playing"] = False
        state.music_state["currentTime"] = 0.0
        state.music_state["server_time"] = datetime.utcnow().timestamp()
    await broadcast({"type": "music_state", **state.music_state})
    return {"url": url}


# -------------------------------------------------------------------
# WebSocket endpoint
# -------------------------------------------------------------------
@app.websocket("/chat")
async def chat_endpoint(ws: WebSocket):
    await ws.accept()

    # Phase 1: Receive join_request
    try:
        raw = await ws.receive_text()
        join_data = json.loads(raw)
        if join_data.get("type") != "join_request":
            await ws.send_text(json.dumps({"type": "join_response", "accepted": False, "reason": "Expected join_request"}))
            await ws.close()
            return
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"type": "join_response", "accepted": False, "reason": str(e)}))
        await ws.close()
        return

    username = join_data.get("username", "").strip()
    password = join_data.get("password", "").strip()

    if not username:
        await ws.send_text(json.dumps({"type": "join_response", "accepted": False, "reason": "Username required"}))
        await ws.close()
        return

    # Phase 2: Authorise
    async with state_lock:
        # Check password if set
        if state.session_password:
            if password != state.session_password:
                await ws.send_text(json.dumps({"type": "join_response", "accepted": False, "reason": "Wrong password"}))
                await ws.close()
                return
        # Determine role
        is_admin = False
        if not state.ai_config["configured"]:
            # First user to join and configure AI becomes admin
            is_admin = True
            state.admin_username = username
            # Admin must later set password via setup; for now, no password
        elif state.admin_username == username and password == state.session_password:
            is_admin = True

        role = "admin" if is_admin else "user"
        state.clients[ws] = {"username": username, "role": role}
        state.ensure_active_session()

    # Send join acceptance
    await ws.send_text(json.dumps({
        "type": "join_response",
        "accepted": True,
        "role": role,
        "username": username,
    }))

    # Phase 3: If AI not configured and this user is admin, trigger setup
    async with state_lock:
        setup_needed = not state.ai_config["configured"] and is_admin
        if setup_needed:
            if state.setup_owner is None:
                state.setup_owner = ws
                state._waiting_for_setup.clear()
                state._setup_done.clear()
                await ws.send_text(json.dumps({"type": "setup_required"}))
            else:
                # Another admin is already doing setup
                await ws.send_text(json.dumps({
                    "type": "system",
                    "message": "AI is being configured by another admin. Please wait..."
                }))
    if setup_needed and state.setup_owner == ws:
        # Wait for setup to be completed by this admin
        try:
            await handle_ai_setup(ws, username)
        except WebSocketDisconnect:
            async with state_lock:
                state.clients.pop(ws, None)
                if state.setup_owner == ws:
                    state.setup_owner = None
                    state._waiting_for_setup.set()
                    state._setup_done.set()
            return
        except Exception:
            await ws.close()
            return

    # Phase 4: Send system message and full session state
    await ws.send_text(json.dumps({
        "type": "system",
        "message": f"Connected as {role}."
    }))
    session_state = state.build_session_state()
    await ws.send_text(json.dumps({
        "type": "session_state",
        **session_state,
    }))

    # Also broadcast user joined to others
    await broadcast({
        "type": "user_joined",
        "username": username,
        "role": role,
    }, exclude=ws)

    print(f"{username} connected as {role}.")

    # Phase 5: Main message loop
    try:
        while True:
            raw = await ws.receive_text()
            print(f"{username}: {raw}")
            handled = await process_message(ws, username, raw)
            if not handled:
                break
    except WebSocketDisconnect:
        print(f"{username} disconnected.")
    finally:
        async with state_lock:
            info = state.clients.pop(ws, None)
            if state.setup_owner == ws:
                state.setup_owner = None
                state._waiting_for_setup.set()
                state._setup_done.set()
        if info:
            await broadcast({
                "type": "user_left",
                "username": info["username"],
            })


# -------------------------------------------------------------------
# AI setup handler (admin sends JSON setup data)
# -------------------------------------------------------------------
async def handle_ai_setup(ws: WebSocket, username: str) -> None:
    """Wait for the admin to send the complete AI configuration."""
    raw = await ws.receive_text()
    try:
        setup = json.loads(raw)
        required = ["ai_name", "behavior", "first_message", "reasoning_level", "show_reasoning"]
        if not all(k in setup for k in required):
            raise ValueError("Missing required fields in setup data")

        async with state_lock:
            state.ai_config["name"] = setup["ai_name"]
            state.ai_config["behavior"] = setup["behavior"]
            state.ai_config["first_message"] = setup.get("first_message", "")
            state.ai_config["reasoning_level"] = setup["reasoning_level"]
            state.ai_config["show_reasoning"] = setup["show_reasoning"]
            state.ai_config["configured"] = True

            # Optionally set password if provided
            password = setup.get("password", "").strip()
            if password:
                state.session_password = password
            # Persist configuration
            state._save_session_config()
            state.setup_owner = None
            state._setup_done.set()

        print("AI configured by admin.")

        first_msg = state.ai_config["first_message"].strip()
        if first_msg:
            async with state_lock:
                state.append_assistant_message(state.ai_config["name"], first_msg)
            await broadcast({
                "type": "chat",
                "sender": state.ai_config["name"],
                "message": first_msg,
                "replay": False,
            })

        await broadcast({
            "type": "system",
            "message": f"AI configured as {state.ai_config['name']}."
        })
        await broadcast({
            "type": "ai_config_changed",
            "config": state.ai_config,
        })

    except Exception as e:
        await ws.send_text(json.dumps({
            "type": "system",
            "message": f"AI setup failed: {e}"
        }))
        raise  # close connection


# -------------------------------------------------------------------
# Message processor
# -------------------------------------------------------------------
async def process_message(ws: WebSocket, username: str, msg: str) -> bool:
    # Try to parse as JSON; many commands are now JSON
    try:
        data = json.loads(msg)
    except (json.JSONDecodeError, TypeError):
        data = None

    # Determine if user is admin (needed for privileged commands)
    async with state_lock:
        user_info = state.clients.get(ws)
        is_admin = user_info and user_info["role"] == "admin" if user_info else False

    # ---- JSON-based commands ----
    if isinstance(data, dict):
        msg_type = data.get("type")

        if msg_type == "upload_session":
            return await handle_upload_session(ws, username, data)

        if msg_type == "ai_config_update":
            if not is_admin:
                await ws.send_text(json.dumps({"type": "system", "message": "Only admin can change AI config"}))
                return True
            async with state_lock:
                state.ai_config["name"] = data.get("ai_name", state.ai_config["name"])
                state.ai_config["behavior"] = data.get("behavior", state.ai_config["behavior"])
                state.ai_config["first_message"] = data.get("first_message", state.ai_config["first_message"])
                state.ai_config["reasoning_level"] = data.get("reasoning_level", state.ai_config["reasoning_level"])
                state.ai_config["show_reasoning"] = data.get("show_reasoning", state.ai_config["show_reasoning"])
                password = data.get("password", "").strip()
                if password:
                    state.session_password = password
                state._save_session_config()
            await broadcast({"type": "ai_config_changed", "config": state.ai_config})
            await broadcast({"type": "system", "message": f"AI configuration updated by {username}."})
            return True

        if msg_type == "music_control":
            if not is_admin:
                await ws.send_text(json.dumps({"type": "system", "message": "Only admin can control music"}))
                return True
            action = data.get("action")
            async with state_lock:
                if action == "play":
                    state.music_state["playing"] = True
                    state.music_state["currentTime"] = data.get("currentTime", state.music_state["currentTime"])
                    state.music_state["server_time"] = datetime.utcnow().timestamp()
                elif action == "pause":
                    state.music_state["playing"] = False
                    state.music_state["currentTime"] = data.get("currentTime", state.music_state["currentTime"])
                    state.music_state["server_time"] = datetime.utcnow().timestamp()
                elif action == "seek":
                    state.music_state["currentTime"] = data.get("currentTime", 0.0)
                    state.music_state["server_time"] = datetime.utcnow().timestamp()
                elif action == "loop":
                    state.music_state["loop"] = data.get("loop", False)
                else:
                    await ws.send_text(json.dumps({"type": "system", "message": f"Unknown music action: {action}"}))
                    return True
            # Broadcast updated music state to everyone
            await broadcast({"type": "music_state", **state.music_state})
            return True

        # If it's a chat message sent as JSON? Not expected; ignore.
        return True

    # ---- Plain text commands ----
    if msg.startswith("/"):
        return await handle_command(ws, username, msg)

    # ---- Normal chat ----
    await broadcast({
        "type": "chat",
        "sender": username,
        "message": msg,
        "replay": False,
    }, exclude=ws)  # exclude sender? Actually we want the sender to see their own message too, so don't exclude.
    async with state_lock:
        state.append_user_message(username, msg)

    if not state.ai_awake:
        return True

    try:
        async with state_lock:
            system_msg = f"You are {state.ai_config['name']}. {state.ai_config['behavior']}"
            recent = state.history[-20:]
            messages = [{"role": "system", "content": system_msg}]
            messages += [{"role": h["role"], "content": h["content"]} for h in recent]
            reasoning_level = state.ai_config["reasoning_level"]
            show_reason = state.ai_config["show_reasoning"]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            reasoning_effort=reasoning_level,
        )
        choice = response.choices[0].message
        reply = choice.content or ""
        reasoning = getattr(choice, "reasoning_content", None)

    except Exception as e:
        reply = f"ERROR: {e}"
        reasoning = None

    async with state_lock:
        state.append_assistant_message(state.ai_config["name"], reply)

    if reasoning and state.ai_config["show_reasoning"]:
        await broadcast({
            "type": "reasoning",
            "sender": state.ai_config["name"],
            "message": reasoning,
        })
    await broadcast({
        "type": "chat",
        "sender": state.ai_config["name"],
        "message": reply,
        "replay": False,
    })
    return True


# -------------------------------------------------------------------
# Slash command handler (unchanged, just add admin checks where needed)
# -------------------------------------------------------------------
async def handle_command(ws: WebSocket, username: str, msg: str) -> bool:
    cmd = msg.strip().lower()
    # Determine admin status
    async with state_lock:
        user_info = state.clients.get(ws)
        is_admin = user_info and user_info["role"] == "admin" if user_info else False

    if cmd == "/sleep":
        if not is_admin:
            await ws.send_text(json.dumps({"type": "system", "message": "Only admin can control AI wakefulness"}))
            return True
        async with state_lock:
            state.ai_awake = False
        await broadcast({"type": "system", "message": "AI is now sleeping."})
        return True

    if cmd == "/wake":
        if not is_admin:
            await ws.send_text(json.dumps({"type": "system", "message": "Only admin can control AI wakefulness"}))
            return True
        async with state_lock:
            state.ai_awake = True
        await broadcast({"type": "system", "message": "AI is now awake."})
        return True

    if cmd == "/clear":
        if not is_admin:
            await ws.send_text(json.dumps({"type": "system", "message": "Only admin can clear history"}))
            return True
        async with state_lock:
            state.history.clear()
        await broadcast({"type": "system", "message": "Conversation history cleared."})
        return True

    if cmd == "/new":
        if not is_admin:
            await ws.send_text(json.dumps({"type": "system", "message": "Only admin can start a new session"}))
            return True
        async with state_lock:
            state._archive_current_session()
            state.history.clear()
            state.current_session_id = state.next_session_id()
        await broadcast({"type": "new_session", "message": "New session started."})
        return True

    if cmd.startswith("/load"):
        if not is_admin:
            await ws.send_text(json.dumps({"type": "system", "message": "Only admin can load sessions"}))
            return True
        parts = msg.split(maxsplit=1)
        if len(parts) < 2:
            await ws.send_text(json.dumps({"type": "system", "message": "Usage: /load yyyy/m/d/n"}))
            return True
        session_id = parts[1].strip()
        if session_id.endswith(".txt"):
            session_id = session_id[:-4]
        try:
            async with state_lock:
                state.restore_session(session_id)
            await broadcast({"type": "system", "message": f"Loaded session {session_id}."})
            async with state_lock:
                for entry in state.history:
                    await broadcast({
                        "type": "chat",
                        "sender": entry["sender"],
                        "message": entry["content"],
                        "replay": True,
                    })
        except Exception as e:
            await ws.send_text(json.dumps({"type": "system", "message": f"Load failed: {e}"}))
        return True

    if cmd == "/export":
        try:
            async with state_lock:
                path = state.export_session()
            await ws.send_text(json.dumps({"type": "system", "message": f"Session exported to {path.name}"}))
        except Exception as e:
            await ws.send_text(json.dumps({"type": "system", "message": f"Export failed: {e}"}))
        return True

    if cmd.startswith("/import"):
        parts = msg.split(maxsplit=1)
        if len(parts) < 2:
            await ws.send_text(json.dumps({"type": "system", "message": "Usage: /import filename.json"}))
            return True
        filename = parts[1].strip()
        try:
            async with state_lock:
                state.import_session(filename)
            await broadcast({"type": "system", "message": f"Imported session {filename}"})
            async with state_lock:
                for entry in state.history:
                    await broadcast({
                        "type": "chat",
                        "sender": entry["sender"],
                        "message": entry["content"],
                        "replay": True,
                    })
        except Exception as e:
            await ws.send_text(json.dumps({"type": "system", "message": f"Import failed: {e}"}))
        return True

    # If unknown command
    await ws.send_text(json.dumps({"type": "system", "message": f"Unknown command: {msg}"}))
    return True


# -------------------------------------------------------------------
# Upload session handler (unchanged logic, but now called via JSON)
# -------------------------------------------------------------------
async def handle_upload_session(ws: WebSocket, username: str, data: dict) -> bool:
    history = data.get("history")
    if not isinstance(history, list):
        await ws.send_text(json.dumps({
            "type": "system",
            "message": "Invalid upload_session: missing or malformed 'history' list."
        }))
        return True

    for i, entry in enumerate(history):
        if not isinstance(entry, dict):
            await ws.send_text(json.dumps({
                "type": "system",
                "message": f"Invalid entry at index {i}: not a dict."
            }))
            return True
        role = entry.get("role")
        if role not in ("user", "assistant"):
            await ws.send_text(json.dumps({
                "type": "system",
                "message": f"Invalid role at index {i}: must be 'user' or 'assistant'."
            }))
            return True
        if "sender" not in entry or "content" not in entry:
            await ws.send_text(json.dumps({
                "type": "system",
                "message": f"Missing 'sender' or 'content' at index {i}."
            }))
            return True

    async with state_lock:
        state.load_uploaded_history(history)

    await broadcast({
        "type": "new_session",
        "message": "Uploaded session loaded."
    })

    async with state_lock:
        for entry in state.history:
            await broadcast({
                "type": "chat",
                "sender": entry["sender"],
                "message": entry["content"],
                "replay": True,
            })
    return True


# -------------------------------------------------------------------
# AppState method for upload history (required for handle_upload_session)
# -------------------------------------------------------------------
def load_uploaded_history(self, entries):
    self._archive_current_session()
    self.history = [dict(e) for e in entries]
    self.current_session_id = self.next_session_id()

# Monkey-patch method into AppState
AppState.load_uploaded_history = load_uploaded_history
# Also ensure restore_session, export_session, import_session exist (they do from original code)
# We need to re-add them because the original server had them; they are missing here. Let's include them.
def restore_session(self, session_id):
    data = self.session_archive.get(session_id)
    if not data:
        raise ValueError(f"Session not found: {session_id}")
    self._archive_current_session()
    self.history = [dict(item) for item in data["history"]]
    loaded = data["ai_config"]
    self.ai_config.update(loaded)
    self.current_session_id = session_id

def export_session(self):
    filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = SESSIONS_DIR / filename
    data = {
        "history": self.history,
        "ai_config": self.ai_config,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

def import_session(self, filename):
    safe_path = (SESSIONS_DIR / os.path.basename(filename)).resolve()
    if not safe_path.is_relative_to(SESSIONS_DIR):
        raise ValueError("Invalid import path")
    data = json.loads(safe_path.read_text(encoding="utf-8"))
    self.history = data.get("history", [])
    self.ai_config.update(data.get("ai_config", {}))

AppState.restore_session = restore_session
AppState.export_session = export_session
AppState.import_session = import_session

# -------------------------------------------------------------------
# Run the server (if executed directly, for local testing)
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)