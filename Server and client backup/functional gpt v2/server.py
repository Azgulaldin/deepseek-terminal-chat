from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import asyncio
import json
import os
import re
import shutil
import time
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, List

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

client: Optional[OpenAI]
if API_KEY:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
else:
    client = None

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
GALLERY_DIR = UPLOADS_DIR / "gallery"
GALLERY_DIR.mkdir(exist_ok=True)

SESSION_CONFIG_FILE = DATA_DIR / "session_config.json"

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def utc_ts() -> float:
    return time.time()


def safe_username(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\-\. ]+", "_", name)
    return name[:64] or "user"


def safe_persona(persona: str) -> str:
    persona = str(persona or "").strip()
    persona = re.sub(r"[\x00-\x1f\x7f]+", " ", persona)
    persona = re.sub(r"\s+", " ", persona)
    return persona[:256]


def safe_session_parts(session_id: str) -> List[str]:
    parts = [p for p in session_id.split("/") if p]
    if len(parts) != 4 or not all(p.isdigit() for p in parts):
        raise ValueError("Invalid session id format. Expected yyyy/mm/dd/n")
    return parts


def session_path(session_id: str) -> Path:
    parts = safe_session_parts(session_id)
    return SESSIONS_DIR / parts[0] / parts[1] / parts[2] / f"{parts[3]}.json"


def latest_existing_counter_for_day(day_dir: Path) -> int:
    if not day_dir.exists():
        return 0
    counters = []
    for p in day_dir.glob("*.json"):
        try:
            counters.append(int(p.stem))
        except ValueError:
            pass
    return max(counters, default=0)


# -------------------------------------------------------------------
# Application state
# -------------------------------------------------------------------
class AppState:
    def __init__(self) -> None:
        self.clients: Dict[WebSocket, Dict[str, Any]] = {}
        self.ai_config: Dict[str, Any] = {
            "name": "DEEPSEEK",
            "behavior": "Helpful and intelligent.",
            "first_message": "",
            "reasoning_level": "medium",
            "show_reasoning": False,
            "configured": False,
            "admin_username": None,
        }
        self.history: List[Dict[str, Any]] = []
        self.ai_awake: bool = True
        self.current_session_id: Optional[str] = None
        self.session_archive_cache: Dict[str, Dict[str, Any]] = {}
        self._day_counters: Dict[str, int] = {}
        self.admin_username: Optional[str] = None
        self.session_password: Optional[str] = None
        self.background_url: Optional[str] = None
        self.music_state: Dict[str, Any] = {
            "url": None,
            "playing": False,
            "currentTime": 0.0,
            "loop": False,
            "server_time": utc_ts(),
        }
        self.profile_pics: Dict[str, str] = {}
        self.gallery: List[Dict[str, Any]] = []
        self.setup_owner: Optional[WebSocket] = None
        self._setup_in_progress: bool = False
        self._load_session_config()
        self._load_gallery_from_disk()

    # ---- Persistence ----
    def _load_session_config(self) -> None:
        if not SESSION_CONFIG_FILE.exists():
            return
        try:
            data = json.loads(SESSION_CONFIG_FILE.read_text(encoding="utf-8"))
            self.ai_config.update({
                "name": data.get("ai_name", self.ai_config["name"]),
                "behavior": data.get("behavior", self.ai_config["behavior"]),
                "first_message": data.get("first_message", self.ai_config["first_message"]),
                "reasoning_level": data.get("reasoning_level", self.ai_config["reasoning_level"]),
                "show_reasoning": data.get("show_reasoning", self.ai_config["show_reasoning"]),
                "configured": True,
                "admin_username": data.get("admin_username"),
            })
            self.admin_username = data.get("admin_username")
            self.session_password = data.get("session_password")
        except Exception:
            pass

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

    def _load_gallery_from_disk(self) -> None:
        self.gallery = []
        for folder, uploader_hint in ((PROFILES_DIR, "profile"), (BACKGROUNDS_DIR, "background")):
            if not folder.exists():
                continue
            for p in sorted(folder.glob("*")):
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                    rel = p.relative_to(DATA_DIR).as_posix()
                    self.gallery.append({
                        "url": f"/{rel}",
                        "uploader": uploader_hint,
                        "timestamp": datetime.utcfromtimestamp(p.stat().st_mtime).isoformat(),
                    })

    def today_key(self) -> str:
        now = datetime.utcnow()
        return f"{now.year:04d}/{now.month:02d}/{now.day:02d}"

    def next_session_id(self) -> str:
        day = self.today_key()
        if day not in self._day_counters:
            day_dir = SESSIONS_DIR / day
            self._day_counters[day] = latest_existing_counter_for_day(day_dir)
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
        payload = {
            "session_id": self.current_session_id,
            "history": [dict(item) for item in self.history],
            "ai_config": dict(self.ai_config),
            "saved_at": datetime.utcnow().isoformat(),
        }
        self.session_archive_cache[self.current_session_id] = payload
        path = session_path(self.current_session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_session_from_disk(self, session_id: str) -> Dict[str, Any]:
        path = session_path(session_id)
        if not path.exists():
            raise ValueError(f"Session not found: {session_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def restore_session(self, session_id: str) -> None:
        payload = self.load_session_from_disk(session_id)
        self._archive_current_session()
        self.history = [dict(item) for item in payload.get("history", [])]
        loaded_cfg = payload.get("ai_config", {})
        if isinstance(loaded_cfg, dict):
            self.ai_config.update(loaded_cfg)
        self.admin_username = self.ai_config.get("admin_username") or self.admin_username
        self.current_session_id = session_id

    def export_session(self) -> Path:
        self.ensure_active_session()
        filename = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        path = SESSIONS_DIR / filename
        data = {
            "session_id": self.current_session_id,
            "history": self.history,
            "ai_config": self.ai_config,
            "exported_at": datetime.utcnow().isoformat(),
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def import_session(self, filename: str) -> None:
        safe_name = os.path.basename(filename)
        path = (SESSIONS_DIR / safe_name).resolve()
        if not str(path).startswith(str(SESSIONS_DIR.resolve())):
            raise ValueError("Invalid import path")
        if not path.exists():
            raise ValueError(f"File not found: {safe_name}")
        data = json.loads(path.read_text(encoding="utf-8"))
        self._archive_current_session()
        self.history = [dict(item) for item in data.get("history", [])]
        loaded_cfg = data.get("ai_config", {})
        if isinstance(loaded_cfg, dict):
            self.ai_config.update(loaded_cfg)
        self.admin_username = self.ai_config.get("admin_username") or self.admin_username
        self.current_session_id = self.next_session_id()

    def load_uploaded_history(self, entries: List[Dict[str, Any]]) -> None:
        self._archive_current_session()
        self.history = [dict(e) for e in entries]
        self.current_session_id = self.next_session_id()

    # ---- History ----
    def append_user_message(self, username: str, message: str) -> None:
        self.history.append({"role": "user", "sender": username, "content": message})

    def append_assistant_message(self, sender: str, reply: str) -> None:
        self.history.append({"role": "assistant", "sender": sender, "content": reply})

    # ---- Media ----
    def add_to_gallery(self, url: str, uploader: str) -> None:
        self.gallery.append({
            "url": url,
            "uploader": uploader,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def get_online_participants(self) -> List[Dict[str, Any]]:
        participants: List[Dict[str, Any]] = []
        for info in self.clients.values():
            participants.append({
                "username": info.get("username", ""),
                "role": info.get("role", "user"),
                "persona": info.get("persona", ""),
            })
        participants.sort(key=lambda item: item["username"].lower())
        return participants

    def build_participant_context(self) -> str:
        participants = self.get_online_participants()
        if not participants:
            return "No other participants are currently online."

        lines = [f"Participant count: {len(participants)}", "Online participants:"]
        for person in participants:
            username = person.get("username") or "UNKNOWN"
            role = person.get("role") or "user"
            persona = person.get("persona") or ""
            if persona:
                lines.append(f"- {username} ({role}) — Persona: {persona}")
            else:
                lines.append(f"- {username} ({role}) — Persona: [not provided]")
        return "\n".join(lines)

    def build_session_state(self) -> Dict[str, Any]:
        participants = self.get_online_participants()
        return {
            "history": self.history,
            "ai_config": self.ai_config,
            "admin_username": self.admin_username,
            "background_url": self.background_url,
            "music_state": self.music_state,
            "profile_pics": self.profile_pics,
            "gallery": self.gallery,
            "online_users": {info["username"]: info["role"] for info in participants},
            "online_user_details": participants,
            "online_user_personas": {info["username"]: info.get("persona", "") for info in participants},
            "participant_count": len(participants),
            "current_session_id": self.current_session_id,
        }


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI()
state = AppState()
state_lock = asyncio.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
async def broadcast(data: dict, exclude: Optional[WebSocket] = None) -> None:
    dead: List[WebSocket] = []
    payload = json.dumps(data)
    for ws in list(state.clients.keys()):
        if ws == exclude:
            continue
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    if dead:
        async with state_lock:
            for ws in dead:
                state.clients.pop(ws, None)


async def send_system(ws: WebSocket, message: str) -> None:
    await ws.send_text(json.dumps({"type": "system", "message": message}))


async def maybe_generate_ai_reply(username: str, message: str) -> None:
    if not state.ai_awake:
        return
    if client is None:
        await broadcast({
            "type": "system",
            "message": "DeepSeek API key is missing. AI replies are disabled.",
        })
        return

    try:
        async with state_lock:
            ai_name = state.ai_config["name"]
            behavior = state.ai_config["behavior"]
            reasoning_level = state.ai_config["reasoning_level"]
            show_reason = state.ai_config["show_reasoning"]
            recent = state.history[-20:]
            participant_context = state.build_participant_context()
            system_msg = (
                f"You are {ai_name}. {behavior}\n\n"
                f"{participant_context}\n\n"
                "Use the participant information above when addressing users, tracking who is present, "
                "and respecting each user's persona when relevant."
            )
            messages = [{"role": "system", "content": system_msg}]
            for h in recent:
                role = h.get("role")
                content = h.get("content", "")
                sender = h.get("sender", "")
                if role in ("user", "assistant"):
                    if sender:
                        messages.append({
                            "role": role,
                            "content": f"{sender}: {content}",
                        })
                    else:
                        messages.append({"role": role, "content": content})

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
        ai_name = state.ai_config["name"]
        state.append_assistant_message(ai_name, reply)

    if reasoning and state.ai_config.get("show_reasoning"):
        await broadcast({"type": "reasoning", "sender": state.ai_config["name"], "message": reasoning})

    await broadcast({"type": "chat", "sender": state.ai_config["name"], "message": reply, "replay": False})


async def handle_ai_setup(ws: WebSocket, username: str) -> None:
    raw = await ws.receive_text()
    try:
        setup = json.loads(raw)
    except Exception:
        raise ValueError("Expected JSON setup message")

    if setup.get("type") not in {"ai_config_update", "setup_config"}:
        raise ValueError("Expected ai_config_update during setup")

    required = ["ai_name", "behavior", "reasoning_level", "show_reasoning"]
    if not all(k in setup for k in required):
        raise ValueError("Missing required setup fields")

    async with state_lock:
        state.ai_config["name"] = str(setup.get("ai_name", state.ai_config["name"]))[:64] or state.ai_config["name"]
        state.ai_config["behavior"] = str(setup.get("behavior", state.ai_config["behavior"]))
        state.ai_config["first_message"] = str(setup.get("first_message", ""))
        state.ai_config["reasoning_level"] = str(setup.get("reasoning_level", "medium"))
        state.ai_config["show_reasoning"] = bool(setup.get("show_reasoning", False))
        state.ai_config["configured"] = True
        state.ai_config["admin_username"] = username
        state.admin_username = username
        password = str(setup.get("password", "")).strip()
        if password:
            state.session_password = password
        state._save_session_config()
        state.setup_owner = None
        state._setup_in_progress = False

    first_msg = state.ai_config["first_message"].strip()
    if first_msg:
        async with state_lock:
            state.append_assistant_message(state.ai_config["name"], first_msg)
        await broadcast({"type": "chat", "sender": state.ai_config["name"], "message": first_msg, "replay": False})

    await broadcast({"type": "system", "message": f"AI configured as {state.ai_config['name']}."})
    await broadcast({"type": "ai_config_changed", "config": state.ai_config})


# -------------------------------------------------------------------
# Upload endpoints
# -------------------------------------------------------------------
@app.post("/upload/profile")
async def upload_profile(file: UploadFile = File(...), username: str = Form(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed")

    uname = safe_username(username)
    ext = os.path.splitext(file.filename or "")[1] or ".png"
    safe_name = f"{uname}_{uuid.uuid4().hex}{ext}"
    dest = PROFILES_DIR / safe_name
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    url = f"/uploads/profiles/{safe_name}"

    async with state_lock:
        state.profile_pics[username] = url
        state.add_to_gallery(url, username)

    await broadcast({"type": "profile_pic_update", "username": username, "url": url})
    await broadcast({"type": "new_image", "url": url, "uploader": username})
    return {"url": url}


@app.post("/upload/background")
async def upload_background(
    file: UploadFile = File(...),
    username: str = Form(...),
    password: str = Form(...),
):
    async with state_lock:
        if username != state.admin_username or password != (state.session_password or ""):
            raise HTTPException(status_code=403, detail="Only admin can change background")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed")

    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    safe_name = f"bg_{uuid.uuid4().hex}{ext}"
    dest = BACKGROUNDS_DIR / safe_name
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    url = f"/uploads/backgrounds/{safe_name}"

    async with state_lock:
        state.background_url = url
        state.add_to_gallery(url, username)

    await broadcast({"type": "background_update", "url": url})
    await broadcast({"type": "new_image", "url": url, "uploader": username})
    return {"url": url}


@app.post("/upload/music")
async def upload_music(
    file: UploadFile = File(...),
    username: str = Form(...),
    password: str = Form(...),
):
    async with state_lock:
        if username != state.admin_username or password != (state.session_password or ""):
            raise HTTPException(status_code=403, detail="Only admin can change music")

    content_type = (file.content_type or "").lower()
    if content_type not in {"audio/mpeg", "audio/mp3", "audio/x-mpeg", "audio/mpeg3", "audio/x-mpeg-3"}:
        raise HTTPException(status_code=400, detail="Only MP3 files allowed")

    safe_name = f"music_{uuid.uuid4().hex}.mp3"
    dest = MUSIC_DIR / safe_name
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    url = f"/uploads/music/{safe_name}"

    async with state_lock:
        state.music_state = {
            "url": url,
            "playing": False,
            "currentTime": 0.0,
            "loop": False,
            "server_time": utc_ts(),
        }

    await broadcast({"type": "music_state", **state.music_state})
    return {"url": url}


# -------------------------------------------------------------------
# WebSocket endpoint
# -------------------------------------------------------------------
@app.websocket("/chat")
async def chat_endpoint(ws: WebSocket):
    await ws.accept()

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

    username = str(join_data.get("username", "")).strip()
    password = str(join_data.get("password", "")).strip()
    persona = safe_persona(join_data.get("persona", ""))
    if not username:
        await ws.send_text(json.dumps({"type": "join_response", "accepted": False, "reason": "Username required"}))
        await ws.close()
        return

    async with state_lock:
        if state.session_password and password != state.session_password:
            await ws.send_text(json.dumps({"type": "join_response", "accepted": False, "reason": "Wrong password"}))
            await ws.close()
            return

        is_admin = False
        if not state.ai_config.get("configured", False):
            is_admin = True
            if state.admin_username is None:
                state.admin_username = username
                state.ai_config["admin_username"] = username
        elif state.admin_username == username and (not state.session_password or password == state.session_password):
            is_admin = True

        role = "admin" if is_admin else "user"
        state.clients[ws] = {"username": username, "role": role, "persona": persona}
        state.ensure_active_session()
        setup_needed = not state.ai_config.get("configured", False) and is_admin and not state._setup_in_progress
        if setup_needed:
            state.setup_owner = ws
            state._setup_in_progress = True

    await ws.send_text(json.dumps({
        "type": "join_response",
        "accepted": True,
        "role": role,
        "username": username,
        "persona": persona,
    }))

    if setup_needed:
        await ws.send_text(json.dumps({"type": "setup_required"}))

    await ws.send_text(json.dumps({"type": "system", "message": f"Connected as {role}."}))
    await ws.send_text(json.dumps({"type": "session_state", **state.build_session_state()}))

    await broadcast({"type": "user_joined", "username": username, "role": role, "persona": persona}, exclude=ws)
    print(f"{username} connected as {role}.")

    try:
        if setup_needed:
            try:
                await handle_ai_setup(ws, username)
            except WebSocketDisconnect:
                raise
            except Exception as e:
                await ws.send_text(json.dumps({"type": "system", "message": f"AI setup failed: {e}"}))
                await ws.close()
                return

        while True:
            raw = await ws.receive_text()
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
                state._setup_in_progress = False
        if info:
            await broadcast({"type": "user_left", "username": info["username"], "role": info.get("role", "user"), "persona": info.get("persona", "")})


# -------------------------------------------------------------------
# Message processing
# -------------------------------------------------------------------
async def process_message(ws: WebSocket, username: str, msg: str) -> bool:
    try:
        data = json.loads(msg)
    except Exception:
        data = None

    async with state_lock:
        user_info = state.clients.get(ws)
        is_admin = bool(user_info and user_info.get("role") == "admin")

    if isinstance(data, dict):
        msg_type = data.get("type")

        if msg_type == "upload_session":
            return await handle_upload_session(ws, username, data)

        if msg_type == "ai_config_update":
            if not is_admin:
                await send_system(ws, "Only admin can change AI config")
                return True
            async with state_lock:
                state.ai_config["name"] = str(data.get("ai_name", state.ai_config["name"]))[:64] or state.ai_config["name"]
                state.ai_config["behavior"] = str(data.get("behavior", state.ai_config["behavior"]))
                state.ai_config["first_message"] = str(data.get("first_message", state.ai_config["first_message"]))
                state.ai_config["reasoning_level"] = str(data.get("reasoning_level", state.ai_config["reasoning_level"]))
                state.ai_config["show_reasoning"] = bool(data.get("show_reasoning", state.ai_config["show_reasoning"]))
                state.ai_config["configured"] = True
                state.ai_config["admin_username"] = username
                state.admin_username = username
                password = str(data.get("password", "")).strip()
                if password:
                    state.session_password = password
                state._save_session_config()
                state.setup_owner = None
                state._setup_in_progress = False
            await broadcast({"type": "ai_config_changed", "config": state.ai_config})
            await broadcast({"type": "system", "message": f"AI configuration updated by {username}."})
            first_msg = state.ai_config["first_message"].strip()
            if first_msg:
                async with state_lock:
                    state.append_assistant_message(state.ai_config["name"], first_msg)
                await broadcast({"type": "chat", "sender": state.ai_config["name"], "message": first_msg, "replay": False})
            return True

        if msg_type == "music_control":
            if not is_admin:
                await send_system(ws, "Only admin can control music")
                return True
            action = data.get("action")
            async with state_lock:
                if action == "play":
                    state.music_state["playing"] = True
                    state.music_state["currentTime"] = float(data.get("currentTime", state.music_state["currentTime"]))
                    state.music_state["server_time"] = utc_ts()
                elif action == "pause":
                    state.music_state["playing"] = False
                    state.music_state["currentTime"] = float(data.get("currentTime", state.music_state["currentTime"]))
                    state.music_state["server_time"] = utc_ts()
                elif action == "seek":
                    state.music_state["currentTime"] = float(data.get("currentTime", 0.0))
                    state.music_state["server_time"] = utc_ts()
                elif action == "loop":
                    state.music_state["loop"] = bool(data.get("loop", False))
                    state.music_state["server_time"] = utc_ts()
                else:
                    await send_system(ws, f"Unknown music action: {action}")
                    return True
            await broadcast({"type": "music_state", **state.music_state})
            return True

        return True

    if msg.startswith("/"):
        return await handle_command(ws, username, msg)

    async with state_lock:
        state.append_user_message(username, msg)

    await broadcast({
        "type": "chat",
        "sender": username,
        "message": msg,
        "replay": False,
    })

    await maybe_generate_ai_reply(username, msg)
    return True


async def handle_command(ws: WebSocket, username: str, msg: str) -> bool:
    cmd = msg.strip()
    lower = cmd.lower()

    async with state_lock:
        user_info = state.clients.get(ws)
        is_admin = bool(user_info and user_info.get("role") == "admin")

    if lower == "/sleep":
        if not is_admin:
            await send_system(ws, "Only admin can control AI wakefulness")
            return True
        async with state_lock:
            state.ai_awake = False
        await broadcast({"type": "system", "message": "AI is now sleeping."})
        return True

    if lower == "/wake":
        if not is_admin:
            await send_system(ws, "Only admin can control AI wakefulness")
            return True
        async with state_lock:
            state.ai_awake = True
        await broadcast({"type": "system", "message": "AI is now awake."})
        return True

    if lower == "/clear":
        if not is_admin:
            await send_system(ws, "Only admin can clear history")
            return True
        async with state_lock:
            state.history.clear()
        await broadcast({"type": "system", "message": "Conversation history cleared."})
        return True

    if lower == "/new":
        if not is_admin:
            await send_system(ws, "Only admin can start a new session")
            return True
        async with state_lock:
            state._archive_current_session()
            state.history.clear()
            state.current_session_id = state.next_session_id()
        await broadcast({"type": "new_session", "message": "New session started."})
        return True

    if lower.startswith("/load"):
        if not is_admin:
            await send_system(ws, "Only admin can load sessions")
            return True
        parts = cmd.split(maxsplit=1)
        if len(parts) < 2:
            await send_system(ws, "Usage: /load yyyy/mm/dd/n")
            return True
        session_id = parts[1].strip()
        if session_id.endswith(".json"):
            session_id = session_id[:-5]
        try:
            async with state_lock:
                state.restore_session(session_id)
            await broadcast({"type": "new_session", "message": f"Loaded session {session_id}."})
            async with state_lock:
                replay = [dict(item) for item in state.history]
            for entry in replay:
                await broadcast({
                    "type": "chat",
                    "sender": entry.get("sender", "UNKNOWN"),
                    "message": entry.get("content", ""),
                    "replay": True,
                })
        except Exception as e:
            await send_system(ws, f"Load failed: {e}")
        return True

    if lower == "/export":
        try:
            async with state_lock:
                path = state.export_session()
            await send_system(ws, f"Session exported to {path.name}")
        except Exception as e:
            await send_system(ws, f"Export failed: {e}")
        return True

    if lower.startswith("/import"):
        parts = cmd.split(maxsplit=1)
        if len(parts) < 2:
            await send_system(ws, "Usage: /import filename.json")
            return True
        filename = parts[1].strip()
        try:
            async with state_lock:
                state.import_session(filename)
            await broadcast({"type": "new_session", "message": f"Imported session {filename}"})
            async with state_lock:
                replay = [dict(item) for item in state.history]
            for entry in replay:
                await broadcast({
                    "type": "chat",
                    "sender": entry.get("sender", "UNKNOWN"),
                    "message": entry.get("content", ""),
                    "replay": True,
                })
        except Exception as e:
            await send_system(ws, f"Import failed: {e}")
        return True

    await send_system(ws, f"Unknown command: {msg}")
    return True


async def handle_upload_session(ws: WebSocket, username: str, data: dict) -> bool:
    history = data.get("history")
    if not isinstance(history, list):
        await send_system(ws, "Invalid upload_session: missing or malformed 'history' list.")
        return True

    cleaned: List[Dict[str, Any]] = []
    for i, entry in enumerate(history):
        if not isinstance(entry, dict):
            await send_system(ws, f"Invalid entry at index {i}: not a dict.")
            return True
        role = entry.get("role")
        sender = entry.get("sender")
        content = entry.get("content")
        if role not in ("user", "assistant"):
            await send_system(ws, f"Invalid role at index {i}: must be 'user' or 'assistant'.")
            return True
        if not isinstance(sender, str) or not isinstance(content, str):
            await send_system(ws, f"Invalid sender/content at index {i}.")
            return True
        cleaned.append({"role": role, "sender": sender, "content": content})

    async with state_lock:
        state.load_uploaded_history(cleaned)

    await broadcast({"type": "new_session", "message": "Uploaded session loaded."})
    for entry in cleaned:
        await broadcast({
            "type": "chat",
            "sender": entry["sender"],
            "message": entry["content"],
            "replay": True,
        })
    return True


# -------------------------------------------------------------------
# Simple HTTP endpoint
# -------------------------------------------------------------------
@app.get("/")
async def home():
    return {"status": "running"}


# -------------------------------------------------------------------
# Local testing
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
