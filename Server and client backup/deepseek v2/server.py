# server.py – DeepSeek roleplay multi‑user WebSocket server
# ------------------------------------------------------------------
# Added: /upload_session via WebSocket JSON command.
# ------------------------------------------------------------------

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import OpenAI
import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise RuntimeError("Environment variable DEEPSEEK_API_KEY is required")

BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"

# secure directory for exported / imported session files
EXPORT_DIR = Path("./sessions").resolve()
EXPORT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# OpenAI client
# -------------------------------------------------------------------
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# -------------------------------------------------------------------
# Global application state
# -------------------------------------------------------------------
class AppState:
    def __init__(self) -> None:
        self.clients: Dict[WebSocket, str] = {}
        self.ai_config: Dict[str, Any] = {
            "name": "DEEPSEEK",
            "behavior": "Helpful and intelligent.",
            "first_message": "",
            "reasoning_level": "medium",
            "show_reasoning": False,
            "configured": False,
        }
        self.history: List[Dict[str, Any]] = []
        self.ai_awake: bool = True
        self.current_session_id: Optional[str] = None
        self.session_archive: Dict[str, Dict[str, Any]] = {}
        self._day_counters: Dict[str, int] = {}
        self.setup_owner: Optional[WebSocket] = None
        self._waiting_for_setup: asyncio.Event = asyncio.Event()
        self._setup_done: asyncio.Event = asyncio.Event()
        self._setup_done.set()

    def today_key(self) -> str:
        now = datetime.now()
        return f"{now.year}/{now.month}/{now.day}"

    def next_session_id(self) -> str:
        day = self.today_key()
        if day not in self._day_counters:
            existing = [int(k.split("/")[-1]) for k in self.session_archive
                        if k.startswith(day)]
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

    def restore_session(self, session_id: str) -> None:
        data = self.session_archive.get(session_id)
        if not data:
            raise ValueError(f"Session not found: {session_id}")
        self._archive_current_session()
        self.history = [dict(item) for item in data["history"]]
        loaded = data["ai_config"]
        self.ai_config.update(loaded)
        self.current_session_id = session_id

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

    def export_session(self) -> Path:
        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = EXPORT_DIR / filename
        data = {
            "history": self.history,
            "ai_config": self.ai_config,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def import_session(self, filename: str) -> None:
        safe_path = (EXPORT_DIR / os.path.basename(filename)).resolve()
        if not safe_path.is_relative_to(EXPORT_DIR):
            raise ValueError("Invalid import path")
        data = json.loads(safe_path.read_text(encoding="utf-8"))
        self.history = data.get("history", [])
        self.ai_config.update(data.get("ai_config", {}))

    # ----- New method for uploaded session -----
    def load_uploaded_history(self, entries: List[Dict[str, Any]]) -> None:
        """Replace current history with uploaded entries and start a new session."""
        self._archive_current_session()
        self.history = [dict(e) for e in entries]
        self.current_session_id = self.next_session_id()


# -------------------------------------------------------------------
# Application and state
# -------------------------------------------------------------------
app = FastAPI()
state = AppState()
state_lock = asyncio.Lock()


# -------------------------------------------------------------------
# Broadcast helper
# -------------------------------------------------------------------
async def broadcast(data: dict) -> None:
    dead = []
    for ws in list(state.clients.keys()):
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            dead.append(ws)
    if dead:
        async with state_lock:
            for ws in dead:
                state.clients.pop(ws, None)


# -------------------------------------------------------------------
# Replay history to a single client
# -------------------------------------------------------------------
async def replay_history(ws: WebSocket) -> None:
    for entry in state.history:
        sender = entry.get("sender", "UNKNOWN")
        if not sender:   # legacy fallback
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
# WebSocket endpoint
# -------------------------------------------------------------------
@app.websocket("/chat")
async def chat_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        username = await ws.receive_text()
    except WebSocketDisconnect:
        return

    async with state_lock:
        state.clients[ws] = username
        state.ensure_active_session()

    print(f"{username} connected.")

    if not state.ai_config["configured"]:
        await handle_initial_setup(ws, username)
        if ws not in state.clients:
            return

    try:
        await ws.send_text(json.dumps({
            "type": "system",
            "message": f"Connected to {state.ai_config['name']}"
        }))
        async with state_lock:
            await replay_history(ws)
    except Exception:
        return

    try:
        while True:
            try:
                raw = await ws.receive_text()
            except WebSocketDisconnect:
                print(f"{username} disconnected.")
                break

            print(f"{username}: {raw}")
            handled = await process_message(ws, username, raw)
            if not handled:
                break
    except Exception as e:
        print(f"{username} error: {e}")
    finally:
        async with state_lock:
            state.clients.pop(ws, None)
            if state.setup_owner == ws:
                state.setup_owner = None
                state._waiting_for_setup.set()
                state._setup_done.set()


# -------------------------------------------------------------------
# Initial AI setup
# -------------------------------------------------------------------
async def handle_initial_setup(ws: WebSocket, username: str) -> None:
    async with state_lock:
        if state.setup_owner is None:
            state.setup_owner = ws
            state._waiting_for_setup.clear()
            state._setup_done.clear()
            await ws.send_text(json.dumps({"type": "setup_required"}))
        else:
            await ws.send_text(json.dumps({
                "type": "system",
                "message": "AI is being configured. Please wait..."
            }))

    if state.setup_owner != ws:
        done, pending = await asyncio.wait(
            [asyncio.create_task(state._setup_done.wait()),
             asyncio.create_task(state._waiting_for_setup.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if ws not in state.clients:
            return
        if not state.ai_config["configured"]:
            return await handle_initial_setup(ws, username)
        else:
            return

    try:
        raw = await ws.receive_text()
    except WebSocketDisconnect:
        print(f"{username} disconnected during setup.")
        async with state_lock:
            if state.setup_owner == ws:
                state.setup_owner = None
                state._waiting_for_setup.set()
                state._setup_done.set()
        return

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
            state.setup_owner = None
            state._setup_done.set()

        print("AI configured.")

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

    except Exception as e:
        try:
            await ws.send_text(json.dumps({
                "type": "system",
                "message": f"AI setup failed: {e}"
            }))
        except Exception:
            pass
        async with state_lock:
            if state.setup_owner == ws:
                state.setup_owner = None
                state._waiting_for_setup.set()
                state._setup_done.set()
        try:
            await ws.close()
        except Exception:
            pass


# -------------------------------------------------------------------
# Message processor
# -------------------------------------------------------------------
async def process_message(ws: WebSocket, username: str, msg: str) -> bool:
    # ------------------------------------------------------------------
    # NEW: detect upload_session JSON command (before any other logic)
    # ------------------------------------------------------------------
    try:
        data = json.loads(msg)
    except (json.JSONDecodeError, TypeError):
        pass
    else:
        if isinstance(data, dict) and data.get("type") == "upload_session":
            return await handle_upload_session(ws, username, data)

    # ---- Scenario reset ----
    if msg.strip().lower() == "/scenario":
        async with state_lock:
            state.ai_config["configured"] = False
            state.ai_config["first_message"] = ""
            state.setup_owner = ws
            state._waiting_for_setup.clear()
            state._setup_done.clear()

        await broadcast({
            "type": "scenario_reset",
            "message": "AI scenario reset. Reconfigure the AI."
        })
        await ws.send_text(json.dumps({"type": "setup_required"}))
        return True

    # ---- AI not configured ----
    async with state_lock:
        configured = state.ai_config["configured"]
        owner = state.setup_owner

    if not configured:
        if owner == ws:
            try:
                setup = json.loads(msg)
                required = ["ai_name", "behavior", "first_message", "reasoning_level", "show_reasoning"]
                if not all(k in setup for k in required):
                    raise ValueError("Missing required fields")
                async with state_lock:
                    state.ai_config["name"] = setup["ai_name"]
                    state.ai_config["behavior"] = setup["behavior"]
                    state.ai_config["first_message"] = setup.get("first_message", "")
                    state.ai_config["reasoning_level"] = setup["reasoning_level"]
                    state.ai_config["show_reasoning"] = setup["show_reasoning"]
                    state.ai_config["configured"] = True
                    state.setup_owner = None
                    state._setup_done.set()

                print("AI configured (reconfiguration).")
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
                return True
            except Exception as e:
                try:
                    await ws.send_text(json.dumps({
                        "type": "system",
                        "message": f"Invalid AI setup: {e}"
                    }))
                except Exception:
                    return False
                return True
        else:
            return True

    # ---- Slash commands ----
    if msg.startswith("/"):
        return await handle_command(ws, username, msg)

    # ---- Normal chat ----
    await broadcast({
        "type": "chat",
        "sender": username,
        "message": msg,
        "replay": False,
    })
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
# Command handler
# -------------------------------------------------------------------
async def handle_command(ws: WebSocket, username: str, msg: str) -> bool:
    cmd = msg.strip().lower()

    if cmd == "/sleep":
        async with state_lock:
            state.ai_awake = False
        await broadcast({"type": "system", "message": "AI is now sleeping."})
        return True

    if cmd == "/wake":
        async with state_lock:
            state.ai_awake = True
        await broadcast({"type": "system", "message": "AI is now awake."})
        return True

    if cmd == "/clear":
        async with state_lock:
            state.history.clear()
        await broadcast({"type": "system", "message": "Conversation history cleared."})
        return True

    if cmd == "/new":
        async with state_lock:
            state._archive_current_session()
            state.history.clear()
            state.current_session_id = state.next_session_id()
        await broadcast({"type": "new_session", "message": "New session started."})
        return True

    if cmd.startswith("/load"):
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
            await broadcast({"type": "system", "message": f"Session exported to {path.name}"})
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

    await ws.send_text(json.dumps({"type": "system", "message": f"Unknown command: {msg}"}))
    return True


# -------------------------------------------------------------------
# NEW: Handle upload_session JSON message
# -------------------------------------------------------------------
async def handle_upload_session(ws: WebSocket, username: str, data: dict) -> bool:
    """
    Expected data format:
    {
        "type": "upload_session",
        "history": [
            {"role": "user", "sender": "...", "content": "..."},
            {"role": "assistant", "sender": "...", "content": "..."},
            ...
        ]
    }
    """
    history = data.get("history")
    if not isinstance(history, list):
        await ws.send_text(json.dumps({
            "type": "system",
            "message": "Invalid upload_session: missing or malformed 'history' list."
        }))
        return True

    # Validate entries
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

    # Replace the current session with the uploaded history
    async with state_lock:
        state.load_uploaded_history(history)

    # Inform clients that a new session has been loaded
    await broadcast({
        "type": "new_session",
        "message": "Uploaded session loaded."
    })

    # Replay the whole history to everyone
    async with state_lock:
        for entry in state.history:
            await broadcast({
                "type": "chat",
                "sender": entry["sender"],
                "message": entry["content"],
                "replay": True,
            })

    return True