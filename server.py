# server.py – DeepSeek roleplay multi‑user WebSocket server
# ------------------------------------------------------------------
# This rewrite addresses:
#   • races when multiple users connect during AI setup
#   • deadlocks after the setup owner disconnects
#   • insecure file import (path traversal)
#   • missing error handling and input validation
#   • unclear state transitions (/scenario, /load, etc.)
# ------------------------------------------------------------------

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import OpenAI
import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

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
# OpenAI client (shared, not used concurrently – single server thread)
# -------------------------------------------------------------------
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# -------------------------------------------------------------------
# Global application state (protected by a single asyncio lock)
# -------------------------------------------------------------------
class AppState:
    """All shared mutable state lives here, accessed only under lock."""

    def __init__(self) -> None:
        # connection manager
        self.clients: Dict[WebSocket, str] = {}      # ws -> username

        # AI configuration
        self.ai_config: Dict[str, Any] = {
            "name": "DEEPSEEK",
            "behavior": "Helpful and intelligent.",
            "first_message": "",
            "reasoning_level": "medium",
            "show_reasoning": False,
            "configured": False,
        }

        # conversation history (list of {role, sender, content})
        self.history: list[Dict[str, Any]] = []

        # wake/sleep toggle
        self.ai_awake: bool = True

        # session management
        self.current_session_id: Optional[str] = None
        self.session_archive: Dict[str, Dict[str, Any]] = {}
        self._day_counters: Dict[str, int] = {}

        # setup coordination
        self.setup_owner: Optional[WebSocket] = None
        self._waiting_for_setup: asyncio.Event = asyncio.Event()
        self._setup_done: asyncio.Event = asyncio.Event()
        self._setup_done.set()  # initially not waiting

    # ---- helper methods (assume lock is held) ----

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
        """Return the current session id, creating one if necessary."""
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
        # SECURITY: only allow files directly inside EXPORT_DIR, no traversal
        safe_path = (EXPORT_DIR / os.path.basename(filename)).resolve()
        if not safe_path.is_relative_to(EXPORT_DIR):
            raise ValueError("Invalid import path")
        data = json.loads(safe_path.read_text(encoding="utf-8"))
        self.history = data.get("history", [])
        self.ai_config.update(data.get("ai_config", {}))


# -------------------------------------------------------------------
# Application and state
# -------------------------------------------------------------------
app = FastAPI()
state = AppState()
state_lock = asyncio.Lock()


# -------------------------------------------------------------------
# Helper: broadcast message to all connected clients
# -------------------------------------------------------------------
async def broadcast(data: dict) -> None:
    """Send JSON data to every alive WebSocket, removing dead ones."""
    dead = []
    # no lock needed – only modifying clients during iteration is safe with list copy
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
# Helper: replay entire history to a single client (already have lock)
# -------------------------------------------------------------------
async def replay_history(ws: WebSocket) -> None:
    """Replay conversation history to a newly connected client."""
    for entry in state.history:
        sender = entry.get("sender", "UNKNOWN")
        if not sender:
            # legacy fallback (should not happen)
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
# Root health endpoint
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

    # 1. Receive username
    try:
        username = await ws.receive_text()
    except WebSocketDisconnect:
        return

    # Register the client
    async with state_lock:
        state.clients[ws] = username
        # Make sure a session exists
        state.ensure_active_session()

    print(f"{username} connected.")

    # 2. AI configuration flow (if needed)
    if not state.ai_config["configured"]:
        await handle_initial_setup(ws, username)
        # If the user disconnected during setup, we're done
        if ws not in state.clients:
            return

    # 3. Send welcome message + replay history
    try:
        await ws.send_text(json.dumps({
            "type": "system",
            "message": f"Connected to {state.ai_config['name']}"
        }))
        async with state_lock:
            await replay_history(ws)
    except Exception:
        return

    # 4. Main message loop
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
                # The client has been disconnected inside process_message
                break
    except Exception as e:
        print(f"{username} error: {e}")
    finally:
        # Cleanup
        async with state_lock:
            state.clients.pop(ws, None)
            if state.setup_owner == ws:
                # The setup owner left – notify any waiters
                state.setup_owner = None
                state._waiting_for_setup.set()   # wake up waiting clients
                state._setup_done.set()          # release any send loop waits


# -------------------------------------------------------------------
# Setup coordination
# -------------------------------------------------------------------
async def handle_initial_setup(ws: WebSocket, username: str) -> None:
    """
    Manage the case where the AI is not yet configured.
    Only one user can be the setup owner; others wait.
    """
    async with state_lock:
        if state.setup_owner is None:
            # Become the setup owner
            state.setup_owner = ws
            state._waiting_for_setup.clear()
            state._setup_done.clear()
            # send setup prompt
            await ws.send_text(json.dumps({"type": "setup_required"}))
        else:
            # Tell this client to wait
            await ws.send_text(json.dumps({
                "type": "system",
                "message": "AI is being configured. Please wait..."
            }))

    # If we are not the owner, wait until setup completes or owner disconnects
    if state.setup_owner != ws:
        # Wait for either _setup_done or _waiting_for_setup
        done, pending = await asyncio.wait(
            [asyncio.create_task(state._setup_done.wait()),
             asyncio.create_task(state._waiting_for_setup.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )
        # Cancel the other task
        for task in pending:
            task.cancel()
        # Check if the connection is still alive
        if ws not in state.clients:
            return
        # If we were woken by _waiting_for_setup (owner left), try to become owner
        if not state.ai_config["configured"]:
            # The AI still isn't configured – retry the flow recursively
            return await handle_initial_setup(ws, username)
        else:
            # AI was configured while we waited
            return

    # ---- We are the setup owner ----
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

    # Parse the setup JSON
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

            # Clear setup owner before broadcasting
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
        # Notify the setup owner of failure
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
        # Close this connection so the user can retry
        try:
            await ws.close()
        except Exception:
            pass


# -------------------------------------------------------------------
# Message processing (runs inside the main loop)
# -------------------------------------------------------------------
async def process_message(ws: WebSocket, username: str, msg: str) -> bool:
    """
    Returns False if the connection should be terminated, True otherwise.
    """
    # ---- Scenario reset (reconfigure AI) ----
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
        # The next message from this user will be processed by handle_initial_setup
        # because the main loop will see ai_config["configured"] == False
        # and call handle_initial_setup again. We signal that by continuing.
        return True

    # ---- If AI is not configured, only the owner can talk ----
    async with state_lock:
        configured = state.ai_config["configured"]
        owner = state.setup_owner

    if not configured:
        if owner == ws:
            # The setup owner is sending the configuration JSON now.
            # We already handled the "setup_required" prompt; process their JSON.
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
            # Not the owner – they shouldn't be sending messages
            return True

    # ---- Commands (available only when AI is configured) ----
    if msg.startswith("/"):
        return await handle_command(ws, username, msg)

    # ---- Normal chat message ----
    # 1. Broadcast to everyone
    await broadcast({
        "type": "chat",
        "sender": username,
        "message": msg,
        "replay": False,
    })

    async with state_lock:
        state.append_user_message(username, msg)

    # 2. AI response (if awake)
    if not state.ai_awake:
        return True

    try:
        async with state_lock:
            system_msg = f"You are {state.ai_config['name']}. {state.ai_config['behavior']}"
            # last 20 messages for context
            recent = state.history[-20:]
            messages = [{"role": "system", "content": system_msg}]
            messages += [{"role": h["role"], "content": h["content"]} for h in recent]
            reasoning_level = state.ai_config["reasoning_level"]
            show_reason = state.ai_config["show_reasoning"]

        # Call DeepSeek (blocking call, but called in async context – okay for this scale)
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

    # Broadcast reasoning (if enabled)
    if reasoning and state.ai_config["show_reasoning"]:
        await broadcast({
            "type": "reasoning",
            "sender": state.ai_config["name"],
            "message": reasoning,
        })

    # Broadcast AI reply
    await broadcast({
        "type": "chat",
        "sender": state.ai_config["name"],
        "message": reply,
        "replay": False,
    })

    return True


# -------------------------------------------------------------------
# Command handler (called only when AI is configured)
# -------------------------------------------------------------------
async def handle_command(ws: WebSocket, username: str, msg: str) -> bool:
    cmd = msg.strip().lower()

    # /sleep
    if cmd == "/sleep":
        async with state_lock:
            state.ai_awake = False
        await broadcast({"type": "system", "message": "AI is now sleeping."})
        return True

    # /wake
    if cmd == "/wake":
        async with state_lock:
            state.ai_awake = True
        await broadcast({"type": "system", "message": "AI is now awake."})
        return True

    # /clear
    if cmd == "/clear":
        async with state_lock:
            state.history.clear()
        await broadcast({"type": "system", "message": "Conversation history cleared."})
        return True

    # /new – start a brand new session
    if cmd == "/new":
        async with state_lock:
            state._archive_current_session()
            state.history.clear()
            state.current_session_id = state.next_session_id()
        await broadcast({"type": "new_session", "message": "New session started."})
        return True

    # /load <session_id>
    if cmd.startswith("/load"):
        parts = msg.split(maxsplit=1)
        if len(parts) < 2:
            await ws.send_text(json.dumps({"type": "system", "message": "Usage: /load yyyy/m/d/n"}))
            return True
        session_id = parts[1].strip()
        # allow .txt extension (from client log files) but strip it
        if session_id.endswith(".txt"):
            session_id = session_id[:-4]
        try:
            async with state_lock:
                state.restore_session(session_id)
            await broadcast({"type": "system", "message": f"Loaded session {session_id}."})
            # replay history to all
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

    # /export
    if cmd == "/export":
        try:
            async with state_lock:
                path = state.export_session()
            await broadcast({"type": "system", "message": f"Session exported to {path.name}"})
        except Exception as e:
            await ws.send_text(json.dumps({"type": "system", "message": f"Export failed: {e}"}))
        return True

    # /import <filename> (must be inside EXPORT_DIR)
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

    # Unknown command
    await ws.send_text(json.dumps({"type": "system", "message": f"Unknown command: {msg}"}))
    return True