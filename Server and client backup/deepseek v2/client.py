# client.py – DeepSeek roleplay terminal client
# ------------------------------------------------------------------
# Connects to the WebSocket server, handles persona setup,
# AI configuration, real‑time chat, local session logging,
# and now: /upload <filename> to restore a session from a local log.
# ------------------------------------------------------------------

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import websockets

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
SERVER_URL = "wss://deepseek-terminal-chat-production-78d3.up.railway.app/chat"

# Local storage directories
BASE_DIR = Path(__file__).resolve().parent
SESSION_ROOT = BASE_DIR / "sessions"
PERSONA_FILE = BASE_DIR / "persona.txt"
AI_PERSONA_FILE = BASE_DIR / "ai_persona.txt"

# Terminal display
WIDTH = 90


def line() -> None:
    """Print a horizontal divider."""
    print("=" * WIDTH)


# -------------------------------------------------------------------
# Persona persistence
# -------------------------------------------------------------------
def save_persona(username: str, persona: str) -> None:
    data = {"username": username, "persona": persona}
    PERSONA_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def load_persona() -> tuple[str, str]:
    data = json.loads(PERSONA_FILE.read_text(encoding="utf-8"))
    return data["username"], data["persona"]


def save_ai_persona(ai_name: str, behavior: str, first_message: str,
                    reasoning_level: str, show_reasoning: bool) -> None:
    data = {
        "ai_name": ai_name,
        "behavior": behavior,
        "first_message": first_message,
        "reasoning_level": reasoning_level,
        "show_reasoning": show_reasoning,
    }
    AI_PERSONA_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def load_ai_persona() -> tuple[str, str, str, str, bool]:
    data = json.loads(AI_PERSONA_FILE.read_text(encoding="utf-8"))
    return (
        data["ai_name"],
        data["behavior"],
        data["first_message"],
        data.get("reasoning_level", "medium"),
        data.get("show_reasoning", False),
    )


# -------------------------------------------------------------------
# Session logging (flat structure: sessions/YYYY-MM-DD_N.txt)
# -------------------------------------------------------------------
class SessionLogger:
    """Writes a transcript of the session to a flat file inside sessions/."""

    def __init__(self, username: str, persona: str) -> None:
        self.username = username
        self.persona = persona
        self.file_path: Optional[Path] = None

    def start(self) -> Path:
        """Create and open a new session log file in sessions/."""
        session_dir = SESSION_ROOT   # BASE_DIR / "sessions"
        session_dir.mkdir(exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")   # "2026-05-29"

        # Find next session number for today
        existing_numbers = []
        for p in session_dir.glob(f"{today}_*.txt"):
            stem = p.stem               # e.g. "2026-05-29_3"
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
        """Append a line to the session log, timestamped."""
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
# Terminal UI helpers
# -------------------------------------------------------------------
def display_chat(sender: str, message: str) -> None:
    """Print a chat message from someone (AI or another user)."""
    line()
    if sender.upper() == sender:   # AI name is usually all caps
        print(f"[{sender}]: {message}")
    else:
        print(f"[{sender}]: {message}")
    line()


def display_reasoning(sender: str, reasoning: str) -> None:
    """Print the AI's internal reasoning (if shown)."""
    line()
    print(f"[{sender} REASONING]")
    print()
    print(reasoning)
    line()


def display_system(message: str) -> None:
    """Print a system notification."""
    line()
    print(message)
    line()


# -------------------------------------------------------------------
# WebSocket receive loop
# -------------------------------------------------------------------
async def receiver(websocket: websockets.WebSocketClientProtocol,
                   username: str,
                   setup_done: asyncio.Event,
                   logger: SessionLogger) -> None:
    """
    Listen for messages from the server and display them.
    Handles: chat, system, reasoning, new_session, scenario_reset,
             setup_required, and silently ignores own messages.
    """
    while True:
        try:
            raw = await websocket.recv()
        except websockets.ConnectionClosed as e:
            print(f"\nDisconnected: {e}")
            break
        except Exception as e:
            print(f"\nUnexpected receive error: {e}")
            break

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        msg_type = data.get("type")

        if msg_type == "system":
            display_system(data["message"])
            logger.log_system(data["message"])

        elif msg_type == "new_session":
            display_system(data["message"])
            logger.log_system(data["message"])
            logger.start()   # start a fresh log file

        elif msg_type == "scenario_reset":
            display_system(data["message"])
            logger.log_system(data["message"])
            setup_done.clear()

        elif msg_type == "setup_required":
            await handle_ai_setup(websocket, logger)
            setup_done.set()

        elif msg_type == "chat":
            sender = data["sender"]
            message = data["message"]
            if sender == username:
                continue
            display_chat(sender, message)
            logger.log_chat(sender, message)

        elif msg_type == "reasoning":
            sender = data["sender"]
            reasoning = data["message"]
            display_reasoning(sender, reasoning)
            logger.log_reasoning(sender, reasoning)

        # (Unknown types are silently ignored)


# -------------------------------------------------------------------
# AI setup flow (when server sends "setup_required")
# -------------------------------------------------------------------
async def handle_ai_setup(websocket: websockets.WebSocketClientProtocol,
                          logger: SessionLogger) -> None:
    """
    Prompt the local user for AI persona settings, then send them to the server.
    This runs inside the receiver loop but blocks until configuration is sent.
    """
    print("\nYou are the first user. Configure the AI.\n")

    reconfigure = input("Reconfigure AI persona? (y/n): ").strip().lower()
    if reconfigure == "y" or not AI_PERSONA_FILE.exists():
        ai_name = input("AI Name: ").strip()
        behavior = input("AI Behavior: ").strip()
        first_message = input("First Message: ").strip()
        print("\nReasoning Intensity:")
        print("1. easy")
        print("2. medium")
        print("3. high")
        choice = input("\nChoose option: ").strip()
        mapping = {"1": "low", "2": "medium", "3": "high"}
        reasoning_level = mapping.get(choice, "medium")
        show_reasoning = input("\nShow reasoning? (y/n): ").strip().lower() == "y"
        save_ai_persona(ai_name, behavior, first_message, reasoning_level, show_reasoning)
        print("\nAI persona saved locally.")
    else:
        ai_name, behavior, first_message, reasoning_level, show_reasoning = load_ai_persona()
        print(f"\nLoaded saved AI persona: {ai_name}")

    setup_data = {
        "ai_name": ai_name,
        "behavior": behavior,
        "first_message": first_message,
        "reasoning_level": reasoning_level,
        "show_reasoning": show_reasoning,
    }
    await websocket.send(json.dumps(setup_data))

    logger.log_system("AI CONFIGURED")
    logger.write(f"AI Name: {ai_name}")
    logger.write(f"Behavior: {behavior}")
    logger.write(f"First Message: {first_message}")
    logger.write(f"Reasoning Level: {reasoning_level}")
    logger.write(f"Show Reasoning: {show_reasoning}")

    print("\nAI configured.\n")


# -------------------------------------------------------------------
# NEW: Upload a local session log to the server
# -------------------------------------------------------------------
def parse_log_file(filepath: Path, ai_name: str, username: str) -> List[Dict[str, str]]:
    """
    Parse a client log file (sessions/YYYY-MM-DD_N.txt) and return
    a list of history entries suitable for the server.
    """
    history: List[Dict[str, str]] = []
    skip_next_reasoning = False

    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if skip_next_reasoning:
                skip_next_reasoning = False
                continue

            # Only process lines that start with a timestamp bracket
            if not line.startswith("[") or "] " not in line:
                continue

            # Split off the timestamp
            idx = line.index("] ")
            after_ts = line[idx + 2:]  # the part after "[HH:MM:SS] "

            # Handle reasoning header: "[AI_NAME REASONING]"
            if after_ts.startswith("[") and after_ts.endswith("]") and " REASONING]" in after_ts:
                skip_next_reasoning = True
                continue

            # System message
            if after_ts.startswith("SYSTEM:"):
                continue

            # Normal chat line: "SENDER: message"
            if ": " not in after_ts:
                continue

            sender, message = after_ts.split(": ", 1)

            # Map "You" to the current username, other senders to user/AI
            if sender == "You":
                history.append({"role": "user", "sender": username, "content": message})
            elif sender.lower() == ai_name.lower():
                history.append({"role": "assistant", "sender": ai_name, "content": message})
            # All other senders are ignored (could be other human players – we skip them)

    return history


async def handle_upload_command(websocket: websockets.WebSocketClientProtocol,
                                msg: str,
                                username: str,
                                logger: SessionLogger) -> None:
    """Process the /upload <filename> command."""
    parts = msg.split(maxsplit=1)
    if len(parts) < 2:
        print("Usage: /upload <filename>")
        print("Example: /upload 2026-05-29_1.txt")
        return

    filename = parts[1].strip()
    # Allow full path or just the name inside sessions/
    filepath = Path(filename)
    if not filepath.is_absolute():
        filepath = SESSION_ROOT / filepath

    if not filepath.exists():
        print(f"File not found: {filepath}")
        return

    # Determine AI name
    ai_name = ""
    if AI_PERSONA_FILE.exists():
        _, _, _, _, _ = load_ai_persona()
        # We only need the name, but load_ai_persona returns more.
        # Let's reload just the name to avoid unpacking mess.
        data = json.loads(AI_PERSONA_FILE.read_text(encoding="utf-8"))
        saved_ai = data.get("ai_name", "")
    else:
        saved_ai = ""

    if saved_ai:
        prompt = f"AI name (press Enter to use '{saved_ai}'): "
    else:
        prompt = "Enter the AI name used in this session: "

    ai_name = await asyncio.get_running_loop().run_in_executor(None, input, prompt)
    ai_name = ai_name.strip() or saved_ai
    if not ai_name:
        print("AI name is required. Aborting upload.")
        return

    # Parse the log
    try:
        history = parse_log_file(filepath, ai_name, username)
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    if not history:
        print("No valid chat messages found in the file.")
        return

    # Send to server
    upload_msg = {
        "type": "upload_session",
        "history": history,
    }
    try:
        await websocket.send(json.dumps(upload_msg))
        print(f"Uploaded {len(history)} messages. The server will now replay them.")
        logger.log_system(f"Uploaded session from {filepath.name}")
    except Exception as e:
        print(f"Failed to send upload: {e}")


# -------------------------------------------------------------------
# WebSocket send loop
# -------------------------------------------------------------------
async def sender(websocket: websockets.WebSocketClientProtocol,
                 setup_done: asyncio.Event,
                 username: str,
                 logger: SessionLogger) -> None:
    """
    Wait until AI setup is complete, then read user input and send it to the server.
    Local commands (like /upload) are handled here.
    """
    await setup_done.wait()
    loop = asyncio.get_running_loop()

    while True:
        if not setup_done.is_set():
            await setup_done.wait()

        msg = await loop.run_in_executor(None, input, "> ")
        msg = msg.strip()
        if not msg:
            continue

        # Local command: /upload
        if msg.startswith("/upload"):
            await handle_upload_command(websocket, msg, username, logger)
            continue

        # All other messages go to the server as-is
        try:
            await websocket.send(msg)
        except websockets.ConnectionClosed:
            break


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------
async def main() -> None:
    line()
    # ---- User persona ----
    if PERSONA_FILE.exists():
        username, persona = load_persona()
        print(f"Loaded saved persona: {username}")
    else:
        username = input("Enter username: ").strip()
        persona = input("Enter persona: ").strip()
        save_persona(username, persona)
        print("Persona saved locally.")
    line()

    logger = SessionLogger(username, persona)
    logger.start()

    # ---- Connect to server ----
    try:
        websocket = await websockets.connect(SERVER_URL)
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    await websocket.send(username)
    logger.log_system(f"CONNECTED AS {username}")
    print(f"Connected as {username}\n")

    setup_done = asyncio.Event()
    setup_done.set()   # assume AI already configured (will be cleared if needed)

    await asyncio.gather(
        receiver(websocket, username, setup_done, logger),
        sender(websocket, setup_done, username, logger),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")