# client.py – DeepSeek roleplay terminal client
# ------------------------------------------------------------------
# Connects to the WebSocket server, handles persona setup,
# AI configuration, real‑time chat, and local session logging.
#
# Rewrite focuses on:
#   - Removing masking input (used getpass incorrectly)
#   - Eliminating duplicate‑message workarounds
#   - Clearer async flow
#   - Robust error handling
# ------------------------------------------------------------------

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

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
            # Extract the number between underscore and .txt
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

        # ---- System message ----
        if msg_type == "system":
            display_system(data["message"])
            logger.log_system(data["message"])

        # ---- New session ----
        elif msg_type == "new_session":
            display_system(data["message"])
            logger.log_system(data["message"])
            logger.start()   # start a fresh log file

        # ---- Scenario reset ----
        elif msg_type == "scenario_reset":
            display_system(data["message"])
            logger.log_system(data["message"])
            # Prepare to reconfigure AI; clear the setup event so
            # the sender loop waits until setup is done again.
            setup_done.clear()

        # ---- Setup required (only the first user gets this) ----
        elif msg_type == "setup_required":
            # This client has been chosen to configure the AI.
            await handle_ai_setup(websocket, logger)
            setup_done.set()   # now the send loop can start

        # ---- Chat message ----
        elif msg_type == "chat":
            sender = data["sender"]
            message = data["message"]
            # Skip our own messages – we don't want a local echo
            if sender == username:
                continue
            display_chat(sender, message)
            logger.log_chat(sender, message)

        # ---- Reasoning ----
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

    # Offer to reload saved AI persona
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

    # Send configuration to server
    setup_data = {
        "ai_name": ai_name,
        "behavior": behavior,
        "first_message": first_message,
        "reasoning_level": reasoning_level,
        "show_reasoning": show_reasoning,
    }
    await websocket.send(json.dumps(setup_data))

    # Log setup details
    logger.log_system("AI CONFIGURED")
    logger.write(f"AI Name: {ai_name}")
    logger.write(f"Behavior: {behavior}")
    logger.write(f"First Message: {first_message}")
    logger.write(f"Reasoning Level: {reasoning_level}")
    logger.write(f"Show Reasoning: {show_reasoning}")

    print("\nAI configured.\n")


# -------------------------------------------------------------------
# WebSocket send loop
# -------------------------------------------------------------------
async def sender(websocket: websockets.WebSocketClientProtocol,
                 setup_done: asyncio.Event) -> None:
    """
    Wait until AI setup is complete (if needed), then read user input
    line by line and send it to the server.
    """
    await setup_done.wait()

    # Use a simple input() prompt – no password masking.
    # We use asyncio.to_thread to avoid blocking the event loop.
    loop = asyncio.get_running_loop()
    while True:
        # Wait for the event to remain set (it may be cleared again if /scenario resets the AI)
        if not setup_done.is_set():
            await setup_done.wait()

        # Read from stdin in a thread
        msg = await loop.run_in_executor(None, input, "> ")
        msg = msg.strip()
        if not msg:
            continue

        # Send to server
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

    # Send username immediately (per protocol)
    await websocket.send(username)
    logger.log_system(f"CONNECTED AS {username}")
    print(f"Connected as {username}\n")

    # This event is set when AI is configured (or if no setup is needed).
    # Initially we must check if the server requires setup; the receiver
    # will clear it if a setup_required or scenario_reset arrives.
    setup_done = asyncio.Event()
    # If the server's AI is already configured, the receiver won't clear it.
    # We set it preemptively; the receiver will clear if needed.
    setup_done.set()

    # Run receiver and sender concurrently
    await asyncio.gather(
        receiver(websocket, username, setup_done, logger),
        sender(websocket, setup_done),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")