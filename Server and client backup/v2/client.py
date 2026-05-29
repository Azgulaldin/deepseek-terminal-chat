import asyncio
import json
import websockets
import os
import getpass
from pathlib import Path
from datetime import datetime

SERVER_URL = "wss://deepseek-terminal-chat-production-78d3.up.railway.app/chat"

WIDTH = 90

PERSONA_FILE = "persona.txt"
AI_PERSONA_FILE = "ai_persona.txt"

BASE_DIR = Path(__file__).resolve().parent
SESSION_ROOT = BASE_DIR / "sessions"

CURRENT_SESSION_FILE = None
CURRENT_PERSONA = ""
PENDING_FIRST_MESSAGE = None


def line():
    print("=" * WIDTH)


# =====================================================
# SESSION LOGGING
# =====================================================

def create_session_file():
    today = datetime.now()
    session_dir = (
        SESSION_ROOT
        / f"{today.year}"
        / f"{today.month}"
        / f"{today.day}"
    )

    session_dir.mkdir(parents=True, exist_ok=True)

    existing_numbers = []
    for path in session_dir.glob("*.txt"):
        if path.stem.isdigit():
            existing_numbers.append(int(path.stem))

    next_number = max(existing_numbers, default=0) + 1
    return session_dir / f"{next_number}.txt"


def start_session_log(username, persona):
    global CURRENT_SESSION_FILE

    CURRENT_SESSION_FILE = create_session_file()

    with open(
        CURRENT_SESSION_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Session started: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Username: {username}\n")
        f.write(f"Persona: {persona}\n")
        f.write("\n")


def log_session(message):
    if CURRENT_SESSION_FILE is None:
        return

    stamp = datetime.now().strftime("%H:%M:%S")

    with open(
        CURRENT_SESSION_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(f"[{stamp}] {message}\n")


def display_own_message(message):
    line()
    print(f"You: {message}".rjust(WIDTH))
    line()
    log_session(f"You: {message}")


# =====================================================
# SAVE PERSONA
# =====================================================

def save_persona(username, persona):

    data = {
        "username": username,
        "persona": persona
    }

    with open(
        PERSONA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# =====================================================
# LOAD PERSONA
# =====================================================

def load_persona():

    with open(
        PERSONA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    username = data["username"]
    persona = data["persona"]

    return username, persona


# =====================================================
# SAVE AI PERSONA
# =====================================================

def save_ai_persona(
    ai_name,
    behavior,
    first_message,
    reasoning_level,
    show_reasoning
):

    data = {
        "ai_name": ai_name,
        "behavior": behavior,
        "first_message": first_message,
        "reasoning_level": reasoning_level,
        "show_reasoning": show_reasoning
    }

    with open(
        AI_PERSONA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# =====================================================
# LOAD AI PERSONA
# =====================================================

def load_ai_persona():

    with open(
        AI_PERSONA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    ai_name = data["ai_name"]
    behavior = data["behavior"]
    first_message = data["first_message"]
    reasoning_level = data.get("reasoning_level", "medium")
    show_reasoning = data.get("show_reasoning", False)

    return (
        ai_name,
        behavior,
        first_message,
        reasoning_level,
        show_reasoning
    )


async def receive_messages(websocket, username, setup_done):

    global PENDING_FIRST_MESSAGE

    while True:

        try:

            raw = await websocket.recv()

            data = json.loads(raw)

            msg_type = data.get("type")

            # ==========================================
            # SYSTEM MESSAGE
            # ==========================================

            if msg_type == "system":

                line()
                print(data["message"])
                line()

                log_session(f"SYSTEM: {data['message']}")

            # ==========================================
            # CHAT MESSAGE
            # ==========================================

            elif msg_type == "chat":

                sender = data["sender"]
                message = data["message"]

                # Skip our own server echo, since we print it locally once
                if sender == username:
                    continue

                # Skip the one first-message copy we already printed locally
                if (
                    PENDING_FIRST_MESSAGE
                    and sender == PENDING_FIRST_MESSAGE[0]
                    and message == PENDING_FIRST_MESSAGE[1]
                ):
                    PENDING_FIRST_MESSAGE = None
                    continue

                line()

                if sender.upper() == sender:

                    print(f"[{sender}]: {message}")

                else:

                    print(f"[{sender}]: {message}")

                line()

                log_session(f"{sender}: {message}")

            # ==========================================
            # REASONING MESSAGE
            # ==========================================

            elif msg_type == "reasoning":

                sender = data["sender"]
                message = data["message"]

                line()

                print(
                    f"[{sender} REASONING]"
                )

                print()

                print(message)

                line()

                log_session(f"[{sender} REASONING]\n{message}")

            # ==========================================
            # SCENARIO RESET
            # ==========================================

            elif msg_type == "scenario_reset":

                setup_done.clear()

                line()
                print(data["message"])
                line()

                log_session(f"SYSTEM: {data['message']}")

                PENDING_FIRST_MESSAGE = None

            # ==========================================
            # NEW SESSION
            # ==========================================

            elif msg_type == "new_session":

                line()
                print(data["message"])
                line()

                log_session(f"SYSTEM: {data['message']}")

                start_session_log(username, CURRENT_PERSONA)

                PENDING_FIRST_MESSAGE = None

            # ==========================================
            # SETUP REQUIRED
            # ==========================================

            elif msg_type == "setup_required":

                line()

                print(
                    "You are the first user."
                )

                print(
                    "Configure the AI."
                )

                line()

                reconfigure_ai = input(
                    "Reconfigure AI persona? (y/n): "
                ).strip().lower()

                line()

                if (
                    reconfigure_ai == "y"
                    or not os.path.exists(
                        AI_PERSONA_FILE
                    )
                ):

                    ai_name = input(
                        "AI Name: "
                    ).strip()

                    behavior = input(
                        "AI Behavior: "
                    ).strip()

                    first_message = input(
                        "First Message: "
                    ).strip()

                    print(
                        "\nReasoning Intensity:"
                    )

                    print("1. easy")
                    print("2. medium")
                    print("3. high")

                    choice = input(
                        "\nChoose option: "
                    ).strip()

                    mapping = {
                        "1": "low",
                        "2": "medium",
                        "3": "high"
                    }

                    reasoning_level = mapping.get(
                        choice,
                        "medium"
                    )

                    show_reasoning = (
                        input(
                            "\nShow reasoning? (y/n): "
                        )
                        .strip()
                        .lower()
                        == "y"
                    )

                    save_ai_persona(
                        ai_name,
                        behavior,
                        first_message,
                        reasoning_level,
                        show_reasoning
                    )

                    line()

                    print(
                        "AI persona saved locally."
                    )

                    line()

                else:

                    (
                        ai_name,
                        behavior,
                        first_message,
                        reasoning_level,
                        show_reasoning
                    ) = load_ai_persona()

                    line()

                    print(
                        "Loaded saved AI persona."
                    )

                    print(
                        f"AI Name: {ai_name}"
                    )

                    line()

                setup_data = {
                    "ai_name": ai_name,
                    "behavior": behavior,
                    "first_message": first_message,
                    "reasoning_level":
                        reasoning_level,
                    "show_reasoning":
                        show_reasoning
                }

                await websocket.send(
                    json.dumps(setup_data)
                )

                line()

                print("AI configured.")

                line()

                log_session("AI CONFIGURED")
                log_session(f"AI Name: {ai_name}")
                log_session(f"Behavior: {behavior}")
                log_session(f"First Message: {first_message}")
                log_session(f"Reasoning Level: {reasoning_level}")
                log_session(f"Show Reasoning: {show_reasoning}")

                # Show the AI's first message locally once setup is complete
                if first_message.strip():

                    PENDING_FIRST_MESSAGE = (ai_name, first_message)

                    line()
                    print(f"[{ai_name}]: {first_message}")
                    line()

                    log_session(f"{ai_name}: {first_message}")

                setup_done.set()

        except Exception as e:

            print(
                f"\nDisconnected: {e}"
            )

            break


async def send_messages(websocket, setup_done):

    while True:

        await setup_done.wait()

        msg = await asyncio.to_thread(
            getpass.getpass,
            "> "
        )

        if not msg:
            continue

        if not setup_done.is_set():
            continue

        display_own_message(msg)

        await websocket.send(msg)


async def main():

    global CURRENT_PERSONA

    line()

    reconfigure = input(
        "Reconfigure persona? (y/n): "
    ).strip().lower()

    line()

    if (
        reconfigure == "y"
        or not os.path.exists(PERSONA_FILE)
    ):

        username = input(
            "Enter username: "
        ).strip()

        persona = input(
            "Enter persona: "
        ).strip()

        save_persona(
            username,
            persona
        )

        line()

        print(
            "Persona saved locally."
        )

        line()

    else:

        username, persona = load_persona()

        line()

        print(
            "Loaded saved persona."
        )

        print(
            f"Username: {username}"
        )

        line()

    CURRENT_PERSONA = persona
    start_session_log(username, persona)

    setup_done = asyncio.Event()

    async with websockets.connect(
        SERVER_URL
    ) as websocket:

        # send username first
        await websocket.send(username)

        line()

        print(
            f"Connected as {username}"
        )

        line()

        log_session(f"CONNECTED AS {username}")

        await asyncio.gather(
            receive_messages(
                websocket,
                username,
                setup_done
            ),
            send_messages(
                websocket,
                setup_done
            )
        )


asyncio.run(main())