import asyncio
import json
import websockets
import os

SERVER_URL = "wss://deepseek-terminal-chat-production-78d3.up.railway.app/chat"

WIDTH = 90

PERSONA_FILE = "persona.txt"
AI_PERSONA_FILE = "ai_persona.txt"


def line():
    print("=" * WIDTH)


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


async def receive_messages(websocket, username):

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

            # ==========================================
            # CHAT MESSAGE
            # ==========================================

            elif msg_type == "chat":

                sender = data["sender"]
                message = data["message"]

                line()

                # Your own message → right side
                if sender == username:

                    print(
                        f"You: {message}".rjust(WIDTH)
                    )

                # AI
                elif sender.upper() == sender:

                    print(f"[{sender}]: {message}")

                # Other users
                else:

                    print(f"[{sender}]: {message}")

                line()

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
                    "behavior": (
                        behavior
                        + "\n\n"
                        + "First message: "
                        + first_message
                    ),
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

        except Exception as e:

            print(
                f"\nDisconnected: {e}"
            )

            break


async def send_messages(websocket):

    while True:

        msg = await asyncio.to_thread(
            input,
            ""
        )

        await websocket.send(msg)


async def main():

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

        await asyncio.gather(
            receive_messages(
                websocket,
                username
            ),
            send_messages(websocket)
        )


asyncio.run(main())