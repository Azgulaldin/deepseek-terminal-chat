import asyncio
import json
import websockets

SERVER_URL = "wss://deepseek-terminal-chat-production-78d3.up.railway.app/chat"

WIDTH = 90


def line():
    print("=" * WIDTH)


# =====================================================
# RECEIVE LOOP
# =====================================================

async def receive_messages(websocket, username):

    try:
        while True:

            raw = await websocket.recv()
            data = json.loads(raw)

            msg_type = data.get("type")

            if msg_type == "system":
                line()
                print(data.get("message", ""))
                line()

            elif msg_type == "chat":

                sender = data.get("sender", "")
                message = data.get("message", "")
                persona = data.get("persona", "")

                line()

                if sender == username:
                    print(f"You: {message}".rjust(WIDTH))

                elif sender and sender.upper() == sender:
                    print(f"[{sender}]: {message}")

                else:
                    if persona:
                        print(f"[{sender} | {persona}]: {message}")
                    else:
                        print(f"[{sender}]: {message}")

                line()

            elif msg_type == "reasoning":

                sender = data.get("sender", "")
                message = data.get("message", "")

                line()
                print(f"[{sender} REASONING]\n")
                print(message)
                line()

            elif msg_type == "setup_required":

                line()
                print("AI setup required")
                line()

                ai_name = input("AI Name: ").strip()
                behavior = input("AI Behavior: ").strip()

                print("\nReasoning Intensity:")
                print("1. easy")
                print("2. medium")
                print("3. high")

                choice = input("\nChoose option: ").strip()

                mapping = {
                    "1": "low",
                    "2": "medium",
                    "3": "high"
                }

                reasoning_level = mapping.get(choice, "medium")

                show_reasoning = input(
                    "\nShow reasoning? (y/n): "
                ).strip().lower() == "y"

                first_message = input(
                    "\nAI First Message: "
                ).strip()

                setup_data = {
                    "ai_name": ai_name,
                    "behavior": behavior,
                    "reasoning_level": reasoning_level,
                    "show_reasoning": show_reasoning,
                    "first_message": first_message
                }

                await websocket.send(json.dumps(setup_data))

                line()
                print("AI configured.")
                line()

            elif msg_type == "request_persona":

                line()
                print("Set your persona:")
                line()

                persona = input("> ").strip()

                await websocket.send(persona)

                line()
                print("Persona set.")
                line()

    except websockets.ConnectionClosed:
        raise


# =====================================================
# SEND LOOP (SAFE)
# =====================================================

async def send_messages(websocket):

    try:
        while True:
            msg = await asyncio.to_thread(input, "")

            await websocket.send(msg)

    except websockets.ConnectionClosed:
        raise


# =====================================================
# CONNECT WITH AUTO RECONNECT
# =====================================================

async def connect(username):

    while True:

        try:
            async with websockets.connect(
                SERVER_URL,
                ping_interval=25,
                ping_timeout=120,
                close_timeout=120,
                max_queue=None
            ) as websocket:

                await websocket.send(username)

                line()
                print(f"Connected as {username}")
                line()

                await asyncio.gather(
                    receive_messages(websocket, username),
                    send_messages(websocket)
                )

        except Exception as e:

            print("\nDisconnected. Reconnecting...")
            print(f"Reason: {e}")

            await asyncio.sleep(3)


# =====================================================
# MAIN
# =====================================================

async def main():

    username = input("Enter username: ").strip()
    await connect(username)


asyncio.run(main())