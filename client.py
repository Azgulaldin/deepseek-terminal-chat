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

                sender = data.get("sender", "")
                message = data.get("message", "")
                persona = data.get("persona", "")

                line()

                # your message
                if sender == username:
                    print(f"You: {message}".rjust(WIDTH))

                # AI (uppercase convention)
                elif sender.upper() == sender:
                    print(f"[{sender}]: {message}")

                # other users
                else:
                    if persona:
                        print(f"[{sender} | {persona}]: {message}")
                    else:
                        print(f"[{sender}]: {message}")

                line()

            # ==========================================
            # REASONING MESSAGE
            # ==========================================

            elif msg_type == "reasoning":

                sender = data.get("sender", "")
                message = data.get("message", "")

                line()
                print(f"[{sender} REASONING]\n")
                print(message)
                line()

            # ==========================================
            # SETUP REQUIRED (AI CONFIG)
            # ==========================================

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
                    "\nAI First Message (scenario intro): "
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

            # ==========================================
            # REQUEST PERSONA
            # ==========================================

            elif msg_type == "request_persona":

                line()
                print("Define your persona (how you appear in chat)")
                line()

                persona = input("> ").strip()

                await websocket.send(persona)

                line()
                print("Persona set.")
                line()

        except Exception as e:
            print(f"\nDisconnected: {e}")
            break


# =====================================================
# SEND LOOP
# =====================================================

async def send_messages(websocket):

    while True:
        msg = await asyncio.to_thread(input, "")
        await websocket.send(msg)


# =====================================================
# MAIN
# =====================================================

async def main():

    username = input("Enter username: ").strip()

    async with websockets.connect(
        SERVER_URL,
        ping_interval=20,
        ping_timeout=20
    ) as websocket:

        # send username first
        await websocket.send(username)

        line()
        print(f"Connected as {username}")
        line()

        await asyncio.gather(
            receive_messages(websocket, username),
            send_messages(websocket)
        )


asyncio.run(main())