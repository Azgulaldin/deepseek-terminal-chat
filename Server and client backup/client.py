import asyncio
import json
import websockets

SERVER_URL = "wss://deepseek-terminal-chat-production-78d3.up.railway.app/chat"

WIDTH = 90


def line():
    print("=" * WIDTH)


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

                ai_name = input(
                    "AI Name: "
                ).strip()

                behavior = input(
                    "AI Behavior: "
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

                setup_data = {
                    "ai_name": ai_name,
                    "behavior": behavior,
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

    username = input(
        "Enter username: "
    ).strip()

    async with websockets.connect(
    SERVER_URL,
    ping_interval=20,
    ping_timeout=20
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