from fastapi import FastAPI, WebSocket
from openai import OpenAI
import json
import os
from datetime import datetime
from starlette.websockets import WebSocketDisconnect

app = FastAPI()

clients = {}
history = []

ai_awake = True
setup_owner = None

ai_config = {
    "name": "DEEPSEEK",
    "behavior": "Helpful and intelligent.",
    "first_message": "",
    "reasoning_level": "medium",
    "show_reasoning": False,
    "configured": False
}


# =====================================================
# API KEY
# =====================================================

API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:

    raise ValueError(
        "DEEPSEEK_API_KEY environment variable not found."
    )

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com/v1"
)


# =====================================================
# BROADCAST
# =====================================================

async def broadcast(data):

    dead = []

    for ws in clients:

        try:
            await ws.send_text(json.dumps(data))

        except:
            dead.append(ws)

    for ws in dead:

        if ws in clients:
            del clients[ws]


# =====================================================
# HISTORY REPLAY
# =====================================================

def normalize_history_entry(entry):

    sender = entry.get("sender")

    if sender:
        return entry

    role = entry.get("role", "")
    content = entry.get("content", "")

    if role == "assistant":
        sender = ai_config["name"]
    else:
        if ": " in content:
            sender, content = content.split(": ", 1)
        else:
            sender = "UNKNOWN"

    return {
        "role": role,
        "sender": sender,
        "content": content
    }


async def replay_history(ws):

    for entry in history:

        entry = normalize_history_entry(entry)

        try:
            await ws.send_text(json.dumps({
                "type": "chat",
                "sender": entry.get("sender", "UNKNOWN"),
                "message": entry.get("content", "")
            }))

        except:
            break


# =====================================================
# SESSION CONTROL
# =====================================================

def apply_ai_setup(setup):

    ai_config["name"] = setup["ai_name"]
    ai_config["behavior"] = setup["behavior"]
    ai_config["first_message"] = setup.get("first_message", "")
    ai_config["reasoning_level"] = setup["reasoning_level"]
    ai_config["show_reasoning"] = setup["show_reasoning"]
    ai_config["configured"] = True


def append_user_message(username, msg):

    history.append({
        "role": "user",
        "sender": username,
        "content": msg
    })


def append_assistant_message(sender, reply):

    history.append({
        "role": "assistant",
        "sender": sender,
        "content": reply
    })


# =====================================================
# EXPORT SESSION
# =====================================================

def export_session():

    filename = (
        "session_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )

    data = {
        "history": history,
        "ai_config": ai_config
    }

    with open(filename, "w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

    return filename


# =====================================================
# IMPORT SESSION
# =====================================================

def import_session(filename):

    global history
    global ai_config

    with open(filename, "r", encoding="utf-8") as f:

        data = json.load(f)

    history = data.get("history", [])

    loaded_config = data.get(
        "ai_config",
        {}
    )

    ai_config.update(loaded_config)


# =====================================================
# ROOT TEST
# =====================================================

@app.get("/")
def home():

    return {
        "status": "server running"
    }


# =====================================================
# WEBSOCKET CHAT
# =====================================================

@app.websocket("/chat")
async def chat(ws: WebSocket):

    global ai_awake
    global setup_owner

    await ws.accept()

    username = await ws.receive_text()

    clients[ws] = username

    print(f"{username} connected.")

    # =================================================
    # FIRST USER CONFIGURES AI
    # =================================================

    if not ai_config["configured"]:

        setup_owner = ws

        await ws.send_text(json.dumps({
            "type": "setup_required"
        }))

        try:

            raw = await ws.receive_text()

        except WebSocketDisconnect:

            print(f"{username} disconnected during setup.")
            if setup_owner == ws:
                setup_owner = None
            return

        try:
            setup = json.loads(raw)

            apply_ai_setup(setup)

            setup_owner = None

            print("AI configured.")

            if ai_config["first_message"].strip():

                first_message = ai_config["first_message"]

                append_assistant_message(
                    ai_config["name"],
                    first_message
                )

                await broadcast({
                    "type": "chat",
                    "sender": ai_config["name"],
                    "message": first_message
                })

        except Exception as e:

            await ws.send_text(json.dumps({
                "type": "system",
                "message": f"AI setup failed: {e}"
            }))

            if setup_owner == ws:
                setup_owner = None

            return

    else:

        await ws.send_text(json.dumps({
            "type": "system",
            "message":
                f"Connected to {ai_config['name']}"
        }))

        await replay_history(ws)

    # =================================================
    # MAIN LOOP
    # =================================================

    try:

        while True:

            try:

                msg = await ws.receive_text()

            except WebSocketDisconnect:

                print(f"{username} disconnected normally.")
                break

            print(f"{username}: {msg}")

            # =========================================
            # IF AI SETUP IS NOT COMPLETE
            # =========================================

            if not ai_config["configured"]:

                if ws != setup_owner:

                    await ws.send_text(json.dumps({
                        "type": "system",
                        "message":
                            "AI is being reconfigured. Please wait."
                    }))

                    continue

                try:

                    setup = json.loads(msg)

                    required_keys = (
                        "ai_name",
                        "behavior",
                        "first_message",
                        "reasoning_level",
                        "show_reasoning"
                    )

                    if not all(k in setup for k in required_keys):
                        raise ValueError("invalid setup data")

                    apply_ai_setup(setup)

                    setup_owner = None

                    print("AI configured.")

                    await broadcast({
                        "type": "system",
                        "message":
                            f"AI configured as {ai_config['name']}."
                    })

                    if ai_config["first_message"].strip():

                        first_message = ai_config["first_message"]

                        append_assistant_message(
                            ai_config["name"],
                            first_message
                        )

                        await broadcast({
                            "type": "chat",
                            "sender": ai_config["name"],
                            "message": first_message
                        })

                except Exception as e:

                    await ws.send_text(json.dumps({
                        "type": "system",
                        "message":
                            f"Invalid AI setup data: {e}"
                    }))

                continue

            # =========================================
            # COMMANDS
            # =========================================

            if msg.startswith("/"):

                command = msg.lower().strip()

                # -------------------------------------
                # /sleep
                # -------------------------------------

                if command == "/sleep":

                    ai_awake = False

                    await broadcast({
                        "type": "system",
                        "message":
                            "AI is now sleeping."
                    })

                    continue

                # -------------------------------------
                # /wake
                # -------------------------------------

                elif command == "/wake":

                    ai_awake = True

                    await broadcast({
                        "type": "system",
                        "message":
                            "AI is now awake."
                    })

                    continue

                # -------------------------------------
                # /clear
                # -------------------------------------

                elif command == "/clear":

                    history.clear()

                    await broadcast({
                        "type": "system",
                        "message":
                            "Conversation history cleared."
                    })

                    continue

                # -------------------------------------
                # /scenario
                # -------------------------------------

                elif command == "/scenario":

                    ai_config["configured"] = False
                    ai_config["first_message"] = ""
                    setup_owner = ws

                    await broadcast({
                        "type": "scenario_reset",
                        "message":
                            "AI scenario reset. Reconfigure the AI."
                    })

                    await ws.send_text(json.dumps({
                        "type": "setup_required"
                    }))

                    continue

                # -------------------------------------
                # /new
                # -------------------------------------

                elif command == "/new":

                    history.clear()

                    await broadcast({
                        "type": "new_session",
                        "message":
                            "New session started."
                    })

                    continue

                # -------------------------------------
                # /export
                # -------------------------------------

                elif command == "/export":

                    filename = export_session()

                    await broadcast({
                        "type": "system",
                        "message":
                            f"Session exported to {filename}"
                    })

                    continue

                # -------------------------------------
                # /import
                # -------------------------------------

                elif command.startswith("/import"):

                    try:

                        parts = msg.split(maxsplit=1)

                        if len(parts) < 2:

                            await ws.send_text(json.dumps({
                                "type": "system",
                                "message":
                                    "Usage: /import filename.json"
                            }))

                            continue

                        filename = parts[1]

                        import_session(filename)

                        await broadcast({
                            "type": "system",
                            "message":
                                f"Imported session {filename}"
                        })

                    except Exception as e:

                        await ws.send_text(json.dumps({
                            "type": "system",
                            "message":
                                f"Import failed: {e}"
                        }))

                    continue

            # =========================================
            # USER MESSAGE BROADCAST
            # =========================================

            await broadcast({
                "type": "chat",
                "sender": username,
                "message": msg
            })

            append_user_message(username, msg)

            # =========================================
            # AI SLEEP CHECK
            # =========================================

            if not ai_awake:
                continue

            # =========================================
            # AI RESPONSE
            # =========================================

            try:

                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {
                            "role": "system",
                            "content":
                                f"You are {ai_config['name']}. "
                                f"{ai_config['behavior']}"
                        }
                    ] + [
                        {
                            "role": item["role"],
                            "content": item["content"]
                        }
                        for item in history[-20:]
                    ],
                    reasoning_effort=ai_config[
                        "reasoning_level"
                    ]
                )

                message = response.choices[0].message

                reply = (
                    message.content
                    if message.content
                    else ""
                )

                reasoning = getattr(
                    message,
                    "reasoning_content",
                    None
                )

            except Exception as e:

                reply = f"ERROR: {e}"
                reasoning = None

            append_assistant_message(
                ai_config["name"],
                reply
            )

            # =========================================
            # REASONING BROADCAST
            # =========================================

            if (
                ai_config["show_reasoning"]
                and reasoning
            ):

                await broadcast({
                    "type": "reasoning",
                    "sender": ai_config["name"],
                    "message": reasoning
                })

            # =========================================
            # FINAL AI MESSAGE
            # =========================================

            await broadcast({
                "type": "chat",
                "sender": ai_config["name"],
                "message": reply
            })

    except Exception as e:

        print(
            f"{username} disconnected. "
            f"Reason: {e}"
        )

    finally:

        if ws in clients:
            del clients[ws]

        if setup_owner == ws:
            setup_owner = None