```python
from fastapi import FastAPI, WebSocket
from openai import OpenAI
import json
import os
from datetime import datetime

app = FastAPI()

clients = {}
history = []

ai_awake = True

ai_config = {
    "name": "DEEPSEEK",
    "behavior": "Helpful and intelligent.",
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

    await ws.accept()

    username = await ws.receive_text()

    clients[ws] = username

    print(f"{username} connected.")

    # =================================================
    # FIRST USER CONFIGURES AI
    # =================================================

    if not ai_config["configured"]:

        await ws.send_text(json.dumps({
            "type": "setup_required"
        }))

        raw = await ws.receive_text()

        setup = json.loads(raw)

        ai_config["name"] = setup["ai_name"]
        ai_config["behavior"] = setup["behavior"]
        ai_config["reasoning_level"] = setup["reasoning_level"]
        ai_config["show_reasoning"] = setup["show_reasoning"]
        ai_config["configured"] = True

        print("AI configured.")

    else:

        await ws.send_text(json.dumps({
            "type": "system",
            "message":
                f"Connected to {ai_config['name']}"
        }))

    # =================================================
    # MAIN LOOP
    # =================================================

    try:

        while True:

            msg = await ws.receive_text()

            print(f"{username}: {msg}")

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

            history.append({
                "role": "user",
                "content": f"{username}: {msg}"
            })

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
                    ] + history[-20:],
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

            history.append({
                "role": "assistant",
                "content": reply
            })

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
```
