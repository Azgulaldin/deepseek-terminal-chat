from fastapi import FastAPI, WebSocket
from openai import OpenAI
import json
import os
from datetime import datetime
import asyncio
import time

app = FastAPI()

clients = {}
personas = {}
history = []

ai_awake = True
scenario_mode = False

# track activity per connection
last_seen = {}

ai_config = {
    "name": "DEEPSEEK",
    "behavior": "Helpful and intelligent.",
    "reasoning_level": "medium",
    "show_reasoning": False,
    "first_message": "",
    "configured": False
}

# =====================================================
# API KEY
# =====================================================

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable not found.")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com/v1"
)

# =====================================================
# HEARTBEAT SYSTEM
# =====================================================

async def heartbeat():
    while True:
        await asyncio.sleep(30)

        now = time.time()
        dead = []

        for ws in list(clients.keys()):
            if now - last_seen.get(ws, now) > 120:
                dead.append(ws)

        for ws in dead:
            try:
                await ws.close()
            except:
                pass
            clients.pop(ws, None)
            personas.pop(ws, None)
            last_seen.pop(ws, None)

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
        clients.pop(ws, None)
        personas.pop(ws, None)
        last_seen.pop(ws, None)

# =====================================================
# DISCONNECT OTHERS
# =====================================================

async def disconnect_all_except(keep_ws):
    for ws in list(clients.keys()):
        if ws != keep_ws:
            try:
                await ws.close()
            except:
                pass

            clients.pop(ws, None)
            personas.pop(ws, None)
            last_seen.pop(ws, None)

# =====================================================
# STARTUP
# =====================================================

@app.on_event("startup")
async def startup():
    asyncio.create_task(heartbeat())

# =====================================================
# ROOT
# =====================================================

@app.get("/")
def home():
    return {"status": "server running"}

# =====================================================
# WEBSOCKET
# =====================================================

@app.websocket("/chat")
async def chat(ws: WebSocket):

    global ai_awake, scenario_mode

    await ws.accept()

    username = await ws.receive_text()
    clients[ws] = username
    last_seen[ws] = time.time()

    print(f"{username} connected.")

    # =================================================
    # AI CONFIG
    # =================================================

    if not ai_config["configured"]:

        await ws.send_text(json.dumps({"type": "setup_required"}))

        raw = await ws.receive_text()
        last_seen[ws] = time.time()

        setup = json.loads(raw)

        ai_config["name"] = setup["ai_name"]
        ai_config["behavior"] = setup["behavior"]
        ai_config["reasoning_level"] = setup["reasoning_level"]
        ai_config["show_reasoning"] = setup["show_reasoning"]
        ai_config["first_message"] = setup.get("first_message", "")
        ai_config["configured"] = True

        print("AI configured.")

    else:
        await ws.send_text(json.dumps({
            "type": "system",
            "message": f"Connected to {ai_config['name']}"
        }))

    # =================================================
    # MAIN LOOP
    # =================================================

    try:
        while True:

            msg = await ws.receive_text()
            last_seen[ws] = time.time()

            print(f"{username}: {msg}")

            # =================================================
            # COMMANDS
            # =================================================

            if msg.startswith("/"):

                command = msg.lower().strip()

                if command == "/sleep":
                    ai_awake = False
                    await broadcast({"type": "system", "message": "AI is now sleeping."})
                    continue

                elif command == "/wake":
                    ai_awake = True
                    await broadcast({"type": "system", "message": "AI is now awake."})
                    continue

                elif command == "/clear":
                    history.clear()
                    await broadcast({"type": "system", "message": "Conversation history cleared."})
                    continue

                elif command == "/scenario":

                    scenario_mode = True

                    await disconnect_all_except(ws)

                    await ws.send_text(json.dumps({
                        "type": "system",
                        "message": "Scenario mode activated. Reconfiguring AI..."
                    }))

                    await ws.send_text(json.dumps({"type": "setup_required"}))

                    raw = await ws.receive_text()
                    last_seen[ws] = time.time()

                    setup = json.loads(raw)

                    ai_config["name"] = setup["ai_name"]
                    ai_config["behavior"] = setup["behavior"]
                    ai_config["reasoning_level"] = setup["reasoning_level"]
                    ai_config["show_reasoning"] = setup["show_reasoning"]
                    ai_config["first_message"] = setup.get("first_message", "")
                    ai_config["configured"] = True

                    history.clear()

                    if ai_config["first_message"]:
                        history.append({
                            "role": "assistant",
                            "content": ai_config["first_message"]
                        })

                    await broadcast({
                        "type": "system",
                        "message": "Scenario initialized."
                    })

                    continue

            # =================================================
            # PERSONA
            # =================================================

            if ws not in personas:
                await ws.send_text(json.dumps({"type": "request_persona"}))
                persona = await ws.receive_text()
                last_seen[ws] = time.time()
                personas[ws] = persona

            # =================================================
            # BROADCAST MESSAGE
            # =================================================

            await broadcast({
                "type": "chat",
                "sender": username,
                "persona": personas.get(ws, ""),
                "message": msg
            })

            history.append({
                "role": "user",
                "content": f"{username}: {msg}"
            })

            if not ai_awake:
                continue

            # =================================================
            # AI RESPONSE
            # =================================================

            try:

                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {
                            "role": "system",
                            "content":
                                f"You are {ai_config['name']}. "
                                f"{ai_config['behavior']}. "
                                f"Scenario intro: {ai_config.get('first_message','')}"
                        }
                    ] + history[-20:],
                    reasoning_effort=ai_config["reasoning_level"]
                )

                message = response.choices[0].message
                reply = message.content or ""
                reasoning = getattr(message, "reasoning_content", None)

            except Exception as e:
                reply = f"ERROR: {e}"
                reasoning = None

            history.append({"role": "assistant", "content": reply})

            if ai_config["show_reasoning"] and reasoning:
                await broadcast({
                    "type": "reasoning",
                    "sender": ai_config["name"],
                    "message": reasoning
                })

            await broadcast({
                "type": "chat",
                "sender": ai_config["name"],
                "message": reply
            })

    except Exception as e:
        print(f"{username} disconnected. Reason: {e}")

    finally:
        clients.pop(ws, None)
        personas.pop(ws, None)
        last_seen.pop(ws, None)