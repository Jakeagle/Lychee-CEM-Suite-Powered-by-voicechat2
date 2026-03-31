import asyncio
import aiohttp
import json
import os
import time
import uuid
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# --- CONFIGURATION ---
SRT_ENDPOINT = "http://localhost:8001/inference"
LLM_ENDPOINT = "http://localhost:11434/api/chat"
TTS_ENDPOINT = "http://localhost:8003/tts"

app = FastAPI()

# --- ABSOLUTE PATH SETUP ---
# Ensures data is saved to /home/[user]/voicechat2/data/
HOME = os.path.expanduser("~")
BASE_DATA_DIR = os.path.join(HOME, "voicechat2", "data", "characters")
CONFIG_PATH = os.path.join(HOME, "voicechat2", "active_config.json")

# Ensure fundamental directories exist
os.makedirs(os.path.join(BASE_DATA_DIR, "bones", "logs"), exist_ok=True)
os.makedirs("ui", exist_ok=True)

app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# --- GLOBAL STATE ---
current_character = "bones"
# This history is reset whenever a Hot-Swap occurs
CONVERSATION_HISTORY = [
    {"role": "system", "content": "You are Bones, a sarcastic animatronic from Lychee Interactive. You have no legs. Be brief (1-2 sentences)."}
]

@app.get("/")
async def get_index():
    return FileResponse("ui/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global current_character, CONVERSATION_HISTORY
    await websocket.accept()
    session_id = str(uuid.uuid4())
    
    # 1. CHECK FOR HOT-SWAP (Goal 6)
    # Check if the Gradio app wrote a new configuration file
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
                if cfg["active_character"] != current_character:
                    current_character = cfg["active_character"]
                    logger.warning(f"HOT-SWAP: Switching personality to {current_character}")
                    
                    # Update System Identity for the new character
                    CONVERSATION_HISTORY = [
                        {"role": "system", "content": f"You are {current_character}, an animatronic character from Lychee Interactive. Stay in character and be brief."}
                    ]
        except Exception as e:
            logger.error(f"Failed to read hot-swap config: {e}")

    logger.info(f"Session {session_id} active for character: {current_character}")
    
    try:
        while True:
            # Handle browser disconnects gracefully
            try:
                data = await websocket.receive()
            except:
                break

            if "bytes" in data:
                audio_bytes = data["bytes"]
                if len(audio_bytes) < 5000: continue # Skip noise

                # 2. TRANSCRIBE (The Ears)
                user_text = ""
                async with aiohttp.ClientSession() as session:
                    form = aiohttp.FormData()
                    form.add_field('file', audio_bytes, filename='input.webm')
                    async with session.post(SRT_ENDPOINT, data=form, timeout=10) as resp:
                        if resp.status == 200:
                            srt_data = await resp.json()
                            user_text = srt_data.get("text", "")

                if not user_text.strip(): continue
                logger.info(f"User Said: {user_text}")
                await websocket.send_json({"type": "transcription", "text": user_text})

                # 3. THINK & SPEAK (The Brain & Voice)
                full_reply = await process_llm_and_tts(websocket, user_text)
                
                # 4. LOGGING (For Character Manager / Training)
                log_dir = os.path.join(BASE_DATA_DIR, current_character, "logs")
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"{session_id}.json")
                
                log_entry = {
                    "instruction": user_text,  # AI Training Key
                    "output": full_reply,      # AI Training Key
                    "user": user_text,         # Display Key
                    "assistant": full_reply,   # Display Key
                    "ts": time.time(),
                    "char": current_character
                }
                
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

    except WebSocketDisconnect:
        logger.info("Connection closed by browser.")
    except Exception as e:
        logger.error(f"Critical Orchestrator Error: {e}")
        traceback.print_exc()

async def process_llm_and_tts(websocket, text):
    global CONVERSATION_HISTORY
    CONVERSATION_HISTORY.append({"role": "user", "content": text})
    full_reply = ""
    sentence_buffer = ""

    async with aiohttp.ClientSession() as session:
        # Connect to Ollama
        payload = {"model": current_character, "messages": CONVERSATION_HISTORY, "stream": True}
        
        async with session.post(LLM_ENDPOINT, json=payload, timeout=30) as resp:
            if resp.status != 200:
                logger.error(f"Ollama Error: {resp.status}")
                return "My internal systems are lagging. Give me a moment."

            async for line in resp.content:
                if not line: continue
                
                chunk = json.loads(line.decode('utf-8'))
                if "message" in chunk:
                    token = chunk["message"].get("content", "")
                    full_reply += token
                    sentence_buffer += token
                    
                    # Stream tokens to the UI (makes text appear as he speaks)
                    await websocket.send_json({"type": "text_chunk", "text": token})

                    # If a sentence is complete, speak it (Low Latency)
                    if any(c in token for c in ".!?"):
                        await send_to_voice(sentence_buffer, websocket)
                        sentence_buffer = ""

            # Catch trailing text
            if sentence_buffer.strip():
                await send_to_voice(sentence_buffer, websocket)

    CONVERSATION_HISTORY.append({"role": "assistant", "content": full_reply})
    
    # Keep Memory Management: prune history if it gets too long for the 3070
    if len(CONVERSATION_HISTORY) > 10: 
        CONVERSATION_HISTORY = [CONVERSATION_HISTORY[0]] + CONVERSATION_HISTORY[-9:]
        
    return full_reply

async def send_to_voice(text, websocket):
    """Sends a single sentence to the VITS server and pushes audio bytes to the browser"""
    if not text.strip(): return
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(TTS_ENDPOINT, json={"text": text}, timeout=10) as resp:
                if resp.status == 200:
                    audio_data = await resp.read()
                    # Browser UI catches these bytes and puts them in the audioQueue
                    await websocket.send_bytes(audio_data)
                else:
                    logger.error(f"TTS Server Error: {resp.status}")
    except Exception as e:
        logger.error(f"Failed to communicate with TTS Server: {e}")

if __name__ == "__main__":
    import uvicorn
    # Live Site runs on Port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)