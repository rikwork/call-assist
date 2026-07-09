import asyncio
import json
import os
import subprocess
import time
from typing import Set, List, Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

import config
from agent import SalesAgent
from vad import VADProcessor

app = FastAPI(title="Zoom Sales Copilot Backend")

# Session state configuration for multi-session support
class SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.connected_clients: Set[WebSocket] = set()
        self.transcript_buffer: List[str] = []
        self.current_hint: dict = {}
        self.speculative_task: Optional[asyncio.Task] = None
        self.last_query_time: float = 0.0
        self.rate_limit_skipped: bool = False
        self.last_speaker: Optional[str] = None
        self.bot_process: Optional[subprocess.Popen] = None
        self.audio_buffer: bytearray = bytearray()

# Global sessions tracker
sessions: Dict[str, SessionState] = {}
sales_agent = SalesAgent()

def get_session(session_id: str) -> SessionState:
    """
    Retrieves or creates an isolated session state.
    """
    if session_id not in sessions:
        sessions[session_id] = SessionState(session_id)
    return sessions[session_id]

@app.get("/")
async def get_index():
    """
    Serves the floating UI index page (Option 1) directly.
    """
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html not found. Please build the UI first.</h1>", status_code=404)

@app.websocket("/ws/{session_id}")
async def client_websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for the floating UI.
    Receives connection and broadcasts real-time hints for a specific session.
    """
    await websocket.accept()
    session = get_session(session_id)
    session.connected_clients.add(websocket)
    print(f"[{session_id}] Frontend UI client connected. Total clients in session: {len(session.connected_clients)}")
    
    # Immediately send the latest state/hint if available
    try:
        if session.current_hint:
            await websocket.send_json({
                "type": "hint",
                "data": session.current_hint,
                "transcript": "\n".join(session.transcript_buffer)
            })
        else:
            await websocket.send_json({
                "type": "status",
                "message": "Connected. Waiting for audio stream..."
            })
            
        while True:
            # Keep connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        session.connected_clients.remove(websocket)
        print(f"[{session_id}] Frontend UI client disconnected. Total clients in session: {len(session.connected_clients)}")

async def run_speculative_query(session_id: str, transcript_text: str, audio_bytes: Optional[bytes] = None):
    """
    Calls the Google Antigravity SDK agent to get hints.
    Processes both text transcripts and multimodal raw audio inputs.
    """
    session = get_session(session_id)
    print(f"\n[{session_id}] [SPECOPS] Triggering speculative query...")
    try:
        # Call the SDK Agent wrapper with audio bytes
        hint_data = await sales_agent.get_hint(transcript_text, audio_bytes)
        if hint_data:
            session.current_hint = hint_data
            
            # Extract values from Gemini multimodal response
            transcription = hint_data.get("customer_transcript")
            role = hint_data.get("speaker_role", "Customer")
            
            if transcription:
                # Format and add to transcript history
                display_speaker = "Менеджер" if role == "Representative" else "Клиент"
                line = f"{display_speaker}: {transcription}"
                session.transcript_buffer.append(line)
                session.last_speaker = role
                print(f"[{session_id}] [SPECOPS-STT] {line}")
                
                # Broadcast transcript update to UI
                payload_trans = {
                    "type": "transcript_update",
                    "line": line,
                    "transcript": "\n".join(session.transcript_buffer)
                }
                for client in list(session.connected_clients):
                    try:
                        await client.send_json(payload_trans)
                    except Exception:
                        pass

            # Broadcast hint details to UI (if Representative, suppress_display flag is added)
            payload_hint = {
                "type": "hint",
                "data": session.current_hint,
                "transcript": "\n".join(session.transcript_buffer)
            }
            if role == "Customer":
                print(f"[{session_id}] [SPECOPS] Hint generated for Customer: {json.dumps(session.current_hint, ensure_ascii=False)}")
            else:
                print(f"[{session_id}] [SPECOPS] Representative turn finished. Suppressing UI hint card update.")
                
            for client in list(session.connected_clients):
                try:
                    await client.send_json(payload_hint)
                except Exception:
                    pass
                
    except asyncio.CancelledError:
        print(f"[{session_id}] [SPECOPS] Speculative query was CANCELLED due to speech resumption.")
        raise
    except Exception as e:
        print(f"[{session_id}] [SPECOPS] Error running query: {e}")

@app.websocket("/audio/{session_id}")
async def audio_stream_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint that receives audio stream chunks and STT transcript events for a specific session.
    """
    await websocket.accept()
    session = get_session(session_id)
    print(f"[{session_id}] Audio streaming source connected.")
    
    vad_processor = VADProcessor(
        sample_rate=config.SAMPLE_RATE,
        frame_duration_ms=config.FRAME_DURATION_MS,
        energy_threshold=300.0  # Simple energy threshold for simulated wave
    )
    
    try:
        while True:
            # Receive binary (PCM chunks) or text (transcript events)
            message = await websocket.receive()
            
            if "bytes" in message:
                audio_chunk = message["bytes"]
                vad_result = vad_processor.process_chunk(audio_chunk)
                
                # Logic: Speech Resumed
                if vad_result["speech_started"]:
                    session.rate_limit_skipped = False
                    session.audio_buffer.clear()  # Clear for fresh speech capture
                    if session.speculative_task and not session.speculative_task.done():
                        session.speculative_task.cancel()
                        print(f"\n[{session_id}] [VAD] Speech started! Speculative task CANCELLED.")
                
                # Logic: Accumulate Audio during speech
                if vad_result["is_speaking"]:
                    session.audio_buffer.extend(audio_chunk)
                
                # Logic: Silence Duration limits
                if not vad_result["is_speaking"]:
                    silence_ms = vad_result["silence_duration_ms"]
                    
                    # Trigger Speculative Task
                    if silence_ms >= config.SPECULATIVE_SILENCE_LIMIT_MS:
                        # Make sure we have at least 0.5s of recorded audio
                        if len(session.audio_buffer) >= 16000 and (session.speculative_task is None or session.speculative_task.done()):
                            now = time.time()
                            # 12 seconds cooldown (max 5 requests per minute for Free Tier)
                            if now - session.last_query_time >= 12.0:
                                session.last_query_time = now
                                session.rate_limit_skipped = False
                                
                                # Fetch and clear the accumulated audio bytes
                                audio_data = bytes(session.audio_buffer)
                                session.audio_buffer.clear()
                                
                                transcript_text = "\n".join(session.transcript_buffer)
                                session.speculative_task = asyncio.create_task(
                                    run_speculative_query(session_id, transcript_text, audio_data)
                                )
                                print(f"[{session_id}] [VAD] Silence reached {silence_ms}ms. Triggered multimodal query.")
                            else:
                                if not session.rate_limit_skipped:
                                    session.rate_limit_skipped = True
                                    print(f"[{session_id}] [VAD] Silence reached, but API query rate-limited to protect Free Tier (cooldown: {12.0 - (now - session.last_query_time):.1f}s).")
                            
            elif "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "transcript":
                    speaker = data.get("speaker", "Unknown")
                    text = data.get("text", "")
                    
                    # Track last speaker
                    session.last_speaker = speaker
                    
                    # Localize speaker names to Russian for UI
                    display_speaker = "Менеджер" if speaker == "Representative" else "Клиент"
                    line = f"{display_speaker}: {text}"
                    session.transcript_buffer.append(line)
                    print(f"\n[{session_id}] [STT] {line}")
                    
                    # Also broadcast transcript update to UI
                    payload = {
                        "type": "transcript_update",
                        "line": line,
                        "transcript": "\n".join(session.transcript_buffer)
                    }
                    for client in list(session.connected_clients):
                        try:
                            await client.send_json(payload)
                        except Exception:
                            pass
                            
    except WebSocketDisconnect:
        print(f"[{session_id}] Audio streaming source disconnected.")
    except Exception as e:
        print(f"[{session_id}] Error in audio stream connection for session {session_id}: {e}")

@app.post("/bot/start")
async def start_bot(data: dict):
    zoom_url = data.get("zoom_url")
    session_id = data.get("session_id", "default_session")
    
    if not zoom_url:
        return {"status": "error", "message": "zoom_url is required"}
        
    session = get_session(session_id)
    if session.bot_process and session.bot_process.poll() is None:
        return {"status": "error", "message": f"Bot is already running for session {session_id}"}
        
    run_mode = os.environ.get("BOT_RUN_MODE", "local")
    print(f"[{session_id}] Launching Zoom Bot in {run_mode} mode for URL: {zoom_url}")
    
    try:
        if run_mode == "docker":
            cmd = [
                "docker", "run", "-d",
                "--network", "host",
                "--name", f"zoom_bot_{session_id}",
                "--rm",
                "call-assist-zoom-bot:latest",
                "./entrypoint.sh",
                "--url", zoom_url,
                "--session", session_id
            ]
            session.bot_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            cmd = [
                "node", "zoom_bot/zoom_bot.js",
                "--url", zoom_url,
                "--session", session_id
            ]
            session.bot_process = subprocess.Popen(cmd)
            
        return {"status": "success", "message": f"Zoom Bot started in {run_mode} mode"}
    except Exception as e:
        print(f"[{session_id}] Failed to start bot: {e}")
        return {"status": "error", "message": f"Failed to start bot: {str(e)}"}

@app.post("/bot/stop")
async def stop_bot(data: dict):
    session_id = data.get("session_id", "default_session")
    session = get_session(session_id)
    
    run_mode = os.environ.get("BOT_RUN_MODE", "local")
    print(f"[{session_id}] Stopping Zoom Bot...")
    
    stopped = False
    if run_mode == "docker":
        try:
            subprocess.run(["docker", "stop", f"zoom_bot_{session_id}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stopped = True
        except Exception as e:
            print(f"[{session_id}] Failed to stop docker container: {e}")
    else:
        if session.bot_process:
            try:
                session.bot_process.terminate()
                session.bot_process.wait(timeout=3)
                stopped = True
            except Exception as e:
                try:
                    session.bot_process.kill()
                    stopped = True
                except Exception:
                    pass
            session.bot_process = None
            
    if stopped:
        return {"status": "success", "message": "Zoom Bot stopped successfully"}
    else:
        return {"status": "error", "message": "No active bot found to stop"}

if __name__ == "__main__":
    # Load port from env
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Zoom Sales Copilot Server on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
