import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import subprocess
import time
from typing import Set, List, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

import config
from agent import SalesAgent
from vad import VADProcessor, extract_stereo_channels
from stt import get_stt_processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, get_stt_processor)
    print("[SERVER] Sber GigaAM STT pre-loaded successfully into memory!")
    yield

app = FastAPI(title="Zoom Sales Copilot Backend", lifespan=lifespan)

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
        self.active_channel: Optional[str] = None
        self.last_manager_speech_time: float = 0.0
        self.bot_process: Optional[subprocess.Popen] = None
        self.audio_buffer: bytearray = bytearray()
        self.pending_gemini_query: bool = False

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

def cleanup_session_if_empty(session_id: str):
    """
    Cleans up session memory if no UI/Audio clients are connected and bot process is inactive.
    """
    session = sessions.get(session_id)
    if session and len(session.connected_clients) == 0 and (session.bot_process is None or session.bot_process.poll() is not None):
        if session.speculative_task and not session.speculative_task.done():
            session.speculative_task.cancel()
        sessions.pop(session_id, None)
        print(f"[{session_id}] Cleaned up empty session state from memory.")

async def broadcast_to_session(session: SessionState, payload: dict):
    """
    Helper method to broadcast JSON messages to all connected UI WebSocket clients.
    """
    for client in list(session.connected_clients):
        try:
            await client.send_json(payload)
        except Exception:
            pass

def is_duplicate_transcript(session: SessionState, text: str, role: str) -> bool:
    """
    Checks if the newly transcribed text matches the last line in the transcript buffer
    from the SAME speaker to prevent repeating accidental VAD duplicates.
    """
    clean_text = text.strip().lower()
    if not clean_text or len(clean_text) < 2:
        return True
    if not session.transcript_buffer:
        return False
    
    clean_words = clean_text.translate(str.maketrans('', '', '.,!?'))
    
    last_line = session.transcript_buffer[-1]
    parts = last_line.split(": ", 1)
    if len(parts) == 2:
        prev_sender, prev_text = parts[0].strip(), parts[1].strip().lower()
        prev_words = prev_text.translate(str.maketrans('', '', '.,!?'))
        current_sender = "Менеджер" if role == "Representative" else "Клиент"
        
        if prev_sender == current_sender and clean_words == prev_words:
            return True
            
    return False

@app.get("/")
async def get_index():
    """
    Serves the floating UI index page directly.
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
        cleanup_session_if_empty(session_id)

async def run_speculative_query(session_id: str, transcript_text: str):
    """
    Calls the Google Antigravity SDK agent to get hints via streaming.
    Text-only input for maximum speed (~300ms latency).
    """
    session = get_session(session_id)
    print(f"\n[{session_id}] [SPECOPS] Triggering text-only speculative query (streaming)...")
    try:
        hint_data = None
        async for item in sales_agent.get_hint_stream(transcript_text):
            if item["type"] == "chunk":
                # Broadcast streaming text chunk to UI clients immediately
                payload_chunk = {
                    "type": "stream_chunk",
                    "text": item["text"],
                    "full_text": item["full_text"]
                }
                await broadcast_to_session(session, payload_chunk)
            if item["type"] == "error":
                payload_err = {
                    "type": "rate_limit_error",
                    "message": item.get("message", "Слишком много запросов")
                }
                await broadcast_to_session(session, payload_err)
                return
            elif item["type"] == "final":
                hint_data = item["data"]

        if hint_data:
            session.current_hint = hint_data
            payload_hint = {
                "type": "hint",
                "data": session.current_hint,
                "transcript": "\n".join(session.transcript_buffer)
            }
            print(f"[{session_id}] [SPECOPS] Hint generated for transcript")
            await broadcast_to_session(session, payload_hint)
                
    except asyncio.CancelledError:
        print(f"[{session_id}] [SPECOPS] Speculative query was CANCELLED due to speech resumption.")
        raise
    except Exception as e:
        print(f"[{session_id}] [SPECOPS] Error running query: {e}")

@app.websocket("/audio/{session_id}")
async def audio_stream_endpoint(websocket: WebSocket, session_id: str, source: str = "browser"):
    """
    WebSocket endpoint that receives audio stream chunks and STT transcript events for a specific session.
    """
    await websocket.accept()
    session = get_session(session_id)
    print(f"[{session_id}] Audio streaming source ({source}) connected.")
    
    if source == "bot":
        payload_init = {
            "type": "bot_status",
            "connected": True,
            "message": "Бот успешно подключился к аудиопотоку встречи!"
        }
        await broadcast_to_session(session, payload_init)
            
    vad_processor = VADProcessor(
        sample_rate=config.SAMPLE_RATE,
        frame_duration_ms=config.FRAME_DURATION_MS,
        energy_threshold=650.0
    )
    
    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                raw_bytes = message["bytes"]
                left_bytes, right_bytes, left_rms, right_rms = extract_stereo_channels(raw_bytes)
                
                threshold = 500.0
                if right_rms >= threshold and right_rms >= left_rms * 0.8:
                    current_speaker = "Customer"
                    current_channel = "right"
                elif left_rms >= threshold:
                    current_speaker = "Representative"
                    current_channel = "left"
                else:
                    current_channel = session.active_channel or "left"
                    current_speaker = "Representative" if current_channel == "left" else "Customer"
                
                audio_chunk = left_bytes if current_channel == "left" else right_bytes
                vad_result = vad_processor.process_chunk(audio_chunk)
                
                if vad_result["speech_started"]:
                    session.active_channel = current_channel
                    session.last_speaker = current_speaker
                    session.rate_limit_skipped = False
                    session.audio_buffer.clear()
                
                if vad_result["is_speaking"]:
                    active_bytes = left_bytes if (session.active_channel or current_channel) == "left" else right_bytes
                    session.audio_buffer.extend(active_bytes)
                
                if not vad_result["is_speaking"]:
                    silence_ms = vad_result["silence_duration_ms"]
                    
                    if silence_ms >= config.SPECULATIVE_SILENCE_LIMIT_MS:
                        if len(session.audio_buffer) >= 4800 and (session.speculative_task is None or session.speculative_task.done()):
                            now = time.time()
                            audio_data = bytes(session.audio_buffer)
                            session.audio_buffer.clear()
                            
                            audio_energy = vad_processor.calculate_energy(audio_data)
                            if audio_energy < 600.0:
                                session.active_channel = None
                                continue
                                
                            if now - session.last_query_time >= 0.05:
                                session.last_query_time = now
                                session.rate_limit_skipped = False
                                
                                try:
                                    stt_processor = get_stt_processor()
                                    loop = asyncio.get_running_loop()
                                    stt_start = time.time()
                                    stt_text = await loop.run_in_executor(None, stt_processor.transcribe, audio_data)
                                    stt_elapsed_ms = (time.time() - stt_start) * 1000
                                    
                                    if stt_text and stt_text.strip():
                                        if session.speculative_task and not session.speculative_task.done():
                                            session.speculative_task.cancel()
                                            print(f"[{session_id}] [VAD] New speech decoded! Cancelled pending Gemini query.")

                                        clean_text = stt_text.strip()
                                        role = session.last_speaker or "Representative"
                                        if not is_duplicate_transcript(session, clean_text, role):
                                            display_speaker = "Менеджер" if role == "Representative" else "Клиент"
                                            line = f"{display_speaker}: {clean_text}"
                                            session.transcript_buffer.append(line)
                                            if role == "Customer":
                                                session.pending_gemini_query = True
                                            print(f"[{session_id}] [LOCAL-STT ⚡ {stt_elapsed_ms:.0f}ms] {line}")
                                            
                                            payload_trans = {
                                                "type": "transcript_update",
                                                "line": line,
                                                "transcript": "\n".join(session.transcript_buffer)
                                            }
                                            await broadcast_to_session(session, payload_trans)
                                except Exception as stt_err:
                                    print(f"[{session_id}] Local STT execution error: {stt_err}")

                                current_speaker_role = session.last_speaker
                                session.active_channel = None

                                if not vad_result["is_speaking"] and silence_ms >= config.FINAL_SILENCE_LIMIT_MS:
                                    if session.pending_gemini_query and session.transcript_buffer:
                                        session.pending_gemini_query = False
                                        if current_speaker_role == "Customer":
                                            transcript_text = "\n".join(session.transcript_buffer)
                                            session.speculative_task = asyncio.create_task(
                                                run_speculative_query(session_id, transcript_text)
                                            )
                                            print(f"[{session_id}] [VAD] Customer speech pause confirmed ({silence_ms}ms). Triggered SPIN query.")
                                        else:
                                            print(f"[{session_id}] [VAD] Representative turn finished ({silence_ms}ms). Suppressing Gemini query.")
                            else:
                                session.active_channel = None
                                if not session.rate_limit_skipped:
                                    session.rate_limit_skipped = True
                                    print(f"[{session_id}] [VAD] Silence reached, but API query rate-limited to protect Free Tier.")
                            
            elif "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "transcript":
                    speaker = data.get("speaker") or "Representative"
                    text = data.get("text", "").strip()
                    
                    if text and not is_duplicate_transcript(session, text, speaker):
                        session.last_speaker = speaker
                        display_speaker = "Менеджер" if speaker == "Representative" else "Клиент"
                        line = f"{display_speaker}: {text}"
                        session.transcript_buffer.append(line)
                        print(f"\n[{session_id}] [STT] {line}")
                        
                        payload = {
                            "type": "transcript_update",
                            "line": line,
                            "transcript": "\n".join(session.transcript_buffer)
                        }
                        await broadcast_to_session(session, payload)

                        if speaker == "Customer":
                            transcript_text = "\n".join(session.transcript_buffer)
                            session.speculative_task = asyncio.create_task(
                                run_speculative_query(session_id, transcript_text)
                            )
                            
    except WebSocketDisconnect:
        print(f"[{session_id}] Audio streaming source disconnected.")
    except Exception as e:
        print(f"[{session_id}] Error in audio stream connection for session {session_id}: {e}")
    finally:
        if source == "bot":
            payload_disc = {
                "type": "bot_status",
                "connected": False,
                "message": "Бот отключился от трансляции аудио."
            }
            await broadcast_to_session(session, payload_disc)
        cleanup_session_if_empty(session_id)

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
        cleanup_session_if_empty(session_id)
        return {"status": "success", "message": "Zoom Bot stopped successfully"}
    else:
        return {"status": "error", "message": "No active bot found to stop"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Zoom Sales Copilot Server on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
