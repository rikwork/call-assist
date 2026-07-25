import asyncio
import json
import os
import subprocess
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
from typing import Set, List, Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

import config
from agent import SalesAgent
from vad import VADProcessor, extract_stereo_channels
from stt import get_stt_processor

app = FastAPI(title="Zoom Sales Copilot Backend")

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, get_stt_processor)
    print("[SERVER] Sber GigaAM STT pre-loaded successfully into memory!")

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

def is_duplicate_transcript(session: SessionState, text: str, role: str) -> bool:
    """
    Checks if the newly transcribed text matches the last line in the transcript buffer
    from the SAME speaker to prevent repeating accidental VAD duplicates.
    """
    clean_text = text.strip()
    if not clean_text:
        return True
    if not session.transcript_buffer:
        return False
    
    last_line = session.transcript_buffer[-1]
    parts = last_line.split(": ", 1)
    if len(parts) == 2:
        last_sender, last_text = parts[0].strip(), parts[1].strip()
        current_sender = "Менеджер" if role == "Representative" else "Клиент"
        if last_sender == current_sender and last_text.lower() == clean_text.lower():
            return True
    return False

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
                for client in list(session.connected_clients):
                    try:
                        await client.send_json(payload_chunk)
                    except Exception:
                        pass
            if item["type"] == "error":
                payload_err = {
                    "type": "rate_limit_error",
                    "message": item.get("message", "Слишком много запросов")
                }
                for client in list(session.connected_clients):
                    try:
                        await client.send_json(payload_err)
                    except Exception:
                        pass
                return
            elif item["type"] == "final":
                hint_data = item["data"]

        if hint_data:
            session.current_hint = hint_data
            
            # Extract values from Gemini multimodal response (Gemini determines speaker role semantically)
            transcription = hint_data.get("customer_transcript")
            role = hint_data.get("speaker_role") or "Representative"
            
            if transcription and transcription.strip():
                clean_text = transcription.strip()
                if not is_duplicate_transcript(session, clean_text, role):
                    # Format and add to transcript history
                    display_speaker = "Менеджер" if role == "Representative" else "Клиент"
                    line = f"{display_speaker}: {clean_text}"
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
    
    # Broadcast connection success to UI clients
    payload_init = {
        "type": "bot_status",
        "connected": True,
        "message": "Бот успешно подключился к аудиопотоку встречи!"
    }
    for client in list(session.connected_clients):
        try:
            await client.send_json(payload_init)
        except Exception:
            pass
            
    vad_processor = VADProcessor(
        sample_rate=config.SAMPLE_RATE,
        frame_duration_ms=config.FRAME_DURATION_MS,
        energy_threshold=650.0  # Filter out Mac/WebAudio background hiss (400-600 RMS)
    )
    
    try:
        while True:
            # Receive binary (PCM chunks) or text (transcript events)
            message = await websocket.receive()
            
            if "bytes" in message:
                raw_bytes = message["bytes"]
                left_bytes, right_bytes, left_rms, right_rms = extract_stereo_channels(raw_bytes)
                
                # Deterministic Speaker Identification: Left = Manager, Right = Customer
                if left_rms >= 600.0 and left_rms >= right_rms:
                    audio_chunk = left_bytes
                    session.last_speaker = "Representative"
                elif right_rms >= 600.0:
                    audio_chunk = right_bytes
                    session.last_speaker = "Customer"
                else:
                    audio_chunk = left_bytes
                    
                vad_result = vad_processor.process_chunk(audio_chunk)
                
                # Logic: Speech Resumed
                if vad_result["speech_started"]:
                    session.rate_limit_skipped = False
                    session.audio_buffer.clear()  # Clear for fresh speech capture
                
                # Logic: Accumulate Audio during speech
                if vad_result["is_speaking"]:
                    session.audio_buffer.extend(audio_chunk)
                
                # Logic: Silence Duration limits
                if not vad_result["is_speaking"]:
                    silence_ms = vad_result["silence_duration_ms"]
                    
                    # Trigger Speculative Task
                    if silence_ms >= config.SPECULATIVE_SILENCE_LIMIT_MS:
                        # Make sure we have at least 0.5s of recorded audio (8000 samples = 16000 bytes)
                        if len(session.audio_buffer) >= 16000 and (session.speculative_task is None or session.speculative_task.done()):
                            now = time.time()
                            # Fetch and clear accumulated audio bytes
                            audio_data = bytes(session.audio_buffer)
                            session.audio_buffer.clear()
                            
                            # Double-check audio energy to ensure real speech exists (not empty background hiss)
                            audio_energy = vad_processor.calculate_energy(audio_data)
                            if audio_energy < 600.0:
                                # Discard empty background noise buffer without firing API query
                                continue
                                
                            # 0.05 seconds cooldown (instant local STT mode)
                            if now - session.last_query_time >= 0.05:
                                session.last_query_time = now
                                session.rate_limit_skipped = False
                                
                                # Stage 1: Instant Local Sber GigaAM STT in RAM (~20ms)
                                try:
                                    stt_processor = get_stt_processor()
                                    loop = asyncio.get_running_loop()
                                    stt_start = time.time()
                                    stt_text = await loop.run_in_executor(None, stt_processor.transcribe, audio_data)
                                    stt_elapsed_ms = (time.time() - stt_start) * 1000
                                    
                                    if stt_text and stt_text.strip():
                                        # New speech decoded! Cancel any pending/running Gemini query task
                                        if session.speculative_task and not session.speculative_task.done():
                                            session.speculative_task.cancel()
                                            print(f"[{session_id}] [VAD] New speech decoded! Cancelled pending Gemini query.")

                                        clean_text = stt_text.strip()
                                        role = session.last_speaker or "Representative"
                                        if not is_duplicate_transcript(session, clean_text, role):
                                            display_speaker = "Менеджер" if role == "Representative" else "Клиент"
                                            line = f"{display_speaker}: {clean_text}"
                                            session.transcript_buffer.append(line)
                                            session.pending_gemini_query = True
                                            print(f"[{session_id}] [LOCAL-STT ⚡ {stt_elapsed_ms:.0f}ms] {line}")
                                            
                                            # Broadcast INSTANT transcript line to UI feed (<20ms!)
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
                                except Exception as stt_err:
                                    print(f"[{session_id}] Local STT execution error: {stt_err}")

                                # Stage 2: Gemini Sales Agent (Only trigger on 380ms pause if speaker has NOT resumed speech and new speech is pending)
                                if not vad_result["is_speaking"] and silence_ms >= config.FINAL_SILENCE_LIMIT_MS:
                                    if session.pending_gemini_query and session.transcript_buffer:
                                        session.pending_gemini_query = False
                                        transcript_text = "\n".join(session.transcript_buffer)
                                        session.speculative_task = asyncio.create_task(
                                            run_speculative_query(session_id, transcript_text)
                                        )
                                        print(f"[{session_id}] [VAD] Speech pause confirmed ({silence_ms}ms). Triggered SPIN query.")
                            else:
                                if not session.rate_limit_skipped:
                                    session.rate_limit_skipped = True
                                    print(f"[{session_id}] [VAD] Silence reached, but API query rate-limited to protect Free Tier (cooldown: {2.5 - (now - session.last_query_time):.1f}s).")
                            
            elif "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "transcript":
                    speaker = data.get("speaker") or "Representative"
                    text = data.get("text", "").strip()
                    
                    if text and not is_duplicate_transcript(session, text, speaker):
                        # Track last speaker
                        session.last_speaker = speaker
                        
                        # Localize speaker names to Russian for UI (Default PC audio is Менеджер)
                        display_speaker = "Менеджер" if speaker == "Representative" else "Клиент"
                        line = f"{display_speaker}: {text}"
                        session.transcript_buffer.append(line)
                        session.pending_gemini_query = True
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
    finally:
        # Broadcast disconnection to UI clients
        payload_disc = {
            "type": "bot_status",
            "connected": False,
            "message": "Бот отключился от трансляции аудио."
        }
        for client in list(session.connected_clients):
            try:
                await client.send_json(payload_disc)
            except Exception:
                pass

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
