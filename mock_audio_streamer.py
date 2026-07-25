import asyncio
import json
import math
import struct
import sys
import websockets

# Dialogue list to simulate a sales call (Localized to Russian)
DIALOGUE = [
    {
        "speaker": "Representative",
        "text": "Здравствуйте! Спасибо, что уделили время на звонок. Как ваши дела?",
        "speech_duration": 3.0,
        "pause_duration": 2.0
    },
    {
        "speaker": "Customer",
        "text": "Приветствую, всё хорошо. Мы сейчас ищем способы повысить эффективность работы нашего отдела продаж.",
        "speech_duration": 4.5,
        "pause_duration": 2.0
    },
    {
        "speaker": "Representative",
        "text": "Отлично! А какие инструменты вы сейчас используете для фиксации звонков и заметок?",
        "speech_duration": 3.5,
        "pause_duration": 2.0
    },
    {
        "speaker": "Customer",
        "text": "Честно говоря, мы ведем всё в таблицах Excel и вручную переносим в CRM. Это наше главное узкое место.",
        "speech_duration": 5.0,
        "pause_duration": 2.0
    },
    {
        "speaker": "Representative",
        "text": "Понимаю. А как эта рутина с ручным вводом влияет на выполнение вашего общего плана продаж?",
        "speech_duration": 4.0,
        "pause_duration": 2.0
    },
    {
        "speaker": "Customer",
        "text": "Каждый менеджер теряет около 5 часов в неделю. Но ваше решение, скорее всего, обойдется нам слишком дорого.",
        "speech_duration": 6.0,
        "pause_duration": 3.0
    }
]

def generate_sine_wave_chunk(duration_ms: int = 30, frequency: float = 440.0, sample_rate: int = 16000, amplitude: float = 2000.0) -> bytes:
    """
    Generates a chunk of mono 16-bit PCM audio containing a sine wave (simulates active speech).
    """
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        sample = int(amplitude * math.sin(2 * math.pi * frequency * t))
        samples.append(sample)
    
    # Pack as little-endian 16-bit signed shorts
    return struct.pack(f"<{num_samples}h", *samples)

def generate_silence_chunk(duration_ms: int = 30, sample_rate: int = 16000) -> bytes:
    """
    Generates a chunk of mono 16-bit PCM audio containing silence (all zeros).
    """
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    return struct.pack(f"<{num_samples}h", *[0] * num_samples)

async def stream_audio():
    session_id = sys.argv[1] if len(sys.argv) > 1 else "default_session"
    uri = f"ws://localhost:8000/audio/{session_id}"
    print(f"Connecting to Zoom Copilot server at {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected successfully. Starting audio simulation...")
            
            for index, turn in enumerate(DIALOGUE):
                speaker = turn["speaker"]
                text = turn["text"]
                speech_sec = turn["speech_duration"]
                pause_sec = turn["pause_duration"]
                
                print(f"\n[Turn {index + 1}] {speaker} is speaking: '{text}'")
                
                # 1. Stream active speech audio chunks
                steps = int(speech_sec / 0.03)
                for _ in range(steps):
                    # Send raw PCM chunk (amplitude > energy threshold)
                    audio_chunk = generate_sine_wave_chunk(duration_ms=30, amplitude=1500.0)
                    await websocket.send(audio_chunk)
                    await asyncio.sleep(0.03)
                
                # 2. Speech finished, send the transcript chunk to simulate STT
                transcript_msg = {
                    "type": "transcript",
                    "speaker": speaker,
                    "text": text
                }
                await websocket.send(json.dumps(transcript_msg))
                print(f"Sent transcription event for {speaker}.")
                
                # 3. Stream silence chunks
                print(f"Pausing for {pause_sec}s...")
                silence_steps = int(pause_sec / 0.03)
                for _ in range(silence_steps):
                    silence_chunk = generate_silence_chunk(duration_ms=30)
                    await websocket.send(silence_chunk)
                    await asyncio.sleep(0.03)
                    
            print("\nSimulation complete. Closing connection.")
    except Exception as e:
        print(f"Error during streaming: {e}")

if __name__ == "__main__":
    asyncio.run(stream_audio())
