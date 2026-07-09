import asyncio
import os
import io
import wave
import time
from dotenv import load_dotenv
from google.antigravity import Agent, LocalAgentConfig
import google.antigravity.types as t
import pydantic
from typing import List, Optional
from enum import Enum

# Load API key from local environment or .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

class SPINStage(str, Enum):
    SITUATION = "Ситуация"
    PROBLEM = "Проблема"
    IMPLICATION = "Извлечение"
    NEED_PAYOFF = "Направляющие вопросы"
    OBJECTION = "Обработка возражений"

class SpeakerRole(str, Enum):
    REPRESENTATIVE = "Representative"
    CUSTOMER = "Customer"

class SalesHint(pydantic.BaseModel):
    stage: SPINStage
    detected_objections: List[str]
    suggested_scripts: List[str]
    next_questions: List[str]
    confidence: float
    customer_transcript: Optional[str]
    speaker_role: Optional[SpeakerRole]

def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, sample_width: int = 2) -> bytes:
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return wav_io.getvalue()

async def main():
    wav_path = os.path.join(os.path.dirname(__file__), 'test_speech.wav')
    if not os.path.exists(wav_path):
        print(f"Error: test_speech.wav not found at {wav_path}!")
        print("Run macOS text-to-speech command first or verify the file exists.")
        return
        
    print(f"Loading test audio: {wav_path}...")
    with open(wav_path, 'rb') as f:
        wav_bytes = f.read()
        
    audio_part = t.from_bytes(data=wav_bytes, mime_type="audio/wav")
    
    agent_config = LocalAgentConfig(
        model="gemini-3.1-flash-lite",
        system_instructions="You are a sales assistant. Identify the speaker role, transcribe the audio phrase in customer_transcript field in Russian, and output the SPIN hints.",
        response_schema=SalesHint,
        temperature=0.2
    )
    
    print("API Key loaded:", "Yes" if os.environ.get("GEMINI_API_KEY") else "No")
    print("Initializing Google Antigravity Agent with gemini-3.1-flash-lite...")
    start_time = time.time()
    try:
        async with Agent(config=agent_config) as agent:
            print("Agent session established. Sending multimodal payload...")
            prompt = "Analyze this audio segment from the sales conversation."
            
            # Send both prompt and audio bytes to the agent
            response = await agent.chat([prompt, audio_part])
            print("\nReceived response! Reading raw stream in real-time:")
            print("="*50)
            
            async for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    print(chunk.text, end="", flush=True)
                else:
                    print(f"\n[Non-text chunk]: {chunk}")
            
            print("\n" + "="*50)
            print("Finished reading stream. Getting last structured output...")
            
            data = await response.structured_output()
            end_time = time.time()
            
            print("\n" + "="*50)
            print(f"⏱️  Query completed in {end_time - start_time:.2f} seconds")
            print("="*50)
            print("Parsed Structured Output:")
            if isinstance(data, pydantic.BaseModel):
                for k, v in data.model_dump().items():
                    print(f"  {k}: {v}")
            else:
                print(data)
            print("="*50)
            
    except Exception as e:
        import traceback
        print("\n❌ Error during execution:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
