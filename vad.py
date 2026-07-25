import math
import struct
from typing import Dict, Any

def extract_stereo_channels(chunk_bytes: bytes) -> tuple[bytes, bytes, float, float]:
    """
    De-interleaves 16-bit 2-channel PCM into mono (left_bytes, right_bytes)
    and calculates RMS energy for each channel.
    Left (Channel 0) = Mic (Representative / Manager)
    Right (Channel 1) = Screen Audio (Customer / Nikita)
    """
    if not chunk_bytes or len(chunk_bytes) < 4:
        return chunk_bytes, chunk_bytes, 0.0, 0.0
        
    num_samples = len(chunk_bytes) // 4  # 2 bytes per sample * 2 channels
    if num_samples == 0:
        return chunk_bytes, chunk_bytes, 0.0, 0.0
        
    fmt = f"<{num_samples * 2}h"
    try:
        samples = struct.unpack(fmt, chunk_bytes)
    except Exception:
        return chunk_bytes, chunk_bytes, 0.0, 0.0
        
    left_samples = samples[0::2]
    right_samples = samples[1::2]
    
    left_bytes = struct.pack(f"<{len(left_samples)}h", *left_samples)
    right_bytes = struct.pack(f"<{len(right_samples)}h", *right_samples)
    
    left_rms = math.sqrt(sum(float(s)**2 for s in left_samples) / len(left_samples)) if left_samples else 0.0
    right_rms = math.sqrt(sum(float(s)**2 for s in right_samples) / len(right_samples)) if right_samples else 0.0
    
    return left_bytes, right_bytes, left_rms, right_rms

class VADProcessor:
    """
    Lightweight energy-based Voice Activity Detection (VAD) processor.
    Processes mono or stereo 16-bit PCM audio chunks (default 16kHz).
    """
    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 30, energy_threshold: float = 300.0):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.energy_threshold = energy_threshold
        
        # State tracking
        self.is_speaking = False
        self.silence_duration_ms = 0
        self.speech_duration_ms = 0
        
        # Hysteresis parameters to prevent rapid flickering
        self.speech_confirm_frames = 2
        self.silence_confirm_frames = 5
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0

    def calculate_energy(self, chunk_bytes: bytes) -> float:
        """
        Calculates the Root Mean Square (RMS) energy of a 16-bit PCM audio chunk.
        """
        if not chunk_bytes:
            return 0.0
            
        # 16-bit PCM is 2 bytes per sample
        num_samples = len(chunk_bytes) // 2
        if num_samples == 0:
            return 0.0
            
        # Unpack binary PCM data (short integers)
        fmt = f"<{num_samples}h"  # Little-endian short
        try:
            samples = struct.unpack(fmt, chunk_bytes)
        except Exception:
            # Fallback if bytes length is odd or unpack fails
            return 0.0
            
        sum_squares = sum(float(sample) ** 2 for sample in samples)
        rms = math.sqrt(sum_squares / num_samples)
        return rms

    def process_chunk(self, chunk_bytes: bytes) -> Dict[str, Any]:
        """
        Processes a single chunk of audio bytes and updates VAD state.
        Returns state flags.
        """
        energy = self.calculate_energy(chunk_bytes)
        is_active = energy > self.energy_threshold
        
        # Calculate actual chunk duration dynamically in ms
        num_samples = len(chunk_bytes) // 2
        chunk_duration_ms = int((num_samples / self.sample_rate) * 1000) if self.sample_rate > 0 else 0
        if chunk_duration_ms == 0:
            chunk_duration_ms = self.frame_duration_ms
            
        speech_started = False
        speech_ended = False
        
        if is_active:
            self.consecutive_silence_frames = 0
            self.consecutive_speech_frames += 1
            
            if not self.is_speaking and self.consecutive_speech_frames >= self.speech_confirm_frames:
                self.is_speaking = True
                speech_started = True
                self.speech_duration_ms = 0
            
            if self.is_speaking:
                self.speech_duration_ms += chunk_duration_ms
                self.silence_duration_ms = 0
        else:
            self.consecutive_speech_frames = 0
            self.consecutive_silence_frames += 1
            
            if self.is_speaking and self.consecutive_silence_frames >= self.silence_confirm_frames:
                self.is_speaking = False
                speech_ended = True
                self.silence_duration_ms = self.consecutive_silence_frames * chunk_duration_ms
            
            if not self.is_speaking:
                self.silence_duration_ms += chunk_duration_ms
                self.speech_duration_ms = 0
                
        return {
            "energy": energy,
            "is_speaking": self.is_speaking,
            "speech_started": speech_started,
            "speech_ended": speech_ended,
            "silence_duration_ms": self.silence_duration_ms,
            "speech_duration_ms": self.speech_duration_ms
        }
