import math
import struct
from typing import Dict, Any

class VADProcessor:
    """
    Lightweight energy-based Voice Activity Detection (VAD) processor.
    Processes mono 16-bit PCM audio chunks (default 16kHz).
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
                self.speech_duration_ms += self.frame_duration_ms
                self.silence_duration_ms = 0
        else:
            self.consecutive_speech_frames = 0
            self.consecutive_silence_frames += 1
            
            if self.is_speaking and self.consecutive_silence_frames >= self.silence_confirm_frames:
                self.is_speaking = False
                speech_ended = True
                self.silence_duration_ms = self.consecutive_silence_frames * self.frame_duration_ms
            
            if not self.is_speaking:
                self.silence_duration_ms += self.frame_duration_ms
                self.speech_duration_ms = 0
                
        return {
            "energy": energy,
            "is_speaking": self.is_speaking,
            "speech_started": speech_started,
            "speech_ended": speech_ended,
            "silence_duration_ms": self.silence_duration_ms,
            "speech_duration_ms": self.speech_duration_ms
        }
