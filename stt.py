import os
import threading
import torch
from typing import Optional

try:
    import gigaam
except ImportError:
    gigaam = None

class GigaAMProcessor:
    def __init__(self, model_name: str = "v2_ctc"):
        print(f"[STT] Initializing Sber GigaAM model '{model_name}' (In-Memory RAM Decoder)...")
        if gigaam is None:
            raise ImportError("gigaam package is not installed.")
        self.model = gigaam.load_model(model_name)
        self._lock = threading.Lock()
        print(f"[STT] Sber GigaAM '{model_name}' in-memory decoder loaded successfully!")

    @torch.inference_mode()
    def transcribe(self, pcm_bytes: bytes) -> str:
        """
        Transcribes PCM 16-bit 16kHz mono audio bytes directly in RAM (~15-25ms latency).
        No temporary files, no disk I/O, no ffmpeg subprocess calls!
        """
        if not pcm_bytes or len(pcm_bytes) < 3200:
            return ""
        
        try:
            with self._lock:
                # Convert raw 16-bit PCM bytes directly into float32 Tensor normalized in [-1.0, 1.0]
                wav_tensor = torch.frombuffer(pcm_bytes, dtype=torch.int16).float() / 32768.0
                wav = wav_tensor.to(self.model._device).to(self.model._dtype).unsqueeze(0)
                length = torch.full([1], wav.shape[-1], device=self.model._device)

                # Direct in-memory forward pass & CTC decode
                encoded, encoded_len = self.model.forward(wav, length)
                decoded = self.model.decoding.decode(self.model.head, encoded, encoded_len)
            
            if decoded and len(decoded) > 0:
                text = str(decoded[0]).strip()
                return text
            return ""
        except Exception as e:
            print(f"[STT] In-Memory GigaAM transcription error: {e}")
            return ""

# Global singleton instance
_stt_instance: Optional[GigaAMProcessor] = None

def get_stt_processor() -> GigaAMProcessor:
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = GigaAMProcessor(model_name="v2_ctc")
    return _stt_instance
