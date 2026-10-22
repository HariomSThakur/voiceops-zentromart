"""Gnani-only speech adapters for STT and TTS."""

from __future__ import annotations

import os
import struct
from pathlib import Path

from .settings import Settings


def _finalize_wav_header(audio: bytes) -> bytes:
    """Replace streaming WAV size sentinels with the completed payload sizes."""
    if len(audio) < 12 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        return audio
    if len(audio) - 8 > 0xFFFFFFFF:
        raise ValueError("The synthesized WAV exceeds the RIFF size limit.")

    finalized = bytearray(audio)
    struct.pack_into("<I", finalized, 4, len(finalized) - 8)
    offset = 12
    while offset + 8 <= len(finalized):
        chunk_id = finalized[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", finalized, offset + 4)[0]
        if chunk_id == b"data" and chunk_size == 0xFFFFFFFF:
            actual_size = len(finalized) - offset - 8
            struct.pack_into("<I", finalized, offset + 4, actual_size)
            break
        if chunk_size == 0xFFFFFFFF:
            break
        offset += 8 + chunk_size + (chunk_size & 1)
    return bytes(finalized)


class GnaniSpeech:
    def __init__(self, settings: Settings) -> None:
        api_key = os.getenv("GNANI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Set GNANI_API_KEY in .env before running the voice pipeline.")
        try:
            from gnani.stt import GnaniSTTClient
            from gnani.tts import GnaniTTSClient, AudioConfig
        except ImportError as exc:
            raise RuntimeError(
                "Gnani SDK is missing. Install requirements.txt (gnani-vachana)."
            ) from exc
        self.settings = settings
        self.stt = GnaniSTTClient(api_key=api_key)
        self.tts = GnaniTTSClient(api_key=api_key)
        self.audio_config = AudioConfig(container="wav", encoding="linear_pcm", sample_rate=48000)

    def transcribe(self, audio_path: Path, language_code: str | None = None) -> str:
        result = self.stt.transcribe(str(audio_path), language_code=language_code or self.settings.stt_language)
        if isinstance(result, dict):
            return str(result.get("transcript", "") or "").strip()
        return str(getattr(result, "transcript", "") or "").strip()

    def synthesize(self, text: str, output_path: Path, language: str | None = None) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio = self.tts.synthesize(
            text,
            voice=self.settings.tts_voice,
            model=self.settings.tts_model,
            language=language or self.settings.tts_language,
            audio_config=self.audio_config,
        )
        output_path.write_bytes(_finalize_wav_header(audio))
