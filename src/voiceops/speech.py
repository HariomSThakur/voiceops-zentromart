"""Gnani-only speech adapters for STT and TTS."""

from __future__ import annotations

import os
from pathlib import Path

from .settings import Settings


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
        output_path.write_bytes(audio)
