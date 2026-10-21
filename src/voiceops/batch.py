"""Batch audio runner used by the submission command."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .agent import SupportAgent
from .settings import Settings
from .speech import GnaniSpeech


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".amr"}


def _output_base(relative: Path) -> Path:
    return Path(*[re.sub(r"[^A-Za-z0-9._-]+", "_", part) for part in relative.parts]).with_suffix("")


def run_batch(
    settings: Settings,
    agent: SupportAgent,
    speech: GnaniSpeech,
    audio_dir: Path,
    output_dir: Path,
    language_code: str | None = None,
) -> dict[str, Any]:
    audio_dir = audio_dir.resolve()
    if not audio_dir.is_dir():
        raise NotADirectoryError(f"Audio input directory does not exist: {audio_dir}")
    files = sorted(
        (p for p in audio_dir.rglob("*") if p.is_file() and p.suffix.casefold() in AUDIO_EXTENSIONS),
        key=lambda p: str(p.relative_to(audio_dir)).casefold(),
    )
    if not files:
        raise FileNotFoundError(f"No supported audio files were found under {audio_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures = 0
    for audio_path in files:
        relative = audio_path.relative_to(audio_dir)
        # Files in one subfolder are successive turns; the root folder is one session.
        session_name = str(relative.parent) if str(relative.parent) != "." else "root"
        session_id = f"batch:{session_name}"
        base = _output_base(relative)
        try:
            heard = speech.transcribe(audio_path, language_code=language_code)
            if heard:
                result = agent.respond(heard, session_id=session_id, mode="rag")
                answer = result.answer
                recorded_heard = result.recorded_user_text
                sources = result.sources
                actions = result.actions
            else:
                answer = "Sorry, I didn't catch that. Could you say it again?"
                recorded_heard = ""
                sources = []
                actions = []
            safe_relative = _output_base(relative)
            text_path = output_dir / safe_relative.with_suffix(".json")
            audio_out_path = output_dir / safe_relative.with_suffix(".wav")
            status = "ok"
            tts_error = None
            try:
                speech.synthesize(answer, audio_out_path)
            except Exception as exc:
                status = "tts_error"
                tts_error = f"{type(exc).__name__}: {exc}"
                failures += 1
            record = {
                "input_file": str(relative),
                "session_id": session_id,
                "status": status,
                "transcript": recorded_heard,
                "reply_text": answer,
                "reply_audio": str(audio_out_path.relative_to(output_dir)) if audio_out_path.exists() else None,
                "retrieved_sources": sources,
                "tool_actions": actions,
                "tts_error": tts_error,
            }
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            failures += 1
            record = {
                "input_file": str(relative),
                "session_id": session_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "retrieved_sources": [],
                "tool_actions": [],
            }
        records.append(record)

    for session_id in {row.get("session_id", "") for row in records}:
        agent.sessions.pop(session_id, None)

    jsonl_path = output_dir / "batch_results.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    summary = {
        "input_directory": str(audio_dir),
        "file_count": len(files),
        "succeeded": sum(row.get("status") == "ok" for row in records),
        "failures": failures,
        "results_file": str(jsonl_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
