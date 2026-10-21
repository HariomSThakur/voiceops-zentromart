"""Runtime configuration for the capstone project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    PROJECT_ROOT / "data" / "dataset" / "Capstone_Project_Dataset_FTTT_2026"
)


def _path_env(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else default.resolve()


@dataclass(frozen=True)
class Settings:
    project_root: Path
    dataset_dir: Path
    index_path: Path
    output_dir: Path
    as_of_date: date
    llm_api_key: str
    llm_base_url: str | None
    llm_model: str
    embedding_api_key: str
    embedding_base_url: str | None
    embedding_model: str
    top_k: int
    chunk_chars: int
    chunk_overlap: int
    stt_language: str
    tts_language: str
    tts_voice: str
    tts_model: str


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    llm_base = os.getenv("LLM_BASE_URL", "").strip() or None
    embedding_base = os.getenv("EMBEDDING_BASE_URL", "").strip() or llm_base
    try:
        as_of = date.fromisoformat(os.getenv("AS_OF_DATE", "2026-10-20"))
    except ValueError as exc:
        raise ValueError("AS_OF_DATE must use YYYY-MM-DD format.") from exc
    return Settings(
        project_root=PROJECT_ROOT,
        dataset_dir=_path_env("DATASET_DIR", DEFAULT_DATASET),
        index_path=_path_env(
            "INDEX_PATH", PROJECT_ROOT / ".cache" / "voiceops_index.sqlite3"
        ),
        output_dir=_path_env("OUTPUT_DIR", PROJECT_ROOT / "sample_outputs"),
        as_of_date=as_of,
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_base_url=llm_base,
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", "").strip()
        or os.getenv("LLM_API_KEY", "").strip(),
        embedding_base_url=embedding_base,
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        top_k=max(1, int(os.getenv("RETRIEVAL_TOP_K", "6"))),
        chunk_chars=max(300, int(os.getenv("CHUNK_CHARS", "1100"))),
        chunk_overlap=max(0, int(os.getenv("CHUNK_OVERLAP", "140"))),
        stt_language=os.getenv("STT_LANGUAGE", "en-IN").strip(),
        tts_language=os.getenv("TTS_LANGUAGE", "en-IN").strip(),
        tts_voice=os.getenv("TTS_VOICE", "Kaveri").strip(),
        tts_model=os.getenv("TTS_MODEL", "timbre-v2.5").strip(),
    )
