"""Markdown and catalog ingestion plus a small persistent SQLite vector index."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from .llm import ModelGateway
from .settings import Settings


SECTION_RE = re.compile(r"^#{1,6}\s+\[([A-Za-z0-9_.:-]+)\]\s*(.*)$")
FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


@dataclass
class Chunk:
    chunk_id: str
    source: str
    section_id: str
    doc_type: str
    priority: int
    status: str
    effective_from: str
    content: str


@dataclass
class RetrievedChunk(Chunk):
    score: float


def _parse_front_matter(text: str) -> dict[str, str]:
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return {}
    values: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip().lower()] = value.strip().strip("\"'")
    return values


def _priority(doc_type: str, source: str) -> int:
    typed = doc_type.casefold()
    if typed == "policy":
        return 6
    if typed == "agent_guide":
        return 5
    if typed == "help":
        return 4
    if typed == "faq":
        return 3
    if typed == "marketing":
        return 1
    if typed == "catalog":
        return 5
    if typed == "pincode_data":
        return 4
    return 2 if source.startswith("kb/") else 1


def _stable_id(source: str, section_id: str, ordinal: int, content: str) -> str:
    raw = f"{source}\0{section_id}\0{ordinal}\0{content}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:20]


def _split_long(text: str, limit: int, overlap: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("; ", start, end), text.rfind(" ", start, end))
            if boundary > start + limit // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return pieces


def _make_chunks(source: str, body: str, metadata: dict[str, str], settings: Settings) -> list[Chunk]:
    doc_type = metadata.get("doc_type", "meta")
    status = metadata.get("status", "current").casefold()
    effective = metadata.get("effective_from", "")
    default_section = metadata.get("doc_id", Path(source).stem.upper())
    sections: list[tuple[str, str, str]] = []
    section_id = default_section
    section_title = metadata.get("title", Path(source).stem)
    lines: list[str] = []
    for line in body.splitlines():
        match = SECTION_RE.match(line.strip())
        if match:
            if "\n".join(lines).strip():
                sections.append((section_id, section_title, "\n".join(lines).strip()))
            section_id = match.group(1)
            section_title = match.group(2).strip() or section_id
            lines = []
        else:
            lines.append(line)
    if "\n".join(lines).strip():
        sections.append((section_id, section_title, "\n".join(lines).strip()))
    if not sections:
        sections = [(section_id, section_title, body.strip())]

    result: list[Chunk] = []
    ordinal = 0
    for sid, title, section_body in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_body) if p.strip()]
        collected = f"[{sid}] {title}\n"
        for paragraph in paragraphs:
            for fragment in _split_long(paragraph, settings.chunk_chars, settings.chunk_overlap):
                if len(collected) + len(fragment) + 2 > settings.chunk_chars and collected.strip():
                    content = collected.strip()
                    result.append(Chunk(
                        chunk_id=_stable_id(source, sid, ordinal, content),
                        source=source, section_id=sid, doc_type=doc_type,
                        priority=_priority(doc_type, source), status=status,
                        effective_from=effective, content=content,
                    ))
                    ordinal += 1
                    carry = content[-settings.chunk_overlap:] if settings.chunk_overlap else ""
                    collected = f"[{sid}] {title}\n{carry}\n" if carry else f"[{sid}] {title}\n"
                collected += fragment + "\n"
        if collected.strip():
            content = collected.strip()
            result.append(Chunk(
                chunk_id=_stable_id(source, sid, ordinal, content),
                source=source, section_id=sid, doc_type=doc_type,
                priority=_priority(doc_type, source), status=status,
                effective_from=effective, content=content,
            ))
            ordinal += 1
    return result


def load_corpus(settings: Settings) -> list[Chunk]:
    """Read current KB sections plus factual catalog/serviceability rows."""
    kb_dir = settings.dataset_dir / "kb"
    if not kb_dir.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at {kb_dir}. Extract the dataset as described in README.md."
        )
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8-sig")
        metadata = _parse_front_matter(text)
        body_match = FRONT_MATTER_RE.match(text)
        body = text[body_match.end():] if body_match else text
        chunks.extend(_make_chunks(f"kb/{path.name}", body, metadata, settings))

    data_dir = settings.dataset_dir / "data"
    catalog_specs = (
        ("products.csv", "sku", "catalog"),
        ("pincodes.csv", "pincode", "pincode_data"),
    )
    for filename, key_name, doc_type in catalog_specs:
        path = data_dir / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = row.get(key_name, "").strip()
                if not key:
                    continue
                lines = [f"{name}: {value}" for name, value in row.items() if value not in (None, "")]
                content = f"[{filename}:{key}]\n" + "\n".join(lines)
                source = f"data/{filename}"
                chunks.append(Chunk(
                    chunk_id=_stable_id(source, f"{filename}:{key}", 0, content),
                    source=source, section_id=f"{filename}:{key}", doc_type=doc_type,
                    priority=_priority(doc_type, source), status="current",
                    effective_from="", content=content,
                ))
    return chunks


def _is_active(chunk: Chunk, as_of_date: date) -> bool:
    if any(word in chunk.status.casefold() for word in ("superseded", "retired", "obsolete", "archived")):
        return False
    if chunk.effective_from:
        try:
            if date.fromisoformat(chunk.effective_from[:10]) > as_of_date:
                return False
        except ValueError:
            pass
    return True


class KnowledgeIndex:
    def __init__(self, settings: Settings, gateway: ModelGateway):
        self.settings = settings
        self.gateway = gateway
        self.path = settings.index_path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, source TEXT NOT NULL, section_id TEXT NOT NULL,
                doc_type TEXT NOT NULL, priority INTEGER NOT NULL, status TEXT NOT NULL,
                effective_from TEXT NOT NULL, content TEXT NOT NULL, embedding TEXT NOT NULL
            )"""
        )
        return connection

    def build(self) -> int:
        corpus = [chunk for chunk in load_corpus(self.settings) if _is_active(chunk, self.settings.as_of_date)]
        if not corpus:
            raise RuntimeError("No current knowledge chunks were found to index.")
        vectors: list[list[float]] = []
        batch_size = 64
        for start in range(0, len(corpus), batch_size):
            vectors.extend(self.gateway.embed([c.content for c in corpus[start:start + batch_size]]))
        if len(vectors) != len(corpus):
            raise RuntimeError("Embedding service returned a different number of vectors than inputs.")
        with self._connect() as db:
            db.execute("DELETE FROM chunks")
            db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(
                    c.chunk_id, c.source, c.section_id, c.doc_type, c.priority,
                    c.status, c.effective_from, c.content, json.dumps(v),
                ) for c, v in zip(corpus, vectors)],
            )
        return len(corpus)

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with self._connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not self.path.exists() or self.count() == 0:
            raise RuntimeError("Vector index is empty. Run `python run.py index` first.")
        query_vector = self.gateway.embed([query])[0]
        with self._connect() as db:
            rows = db.execute(
                "SELECT chunk_id, source, section_id, doc_type, priority, status, effective_from, content, embedding FROM chunks"
            ).fetchall()
        ranked: list[RetrievedChunk] = []
        qnorm = sum(x * x for x in query_vector) ** 0.5 or 1.0
        query_terms = set(re.findall(r"[a-z0-9]+", query.casefold()))
        for row in rows:
            (chunk_id, source, section_id, doc_type, priority, status,
             effective_from, content, encoded_vector) = row
            chunk = Chunk(chunk_id, source, section_id, doc_type, priority, status, effective_from, content)
            if not _is_active(chunk, self.settings.as_of_date):
                continue
            vector = json.loads(encoded_vector)
            denom = qnorm * (sum(x * x for x in vector) ** 0.5 or 1.0)
            cosine = sum(a * b for a, b in zip(query_vector, vector)) / denom
            content_terms = set(re.findall(r"[a-z0-9]+", content.casefold()))
            overlap = len(query_terms & content_terms) / max(1, len(query_terms))
            # Retrieval remains semantic-first; this small lexical signal helps exact SKU/section queries.
            score = cosine + 0.08 * overlap + 0.006 * priority
            ranked.append(RetrievedChunk(**asdict(chunk), score=score))
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:(top_k or self.settings.top_k)]
