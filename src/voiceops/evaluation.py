"""Practice-set runner with transparent, lightweight answer checks."""

from __future__ import annotations

import csv
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .agent import AgentResult, SupportAgent
from .database import normalize_digits
from .settings import Settings


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class EvaluationProgress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0
        self.started_at = time.perf_counter()
        width = 24
        print(
            f"[{ '-' * width }] 0/{self.total} (0%) "
            "| Evaluation started; ETA appears after the first response",
            flush=True,
        )

    def advance(self, label: str) -> None:
        self.completed += 1
        elapsed = time.perf_counter() - self.started_at
        average_seconds = elapsed / self.completed
        remaining = max(0, self.total - self.completed)
        eta = average_seconds * remaining
        fraction = self.completed / self.total
        width = 24
        filled = min(width, (width * self.completed + self.total - 1) // self.total)
        percent = min(100, (100 * self.completed + self.total - 1) // self.total)
        bar = "#" * filled + "-" * (width - filled)
        line = (
            f"[{bar}] {self.completed}/{self.total} ({percent}%) "
            f"| {label} | elapsed {_format_duration(elapsed)} "
            f"| ETA {_format_duration(eta)}"
        )
        print("\r" + line.ljust(140), end="\n" if self.completed >= self.total else "", flush=True)


def _normal(text: str) -> str:
    text = normalize_digits(text).casefold()
    return " ".join(re.findall(r"[\w]+", text, flags=re.UNICODE))


def _split_facts(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s*;\s*", text or "") if part.strip()]


def _fact_present(fact: str, normalized_answer: str) -> bool:
    for alternative in re.split(r"\s+/\s*", fact):
        normalized_fact = _normal(alternative)
        if normalized_fact and normalized_fact in normalized_answer:
            return True
    return False


def _check_answer(row: dict[str, str], result: AgentResult) -> dict[str, Any]:
    required = _split_facts(row.get("must_include", ""))
    forbidden = _split_facts(row.get("must_not_include", ""))
    normalized_answer = _normal(result.answer)
    required_ok = [fact for fact in required if _fact_present(fact, normalized_answer)]
    forbidden_hit = [fact for fact in forbidden if _fact_present(fact, normalized_answer)]
    gold = [part.strip() for part in re.split(r"[;,|]", row.get("gold_evidence", "")) if part.strip()]
    retrieved = {str(source.get("section_id", "")).casefold() for source in result.sources}
    evidence_hits = [item for item in gold if item.casefold() in retrieved]
    return {
        "must_include_total": len(required),
        "must_include_passed": len(required_ok),
        "must_include_complete": len(required) == len(required_ok),
        "must_not_include_violations": forbidden_hit,
        "evidence_total": len(gold),
        "evidence_hits": evidence_hits,
        "evidence_hit": bool(evidence_hits) if gold else None,
        "expected_decision": row.get("expected_decision", ""),
        "decision_requires_manual_review": True,
    }


def _evaluate_question_case(
    agent: SupportAgent,
    task_index: int,
    row: dict[str, str],
    mode: str,
    input_kind: str,
) -> dict[str, Any]:
    text = row.get(input_kind, "").strip()
    session_id = (
        f"evaluation:question:{task_index}:{mode}:"
        f"{row.get('id', 'unknown')}:{input_kind}"
    )
    try:
        result = agent.respond(text, session_id=session_id, mode=mode)
        return {
            "case_type": "question",
            "case_id": row.get("id", ""),
            "input_kind": input_kind,
            "mode": mode,
            "input": result.recorded_user_text,
            "answer": result.answer,
            "retrieved_sources": result.sources,
            "tool_actions": result.actions,
            "checks": _check_answer(row, result),
        }
    finally:
        agent.reset(session_id)


def _evaluate_questions(
    agent: SupportAgent,
    rows: list[dict[str, str]],
    output: list[dict[str, Any]],
    modes: list[str],
    include_asr: bool,
    progress: EvaluationProgress,
    workers: int,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    tasks = [
        (row, mode, input_kind)
        for mode in modes
        for row in rows
        for input_kind in (["question", "asr_text"] if include_asr else ["question"])
        if row.get(input_kind, "").strip()
    ]

    if workers == 1:
        for index, (row, mode, input_kind) in enumerate(tasks):
            record = _evaluate_question_case(agent, index, row, mode, input_kind)
            output.append(record)
            if checkpoint:
                checkpoint(record)
            progress.advance(f"question {row.get('id', 'unknown')} ({mode})")
        return

    ordered: list[dict[str, Any] | None] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="voiceops-eval") as executor:
        futures = {
            executor.submit(_evaluate_question_case, agent, index, row, mode, input_kind): (
                index, row, mode
            )
            for index, (row, mode, input_kind) in enumerate(tasks)
        }
        for future in as_completed(futures):
            index, row, mode = futures[future]
            record = future.result()
            ordered[index] = record
            if checkpoint:
                checkpoint(record)
            progress.advance(f"question {row.get('id', 'unknown')} ({mode})")
    output.extend(record for record in ordered if record is not None)


def _evaluate_conversations(
    agent: SupportAgent,
    rows: list[dict[str, str]],
    output: list[dict[str, Any]],
    progress: EvaluationProgress,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.get("conversation_id", "unknown"), []).append(row)
    for conversation_id, turns in groups.items():
        session_id = f"evaluation:conversation:{conversation_id}"
        for row in sorted(turns, key=lambda item: int(item.get("turn", "0") or 0)):
            result = agent.respond(row.get("customer_message", ""), session_id=session_id, mode="rag")
            checks = _check_answer(row, result)
            checks["expected_tool_names"] = sorted(set(re.findall(r"\b(lookup_order|create_ticket)\b", row.get("expected_tool_calls", ""))))
            record = {
                "case_type": "conversation_turn",
                "conversation_id": conversation_id,
                "turn": row.get("turn", ""),
                "mode": "rag",
                "input": result.recorded_user_text,
                "answer": result.answer,
                "retrieved_sources": result.sources,
                "tool_actions": result.actions,
                "checks": checks,
                "scenario": row.get("scenario", ""),
            }
            output.append(record)
            if checkpoint:
                checkpoint(record)
            progress.advance(
                f"conversation {conversation_id}, turn {row.get('turn', 'unknown')}"
            )
        agent.reset(session_id)


def run_evaluation(
    settings: Settings,
    agent: SupportAgent,
    modes: list[str] | None = None,
    include_asr: bool = False,
    workers: int = 4,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    practice_dir = settings.dataset_dir / "practice"
    questions_path = practice_dir / "questions.csv"
    conversations_path = practice_dir / "conversations.csv"
    if not questions_path.exists() or not conversations_path.exists():
        raise FileNotFoundError(f"Practice CSVs not found under {practice_dir}")
    with questions_path.open(encoding="utf-8-sig", newline="") as stream:
        questions = list(csv.DictReader(stream))
    with conversations_path.open(encoding="utf-8-sig", newline="") as stream:
        conversations = list(csv.DictReader(stream))

    selected_modes = modes or ["prompt-only", "rag"]
    records: list[dict[str, Any]] = []
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = settings.output_dir / "evaluation.jsonl"
    result_path.write_text("", encoding="utf-8")
    checkpoint_lock = threading.Lock()

    def checkpoint(record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with checkpoint_lock, result_path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()

    question_runs = sum(
        1
        for _mode in selected_modes
        for row in questions
        for input_kind in (["question", "asr_text"] if include_asr else ["question"])
        if row.get(input_kind, "").strip()
    )
    progress = EvaluationProgress(question_runs + len(conversations))
    _evaluate_questions(
        agent,
        questions,
        records,
        selected_modes,
        include_asr,
        progress,
        workers,
        checkpoint,
    )
    _evaluate_conversations(agent, conversations, records, progress, checkpoint)
    result_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    summary = {
        "question_rows": len(questions),
        "conversation_turns": len(conversations),
        "modes": selected_modes,
        "include_asr_text": include_asr,
        "records": len(records),
        "result_file": str(result_path),
        "note": "Automated checks are string/evidence heuristics; review decisions and language quality manually.",
    }
    (settings.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
