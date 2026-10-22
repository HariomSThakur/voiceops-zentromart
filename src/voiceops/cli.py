"""Command-line interface for indexing, chat, evaluation, and audio batches."""

from __future__ import annotations

import argparse
from pathlib import Path

from .agent import SupportAgent
from .batch import run_batch
from .database import DatasetTools
from .evaluation import run_evaluation
from .knowledge import KnowledgeIndex
from .llm import ModelGateway
from .settings import load_settings
from .speech import GnaniSpeech


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _runtime():
    settings = load_settings()
    gateway = ModelGateway(settings)
    index = KnowledgeIndex(settings, gateway)
    dataset_tools = DatasetTools(settings)
    agent = SupportAgent(settings, gateway, index, dataset_tools)
    return settings, gateway, index, dataset_tools, agent


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ZentroMart VoiceOps capstone: grounded support chat and Gnani voice batches."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("index", help="Build the current-policy and catalog embedding index.")

    ask = commands.add_parser("ask", help="Ask one text question.")
    ask.add_argument("--text", required=True, help="Customer message.")
    ask.add_argument("--session", default="cli", help="Conversation key for follow-up turns.")
    ask.add_argument("--mode", choices=("rag", "prompt-only"), default="rag")

    batch = commands.add_parser("batch", help="Transcribe, answer, and speak every audio file in a folder.")
    batch.add_argument("--audio-dir", required=True, type=Path, help="Folder of audio files; subfolders group turns into a conversation.")
    batch.add_argument("--output-dir", type=Path, default=None, help="Where JSON/WAV results are saved.")
    batch.add_argument("--language", default=None, help="Gnani STT language code (default: STT_LANGUAGE).")

    evaluate = commands.add_parser("evaluate", help="Run the supplied question and conversation practice sets.")
    evaluate.add_argument("--mode", choices=("all", "rag", "prompt-only"), default="all")
    evaluate.add_argument("--include-asr", action="store_true", help="Also evaluate asr_text variants of single questions.")
    evaluate.add_argument("--output-dir", type=Path, default=None)
    evaluate.add_argument(
        "--workers",
        type=_positive_int,
        default=4,
        help="Concurrent workers for independent question cases (use 1 for sequential execution).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    settings, _gateway, index, _dataset_tools, agent = _runtime()

    if args.command == "index":
        count = index.build()
        print(f"Indexed {count} active knowledge/catalog chunks into {index.path}")
        return 0

    if args.command == "ask":
        if args.mode == "rag" and index.count() == 0:
            raise RuntimeError("The knowledge index is empty. Run `python run.py index` first.")
        result = agent.respond(args.text, session_id=args.session, mode=args.mode)
        print(result.answer)
        if result.sources:
            print("\nRetrieved sections:")
            for source in result.sources:
                print(f"- {source['section_id']} ({source['source']})")
        if result.actions:
            print("\nTool actions:")
            for action in result.actions:
                print(f"- {action}")
        return 0

    if args.command == "batch":
        if index.count() == 0:
            raise RuntimeError("The knowledge index is empty. Run `python run.py index` first.")
        speech = GnaniSpeech(settings)
        output_dir = args.output_dir or settings.output_dir / "batch"
        summary = run_batch(settings, agent, speech, args.audio_dir, output_dir, args.language)
        print(f"Processed {summary['file_count']} files; {summary['failures']} failed.")
        print(f"Results: {summary['results_file']}")
        return 1 if summary["failures"] else 0

    if args.command == "evaluate":
        if args.mode in ("all", "rag") and index.count() == 0:
            raise RuntimeError("The knowledge index is empty. Run `python run.py index` first.")
        if args.output_dir:
            # Keep evaluation outputs in the requested location without changing the environment.
            from dataclasses import replace
            settings = replace(settings, output_dir=args.output_dir.resolve())
            index.settings = settings
            agent.settings = settings
        modes = ["prompt-only", "rag"] if args.mode == "all" else [args.mode]
        summary = run_evaluation(
            settings,
            agent,
            modes=modes,
            include_asr=args.include_asr,
            workers=args.workers,
        )
        print(f"Wrote {summary['records']} records to {summary['result_file']}")
        print(summary["note"])
        return 0

    return 2
