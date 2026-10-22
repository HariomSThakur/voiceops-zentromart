"""Create a reproducible submission ZIP without credentials or dataset copies."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "README.md",
    ".env.example",
    "requirements.txt",
    "run.py",
    "report.pdf",
    "demo.mp4",
]
EXCLUDED_PARTS = {".git", ".venv", "venv", ".cache", "dataset", "report", "sample_outputs", "__pycache__", ".pytest_cache"}


def safe_to_package(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.name in {".env", ".env.local"} or path.suffix.casefold() in {".zip", ".pyc", ".pyo"}:
        return False
    if path.name == "tickets_created.csv":
        return False
    return path.is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-name", required=True, help="Team slug used in the output ZIP name.")
    parser.add_argument("--output", type=Path, default=None, help="Optional destination ZIP path.")
    args = parser.parse_args()

    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(
            "Submission is not ready; missing required deliverables: " + ", ".join(missing)
        )
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", args.team_name.strip()).strip("_")
    if not slug:
        raise SystemExit("--team-name must contain at least one letter or digit.")
    output = (args.output or ROOT / f"{slug}_voiceops.zip").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates = sorted(path for path in ROOT.rglob("*") if safe_to_package(path))
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=8) as archive:
        for path in candidates:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"Created {output} ({len(candidates)} files); dataset, credentials, and caches excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
