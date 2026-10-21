"""Dataset helper adapter and fixed-date business-day utilities."""

from __future__ import annotations

import importlib.util
import re
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .settings import Settings


def normalize_digits(text: str) -> str:
    result: list[str] = []
    for char in text:
        try:
            result.append(str(unicodedata.digit(char)))
        except (TypeError, ValueError):
            result.append(char)
    return "".join(result)


def find_order_id(text: str) -> str | None:
    normalized = normalize_digits(text)
    match = re.search(r"(?<!\d)(\d{4}[\s-]?\d{4})(?!\d)", normalized)
    return re.sub(r"\D", "", match.group(1)) if match else None


def find_verification_value(text: str, order_id: str | None = None) -> str | None:
    normalized = normalize_digits(text)
    # Remove the whole order ID first so a spaced ID such as "8978 2837"
    # cannot be misread as two separate four-digit verification values.
    if order_id:
        for match in list(re.finditer(r"(?<!\d)\d{4}[\s-]?\d{4}(?!\d)", normalized)):
            if re.sub(r"\D", "", match.group(0)) == order_id:
                normalized = normalized[:match.start()] + (" " * (match.end() - match.start())) + normalized[match.end():]
    # Do not interpret an order number as authentication data.
    candidates = re.findall(r"(?<!\d)(?:\d{4}|\d{6})(?!\d)", normalized)
    candidates = [c for c in candidates if c != order_id]
    return candidates[-1] if candidates else None


class DatasetTools:
    """Loads the supplied offline tools without copying or modifying their source."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._module: Any | None = None
        self._load_error: str | None = None

    def _load(self) -> Any | None:
        if self._module is not None:
            return self._module
        if self._load_error:
            return None
        path = self.settings.dataset_dir / "tools" / "tools.py"
        if not path.exists():
            self._load_error = f"Dataset tools not found at {path}"
            return None
        try:
            spec = importlib.util.spec_from_file_location("voiceops_capstone_dataset_tools", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module from {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._module = module
            return module
        except Exception as exc:  # surfaced as a safe tool error, not a private traceback
            self._load_error = f"Could not load supplied dataset tools: {type(exc).__name__}"
            return None

    def lookup_order(self, order_id: str, verification_value: str | None = None) -> dict[str, Any]:
        module = self._load()
        if module is None:
            return {"found": False, "verified": False, "error": self._load_error or "Dataset tools unavailable."}
        try:
            return module.lookup_order(order_id, verification_value=verification_value)
        except Exception as exc:
            return {"found": False, "verified": False, "error": f"Order lookup failed: {type(exc).__name__}"}

    def create_ticket(self, **fields: Any) -> dict[str, Any]:
        module = self._load()
        if module is None:
            return {"created": False, "error": self._load_error or "Dataset tools unavailable."}
        try:
            return module.create_ticket(**fields)
        except Exception as exc:
            return {"created": False, "error": f"Ticket creation failed: {type(exc).__name__}"}


class Calendar:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.holidays: set[date] = set()
        path = settings.dataset_dir / "data" / "holidays.csv"
        if path.exists():
            import csv
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    try:
                        self.holidays.add(date.fromisoformat(row.get("date", "")))
                    except ValueError:
                        continue

    def add_business_days(self, start: date, count: int) -> date:
        """Return the date after `count` business days; Sunday/holidays do not count."""
        if count < 0:
            raise ValueError("Business-day count must be non-negative.")
        current = start
        elapsed = 0
        while elapsed < count:
            current += timedelta(days=1)
            if current.weekday() != 6 and current not in self.holidays:
                elapsed += 1
        return current

    def reference_date(self) -> date:
        return self.settings.as_of_date
