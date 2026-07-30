from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class JsonlCache:
    """Append-only JSONL cache keyed by selected metadata fields."""

    def __init__(self, path: str | Path, key_fields: Iterable[str]):
        self.path = Path(path)
        self.key_fields = tuple(key_fields)
        self._records: dict[tuple[str, ...], dict[str, Any]] = {}
        self._load()

    def _make_key(self, key_data: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(key_data.get(field, "")) for field in self.key_fields)

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key_data = obj.get("key", {})
                value = obj.get("value", {})
                self._records[self._make_key(key_data)] = value

    def get(self, key_data: dict[str, Any]) -> dict[str, Any] | None:
        value = self._records.get(self._make_key(key_data))
        return dict(value) if value is not None else None

    def set(self, key_data: dict[str, Any], value: dict[str, Any]) -> None:
        key = self._make_key(key_data)
        self._records[key] = dict(value)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key_data, "value": value}, ensure_ascii=False) + "\n")

    def __contains__(self, key_data: dict[str, Any]) -> bool:
        return self._make_key(key_data) in self._records
