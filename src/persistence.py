"""Utilities for reading/writing JSON & JSONL artifacts on disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping
import uuid

from .models import Problem

ENCODING = "utf-8"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class JsonlWriter:
    """Append-only JSONL writer with automatic flush."""

    def __init__(self, path: Path, mode: str = "a", encoding: str = ENCODING):
        self.path = path
        self.mode = mode
        self.encoding = encoding
        self._fh = None

    def __enter__(self) -> "JsonlWriter":
        ensure_parent_dir(self.path)
        self._fh = self.path.open(self.mode, encoding=self.encoding)
        return self

    def write(self, payload: Mapping) -> None:
        if self._fh is None:
            raise RuntimeError("JsonlWriter must be used inside a context")
        json.dump(payload, self._fh, ensure_ascii=False)
        self._fh.write("\n")
        self._fh.flush()

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def write_jsonl(path: Path, records: Iterable[Mapping]) -> None:
    with JsonlWriter(path) as writer:
        for payload in records:
            writer.write(payload)


def read_jsonl(path: Path) -> Iterator[Dict]:
    if not path.exists():
        return
    with path.open("r", encoding=ENCODING) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def atomic_write_json(path: Path, payload: Mapping) -> None:
    ensure_parent_dir(path)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    with tmp_path.open("w", encoding=ENCODING) as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    tmp_path.replace(path)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding=ENCODING) as fh:
        return json.load(fh)


def load_problem(path: Path) -> Problem:
    return Problem.from_json(load_json(path))
