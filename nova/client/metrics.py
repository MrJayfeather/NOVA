import json
import time
from pathlib import Path


class Metrics:
    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, **fields) -> None:
        record = {"ts": time.time(), "kind": kind, **fields}
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
