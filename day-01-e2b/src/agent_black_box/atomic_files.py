from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4


def write_text_atomic(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    data = content.encode(encoding)
    write_bytes_atomic(path, data)


def write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with tmp_path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
