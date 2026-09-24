"""Shared inventory helpers for the documentation-consistency tests."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every documentation file the consistency tests cover, sorted for stable output.
DOC_PATHS: tuple[Path, ...] = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "README.zh-CN.md",
    *sorted((REPO_ROOT / "docs").rglob("*.md")),
)

_FENCE_RE = re.compile(r"^\s*```")


def read_doc(path: Path) -> str:
    """Read a documentation file, tolerating the UTF-8 BOM used by zh docs."""
    return path.read_text(encoding="utf-8-sig")


def iter_fenced_blocks(text: str) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(language, lines)`` for every fenced code block in the text."""
    language: str | None = None
    block: list[str] = []
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            if language is None:
                language = line.strip()[3:].strip().lower()
                block = []
            else:
                yield language, block
                language = None
                block = []
            continue
        if language is not None:
            block.append(line)
