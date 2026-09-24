"""Both READMEs must carry the same generator-managed blocks in sync."""

from __future__ import annotations

import re

from tests.docs_inventory import REPO_ROOT, read_doc
from yolo_agent.tools.docs_status import (
    STATUS_END,
    STATUS_START,
    collect_status,
    render_status_en,
    render_status_zh,
)

MARKER_PAIRS: dict[str, tuple[str, str]] = {
    "status": (STATUS_START, STATUS_END),
    "paper-adapter-coverage": (
        "<!-- paper-adapter-coverage:start -->",
        "<!-- paper-adapter-coverage:end -->",
    ),
    "capability-maturity": (
        "<!-- capability-maturity:start -->",
        "<!-- capability-maturity:end -->",
    ),
}

DOCS_LINK_RE = re.compile(r"\]\((?:\./)?docs/[^)#?]+\)")


def _extract_block(text: str, start: str, end: str) -> str:
    first = text.find(start)
    last = text.find(end)
    assert first != -1, f"missing start marker: {start}"
    assert last != -1, f"missing end marker: {end}"
    return text[first + len(start) : last].strip("\n")


def test_readmes_carry_each_marker_pair_exactly_once() -> None:
    for name in ("README.md", "README.zh-CN.md"):
        text = read_doc(REPO_ROOT / name)
        for label, (start, end) in MARKER_PAIRS.items():
            assert text.count(start) == 1, f"{name}: {label} start marker count != 1"
            assert text.count(end) == 1, f"{name}: {label} end marker count != 1"


def test_readme_status_blocks_match_structured_source() -> None:
    status = collect_status(REPO_ROOT)
    expected = {
        "README.md": render_status_en(status),
        "README.zh-CN.md": render_status_zh(status),
    }
    for name, block in expected.items():
        text = read_doc(REPO_ROOT / name)
        current = _extract_block(text, STATUS_START, STATUS_END)
        assert current == block, f"{name}: generated status block is stale"


def test_readme_document_maps_stay_in_sync() -> None:
    en = read_doc(REPO_ROOT / "README.md")
    zh = read_doc(REPO_ROOT / "README.zh-CN.md")
    assert len(DOCS_LINK_RE.findall(en)) == len(DOCS_LINK_RE.findall(zh)), (
        "English and Chinese READMEs link a different number of docs/ pages"
    )
