"""Track the key package paths that docs/architecture.md depends on.

The YAML manifest lists, per architecture area, the package paths the document
describes. The tool snapshots the tracked Python file set per area; when the
set changes the check fails with an "architecture review required" message so a
human reviews the document and regenerates the snapshot. The tool never writes
architecture content itself.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

import yaml

SOURCES_PATH = Path("docs/ARCHITECTURE_SOURCES.yaml")


def load_spec(root: Path) -> dict:
    return yaml.safe_load((root / SOURCES_PATH).read_text(encoding="utf-8")) or {}


def collect_files(root: Path, paths: Sequence[str]) -> list[str]:
    """Resolve each path to a sorted list of tracked .py files (relative posix)."""
    files: set[str] = set()
    for raw in paths:
        full = root / raw
        if not full.exists():
            raise FileNotFoundError(f"architecture source path does not exist: {raw}")
        if full.is_file():
            files.add(Path(raw).as_posix())
        else:
            for found in full.rglob("*.py"):
                if "__pycache__" in found.parts:
                    continue
                files.add(found.relative_to(root).as_posix())
    return sorted(files)


def compute_snapshot(files: Sequence[str]) -> str:
    joined = "\n".join(files)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def evaluate(root: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return (recorded snapshots, actual files per area)."""
    spec = load_spec(root)
    recorded = spec.get("snapshot") or {}
    actual_files: dict[str, list[str]] = {}
    for area, config in (spec.get("areas") or {}).items():
        actual_files[area] = collect_files(root, config.get("paths") or [])
    return recorded, actual_files


def check(root: Path) -> list[str]:
    """Return human-readable mismatches; empty list means architecture is fresh."""
    recorded, actual_files = evaluate(root)
    problems: list[str] = []
    for area, files in actual_files.items():
        current = compute_snapshot(files)
        if recorded.get(area) != current:
            problems.append(
                f"architecture review required: key packages for area '{area}' "
                f"changed ({len(files)} tracked files); review docs/architecture.md "
                f"then run `python -m yolo_agent.tools.architecture_sources --update`"
            )
    missing = sorted(set(recorded) - set(actual_files))
    for area in missing:
        problems.append(f"snapshot records unknown area '{area}'")
    return problems


def update(root: Path) -> None:
    spec_path = root / SOURCES_PATH
    spec = load_spec(root)
    spec["snapshot"] = {
        area: compute_snapshot(files)
        for area, files in sorted(evaluate(root)[1].items())
    }
    spec_path.write_text(
        yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m yolo_agent.tools.architecture_sources",
        description="Check or refresh the architecture source snapshot.",
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Recompute the per-area snapshots and write them back to the YAML.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.update:
        update(args.root)
        print("architecture source snapshots updated")
        return 0
    problems = check(args.root)
    if problems:
        for problem in problems:
            print(problem)
        return 1
    print("architecture sources are fresh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
