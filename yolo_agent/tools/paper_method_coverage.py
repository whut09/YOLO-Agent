"""Generate paper-method to adapter coverage from a frozen or local registry."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yolo_agent.research.method_profiles import PaperMethodProfileBuilder
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.snapshot import ResearchSnapshot
from yolo_agent.research.component_aliases import ComponentAliasResolver


def build_report(
    *,
    root: Path | str = "research",
    snapshot: Path | str | None = None,
):
    """Build coverage without changing papers, contracts, or maturity state."""
    registry_root = Path(snapshot) if snapshot is not None else Path(root)
    snapshot_hash = None
    if snapshot is not None:
        manifest = ResearchSnapshot.from_snapshot_dir(registry_root)
        failures = manifest.verify(registry_root)
        if failures:
            raise ValueError("invalid research snapshot: " + "; ".join(failures))
        snapshot_hash = manifest.snapshot_hash
    papers = PaperRegistry(registry_root).list()
    report = PaperMethodProfileBuilder(
        ComponentAliasResolver.from_yaml()
    ).build(papers)
    return report.model_copy(update={"snapshot_hash": snapshot_hash})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate paper-to-adapter MethodProfile coverage."
    )
    parser.add_argument("--root", type=Path, default=Path("research"))
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Use a verified frozen ResearchSnapshot directory instead of live research.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/paper-method-coverage.yaml"),
    )
    args = parser.parse_args(argv)
    report = build_report(root=args.root, snapshot=args.snapshot)
    report.to_yaml(args.report, exclude_none=True, sort_keys=False)
    print(f"Generated paper method coverage report: {args.report}")
    coverage = report.compatible_mechanism_coverage
    print(
        "Mechanisms: "
        f"referenced={coverage.referenced_mechanism_count} "
        f"adaptable={coverage.potentially_adaptable_mechanism_count} "
        f"reusable_adapter={coverage.reusable_adapter_mechanism_count} "
        f"runtime_ready={coverage.runtime_ready_mechanism_count}"
    )
    print(
        "Coverage: "
        f"adapter={coverage.compatible_adapter_coverage_ratio:.1%} "
        f"runtime_ready={coverage.runtime_ready_coverage_ratio:.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_report", "main"]
