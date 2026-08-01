"""CLI for the four-denominator executable paper coverage baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yolo_agent.research.executable_coverage import (
    ExecutablePaperCoverageAuditor,
    method_coverage_file_hash,
)
from yolo_agent.research.executable_coverage_inputs import (
    load_live_coverage_inputs,
    load_method_coverage,
    load_snapshot_coverage_inputs,
)
from yolo_agent.research.executable_coverage_report import (
    write_executable_coverage_artifacts,
)
from yolo_agent.research.snapshot import load_research_snapshot


def build_executable_coverage_baseline(
    *,
    snapshot: Path | str | None = None,
    research_root: Path | str = "research",
    method_coverage: Path | str | None = None,
    maturity_registry: Path | str = "runs/component_maturity_registry.yaml",
):
    """Build from one frozen snapshot or explicit live inputs."""
    if method_coverage is not None:
        inputs = load_live_coverage_inputs(
            method_coverage_path=method_coverage,
            maturity_registry_path=maturity_registry,
        )
    else:
        resolved = load_research_snapshot(research_root, snapshot)
        if resolved is None:
            raise ValueError(
                "research snapshot is unavailable; build one before auditing coverage"
            )
        _, snapshot_dir = resolved
        inputs = load_snapshot_coverage_inputs(snapshot_dir)
    report = load_method_coverage(inputs)
    return ExecutablePaperCoverageAuditor(
        contracts=inputs.contracts,
        maturity=inputs.maturity,
    ).build(
        report,
        source_method_coverage_hash=method_coverage_file_hash(
            inputs.method_coverage_path
        ),
        source_taxonomy_hash=method_coverage_file_hash(
            inputs.taxonomy_path
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit executable paper coverage with four explicit denominators."
    )
    parser.add_argument("--research-root", type=Path, default=Path("research"))
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument(
        "--method-coverage",
        type=Path,
        help="Explicit live paper_method_coverage.yaml; otherwise use a snapshot.",
    )
    parser.add_argument(
        "--maturity-registry",
        type=Path,
        default=Path("runs/component_maturity_registry.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/coverage_baseline.yaml"),
    )
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    report = build_executable_coverage_baseline(
        snapshot=args.snapshot,
        research_root=args.research_root,
        method_coverage=args.method_coverage,
        maturity_registry=args.maturity_registry,
    )
    markdown = args.markdown or args.output.with_suffix(".md")
    write_executable_coverage_artifacts(
        report,
        yaml_path=args.output,
        markdown_path=markdown,
    )
    print("Executable Paper Coverage Baseline")
    print("----------------------------------")
    for name in (
        "all_papers",
        "yolo26_compatible_papers",
        "adaptable_component_papers",
        "exact_reproduction_candidates",
    ):
        print(f"{name}: {report.denominators[name].paper_count}")
    print(f"reusable_adapter_papers: {report.reusable_adapter_paper_count}")
    print(f"runtime_ready_papers: {report.runtime_ready_paper_count}")
    print(f"yaml: {args.output}")
    print(f"markdown: {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_executable_coverage_baseline", "main"]
