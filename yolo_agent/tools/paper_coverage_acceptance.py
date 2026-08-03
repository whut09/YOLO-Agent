"""CLI for final paper implementation coverage acceptance."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from yolo_agent.components.maturity_registry import (
    ComponentMaturityRegistry,
    adapter_source_hash,
    installed_ultralytics_version,
)
from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.coverage_acceptance import (
    PaperCoverageAcceptanceBuilder,
    PaperCoverageAcceptanceReport,
    file_sha256,
    render_coverage_acceptance_markdown,
)
from yolo_agent.research.executable_coverage_schemas import (
    ExecutablePaperCoverageBaseline,
)
from yolo_agent.research.method_profiles import PaperMethodCoverageReport


def build_final_coverage_acceptance(
    *,
    method_coverage_path: Path | str,
    executable_coverage_path: Path | str,
    maturity_registry_path: Path | str,
) -> PaperCoverageAcceptanceReport:
    method_path = Path(method_coverage_path)
    executable_path = Path(executable_coverage_path)
    registry_path = Path(maturity_registry_path)
    registry = ComponentMaturityRegistry(registry_path)
    resolver = ComponentAliasResolver.from_yaml()
    runtime_version = installed_ultralytics_version()
    effective = {}
    for component_id, contract in resolver.contracts.items():
        try:
            source_hash = adapter_source_hash(contract)
        except (AttributeError, ImportError, TypeError, ValueError):
            effective[component_id] = contract
            continue
        resolved, _, _ = registry.resolve(
            contract,
            adapter_hash=source_hash,
            ultralytics_version=runtime_version,
        )
        effective[component_id] = resolved
    return PaperCoverageAcceptanceBuilder(
        effective_contracts=effective
    ).build(
        PaperMethodCoverageReport.from_yaml(method_path),
        ExecutablePaperCoverageBaseline.from_yaml(executable_path),
        source_method_coverage_hash=file_sha256(method_path),
        source_registry_hash=(
            file_sha256(registry_path) if registry_path.is_file() else None
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate traceable paper implementation coverage thresholds."
    )
    parser.add_argument(
        "--method-coverage",
        type=Path,
        default=Path("research/production/paper_method_coverage.yaml"),
    )
    parser.add_argument(
        "--executable-coverage",
        type=Path,
        default=Path("research/production/coverage_baseline.yaml"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("runs/component_maturity_registry.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/paper-coverage-acceptance.yaml"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Compact Markdown summary; defaults beside --output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_final_coverage_acceptance(
        method_coverage_path=args.method_coverage,
        executable_coverage_path=args.executable_coverage,
        maturity_registry_path=args.registry,
    )
    report.to_yaml(args.output, exclude_none=True, sort_keys=False)
    markdown = args.markdown or args.output.with_suffix(".md")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(
        render_coverage_acceptance_markdown(report),
        encoding="utf-8",
    )
    for metric in report.metrics.values():
        print(
            f"{metric.metric_id}: {metric.numerator}/{metric.denominator} "
            f"({metric.ratio:.1%}) target={metric.target:.1%} "
            f"status={'passed' if metric.passed else 'failed'}"
        )
    if report.status == "failed":
        print("Highest-yield gaps:")
        for item in report.next_mechanisms[:10]:
            print(
                f"- {item.mechanism_id}: papers={item.covered_paper_count}; "
                f"{item.reason}"
            )
        return 1
    print("Paper implementation coverage acceptance passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_final_coverage_acceptance", "main"]
