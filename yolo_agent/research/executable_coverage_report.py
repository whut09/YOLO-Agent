"""Human-readable rendering for executable paper coverage."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml

from yolo_agent.research.executable_coverage_schemas import (
    ExecutablePaperCoverageBaseline,
)


def render_executable_coverage_markdown(
    report: ExecutablePaperCoverageBaseline,
) -> str:
    """Render every paper field without changing the machine-readable report."""
    lines = [
        "# Executable Paper Coverage Baseline",
        "",
        "This audit separates paper metadata, reusable adapters, and valid runtime "
        "evidence. Component adaptation is not exact paper reproduction.",
        "",
        "## Denominators",
        "",
        "| Denominator | Papers | Definition |",
        "|---|---:|---|",
    ]
    for name in (
        "all_papers",
        "yolo26_compatible_papers",
        "adaptable_component_papers",
        "exact_reproduction_candidates",
    ):
        denominator = report.denominators[name]
        lines.append(
            f"| `{name}` | {denominator.paper_count} | "
            f"{_cell(denominator.definition)} |"
        )
    lines.extend(
        [
            "",
            "## Execution Summary",
            "",
            f"- Papers with reusable adapter candidates: {report.reusable_adapter_paper_count}",
            f"- Papers with valid runtime-ready adapters: {report.runtime_ready_paper_count}",
            f"- Source method coverage hash: `{report.source_method_coverage_hash}`",
            f"- Source maturity hash: `{report.source_maturity_hash or 'not_available'}`",
            f"- Report hash: `{report.report_hash}`",
            "",
            "## Per-Paper Audit",
            "",
            "| Paper | Compatibility | Scope | Blocking fields | Mechanisms | "
            "Reusable adapters | Runtime-ready | Hooks | Implementation cost | "
            "Resource cost | Exact | Exclusion |",
            "|---|---|---|---|---|---|---|---|---|---|---:|---|",
        ]
    )
    for item in report.entries:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_cell(item.paper_id)}`",
                    item.compatibility_class,
                    item.adaptation_scope,
                    _items(item.blocking_fields),
                    _items(item.canonical_mechanisms),
                    _items(item.reusable_adapter_candidates),
                    _items(item.runtime_ready_adapters),
                    _items(item.required_runtime_hooks),
                    _cell(
                        f"{item.implementation_cost.level}; "
                        + "; ".join(item.implementation_cost.rationale)
                    ),
                    _cell(
                        f"{item.expected_resource_cost.level}; "
                        f"latency={item.expected_resource_cost.latency}; "
                        f"size={item.expected_resource_cost.model_size}; "
                        f"vram={item.expected_resource_cost.vram}; "
                        f"training={item.expected_resource_cost.training_compute}"
                    ),
                    "yes" if item.exact_reproduction_possible else "no",
                    _cell(item.exclusion_reason or "none"),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_executable_coverage_artifacts(
    report: ExecutablePaperCoverageBaseline,
    *,
    yaml_path: Path | str,
    markdown_path: Path | str,
) -> tuple[Path, Path]:
    """Atomically write both report representations."""
    yaml_output = Path(yaml_path)
    markdown_output = Path(markdown_path)
    _atomic_text(
        yaml_output,
        yaml.safe_dump(
            report.model_dump(mode="json", exclude_none=True),
            sort_keys=False,
        ),
    )
    _atomic_text(markdown_output, render_executable_coverage_markdown(report))
    return yaml_output, markdown_output


def _items(values: list[str]) -> str:
    return _cell("<br>".join(values) if values else "none")


def _cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


__all__ = [
    "render_executable_coverage_markdown",
    "write_executable_coverage_artifacts",
]
