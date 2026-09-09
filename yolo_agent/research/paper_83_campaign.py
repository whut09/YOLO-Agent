"""Freeze the historical README-certified paper campaign by exact identity.

This module is an offline audit utility.  It never builds a trainer, probes a
GPU, or derives campaign membership from current runtime maturity.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from collections.abc import Iterable

import yaml

from yolo_agent.research.coverage_acceptance import PaperCoverageAcceptanceReport
from yolo_agent.research.executable_coverage_schemas import (
    ExecutablePaperCoverageBaseline,
    PaperExecutableCoverageEntry,
)
from yolo_agent.research.method_profiles import PaperMethodCoverageReport
from yolo_agent.research.paper_execution_inventory import (
    PaperExecutionInventory,
    PaperExecutionSpec,
)
from yolo_agent.research.schemas import PaperRecord
from yolo_agent.research.paper_83_campaign_schemas import (
    Paper83Paper,
)

from yolo_agent.research.paper_83_campaign_schemas import (
    PAPER_83_COUNT,
    PAPER_85_COUNT,
    Paper83Campaign,
    Paper83Manifest,
    Paper83MembershipSource,
    Paper83RepositoryLineage,
    calculate_membership_hash,
    ReadmeCoverageDeclaration,
)


class Paper83CampaignError(ValueError):
    """Raised when the historical paper-83 source cannot be verified."""


_RATIO_ROW = re.compile(
    r"\|\s*{label}\s*\|\s*(?P<numerator>\d+)\s*/\s*(?P<denominator>\d+)\b",
)
_HASH_LINE = re.compile(
    r"(?P<label>Audit snapshot|Acceptance hash)\s*:\s*`(?P<hash>[0-9a-f]{64})`",
    re.IGNORECASE,
)


def parse_readme_coverage(readme_path: Path | str) -> ReadmeCoverageDeclaration:
    """Parse coverage numbers and lineage hashes from the current README."""

    path = Path(readme_path)
    if not path.is_file():
        raise Paper83CampaignError(f"README_NOT_FOUND={path}")
    text = path.read_text(encoding="utf-8-sig")
    method = _parse_ratio_row(text, "Compatible papers with valid MethodProfile")
    certified = _parse_ratio_row(text, "Compatible papers reusing a certified adapter")
    hashes = {
        match.group("label").casefold(): match.group("hash")
        for match in _HASH_LINE.finditer(text)
    }
    try:
        audit_snapshot_hash = hashes["audit snapshot"]
        acceptance_hash = hashes["acceptance hash"]
    except KeyError as exc:
        raise Paper83CampaignError(
            "README lineage hashes are incomplete; expected Audit snapshot and Acceptance hash"
        ) from exc
    return ReadmeCoverageDeclaration(
        method_profile_numerator=method[0],
        method_profile_denominator=method[1],
        certified_adapter_numerator=certified[0],
        certified_adapter_denominator=certified[1],
        audit_snapshot_hash=audit_snapshot_hash,
        acceptance_hash=acceptance_hash,
    )


def assert_readme_campaign_shape(
    declaration: ReadmeCoverageDeclaration,
) -> None:
    """Fail closed unless README still declares the requested 83/85 campaign."""

    if (
        declaration.certified_adapter_numerator != PAPER_83_COUNT
        or declaration.certified_adapter_denominator != PAPER_85_COUNT
        or declaration.method_profile_numerator != PAPER_85_COUNT
        or declaration.method_profile_denominator != PAPER_85_COUNT
    ):
        raise Paper83CampaignError(
            "README campaign shape changed: "
            f"README_EXPECTED_CERTIFIED={PAPER_83_COUNT} "
            f"README_ACTUAL_CERTIFIED={declaration.certified_adapter_numerator} "
            f"README_COMPATIBLE_DENOMINATOR={declaration.certified_adapter_denominator}"
        )


def _parse_ratio_row(text: str, label: str) -> tuple[int, int]:
    pattern = re.compile(_RATIO_ROW.pattern.format(label=re.escape(label)))
    match = pattern.search(text)
    if match is None:
        raise Paper83CampaignError(f"README_COVERAGE_ROW_NOT_FOUND={label}")
    return int(match.group("numerator")), int(match.group("denominator"))


def find_acceptance_artifact(
    expected_hash: str,
    *,
    search_roots: Iterable[Path | str] = (
        "docs",
        "runs/coverage-audit",
        "research/production",
        "configs",
        "tests",
    ),
) -> Path:
    """Find the report whose serialized report hash matches README exactly."""

    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise Paper83CampaignError(f"INVALID_README_ACCEPTANCE_HASH={expected_hash}")
    candidates: list[Path] = []
    other_reports: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for root_value in search_roots:
        root = Path(root_value)
        paths = [root] if root.is_file() else (
            sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml"))
            if root.is_dir()
            else []
        )
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                text = path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeError):
                continue
            raw = _yaml_mapping(text)
            report_hash = raw.get("report_hash") if raw else None
            if isinstance(report_hash, str) and re.fullmatch(
                r"[0-9a-f]{64}", report_hash
            ):
                other_reports.append((path, report_hash))
            if report_hash == expected_hash:
                candidates.append(path)
    if not candidates:
        details = "\n".join(
            f"OTHER_ACCEPTANCE_ARTIFACT={path} REPORT_HASH={report_hash}"
            for path, report_hash in sorted(other_reports)
        )
        suffix = f"\n{details}" if details else ""
        raise Paper83CampaignError(
            "README_ACCEPTANCE_HASH="
            f"{expected_hash}\nMATCHING_ACCEPTANCE_ARTIFACT=NOT_FOUND{suffix}"
        )
    if len(candidates) > 1:
        paths = ", ".join(str(path) for path in sorted(candidates))
        raise Paper83CampaignError(
            f"README_ACCEPTANCE_HASH={expected_hash}\n"
            f"MATCHING_ACCEPTANCE_ARTIFACT=AMBIGUOUS:{paths}"
        )
    return candidates[0]


def load_exact_acceptance_report(
    path: Path | str,
    expected_hash: str,
) -> PaperCoverageAcceptanceReport:
    """Load and verify the exact acceptance report referenced by README."""

    report = PaperCoverageAcceptanceReport.from_yaml(path)
    if report.report_hash != expected_hash:
        raise Paper83CampaignError(
            f"acceptance report hash mismatch: expected {expected_hash}, "
            f"got {report.report_hash}"
        )
    calculated = report.calculate_hash()
    if calculated != report.report_hash:
        raise Paper83CampaignError(
            f"acceptance report integrity mismatch: calculated {calculated}, "
            f"stored {report.report_hash}"
        )
    return report


def extract_frozen_paper_ids(
    report: PaperCoverageAcceptanceReport,
) -> list[str]:
    """Return the only membership source allowed for the paper-83 campaign."""

    metric = report.metrics.get("compatible_papers_certified_adapter")
    if metric is None:
        raise Paper83CampaignError(
            "acceptance report is missing metric compatible_papers_certified_adapter"
        )
    if metric.metric_id != "compatible_papers_certified_adapter":
        raise Paper83CampaignError(
            f"unexpected acceptance metric identity: {metric.metric_id}"
        )
    numerator_ids = list(metric.numerator_ids)
    denominator_ids = list(metric.denominator_ids)
    if metric.numerator != PAPER_83_COUNT:
        raise Paper83CampaignError(
            f"FROZEN_CAMPAIGN_PAPERS expected {PAPER_83_COUNT} actual {metric.numerator}"
        )
    if metric.denominator != PAPER_85_COUNT:
        raise Paper83CampaignError(
            "FROZEN_COMPATIBLE_DENOMINATOR expected "
            f"{PAPER_85_COUNT} actual {metric.denominator}"
        )
    if len(numerator_ids) != PAPER_83_COUNT:
        raise Paper83CampaignError(
            f"acceptance numerator ID count expected {PAPER_83_COUNT} "
            f"actual {len(numerator_ids)}"
        )
    if len(denominator_ids) != PAPER_85_COUNT:
        raise Paper83CampaignError(
            f"acceptance denominator ID count expected {PAPER_85_COUNT} "
            f"actual {len(denominator_ids)}"
        )
    if numerator_ids != sorted(set(numerator_ids)):
        raise Paper83CampaignError(
            "acceptance numerator_ids must be sorted and unique"
        )
    if denominator_ids != sorted(set(denominator_ids)):
        raise Paper83CampaignError(
            "acceptance denominator_ids must be sorted and unique"
        )
    if not set(numerator_ids).issubset(denominator_ids):
        raise Paper83CampaignError(
            "acceptance numerator_ids must be a subset of denominator_ids"
        )
    traces = {item.paper_id: item for item in report.paper_traces}
    missing_traces = sorted(set(numerator_ids) - set(traces))
    if missing_traces:
        raise Paper83CampaignError(
            "acceptance numerator is missing paper traces: "
            + ", ".join(missing_traces)
        )
    missing_certification = sorted(
        paper_id
        for paper_id in numerator_ids
        if not traces[paper_id].certified_adapter_ids
    )
    if missing_certification:
        raise Paper83CampaignError(
            "acceptance numerator has no certified adapter trace: "
            + ", ".join(missing_certification)
        )
    return numerator_ids


def load_current_snapshot_hash(
    research_root: Path | str = "research",
) -> str | None:
    """Read the current snapshot pointer without using it for membership."""

    pointer = Path(research_root) / "latest_snapshot.yaml"
    if not pointer.is_file():
        return None
    raw = _yaml_mapping(pointer.read_text(encoding="utf-8-sig"))
    value = raw.get("snapshot_hash")
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise Paper83CampaignError(
            f"CURRENT_RESEARCH_SNAPSHOT_HASH_INVALID={value}"
        )
    return value


def acceptance_lineage_status(
    readme_snapshot_hash: str,
    current_snapshot_hash: str | None,
) -> str:
    """Classify historical lineage while preserving the acceptance membership."""

    return "exact" if current_snapshot_hash == readme_snapshot_hash else "historical"


def load_paper_records_by_id(
    paper_source: Path | str = "research",
) -> dict[str, PaperRecord]:
    """Load current paper records and index them by exact paper ID."""

    source = Path(paper_source)
    path = source / "papers.jsonl" if source.is_dir() else source
    if not path.is_file():
        raise Paper83CampaignError(f"PAPER_RECORDS_NOT_FOUND={path}")
    records: dict[str, PaperRecord] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = PaperRecord.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValueError) as exc:
            raise Paper83CampaignError(
                f"PAPER_RECORD_INVALID={path}:{line_number}:{exc}"
            ) from exc
        if record.paper_id in records:
            raise Paper83CampaignError(
                f"DUPLICATE_PAPER_RECORD_ID={record.paper_id}"
            )
        records[record.paper_id] = record
    return records


def load_current_method_coverage(
    path: Path | str = "research/production/paper_method_coverage.yaml",
) -> PaperMethodCoverageReport:
    """Load the current MethodProfile/decision audit without changing membership."""

    return PaperMethodCoverageReport.from_yaml(path)


def load_current_executable_entries(
    path: Path | str = "research/production/coverage_baseline.yaml",
) -> dict[str, PaperExecutableCoverageEntry]:
    """Load optional current executable mappings indexed by exact paper ID."""

    input_path = Path(path)
    if not input_path.is_file():
        return {}
    report = ExecutablePaperCoverageBaseline.from_yaml(input_path)
    return {item.paper_id: item for item in report.entries}


def load_current_inventory_entries(
    path: Path | str = "runs/coverage-audit/paper_execution_inventory.yaml",
) -> dict[str, PaperExecutionSpec]:
    """Load optional current paper execution statuses indexed by exact ID."""

    input_path = Path(path)
    if not input_path.is_file():
        return {}
    inventory = PaperExecutionInventory.from_yaml(input_path)
    return {item.paper_id: item for item in inventory.records}


def resolve_current_paper_metadata(
    frozen_paper_ids: Iterable[str],
    *,
    acceptance_report: PaperCoverageAcceptanceReport,
    paper_records: dict[str, PaperRecord],
    method_coverage: PaperMethodCoverageReport,
    executable_entries: dict[str, PaperExecutableCoverageEntry] | None = None,
    inventory_entries: dict[str, PaperExecutionSpec] | None = None,
) -> list[Paper83Paper]:
    """Resolve current metadata for a fixed historical ID set by exact identity."""

    ids = list(frozen_paper_ids)
    traces = {item.paper_id: item for item in acceptance_report.paper_traces}
    profiles = {item.paper_id: item for item in method_coverage.profiles}
    decisions = {item.paper_id: item for item in method_coverage.decisions}
    entries = executable_entries or {}
    current_inventory = inventory_entries or {}
    missing = {
        "paper_records": sorted(set(ids) - set(paper_records)),
        "method_profiles": sorted(set(ids) - set(profiles)),
        "acceptance_traces": sorted(set(ids) - set(traces)),
    }
    missing = {name: values for name, values in missing.items() if values}
    if missing:
        detail = "; ".join(
            f"{name}={','.join(values)}" for name, values in sorted(missing.items())
        )
        raise Paper83CampaignError(f"EXACT_PAPER_METADATA_MISSING={detail}")

    papers: list[Paper83Paper] = []
    for paper_id in ids:
        paper = paper_records[paper_id]
        profile = profiles[paper_id]
        decision = decisions.get(paper_id)
        executable = entries.get(paper_id)
        current = current_inventory.get(paper_id)
        trace = traces[paper_id]
        components = set(profile.canonical_component_ids)
        adapters = set()
        if decision is not None:
            components.update(decision.canonical_component_ids)
            adapters.update(decision.reusable_adapter_ids)
        if executable is not None:
            components.update(executable.canonical_mechanisms)
            adapters.update(executable.reusable_adapter_candidates)
            adapters.update(executable.runtime_ready_adapters)
        if current is not None:
            components.update(current.canonical_component_ids)
            adapters.update(current.reusable_adapter_ids)
            adapters.update(current.runtime_ready_adapters)
        disposition = (
            current.current_disposition
            if current is not None
            else _derive_current_disposition(
                profile=profile,
                decision=decision,
                executable=executable,
                adapters=adapters,
            )
        )
        papers.append(
            Paper83Paper(
                paper_id=paper_id,
                title=paper.title,
                year=paper.year,
                source=paper.source,
                method_profile_id=profile.profile_id,
                frozen_certified_adapter_ids=sorted(
                    set(trace.certified_adapter_ids)
                ),
                current_component_ids=sorted(components),
                current_adapter_ids=sorted(adapters),
                current_disposition=disposition,
            )
        )
    return sorted(papers, key=lambda item: item.paper_id)


def _derive_current_disposition(
    *,
    profile: object,
    decision: object | None,
    executable: PaperExecutableCoverageEntry | None,
    adapters: set[str],
) -> str:
    decision_kind = getattr(decision, "decision", None)
    compatibility = getattr(executable, "compatibility_class", None)
    if decision_kind == "separate_detector_family" or compatibility in {
        "separate_detector_family",
        "incompatible",
    }:
        return "incompatible"
    if executable is not None and executable.runtime_ready_adapters:
        return "runtime_ready"
    if adapters:
        return "blocked_runtime"
    if getattr(profile, "canonical_component_ids", None):
        return "implementation_request"
    return "implementation_request"


def build_paper_83_manifest(
    *,
    readme_path: Path | str = "README.md",
    acceptance_search_roots: Iterable[Path | str] = (
        "docs",
        "runs/coverage-audit",
        "research/production",
        "configs",
        "tests",
    ),
    paper_source: Path | str = "research",
    method_coverage_path: Path | str = (
        "research/production/paper_method_coverage.yaml"
    ),
    executable_coverage_path: Path | str = (
        "research/production/coverage_baseline.yaml"
    ),
    current_inventory_path: Path | str = (
        "runs/coverage-audit/paper_execution_inventory.yaml"
    ),
    research_root: Path | str = "research",
    repository_commit: str | None = None,
) -> Paper83Manifest:
    """Build the frozen campaign while auditing mutable current metadata."""

    declaration = parse_readme_coverage(readme_path)
    assert_readme_campaign_shape(declaration)
    acceptance_path = find_acceptance_artifact(
        declaration.acceptance_hash,
        search_roots=acceptance_search_roots,
    )
    acceptance = load_exact_acceptance_report(
        acceptance_path,
        declaration.acceptance_hash,
    )
    frozen_ids = extract_frozen_paper_ids(acceptance)
    papers = load_paper_records_by_id(paper_source)
    method_coverage = load_current_method_coverage(method_coverage_path)
    current_papers = resolve_current_paper_metadata(
        frozen_ids,
        acceptance_report=acceptance,
        paper_records=papers,
        method_coverage=method_coverage,
        executable_entries=load_current_executable_entries(
            executable_coverage_path
        ),
        inventory_entries=load_current_inventory_entries(current_inventory_path),
    )
    current_snapshot = load_current_snapshot_hash(research_root)
    commit = repository_commit or _git_head()
    campaign = Paper83Campaign(
        name="paper-83",
        membership_source=Paper83MembershipSource(
            type="coverage_acceptance_metric_numerator",
            metric_id="compatible_papers_certified_adapter",
            acceptance_hash=acceptance.report_hash,
            source_method_coverage_hash=acceptance.source_method_coverage_hash,
            source_executable_coverage_hash=acceptance.source_executable_coverage_hash,
            source_registry_hash=acceptance.source_registry_hash,
        ),
        repository=Paper83RepositoryLineage(
            git_commit=commit,
            readme_audit_snapshot_hash=declaration.audit_snapshot_hash,
            current_research_snapshot_hash=current_snapshot,
            acceptance_lineage_status=acceptance_lineage_status(
                declaration.audit_snapshot_hash,
                current_snapshot,
            ),
        ),
        membership_hash=calculate_membership_hash(frozen_ids),
    )
    return Paper83Manifest(
        campaign=campaign,
        papers=current_papers,
    )


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Paper83CampaignError(f"GIT_HEAD_UNAVAILABLE={exc}") from exc
    value = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise Paper83CampaignError(f"GIT_HEAD_INVALID={value}")
    return value


def write_paper_83_manifest(
    manifest: Paper83Manifest,
    path: Path | str = "configs/research/paper_83_manifest.yaml",
) -> Path:
    """Write the campaign manifest without adding runtime or training evidence."""

    return manifest.to_yaml(path, exclude_none=True, sort_keys=False)


def render_paper_83_status_markdown(manifest: Paper83Manifest) -> str:
    """Render a paper-level status audit without overstating implementation."""

    source = manifest.campaign.membership_source
    repository = manifest.campaign.repository
    lines = [
        "# Paper-83 Implementation Status",
        "",
        "Membership is frozen from the numerator_ids of the README-referenced "
        "CoverageAcceptanceReport.",
        "Shared certified adapter coverage is NOT paper implementation.",
        "",
        "## Frozen Campaign",
        "",
        f"- Papers: {manifest.paper_count}",
        f"- Acceptance metric: `{source.metric_id}`",
        f"- Acceptance hash: `{source.acceptance_hash}`",
        f"- Membership hash: `{manifest.campaign.membership_hash}`",
        f"- README audit snapshot: `{repository.readme_audit_snapshot_hash}`",
        f"- Current ResearchSnapshot: `"
        f"{repository.current_research_snapshot_hash or 'not_available'}`",
        f"- Acceptance lineage: `{repository.acceptance_lineage_status}`",
        f"- Repository commit: `{repository.git_commit}`",
        "",
        "`actual implementation status` is deliberately `not_yet_audited` for "
        "every paper until a paper-specific implementation contract is formally "
        "defined and independently verified.",
        "",
        "## Per-Paper Audit",
        "",
        "| Paper ID | Title | Frozen certified adapter mapping | Current component mapping | "
        "Current adapter mapping | Current disposition | Actual implementation status | Known gaps |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in manifest.papers:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_md_cell(item.paper_id)}`",
                    _md_cell(item.title),
                    _md_items(item.frozen_certified_adapter_ids),
                    _md_items(item.current_component_ids),
                    _md_items(item.current_adapter_ids),
                    item.current_disposition,
                    "not_yet_audited",
                    _md_cell(_current_gap(item)),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_paper_83_status_markdown(
    manifest: Paper83Manifest,
    path: Path | str = "docs/paper-83-implementation-status.md",
) -> Path:
    """Write the human-readable paper-83 audit."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_paper_83_status_markdown(manifest), encoding="utf-8")
    return output


def _current_gap(item: Paper83Paper) -> str:
    if not item.current_component_ids:
        return "current component mapping is absent"
    if not item.current_adapter_ids:
        return "current adapter mapping is absent"
    if item.current_disposition == "runtime_ready":
        return "shared runtime adapter exists; paper-specific reproduction is unverified"
    if item.current_disposition == "blocked_runtime":
        return "current runtime evidence or control gate is incomplete"
    if item.current_disposition == "incompatible":
        return "current YOLO26 route is marked incompatible"
    if item.current_disposition == "evidence_recovery":
        return "paper-specific evidence recovery is required"
    if item.current_disposition == "already_tested":
        return "existing result is not an exact paper reproduction"
    return "paper-specific implementation contract and evidence are not audited"


def _md_items(values: list[str]) -> str:
    return _md_cell("<br>".join(values) if values else "none")


def _md_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def main(argv: list[str] | None = None) -> int:
    """Generate the frozen campaign artifacts using only offline inputs."""

    parser = argparse.ArgumentParser(
        description="Freeze the README-referenced paper-83 acceptance numerator."
    )
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--paper-source", type=Path, default=Path("research"))
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
        "--current-inventory",
        type=Path,
        default=Path("runs/coverage-audit/paper_execution_inventory.yaml"),
    )
    parser.add_argument("--research-root", type=Path, default=Path("research"))
    parser.add_argument(
        "--acceptance-root",
        type=Path,
        action="append",
        dest="acceptance_roots",
        help="Directory or file to search; may be repeated.",
    )
    parser.add_argument("--repository-commit")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/research/paper_83_manifest.yaml"),
    )
    parser.add_argument(
        "--status-output",
        type=Path,
        default=Path("docs/paper-83-implementation-status.md"),
    )
    args = parser.parse_args(argv)
    roots = args.acceptance_roots or [
        Path("docs"),
        Path("runs/coverage-audit"),
        Path("research/production"),
        Path("configs"),
        Path("tests"),
    ]
    try:
        manifest = build_paper_83_manifest(
            readme_path=args.readme,
            acceptance_search_roots=roots,
            paper_source=args.paper_source,
            method_coverage_path=args.method_coverage,
            executable_coverage_path=args.executable_coverage,
            current_inventory_path=args.current_inventory,
            research_root=args.research_root,
            repository_commit=args.repository_commit,
        )
        acceptance = manifest.campaign.membership_source
        write_paper_83_manifest(manifest, args.output)
        write_paper_83_status_markdown(manifest, args.status_output)
    except (OSError, Paper83CampaignError, ValueError) as exc:
        print(str(exc))
        return 2
    print(f"GIT_HEAD={manifest.campaign.repository.git_commit}")
    print(f"README_ACCEPTANCE_HASH={acceptance.acceptance_hash}")
    print(f"FOUND_ACCEPTANCE_HASH={acceptance.acceptance_hash}")
    print(f"FROZEN_COMPATIBLE_DENOMINATOR={PAPER_85_COUNT}")
    print(f"FROZEN_CAMPAIGN_PAPERS={manifest.paper_count}")
    print(f"MEMBERSHIP_HASH={manifest.campaign.membership_hash}")
    print("TRAINING_EXECUTED=false")
    print(f"MANIFEST={args.output}")
    print(f"STATUS={args.status_output}")
    return 0


def _yaml_mapping(text: str) -> dict[str, object]:
    try:
        value = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return {}
    return value if isinstance(value, dict) else {}


__all__ = [
    "Paper83CampaignError",
    "assert_readme_campaign_shape",
    "acceptance_lineage_status",
    "find_acceptance_artifact",
    "extract_frozen_paper_ids",
    "load_current_executable_entries",
    "load_current_inventory_entries",
    "load_current_method_coverage",
    "load_current_snapshot_hash",
    "load_exact_acceptance_report",
    "load_paper_records_by_id",
    "parse_readme_coverage",
    "resolve_current_paper_metadata",
    "build_paper_83_manifest",
    "write_paper_83_manifest",
    "render_paper_83_status_markdown",
    "write_paper_83_status_markdown",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
