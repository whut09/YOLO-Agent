"""Freeze the historical README-certified paper campaign by exact identity.

This module is an offline audit utility.  It never builds a trainer, probes a
GPU, or derives campaign membership from current runtime maturity.
"""

from __future__ import annotations

import re
from pathlib import Path
from collections.abc import Iterable

import yaml

from yolo_agent.research.coverage_acceptance import PaperCoverageAcceptanceReport

from yolo_agent.research.paper_83_campaign_schemas import (
    PAPER_83_COUNT,
    PAPER_85_COUNT,
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


def _yaml_mapping(text: str) -> dict[str, object]:
    try:
        value = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return {}
    return value if isinstance(value, dict) else {}


__all__ = [
    "Paper83CampaignError",
    "assert_readme_campaign_shape",
    "find_acceptance_artifact",
    "extract_frozen_paper_ids",
    "load_exact_acceptance_report",
    "parse_readme_coverage",
]
