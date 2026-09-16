"""Rendering and persistence for the Paper-83 exactness audit.

The exactness audit is allowed to complete with fewer than 83/83 ready
papers.  The writer therefore always produces the YAML artifact, the gap
queue, and the markdown status report; callers decide separately whether a
gate passes.  Producing the audit is never a failure.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from yolo_agent.research.paper_exactness_schemas import (
    BLOCKER_CATEGORY_ORDER,
    EXACTNESS_CHECK_COUNT,
    EXACTNESS_CHECK_IDS,
    IMPLEMENTATION_READY,
    OUT_OF_SCOPE_STATUS,
    PaperExactnessAudit,
    PaperExactnessGapQueue,
)

_TRAINING_LOCK_NOTE = (
    "Training remains locked: this audit inventories evidence only and never "
    "starts model training."
)


def _check_labels() -> dict[str, str]:
    return {
        "membership_valid": "paper membership valid",
        "method_profile_valid": "MethodProfile valid",
        "mechanism_evidence_available": "mechanism evidence available",
        "implementation_spec_complete": "PaperImplementationSpec complete",
        "not_generic_only": "not generic-only",
        "not_alias_only": "not alias-only",
        "not_metadata_only": "not metadata-only",
        "not_no_op": "not a no-op",
        "paper_specific_composition": "paper-specific composition present",
        "runtime_hook_real": "runtime hook real",
        "runtime_fingerprint_present": "source/runtime fingerprint present",
        "unit_tests_present": "unit tests",
        "non_mock_smoke_present": "non-mock smoke",
        "compatibility_tests_present": "compatibility tests",
        "rollback_path_present": "rollback path",
        "shared_primitive_correctly_referenced": "shared primitives referenced, not copied",
        "core_mechanism_covered": "core mechanism covered",
    }


def summarize_exactness_audit(audit: PaperExactnessAudit) -> dict[str, int]:
    """Aggregate per-check failure counts across all records."""

    failures: Counter[str] = Counter()
    for record in audit.records:
        for check_id in record.failed_checks:
            failures[check_id] += 1
    return {check_id: failures.get(check_id, 0) for check_id in EXACTNESS_CHECK_IDS}


def render_paper_83_exactness_audit(
    audit: PaperExactnessAudit,
    gap_queue: PaperExactnessGapQueue | None = None,
) -> str:
    """Render the markdown status report for the exactness audit."""

    summary = audit.summary
    lines: list[str] = []
    lines.append("# Paper-83 Exactness Audit")
    lines.append("")
    lines.append(
        "Per-paper implementation-exactness inventory over the frozen 83. "
        "An audit result below 83/83 ready is a legitimate outcome: the "
        "report, artifacts, gap queue, and tests are still produced."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("```")
    lines.append(f"Total: {summary.total}")
    lines.append(f"Ready: {summary.ready}")
    lines.append(f"Blocked: {summary.blocked}")
    lines.append(f"Out of scope: {summary.out_of_scope}")
    lines.append(
        "Papers passing all "
        f"{EXACTNESS_CHECK_COUNT} checks: {summary.full_check_pass_count}"
    )
    lines.append("```")
    lines.append("")
    lines.append(_TRAINING_LOCK_NOTE)
    lines.append("")

    lines.append("## Status vocabulary")
    lines.append("")
    lines.append(
        "Only `implementation_ready` grants implementation status. Blocked "
        "statuses use the fixed blocker categories; component-level terms "
        "(`covered`, `mapped`, `certified_adapter`) are not readiness states."
    )
    lines.append("")

    lines.append("## Results by status")
    lines.append("")
    lines.append("| Status | Papers |")
    lines.append("| --- | --- |")
    for status in (IMPLEMENTATION_READY, *BLOCKER_CATEGORY_ORDER, OUT_OF_SCOPE_STATUS):
        count = summary.by_status.get(status, 0)
        if count:
            lines.append(f"| `{status}` | {count} |")
    lines.append("")

    lines.append("## Failing checks across the campaign")
    lines.append("")
    lines.append("| Check | Failing papers |")
    lines.append("| --- | --- |")
    labels = _check_labels()
    for check_id, count in summarize_exactness_audit(audit).items():
        lines.append(f"| {labels.get(check_id, check_id)} | {count} |")
    lines.append("")

    lines.append("## Blocked papers by category")
    lines.append("")
    for category in BLOCKER_CATEGORY_ORDER:
        records = [
            record for record in audit.records if record.status == category
        ]
        if not records:
            continue
        lines.append(f"### `{category}` ({len(records)} papers)")
        lines.append("")
        for record in records:
            failed = ", ".join(record.failed_checks) or "n/a"
            lines.append(f"- `{record.paper_id}` — failed: {failed}")
        lines.append("")

    if summary.out_of_scope:
        lines.append("## Out-of-scope papers")
        lines.append("")
        for record in audit.records:
            if record.status == OUT_OF_SCOPE_STATUS:
                lines.append(f"- `{record.paper_id}` — {record.title}")
        lines.append("")

    if gap_queue is not None:
        lines.append("## Gap queue")
        lines.append("")
        lines.append("```")
        lines.append(f"Gap entries: {len(gap_queue.gaps)}")
        lines.append(f"Blocked papers: {gap_queue.blocked_paper_count}")
        for category, count in gap_queue.gaps_by_category.items():
            lines.append(f"{category}: {count}")
        lines.append("```")
        lines.append("")
        lines.append(
            "Every gap entry carries a recommended fix, an estimated scope, "
            "and its dependency; see `artifacts/paper_83_gap_queue.yaml`."
        )
        lines.append("")

    lines.append("## Ready papers")
    lines.append("")
    ready = [record for record in audit.records if record.status == IMPLEMENTATION_READY]
    if ready:
        for record in ready:
            lines.append(f"- `{record.paper_id}` — {record.title}")
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_paper_exactness_artifacts(
    audit: PaperExactnessAudit,
    gap_queue: PaperExactnessGapQueue,
    *,
    yaml_path: Path | str,
    queue_path: Path | str,
    markdown_path: Path | str,
) -> tuple[Path, Path, Path]:
    """Write the audit YAML, gap-queue YAML, and markdown status report."""

    written_yaml = audit.to_yaml(yaml_path)
    written_queue = gap_queue.to_yaml(queue_path)
    markdown = Path(markdown_path)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(
        render_paper_83_exactness_audit(audit, gap_queue), encoding="utf-8"
    )
    return written_yaml, written_queue, markdown


def render_exactness_cli_summary(
    audit: PaperExactnessAudit,
    gap_queue: PaperExactnessGapQueue,
) -> list[str]:
    """Concise CLI summary lines for the exactness audit."""

    summary = audit.summary
    lines = [
        "Paper-83 Exactness Audit",
        "------------------------",
        "Training: not started (offline audit only)",
        f"Total: {summary.total}",
        f"Ready: {summary.ready}",
        f"Blocked: {summary.blocked}",
        f"Out of scope: {summary.out_of_scope}",
        "Statuses: "
        + " ".join(f"{name}={count}" for name, count in summary.by_status.items()),
        f"Gap entries: {len(gap_queue.gaps)} across {gap_queue.blocked_paper_count} papers",
    ]
    return lines


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "render_exactness_cli_summary",
    "render_paper_83_exactness_audit",
    "summarize_exactness_audit",
    "utc_now_iso",
    "write_paper_exactness_artifacts",
]
