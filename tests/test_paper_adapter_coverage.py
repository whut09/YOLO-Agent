from __future__ import annotations

from pathlib import Path

from yolo_agent.tools.paper_adapter_coverage import (
    PaperAdapterCoverageReport,
    PaperCatalogAudit,
    build_report,
    generate,
)


AUDIT_PATH = Path("configs/paper_catalog_audit.yaml")
REPORT_PATH = Path("docs/paper-adapter-coverage.yaml")


def test_coverage_separates_papers_implementation_and_runtime() -> None:
    report = build_report(PaperCatalogAudit.from_yaml(AUDIT_PATH))

    assert report.paper_count == 728
    assert report.implemented_adapter_count == 13
    assert report.runtime_integrated_count == 0
    assert report.pilot_reproduced_count == 0
    assert report.maturity_counts["adapter_implemented"] == 13
    assert report.maturity_counts["smoke_passed"] == 0


def test_committed_paper_adapter_coverage_is_current() -> None:
    assert generate(audit_path=AUDIT_PATH, report_path=REPORT_PATH, check=True)
    report = PaperAdapterCoverageReport.from_yaml(REPORT_PATH)
    assert report.snapshot_hash == PaperCatalogAudit.from_yaml(AUDIT_PATH).snapshot_hash
