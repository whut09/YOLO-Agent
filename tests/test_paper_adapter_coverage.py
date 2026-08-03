from __future__ import annotations

from pathlib import Path

from yolo_agent.components.maturity import maturity_artifact, transition_maturity
from yolo_agent.components.maturity_registry import (
    ComponentMaturityRegistry,
    adapter_source_hash,
    installed_ultralytics_version,
)
from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.tools.paper_adapter_coverage import (
    LocalPaperAdapterCoverageReport,
    PaperAdapterCoverageReport,
    PaperCatalogAudit,
    build_local_report,
    build_report,
    generate,
)


AUDIT_PATH = Path("configs/paper_catalog_audit.yaml")
REPORT_PATH = Path("docs/paper-adapter-coverage.yaml")


def test_coverage_separates_papers_implementation_and_runtime() -> None:
    report = build_report(PaperCatalogAudit.from_yaml(AUDIT_PATH))

    assert report.paper_count == 728
    assert report.implemented_adapter_count == 54
    assert report.runtime_integrated_count == 0
    assert report.pilot_reproduced_count == 0
    assert report.maturity_counts["adapter_implemented"] == 54
    assert report.maturity_counts["smoke_passed"] == 0


def test_committed_paper_adapter_coverage_is_current() -> None:
    assert generate(audit_path=AUDIT_PATH, report_path=REPORT_PATH, check=True)
    report = PaperAdapterCoverageReport.from_yaml(REPORT_PATH)
    assert report.snapshot_hash == PaperCatalogAudit.from_yaml(AUDIT_PATH).snapshot_hash


def test_local_coverage_applies_registry_without_changing_source_counts(
    tmp_path: Path,
) -> None:
    resolver = ComponentAliasResolver.from_yaml()
    source = resolver.contracts["loss.quality.correlation"]
    updated = source
    for maturity in ("runtime_integrated", "unit_tested", "smoke_passed"):
        path = tmp_path / f"{maturity}.yaml"
        path.write_text(f"stage: {maturity}\n", encoding="utf-8")
        updated = transition_maturity(
            updated,
            maturity,
            reason="test certification",
            artifact=maturity_artifact(
                component_id=source.component_id,
                target_maturity=maturity,
                artifact_path=path,
                status="passed",
                producer="pytest",
                protocol_hash="protocol-1",
            ),
        )
    registry_path = tmp_path / "component_maturity_registry.yaml"
    ComponentMaturityRegistry(registry_path).record_contract(
        updated,
        adapter_hash=adapter_source_hash(source),
        code_commit="test-commit",
        ultralytics_version=installed_ultralytics_version(),
        protocol_hash="protocol-1",
    )

    local = build_local_report(
        PaperCatalogAudit.from_yaml(AUDIT_PATH),
        registry_path=registry_path,
        protocol_hash="protocol-1",
    )
    output = tmp_path / "local-coverage.yaml"
    local.to_yaml(output, exclude_none=True, sort_keys=False)
    restored = LocalPaperAdapterCoverageReport.from_yaml(output)

    assert "loss.quality.correlation" in restored.smoke_passed_ids
    assert restored.overlay_applied_count == 1
    assert restored.maturity_counts["smoke_passed"] == 1
    assert build_report(PaperCatalogAudit.from_yaml(AUDIT_PATH)).runtime_integrated_count == 0
