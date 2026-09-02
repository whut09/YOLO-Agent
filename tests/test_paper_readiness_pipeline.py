"""Regression tests for the one-command paper readiness pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_paper_readiness import _inventory
from yolo_agent.certification.paper_readiness import PaperReadinessPreflight
from yolo_agent.tools.paper_readiness import run_paper_readiness


def test_pipeline_materializes_all_papers_with_fail_closed_assets(
    tmp_path: Path,
) -> None:
    inventory_path = tmp_path / "inventory.yaml"
    _inventory().to_yaml(inventory_path, sort_keys=False)
    data = tmp_path / "coco.yaml"
    data.write_text("path: .\ntrain: train\nval: val\n", encoding="utf-8")
    output = tmp_path / "readiness" / "report.yaml"

    report = run_paper_readiness(
        research_root=tmp_path,
        registry_path=tmp_path / "maturity.yaml",
        model="yolo26n.pt",
        data=data,
        output_path=output,
        inventory_path=inventory_path,
        run_cpu_certification=False,
    )

    assert report.paper_count == 83
    assert len(report.records) == 83
    assert report.asset_registry_path == str(
        (output.parent / "paper_asset_registry.yaml").resolve()
    )
    assert report.asset_registry_hash != "missing"
    assert report.training_started is False
    assert not any(item.asha_eligibility for item in report.records)
    assert all(item.asset_registry_hash == report.asset_registry_hash for item in report.records)


def test_explicit_asset_registry_must_match_requirements_and_inventory(
    tmp_path: Path,
) -> None:
    inventory_path = tmp_path / "inventory.yaml"
    _inventory().to_yaml(inventory_path, sort_keys=False)
    data = tmp_path / "coco.yaml"
    data.write_text("names: [object]\n", encoding="utf-8")
    assets = tmp_path / "assets.yaml"
    assets.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="PaperAssetRegistry"):
        run_paper_readiness(
            research_root=tmp_path,
            registry_path=tmp_path / "maturity.yaml",
            model="yolo26n.pt",
            data=data,
            output_path=tmp_path / "report.yaml",
            inventory_path=inventory_path,
            assets_path=assets,
            run_cpu_certification=False,
        )


def test_preflight_keeps_one_paper_failure_isolated(tmp_path: Path) -> None:
    data = tmp_path / "coco.yaml"
    data.write_text("names: [object]\n", encoding="utf-8")
    report = PaperReadinessPreflight().run(
        inventory=_inventory(),
        registry_path=tmp_path / "maturity.yaml",
        model="yolo26n.pt",
        data=data,
        output_path=tmp_path / "report.yaml",
        run_cpu_certification=False,
    )

    assert report.paper_count == 83
    assert {item.paper_id for item in report.records} == {
        f"paper:{index:03d}" for index in range(83)
    }
    assert all(item.exact_blocker for item in report.records)


def test_report_contains_separate_runtime_evidence_checks(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.yaml"
    _inventory().to_yaml(inventory_path, sort_keys=False)
    data = tmp_path / "coco.yaml"
    data.write_text("names: [object]\n", encoding="utf-8")
    report = run_paper_readiness(
        research_root=tmp_path,
        registry_path=tmp_path / "maturity.yaml",
        model="yolo26n.pt",
        data=data,
        output_path=tmp_path / "report.yaml",
        inventory_path=inventory_path,
        run_cpu_certification=False,
    )

    row = report.records[0]
    assert row.domain_evidence_result.status == "not_applicable"
    assert row.manifest_evidence_result.status == "not_applicable"
    assert row.protocol_evidence_result.passed is True
    assert row.asha_eligibility is False


def test_non_yolo26n_model_cannot_become_asha_eligible(tmp_path: Path) -> None:
    report = PaperReadinessPreflight().run(
        inventory=_inventory(),
        registry_path=tmp_path / "maturity.yaml",
        model="yolo26s.pt",
        data=tmp_path / "missing.yaml",
        output_path=tmp_path / "report.yaml",
        run_cpu_certification=False,
    )

    assert not any(item.asha_eligibility for item in report.records)
    assert all(
        item.exact_blocker in {"dataset_manifest_missing", "student_model_yolo26n_required"}
        or "student_model_yolo26n_required" in item.exact_blocker
        for item in report.records
    )
