"""Tests for frozen and live executable coverage input boundaries."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.research.executable_coverage_inputs import (
    load_live_coverage_inputs,
    load_snapshot_coverage_inputs,
)
from yolo_agent.research.method_profiles import PaperMethodCoverageReport
from yolo_agent.research.maturity_snapshot import EffectiveComponentMaturityManifest
from yolo_agent.research.snapshot import freeze_research_snapshot


def test_live_inputs_do_not_treat_source_maturity_as_runtime_evidence(
    tmp_path: Path,
) -> None:
    coverage = tmp_path / "paper_method_coverage.yaml"
    PaperMethodCoverageReport().to_yaml(coverage)
    registry = tmp_path / "component_maturity_registry.yaml"
    registry.write_text(
        "schema_version: component_maturity_registry.v1\noverlays: []\n",
        encoding="utf-8",
    )

    inputs = load_live_coverage_inputs(
        method_coverage_path=coverage,
        maturity_registry_path=registry,
        ultralytics_version="8.4.test",
    )

    assert inputs.contracts
    assert inputs.maturity.entries == []


def test_snapshot_inputs_read_only_frozen_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "research"
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    method = inputs / "paper_method_coverage.yaml"
    PaperMethodCoverageReport().to_yaml(method)
    maturity = inputs / "effective_component_maturity.yaml"
    EffectiveComponentMaturityManifest().to_yaml(maturity)
    contracts = inputs / "component_contracts.yaml"
    contracts.write_text(
        yaml.safe_dump(
            {
                "components": {
                    "fixture.component": {
                        "display_name": "Fixture",
                        "category": "loss",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    generic = inputs / "generic.yaml"
    generic.write_text("items: []\n", encoding="utf-8")
    artifacts = {
        "papers": generic,
        "component_taxonomy": generic,
        "classifications": generic,
        "component_extractions": generic,
        "component_contracts": contracts,
        "compatibility_reviews": generic,
        "recipes": generic,
        "reproduction_queue": generic,
        "paper_method_coverage": method,
        "effective_component_maturity": maturity,
    }
    _, snapshot_dir = freeze_research_snapshot(
        root,
        artifacts,
        paper_count=0,
        component_count=1,
        recipe_count=0,
        paper_method_coverage_version="coverage-v1",
        effective_maturity_version="maturity-v1",
    )

    loaded = load_snapshot_coverage_inputs(snapshot_dir)

    assert loaded.snapshot_hash
    assert list(loaded.contracts) == ["fixture.component"]
    assert loaded.method_coverage_path.parent == snapshot_dir
    assert loaded.taxonomy_path.parent == snapshot_dir
