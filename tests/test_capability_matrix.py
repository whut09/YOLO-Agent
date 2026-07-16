"""Capability maturity manifest and generated-document tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.tools.capability_matrix import (
    CapabilityManifest,
    generate,
    render_detail_document,
    render_readme_matrix,
    validate_source_paths,
)


CONFIG_PATH = Path("configs/capability_maturity.yaml")


def test_capability_manifest_separates_execution_and_reproduction() -> None:
    manifest = CapabilityManifest.from_yaml(CONFIG_PATH)
    capabilities = {item.capability_id: item for item in manifest.capabilities}

    assert capabilities["pilot_auto_training"].status == "executable"
    assert capabilities["candidate_coco_error_facts"].status == "incomplete"
    assert capabilities["error_delta_next_round"].status == "partial"
    assert capabilities["three_seed_confirmation"].status == "supported_not_automatic"
    assert capabilities["stable_two_map_gain"].status == "not_guaranteed"
    assert capabilities["three_seed_confirmation"].code_present is True
    assert capabilities["three_seed_confirmation"].local_reproduction == "not_claimed"


def test_capability_manifest_source_references_exist() -> None:
    manifest = CapabilityManifest.from_yaml(CONFIG_PATH)
    assert validate_source_paths(manifest) == []


def test_generated_matrix_explains_real_boundaries() -> None:
    manifest = CapabilityManifest.from_yaml(CONFIG_PATH)
    matrix = render_readme_matrix(manifest, language="zh")
    document = render_detail_document(manifest)

    assert "代码存在" in matrix
    assert "自动执行" in matrix
    assert "本地复现" in matrix
    assert "`not guaranteed`" in matrix
    assert "代码存在不代表可以自动执行" in document
    assert "yolo_agent/agents/asha_scheduler.py" in document


def test_committed_capability_docs_are_current() -> None:
    assert generate(
        config_path=CONFIG_PATH,
        document_path=Path("docs/capability-maturity.md"),
        readme_path=Path("README.md"),
        readme_en_path=Path("README.en.md"),
        check=True,
    ) is True
