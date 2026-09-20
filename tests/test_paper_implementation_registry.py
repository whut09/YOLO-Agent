from pathlib import Path

from yolo_agent.research.paper_implementation_registry import (
    PaperImplementationRegistryBuilder,
    _load_contract_map,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONTRACTS = ROOT / "research" / "production" / "component_contracts.yaml"
LOCAL_CONTRACTS = ROOT / "configs" / "components"


def test_local_contracts_fill_metadata_only_production_rows() -> None:
    contracts = _load_contract_map(
        PRODUCTION_CONTRACTS,
        overlay_path=LOCAL_CONTRACTS,
    )

    head = contracts["detection_head.task_aligned"]
    pyramid = contracts["feature_pyramid.multi_scale"]

    assert head.implementation_path == (
        "yolo_agent.components.adapters.heads.task_aligned"
    )
    assert head.adapter_class == "TaskAlignedHeadAdapter"
    assert pyramid.implementation_path == (
        "yolo_agent.components.adapters.neck.feature_pyramid_adapter"
    )
    assert pyramid.adapter_class == "FeaturePyramidMultiScaleAdapter"


def test_contract_overlay_does_not_grant_maturity_without_artifacts() -> None:
    contracts = _load_contract_map(
        PRODUCTION_CONTRACTS,
        overlay_path=LOCAL_CONTRACTS,
    )

    # Post-closure refresh: the frozen production snapshot now carries hash-
    # verified non-mock certification artifacts for these two components
    # (captured from the current-code recertification pass), so their
    # maturity comes from real evidence — not from the configs overlay.
    for component_id in (
        "detection_head.task_aligned",
        "feature_pyramid.multi_scale",
    ):
        contract = contracts[component_id]
        assert contract.maturity == "smoke_passed"
        assert contract.maturity_artifacts
        assert all(not artifact.mock for artifact in contract.maturity_artifacts)
        assert contract.can_execute

    # The overlay anti-grant invariant itself: a frozen row without
    # certification artifacts keeps its source maturity — the configs
    # overlay fills implementation metadata only, never maturity.
    uncertified = contracts["attention.spatial"]
    assert uncertified.maturity == "adapter_implemented"
    assert uncertified.maturity_artifacts == []
    assert not uncertified.can_execute


def test_registry_reports_discovered_adapters_but_keeps_runtime_blockers() -> None:
    registry = PaperImplementationRegistryBuilder(
        contracts_path=PRODUCTION_CONTRACTS,
        contract_overlay_path=LOCAL_CONTRACTS,
    ).build()

    by_id = {item.paper_id: item for item in registry.records}
    head_paper = by_id["arxiv:2108.07755"]
    pyramid_paper = by_id["arxiv:2309.11331"]

    assert "detection_head.task_aligned" in head_paper.adapter_ids
    assert not any(
        item.endswith("missing_runtime:implementation_identity:detection_head.task_aligned")
        for item in head_paper.blockers
    )
    assert "feature_pyramid.multi_scale" in pyramid_paper.adapter_ids
    assert not any(
        item.endswith("missing_runtime:implementation_identity:feature_pyramid.multi_scale")
        for item in pyramid_paper.blockers
    )
    # Gap closure supplied runtime evidence for every adapter these papers
    # bind, so both are implementation-ready now.  The "overlay does not grant
    # maturity" invariant lives in the contract test above; the blocking
    # behavior lives in the synthetic gate tests in test_paper_implementation_gate.py.
    assert head_paper.is_implementation_ready
    assert pyramid_paper.is_implementation_ready
