from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from yolo_agent.research.paper_execution_schemas import (
    PaperExecutionInventory,
    PaperExecutionSpec,
)


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _spec(*, paper_id: str = "arxiv:0000.0001", **updates: object) -> PaperExecutionSpec:
    payload: dict[str, object] = {
        "paper_id": paper_id,
        "profile_id": f"profile:{paper_id}",
        "title": "A paper method",
        "source_locations": ["paper_record.title"],
        "canonical_component_ids": ["loss.quality.correlation"],
        "paper_specific_mechanism_ids": ["loss.quality.correlation"],
        "runtime_ready_adapters": ["loss.quality.correlation"],
        "execution_fingerprint": _fingerprint(paper_id),
        "current_disposition": "runtime_ready",
        "disposition_reason": "paper-specific runtime adapter is verified",
    }
    payload.update(updates)
    return PaperExecutionSpec.model_validate(payload)


def test_generic_mechanism_cannot_be_runtime_ready() -> None:
    with pytest.raises(ValueError, match="generic mechanisms"):
        _spec(
            generic_component_ids=["distillation.yolo26_teacher_student"],
            paper_specific_mechanism_ids=[],
        )


def test_generic_mechanism_may_request_implementation() -> None:
    item = _spec(
        canonical_component_ids=["domain_adaptation.general"],
        generic_component_ids=["domain_adaptation.general"],
        paper_specific_mechanism_ids=[],
        runtime_ready_adapters=[],
        current_disposition="implementation_request",
        disposition_reason="paper-specific domain adaptation branch is unresolved",
    )
    assert item.current_disposition == "implementation_request"


def test_paper_specific_ids_must_match_resolution_records() -> None:
    with pytest.raises(ValueError, match="must match resolution records"):
        _spec(
            paper_specific_mechanism_ids=["pseudo_iou_quality"],
            paper_mechanism_resolutions=[{
                "paper_id": "arxiv:0000.0001",
                "original_method_name": "correlation quality",
                "paper_specific_mechanism_id": "correlation_quality",
                "canonical_component_id": "loss.quality.correlation",
                "implementation_family": "quality_alignment",
                "paper_config_signature": "a" * 64,
                "compatibility": "compatible",
                "required_adapter": "loss.quality.correlation",
                "execution_fingerprint": "b" * 64,
            }],
        )


def test_paper_specific_id_may_differ_from_canonical_component() -> None:
    item = _spec(
        paper_specific_mechanism_ids=["correlation_quality"],
        paper_mechanism_resolutions=[{
            "paper_id": "arxiv:0000.0001",
            "original_method_name": "correlation quality",
            "paper_specific_mechanism_id": "correlation_quality",
            "canonical_component_id": "loss.quality.correlation",
            "implementation_family": "quality_alignment",
            "paper_config_signature": "a" * 64,
            "compatibility": "compatible",
            "required_adapter": "loss.quality.correlation",
            "execution_fingerprint": "b" * 64,
        }],
    )

    assert item.paper_specific_mechanism_ids == ["correlation_quality"]


def test_inventory_rejects_duplicate_paper_ids() -> None:
    with pytest.raises(ValueError, match="duplicate paper IDs"):
        PaperExecutionInventory(
            source_method_coverage_hash="a" * 64,
            all_paper_count=2,
            compatible_paper_count=2,
            exact_reproduction_candidates=0,
            records=[_spec(), _spec()],
            generic_mechanism_counts={},
        )


def test_inventory_hash_is_deterministic() -> None:
    inventory = PaperExecutionInventory(
        source_method_coverage_hash="a" * 64,
        all_paper_count=1,
        compatible_paper_count=1,
        exact_reproduction_candidates=0,
        records=[_spec()],
        generic_mechanism_counts={},
    )
    hashed = inventory.with_hash()
    assert len(hashed.inventory_hash) == 64
    assert hashed.with_hash().inventory_hash == hashed.inventory_hash


def test_inventory_hash_ignores_generation_timestamps() -> None:
    inventory = PaperExecutionInventory(
        source_method_coverage_hash="a" * 64,
        all_paper_count=1,
        compatible_paper_count=1,
        exact_reproduction_candidates=0,
        records=[_spec()],
        generic_mechanism_counts={},
    ).with_hash()
    later = inventory.model_copy(
        update={
            "generated_at": datetime(2030, 1, 1, tzinfo=timezone.utc),
            "records": [
                inventory.records[0].model_copy(
                    update={"generated_at": datetime(2030, 1, 1, tzinfo=timezone.utc)}
                )
            ],
        }
    )

    assert later.with_hash().inventory_hash == inventory.inventory_hash
