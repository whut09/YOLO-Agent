from __future__ import annotations

import hashlib

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


def test_mechanism_partitions_must_match_canonical_components() -> None:
    with pytest.raises(ValueError, match="canonical component IDs"):
        _spec(paper_specific_mechanism_ids=["loss.quality.pseudo_iou"])


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
