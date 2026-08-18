"""PaperRecipeSpec tests. No GPU training."""

from __future__ import annotations

import pytest

from pydantic import ValidationError

from yolo_agent.recipes.paper_recipe_spec import (
    PaperRecipeSpec,
    compute_paper_recipe_fingerprint,
    queue_disposition,
)


def _spec(**overrides: object) -> PaperRecipeSpec:
    payload = {
        "recipe_id": "paper_test",
        "paper_ids": ["arxiv:2103.14259"],
        "method_profile_ids": ["method-profile-test"],
        "paper_specific_mechanism_id": "assigner.optimal_transport",
        "canonical_component_ids": ["assigner.optimal_transport"],
        "changed_variables": {"assigner.kind": "optimal_transport"},
        "runtime_plugin": "assigner.optimal_transport",
        "protocol_hash": "a" * 64,
        "required_evidence": ["paper_protocol_contract"],
        "expected_metrics": ["map50_95"],
        "stop_conditions": ["pilot_no_gain"],
        "compatibility_requirements": ["imgsz_640"],
        "target_error_facts": [{"fact_type": "assignment_conflict"}],
        "disposition": "queued",
    }
    payload.update(overrides)
    return PaperRecipeSpec.model_validate(payload)


def test_generic_mechanism_is_rejected() -> None:
    with pytest.raises(ValidationError, match="generic"):
        _spec(paper_specific_mechanism_id="domain_adaptation.general")


def test_empty_facts_cannot_be_queued() -> None:
    with pytest.raises(ValidationError, match="empty target_error_facts"):
        _spec(target_error_facts=[], disposition="queued")


def test_inference_only_cannot_be_queued() -> None:
    with pytest.raises(ValidationError, match="inference-only"):
        _spec(inference_only=True, disposition="queued")


def test_fingerprint_includes_required_identity_fields() -> None:
    spec = _spec()
    assert spec.execution_fingerprint == compute_paper_recipe_fingerprint(spec)
    assert len(spec.execution_fingerprint) == 64
    other = _spec(seed=7, recipe_id="paper_test_seed")
    assert other.execution_fingerprint != spec.execution_fingerprint


def test_queue_disposition_matrix() -> None:
    assert queue_disposition(target_error_facts=[], inference_only=False, has_runtime_adapter=True) == "evidence_recovery"
    assert queue_disposition(target_error_facts=[{"fact_type": "x"}], inference_only=True, has_runtime_adapter=True) == "blocked_runtime"
    assert queue_disposition(target_error_facts=[{"fact_type": "x"}], inference_only=False, has_runtime_adapter=False) == "implementation_request"
    assert queue_disposition(target_error_facts=[{"fact_type": "x"}], inference_only=False, has_runtime_adapter=True) == "queued"
