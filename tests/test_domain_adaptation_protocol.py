"""CPU-only tests for typed domain-adaptation protocols and route boundaries."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.domain_adaptation.branch_runtime import (
    DomainAdaptationBranchAdapter,
)
from yolo_agent.components.adapters.domain_adaptation.branches import (
    CANONICAL_DOMAIN_BRANCHES,
    DomainProtocolError,
    canonical_branch_id,
    default_domain_adaptation_registry,
)
from yolo_agent.components.adapters.domain_adaptation.domain_evidence import (
    DomainDatasetManifest,
    DomainProtocolResolution,
    resolve_domain_protocol,
)
from yolo_agent.components.contracts import ComponentContract


def _manifest(role: str, domain_id: str, dataset_hash: str, labels: str = "labeled") -> DomainDatasetManifest:
    return DomainDatasetManifest(
        path=f"{role}.manifest.yaml",
        sha256=f"{role}-sha256",
        dataset_hash=dataset_hash,
        domain_id=domain_id,
        domain_name=role,
        role=role,
        split=f"{role}_train",
        label_availability=labels,
    )


def _protocol(branch_id: str) -> DomainProtocolResolution:
    source_free = branch_id == "source_free_adaptation"
    return resolve_domain_protocol(
        source=None if source_free else _manifest("source", "0", "source-dataset"),
        target=_manifest("target", "1", "target-dataset", "unlabeled"),
        adaptation_mode="source_free" if source_free else "unsupervised",
        source_free=source_free,
        source_model_checkpoint_sha256="model-sha256" if source_free else None,
        source_model_protocol_hash="model-protocol" if source_free else None,
    )


def _contract(component_id: str) -> ComponentContract:
    return ComponentContract(
        component_id=component_id,
        display_name=component_id,
        category="domain_adaptation",
        implementation_path="yolo_agent.components.adapters.domain_adaptation.branch_runtime",
        adapter_class="DomainAdaptationBranchAdapter",
        maturity="adapter_implemented",
    )


def test_complete_domain_pair_is_typed_and_hashed() -> None:
    result = _protocol("feature_alignment")
    assert result.ok is True
    assert result.pair is not None
    assert result.protocol_hash
    payload = result.runtime_payload()
    assert payload["source_dataset_hash"] != payload["target_dataset_hash"]
    assert payload["source_split"] != payload["target_split"]
    assert payload["domain_protocol_hash"] == result.protocol_hash


@pytest.mark.parametrize(
    "source,target,expected",
    [
        (None, _manifest("target", "1", "target"), "source_domain_manifest_missing"),
        (_manifest("source", "0", "source"), None, "target_domain_manifest_missing"),
        (_manifest("source", "0", "same"), _manifest("target", "1", "same"), "source_target_manifest_identity_collision"),
    ],
)
def test_incomplete_or_colliding_domain_evidence_fails_closed(source, target, expected: str) -> None:
    result = resolve_domain_protocol(
        source=source,
        target=target,
        adaptation_mode="unsupervised",
    )
    assert result.ok is False
    assert expected in result.reason_codes
    assert result.recovery_action


def test_coco_supervised_manifest_cannot_be_relabelled_as_domain() -> None:
    source = _manifest("source", "0", "coco")
    source.is_coco_supervised = True
    target = _manifest("target", "1", "target")
    with pytest.raises(ValidationError, match="COCO"):
        resolve_domain_protocol(source=source, target=target, adaptation_mode="unsupervised")


def test_source_free_requires_target_and_source_model_protocol() -> None:
    missing = resolve_domain_protocol(
        source=None,
        target=_manifest("target", "1", "target", "unlabeled"),
        adaptation_mode="source_free",
        source_free=True,
    )
    assert missing.ok is False
    assert "source_free_source_model_evidence_missing" in missing.reason_codes
    ready = _protocol("source_free_adaptation")
    assert ready.ok is True
    assert ready.pair is not None
    assert ready.pair.source_domain_id == "source_model"


def test_each_canonical_route_has_independent_contract_and_payload() -> None:
    registry = default_domain_adaptation_registry()
    fingerprints = set()
    changed = set()
    for branch_id in CANONICAL_DOMAIN_BRANCHES:
        branch = registry.get(branch_id)
        assert branch.component_id == f"domain_adaptation.{branch_id}"
        assert branch.payload_schema
        assert branch.required_evidence
        fingerprints.add(branch.execution_fingerprint)
        changed.add(branch.changed_variable)
    assert len(fingerprints) == len(CANONICAL_DOMAIN_BRANCHES)
    assert len(changed) == len(CANONICAL_DOMAIN_BRANCHES)


def test_legacy_aliases_are_explicit_only() -> None:
    assert canonical_branch_id("adversarial_feature_alignment") == "adversarial_alignment"
    with pytest.raises(DomainProtocolError, match="unknown"):
        canonical_branch_id("adversarial_alignment_typo")


@pytest.mark.parametrize("branch_id", CANONICAL_DOMAIN_BRANCHES)
def test_runtime_payload_requires_complete_domain_evidence(branch_id: str) -> None:
    branch = default_domain_adaptation_registry().get(branch_id)
    adapter = DomainAdaptationBranchAdapter(branch_id)
    context = AdapterContext(
        contract=_contract(branch.component_id),
        detector_family="yolo26",
        options={"branch_id": branch_id},
    )
    with pytest.raises(DomainProtocolError, match="DomainProtocolResolution"):
        adapter.build_runtime_payload(
            context,
            protocol_hash="protocol",
            base_command=["train"],
            generated_config={},
        )


@pytest.mark.parametrize("branch_id", CANONICAL_DOMAIN_BRANCHES)
def test_runtime_payload_contains_route_identity(branch_id: str) -> None:
    branch = default_domain_adaptation_registry().get(branch_id)
    adapter = DomainAdaptationBranchAdapter(branch_id)
    options = {"branch_id": branch_id, "domain_protocol": _protocol(branch_id).model_dump(mode="json")}
    if branch.runtime_strategy == "target_pseudo_label_consistency":
        options["pseudo_label_manifest"] = "pseudo-labels.yaml"
    elif branch.runtime_strategy in {"domain_teacher_distillation", "cross_domain_teacher"}:
        options.update({"teacher_checkpoint": "teacher.pt", "teacher_sha256": "teacher-sha256"})
    elif branch.runtime_strategy == "source_free_target_adaptation":
        options.update({"source_model_checkpoint": "source.pt", "source_model_sha256": "model-sha256"})
    elif branch.runtime_strategy == "cross_domain_contrastive":
        options["contrastive_pair_manifest"] = "pairs.yaml"
    elif branch.runtime_strategy == "active_query_selection":
        options.update({"query_manifest": "queries.yaml", "label_budget": 10})
    context = AdapterContext(
        contract=_contract(branch.component_id),
        detector_family="yolo26",
        options=options,
    )
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="protocol",
        base_command=["train"],
        generated_config={},
    )
    assert payload is not None
    assert payload.component_ids == [branch.component_id]
    assert payload.changed_variables[branch.changed_variable] == 0.05
    plugin_options = payload.loss_plugin[0].options
    assert plugin_options["domain_protocol_hash"]
    assert plugin_options["runtime_strategy"] == branch.runtime_strategy
