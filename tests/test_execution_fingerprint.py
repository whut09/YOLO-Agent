from __future__ import annotations

from tests.paired_result_helpers import verified_paired_result
from tests.paper_materialization_fixtures import node
from yolo_agent.core.execution_fingerprint import (
    canonical_component_ids,
    execution_fingerprint,
    execution_identity_payload,
    paired_evidence_is_valid,
)
from yolo_agent.core.policy_memory import ActionFingerprint


def _paper_node(*, component: str, paper_id: str, protocol: str = "protocol-640"):
    base = node("paper-candidate")
    candidate = base.candidate_config.model_copy(update={"components": [component]})
    metadata = dict(base.command_spec.metadata)
    metadata.update(
        {
            "paper_id": paper_id,
            "component_recipe_id": "recipe-quality",
            "component_recipe_version": "v1",
            "baseline_protocol_hash": protocol,
            "dataset_manifest_sha256": "dataset-1",
            "fidelity": "pilot_10",
            "teacher_checkpoint_sha256": "teacher-a",
            "graph_identity_hash": "graph-a",
            "adapter_runtime_payload_hash": "payload-a",
            "ablation_combination_id": "atomic",
        }
    )
    return base.model_copy(
        update={
            "candidate_config": candidate,
            "command_spec": base.command_spec.model_copy(update={"metadata": metadata}),
        }
    )


def test_aliases_resolve_to_one_canonical_execution_component() -> None:
    assert canonical_component_ids(["rtmdet_large_kernel_neck"]) == [
        "neck.rtmdet_large_kernel"
    ]
    assert canonical_component_ids(["neck.rtmdet_large_kernel"]) == [
        "neck.rtmdet_large_kernel"
    ]


def test_paper_provenance_does_not_duplicate_same_execution() -> None:
    first = _paper_node(component="neck.rtmdet_large_kernel", paper_id="paper-a")
    second = _paper_node(component="neck.rtmdet_large_kernel", paper_id="paper-b")

    assert execution_fingerprint(first) == execution_fingerprint(second)


def test_same_paper_different_component_or_teacher_is_not_deduplicated() -> None:
    first = _paper_node(component="neck.rtmdet_large_kernel", paper_id="paper-a")
    different_component = _paper_node(component="loss.quality.correlation", paper_id="paper-a")
    different_teacher = _paper_node(component="neck.rtmdet_large_kernel", paper_id="paper-a")
    teacher_metadata = dict(different_teacher.command_spec.metadata)
    teacher_metadata["teacher_checkpoint_sha256"] = "teacher-b"
    different_teacher.command_spec = different_teacher.command_spec.model_copy(
        update={"metadata": teacher_metadata}
    )

    assert execution_fingerprint(first) != execution_fingerprint(different_component)
    assert execution_fingerprint(first) != execution_fingerprint(different_teacher)


def test_execution_identity_contains_protocol_runtime_and_combination_dimensions() -> None:
    identity = execution_identity_payload(
        _paper_node(component="neck.rtmdet_large_kernel", paper_id="paper-a")
    )

    assert identity["model_checkpoint_identity"] == "yolo26n.pt"
    assert identity["canonical_component_ids"] == ["neck.rtmdet_large_kernel"]
    assert identity["recipe_id"] == "recipe-quality"
    assert identity["recipe_version"] == "v1"
    assert identity["dataset_manifest_hash"] == "dataset-1"
    assert identity["baseline_protocol_hash"] == "protocol-640"
    assert identity["imgsz"] == 640
    assert identity["fidelity"] == "pilot_10"
    assert identity["teacher_checkpoint_hash"] == "teacher-a"
    assert identity["graph_identity_hash"] == "graph-a"
    assert identity["runtime_payload_hash"] == "payload-a"
    assert identity["combination_id"] == "atomic"


def test_old_or_protocol_mismatched_paired_artifact_is_not_already_tested() -> None:
    paired = verified_paired_result(
        candidate_id="candidate",
        node_id="node-candidate",
        delta=0.01,
        protocol_hash="old-protocol",
    )

    assert not paired_evidence_is_valid(
        paired,
        expected_candidate_id="candidate",
        expected_protocol_hash="current-protocol",
        expected_dataset_manifest_hash="dataset",
    )
    assert paired_evidence_is_valid(
        paired,
        expected_candidate_id="candidate",
        expected_protocol_hash="old-protocol",
        expected_dataset_manifest_hash="dataset",
    )


def test_policy_memory_fingerprint_ignores_paper_ids_but_keeps_execution_fields() -> None:
    common = dict(
        action="recipe-quality",
        recipe_id="recipe-quality",
        recipe_version="v1",
        component_ids=["neck.rtmdet_large_kernel"],
        dataset_signature="dataset-1",
        protocol_hash="protocol-640",
        fidelity="pilot_10",
        seed=42,
        model_checkpoint_identity="yolo26n.pt",
        effective_overrides={"neck.kernel": 5},
        dataset_manifest_hash="dataset-1",
        baseline_protocol_hash="protocol-640",
        teacher_checkpoint_hash="teacher-a",
        graph_identity_hash="graph-a",
        runtime_payload_hash="payload-a",
    )
    first = ActionFingerprint(paper_ids=["paper-a"], **common)
    second = ActionFingerprint(paper_ids=["paper-b"], **common)
    changed = ActionFingerprint(paper_ids=["paper-a"], **{**common, "teacher_checkpoint_hash": "teacher-b"})

    assert first.fingerprint_sha256 == second.fingerprint_sha256
    assert first.fingerprint_sha256 != changed.fingerprint_sha256
