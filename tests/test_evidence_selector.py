"""Strict metric evidence provenance selector tests."""

from __future__ import annotations

from yolo_agent.core.evidence_selector import EvidenceSelector, select_metric_evidence
from yolo_agent.core.experiment_graph import MetricEvidence


def _record(**updates: object) -> MetricEvidence:
    data = {
        "candidate_id": "candidate",
        "node_id": "node_candidate",
        "run_id": "child",
        "origin_run_id": "child",
        "metric_name": "map50_95",
        "value": 0.4,
        "split": "val2017",
        "protocol_hash": "protocol-1",
        "dataset_manifest_sha256": "manifest-1",
        "seed": 1,
        "verified": True,
    }
    data.update(updates)
    return MetricEvidence.model_validate(data)


def test_current_selector_excludes_inherited_baseline_context() -> None:
    current = _record()
    inherited = _record(
        origin_run_id="parent",
        evidence_role="baseline_reference",
        inheritance_depth=1,
        source="inherited:parent:test",
        value=0.9,
    )

    selection = select_metric_evidence(
        [current, inherited],
        EvidenceSelector(
            current_run_id="child",
            current_run_only=True,
            current_node_only=["node_candidate"],
            inherited_context=False,
            baseline_reference=False,
            same_protocol_hash="protocol-1",
            same_dataset_manifest="manifest-1",
            same_split="val2017",
            same_seed=1,
        ),
    )

    assert selection.records == [current]
    assert selection.rejected_by["not_current_run"] == 1


def test_current_run_baseline_reference_is_not_treated_as_inherited() -> None:
    baseline = _record(
        candidate_id="baseline",
        node_id="node_baseline",
        evidence_role="baseline_reference",
    )

    selection = select_metric_evidence(
        [baseline],
        EvidenceSelector(
            current_run_id="child",
            current_run_only=True,
            current_node_only=["node_baseline"],
            inherited_context=False,
            baseline_reference=True,
            same_protocol_hash="protocol-1",
        ),
    )

    assert selection.records == [baseline]
    assert selection.rejected_by == {}


def test_parent_run_baseline_reference_is_rejected_from_current_run() -> None:
    inherited = _record(
        candidate_id="baseline",
        node_id="node_baseline",
        evidence_role="baseline_reference",
        origin_run_id="parent",
        inheritance_depth=1,
    )

    selection = select_metric_evidence(
        [inherited],
        EvidenceSelector(
            current_run_id="child",
            current_run_only=True,
            baseline_reference=True,
        ),
    )

    assert selection.records == []
    assert selection.rejected_by == {"not_current_run": 1}


def test_selector_rejects_protocol_manifest_split_and_seed_mismatches() -> None:
    records = [
        _record(protocol_hash="wrong"),
        _record(dataset_manifest_sha256="wrong"),
        _record(split="test2017"),
        _record(seed=2),
    ]
    selection = select_metric_evidence(
        records,
        EvidenceSelector(
            current_run_id="child",
            current_run_only=True,
            same_protocol_hash="protocol-1",
            same_dataset_manifest="manifest-1",
            same_split="val2017",
            same_seed=1,
        ),
    )

    assert selection.records == []
    assert selection.rejected_by == {
        "protocol_hash_mismatch": 1,
        "dataset_manifest_mismatch": 1,
        "split_mismatch": 1,
        "seed_mismatch": 1,
    }


def test_protocol_can_be_resolved_from_node_companion_record() -> None:
    metric = _record(protocol_hash=None)
    companion = _record(
        protocol_hash=None,
        metric_name="baseline_protocol_hash",
        value="protocol-1",
        split="protocol",
    )

    selection = select_metric_evidence(
        [metric, companion],
        EvidenceSelector(
            current_run_id="child",
            current_run_only=True,
            current_node_only=["node_candidate"],
            same_protocol_hash="protocol-1",
            same_split="val2017",
        ),
    )

    assert selection.records == [metric]
