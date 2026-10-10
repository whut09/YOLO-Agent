"""fix 37: stale blocked_runtime dispositions must not retire healed candidates."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from yolo_agent.agents.auto_optimization_loop import _stale_blocked_runtime_candidate_ids
from yolo_agent.agents.paper_proposal_schemas import PaperCandidateCoverage
from yolo_agent.core import ExperimentNode

CONTRACT_YAML = """
components:
  loss.retry_probe:
    schema_version: component_contract.v1
    display_name: Retry Probe
    category: loss
    implementation_path: yolo_agent.components.adapters.dummy
    adapter_class: DummyAdapter
    changed_variable: loss.retry_probe.weight
    insertion_point: loss
    supported_detector_families: [yolo26]
    maturity: smoke_passed
    training_only: true
    inference_only: false
    changes_model_graph: false
    affects_latency: training_only
    affects_model_size: none
    supports_amp: true
    supports_ddp: true
    supports_onnx: true
    supports_tensorrt: true
    fixed_imgsz_compatible: true
    tests_required: []
    known_risks: []
    maturity_artifacts:
      - component_id: loss.retry_probe
        target_maturity: smoke_passed
        artifact_type: smoke_report
        artifact_path: {artifact_path}
        artifact_sha256: {artifact_sha256}
        status: passed
        producer: test
        mock: false
"""


def _node(candidate_id: str, components: list[str]) -> ExperimentNode:
    return ExperimentNode.model_validate(
        {
            "node_id": f"node_{candidate_id}",
            "data_version": "coco2017",
            "candidate_config": {
                "candidate_id": candidate_id,
                "framework": "ultralytics",
                "base_model": "yolo26n.pt",
                "scale": "n",
                "components": components,
                "train_overrides": {},
            },
        }
    )


def _coverage(candidate_ids: list[str]) -> PaperCandidateCoverage:
    return PaperCandidateCoverage.model_validate(
        {
            "run_id": "run-fix37",
            "records": [
                {
                    "run_id": "run-fix37",
                    "recipe_id": candidate_id,
                    "recipe_version": "v1",
                    "candidate_id": candidate_id,
                    "source_stage": "runtime_readiness",
                    "disposition": "blocked_runtime",
                    "reason_codes": ["automatic_runtime_readiness_failed"],
                }
                for candidate_id in candidate_ids
            ],
        }
    )


def _child(snapshot_dir: Path) -> SimpleNamespace:
    metadata = {
        "research_snapshot_verified": True,
        "research_snapshot_path": str(snapshot_dir),
    }
    return SimpleNamespace(context=SimpleNamespace(metadata=metadata))


def _write_snapshot(tmp_path) -> Path:
    import hashlib

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    report = snapshot / "smoke_report.yaml"
    report.write_text("component_id: loss.retry_probe\nresult: passed\n", encoding="utf-8")
    contract = CONTRACT_YAML.format(
        artifact_path=report.as_posix(),
        artifact_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
    )
    (snapshot / "component_contracts.yaml").write_text(contract, encoding="utf-8")
    return snapshot


def test_blocked_runtime_is_retryable_once_contracts_resolve_valid(tmp_path) -> None:
    snapshot = _write_snapshot(tmp_path)

    healed = "paper_recipe_retry_healed"
    still_broken = "paper_recipe_still_broken"
    plan = SimpleNamespace(
        deferred_nodes=[
            _node(healed, ["loss.retry_probe"]),
            _node(still_broken, ["loss.does_not_exist"]),
        ],
        execution_nodes=[],
    )
    coverage = _coverage([healed, still_broken])

    retryable = _stale_blocked_runtime_candidate_ids(_child(snapshot), coverage, plan)

    assert healed in retryable
    assert still_broken not in retryable


def test_blocked_runtime_without_nodes_is_not_retryable(tmp_path) -> None:
    snapshot = _write_snapshot(tmp_path)

    plan = SimpleNamespace(deferred_nodes=[], execution_nodes=[])
    coverage = _coverage(["paper_recipe_absent"])

    assert (
        _stale_blocked_runtime_candidate_ids(_child(snapshot), coverage, plan)
        == set()
    )
