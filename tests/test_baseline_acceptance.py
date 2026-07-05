"""COCO baseline acceptance gate tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.adapters.ultralytics.baseline_acceptance import (
    BaselineAcceptanceConfig,
    BaselineAcceptanceGate,
    BaselineAcceptanceResult,
)
from yolo_agent.adapters.ultralytics.training import UltralyticsTrainingConfig
from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluator
from yolo_agent.agents.strategy_policy import CandidatePolicy
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.task_spec import MetricPriority, TaskSpec


def test_baseline_acceptance_gate_accepts_verified_three_seed_coco_baseline(tmp_path: Path) -> None:
    """A trusted baseline needs verified mAP, artifacts, fixed imgsz, sha match, and seed count."""
    store = EvidenceStore(tmp_path / "runs")
    for seed in [1, 2, 3]:
        _write_seed_evidence(store, tmp_path, "exp001", seed)

    result = BaselineAcceptanceGate().check(
        store.load_run("exp001"),
        expected_dataset_manifest_sha256="coco-sha",
        actual_dataset_manifest_sha256="coco-sha",
    )

    assert result.baseline_trusted is True
    assert result.baseline_rejection_reason == []
    assert result.accepted_seed_count == 3
    assert result.accepted_nodes == ["node_baseline_seed1", "node_baseline_seed2", "node_baseline_seed3"]


def test_baseline_acceptance_gate_rejects_missing_required_artifact(tmp_path: Path) -> None:
    """A metric alone is not enough; the baseline must carry auditable train artifacts."""
    store = EvidenceStore(tmp_path / "runs")
    _write_seed_evidence(store, tmp_path, "exp001", 1, include_best=False)

    result = BaselineAcceptanceGate(BaselineAcceptanceConfig(minimum_seeds=1)).check(
        store.load_run("exp001"),
        actual_dataset_manifest_sha256="coco-sha",
    )

    assert result.baseline_trusted is False
    assert "node_baseline_seed1:missing_artifact:best_pt" in result.baseline_rejection_reason


def test_baseline_acceptance_gate_rejects_non_comparable_imgsz(tmp_path: Path) -> None:
    """COCO baseline acceptance should enforce the fixed 640 input-size protocol."""
    store = EvidenceStore(tmp_path / "runs")
    _write_seed_evidence(store, tmp_path, "exp001", 1, imgsz=960)

    result = BaselineAcceptanceGate(BaselineAcceptanceConfig(minimum_seeds=1)).check(
        store.load_run("exp001"),
        actual_dataset_manifest_sha256="coco-sha",
    )

    assert result.baseline_trusted is False
    assert "node_baseline_seed1:imgsz_mismatch:960!=640" in result.baseline_rejection_reason


def test_candidate_full_policy_waits_for_trusted_baseline() -> None:
    """candidate_full proposals must not become experiment nodes before baseline acceptance."""
    proposal = CandidatePolicy(
        policy_id="candidate_full_nwd",
        source="rule_engine",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.nwd"],
    )
    config = UltralyticsTrainingConfig(
        model="yolo26n.pt",
        data=Path("configs/datasets/coco.yaml"),
        budget_profile="candidate_full",
    )
    baseline = BaselineAcceptanceResult(
        baseline_trusted=False,
        baseline_rejection_reason=["insufficient_confirmed_seeds:1/3"],
    )

    evaluation = _evaluator().evaluate_one(
        proposal,
        _task(),
        training_config=config,
        baseline_acceptance=baseline,
    )

    assert evaluation.decision == "needs_evidence"
    assert evaluation.missing_evidence == ["baseline_trusted"]
    assert evaluation.candidate_config is None
    assert "insufficient_confirmed_seeds:1/3" in evaluation.warnings


def test_candidate_full_policy_runs_after_trusted_baseline() -> None:
    """A trusted baseline unlocks candidate_full planning."""
    proposal = CandidatePolicy(
        policy_id="candidate_full_nwd",
        source="rule_engine",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        components=["loss.bbox.nwd"],
    )
    config = UltralyticsTrainingConfig(
        model="yolo26n.pt",
        data=Path("configs/datasets/coco.yaml"),
        budget_profile="candidate_full",
    )
    baseline = BaselineAcceptanceResult(baseline_trusted=True, accepted_seed_count=3)

    evaluation = _evaluator().evaluate_one(
        proposal,
        _task(),
        training_config=config,
        baseline_acceptance=baseline,
    )

    assert evaluation.decision == "accepted"
    assert evaluation.experiment_node is not None
    assert evaluation.experiment_node.command_spec is not None
    assert evaluation.experiment_node.command_spec.metadata["training_budget_profile"] == "candidate_full"


def _write_seed_evidence(
    store: EvidenceStore,
    tmp_path: Path,
    run_id: str,
    seed: int,
    imgsz: int = 640,
    include_best: bool = True,
) -> None:
    node_id = f"node_baseline_seed{seed}"
    candidate_id = "yolo26s_coco_baseline"
    ultra_dir = tmp_path / "ultra" / node_id
    weights = ultra_dir / "weights"
    weights.mkdir(parents=True)
    results_csv = ultra_dir / "results.csv"
    args_yaml = ultra_dir / "args.yaml"
    best_pt = weights / "best.pt"
    results_csv.write_text(
        "epoch,metrics/mAP50-95(B)\n0,0.40\n",
        encoding="utf-8",
    )
    args_yaml.write_text(f"imgsz: {imgsz}\nepochs: 100\n", encoding="utf-8")
    if include_best:
        best_pt.write_bytes(b"weights")

    store.create_run(run_id)
    store.log_artifact_manifest(run_id, f"{node_id}_results_csv", results_csv, "ultralytics_train")
    store.log_artifact_manifest(run_id, f"{node_id}_args_yaml", args_yaml, "ultralytics_train")
    if include_best:
        store.log_artifact_manifest(run_id, f"{node_id}_best_pt", best_pt, "ultralytics_train")
    store.log_candidate_metrics(
        run_id,
        candidate_id,
        node_id,
        {"map50_95": 0.4},
        dataset_version="coco2017",
        split="val",
        source="ultralytics_train",
        validator="ultralytics_results_importer",
    )
    store.log_candidate_metrics(
        run_id,
        candidate_id,
        node_id,
        {
            "training_budget_profile": "baseline_confirm",
            "fast_baseline_seed": seed,
        },
        dataset_version="coco2017",
        split="runtime",
        source="fast_baseline_gate",
        validator="fast_baseline_gate",
    )


def _task() -> TaskSpec:
    return TaskSpec(
        task_type="detect",
        scene="generic",
        class_names=["object"],
        primary_metric=MetricPriority(name="map50_95"),
    )


def _evaluator() -> LoopPolicyEvaluator:
    return LoopPolicyEvaluator(ComponentRegistry.from_path("configs/components"), fixed_imgsz=640)
