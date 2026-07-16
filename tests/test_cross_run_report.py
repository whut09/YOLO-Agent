"""Cross-run comparison report tests."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.cli import main
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import ExperimentNode, ExperimentPlan
from yolo_agent.reports.cross_run_report import generate_cross_run_comparison_report


def _candidate(candidate_id: str, components: list[str] | None = None) -> CandidateConfig:
    return CandidateConfig(
        candidate_id=candidate_id,
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        components=components or [],
    )


def _write_run(
    root: Path,
    run_id: str,
    candidate_id: str,
    metrics: dict[str, float],
    changed_variables: dict[str, object] | None = None,
    components: list[str] | None = None,
    baseline_metrics: dict[str, float] | None = None,
    dataset_version: str = "dataset-v1",
    manifest_sha: str = "sha-shared",
) -> Path:
    store = EvidenceStore(root)
    run_dir = store.create_run(run_id)
    (run_dir / "run_context.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": run_id,
                "run_root": str(root),
                "task_path": "task.yaml",
                "data_yaml": "data.yaml",
                "dataset_version": dataset_version,
                "dataset_manifest_sha256": manifest_sha,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    node_id = f"node_{candidate_id}"
    nodes = []
    if baseline_metrics is not None:
        nodes.append(
            ExperimentNode(
                node_id="node_baseline",
                candidate_config=_candidate("baseline"),
                data_version=dataset_version,
                command="yolo train ...",
            )
        )
    nodes.append(
        ExperimentNode(
            node_id=node_id,
            candidate_config=_candidate(candidate_id, components),
            data_version=dataset_version,
            command="yolo train ...",
            parent_id="baseline" if baseline_metrics is not None else None,
            changed_variables=changed_variables or {},
        )
    )
    ExperimentPlan(
        plan_id=f"{run_id}_plan",
        nodes=nodes,
    ).to_yaml(run_dir / "experiment_plan.yaml")
    if baseline_metrics is not None:
        store.log_candidate_metrics(
            run_id,
            candidate_id="baseline",
            node_id="node_baseline",
            metrics=baseline_metrics,
            dataset_version=dataset_version,
            dataset_manifest_sha256=manifest_sha,
            source="test",
        )
    store.log_candidate_metrics(
        run_id,
        candidate_id=candidate_id,
        node_id=node_id,
        metrics=metrics,
        dataset_version=dataset_version,
        dataset_manifest_sha256=manifest_sha,
        source="test",
    )
    store.log_metrics(run_id, metrics)
    (run_dir / "evidence_status.json").write_text(
        json.dumps({"ok": True, "trusted": True, "statuses": [], "missing_required": [], "warning": None}),
        encoding="utf-8",
    )
    return run_dir


def test_cross_run_report_compares_dataset_pareto_delta_and_contribution(tmp_path: Path) -> None:
    """Cross-run report should show dataset consistency, metric deltas, and positive actions."""
    runs_root = tmp_path / "runs"
    run1 = _write_run(
        runs_root,
        "exp001",
        "baseline",
        {"map50": 0.6, "recall": 0.7, "latency_ms": 12, "model_size_mb": 5},
    )
    run2 = _write_run(
        runs_root,
        "exp002",
        "nwd",
        {"map50": 0.68, "recall": 0.76, "latency_ms": 10, "model_size_mb": 5},
        changed_variables={"bbox_loss": ["loss.bbox.nwd"]},
        components=["loss.bbox.nwd"],
        baseline_metrics={"map50": 0.6, "recall": 0.7, "latency_ms": 12, "model_size_mb": 5},
    )

    markdown = generate_cross_run_comparison_report([run1, run2], tmp_path / "comparison.md")

    assert "# YOLO Agent Cross-Run Comparison" in markdown
    assert "Dataset version consistent: `True`" in markdown
    assert "Dataset manifest SHA consistent: `True`" in markdown
    assert "| exp001 -> exp002 | +0.08 | +0.06 | -2 |" in markdown
    assert (
        "`exp002`: possible contribution from single-variable ablation `nwd` changed `bbox_loss=['loss.bbox.nwd']`; "
        "parent=`baseline`; improved map50, recall, latency_ms; deltas "
        "map50=+0.08, recall=+0.06, latency_ms=-2; confidence=insufficient_repeated_seeds:1/3"
    ) in markdown
    assert "Added: nwd" in markdown
    assert "Removed: baseline" in markdown


def test_loop_compare_cli_writes_markdown(tmp_path: Path) -> None:
    """loop compare CLI should write a comparison report."""
    runs_root = tmp_path / "runs"
    run1 = _write_run(
        runs_root,
        "exp001",
        "baseline",
        {"map50": 0.6, "recall": 0.7, "latency_ms": 12},
    )
    run2 = _write_run(
        runs_root,
        "exp002",
        "fast",
        {"map50": 0.58, "recall": 0.7, "latency_ms": 8},
        changed_variables={"assigner": ["assigner.stal"]},
        components=["assigner.stal"],
        baseline_metrics={"map50": 0.6, "recall": 0.7, "latency_ms": 12},
    )
    out_path = tmp_path / "comparison.md"

    assert main(["loop", "compare", "--runs", str(run1), str(run2), "--out", str(out_path)]) == 0

    text = out_path.read_text(encoding="utf-8")
    assert "YOLO Agent Cross-Run Comparison" in text
    assert "| exp001 -> exp002 | +0 | +0 | +0 |" in text
    assert (
        "`exp002`: possible contribution from single-variable ablation `fast` changed `assigner=['assigner.stal']`; "
        "parent=`baseline`; improved latency_ms; deltas map50=-0.02, recall=+0, latency_ms=-4; "
        "confidence=insufficient_repeated_seeds:1/3"
    ) in text


def test_cross_run_report_does_not_attribute_multi_variable_candidate(tmp_path: Path) -> None:
    """Contribution text should skip candidates that changed multiple variables."""
    runs_root = tmp_path / "runs"
    run1 = _write_run(
        runs_root,
        "exp001",
        "baseline",
        {"map50": 0.6, "recall": 0.7, "latency_ms": 12},
    )
    run2 = _write_run(
        runs_root,
        "exp002",
        "multi",
        {"map50": 0.72, "recall": 0.78, "latency_ms": 13},
        changed_variables={
            "bbox_loss": ["loss.bbox.nwd"],
            "head_component": ["head.p2_small_object"],
        },
        components=["loss.bbox.nwd", "head.p2_small_object"],
        baseline_metrics={"map50": 0.6, "recall": 0.7, "latency_ms": 12},
    )

    markdown = generate_cross_run_comparison_report([run1, run2], tmp_path / "comparison.md")

    assert "| exp001 -> exp002 | +0.12 | +0.08 | +1 |" in markdown
    assert "single-variable ablation `multi`" not in markdown
    assert "No single-variable ablation contribution with trusted parent evidence detected." in markdown


def test_cross_run_report_marks_single_seed_confidence_interval_as_possible(tmp_path: Path) -> None:
    """Confidence intervals alone do not confirm contribution without repeated seeds."""
    runs_root = tmp_path / "runs"
    run1 = _write_run(
        runs_root,
        "exp001",
        "baseline",
        {"map50": 0.6, "map50_ci_low": 0.58, "map50_ci_high": 0.62, "latency_ms": 12},
    )
    run2 = _write_run(
        runs_root,
        "exp002",
        "nwd",
        {"map50": 0.7, "map50_ci_low": 0.68, "map50_ci_high": 0.72, "latency_ms": 12},
        changed_variables={"bbox_loss": ["loss.bbox.nwd"]},
        components=["loss.bbox.nwd"],
        baseline_metrics={"map50": 0.6, "map50_ci_low": 0.58, "map50_ci_high": 0.62, "latency_ms": 12},
    )

    markdown = generate_cross_run_comparison_report([run1, run2], tmp_path / "comparison.md")

    assert (
        "`exp002`: possible contribution from single-variable ablation `nwd` "
        "changed `bbox_loss=['loss.bbox.nwd']`; parent=`baseline`; improved map50; "
        "deltas map50=+0.1, latency_ms=+0; "
        "confidence=insufficient_repeated_seeds:1/3;confidence_interval_present_but_not_confirmatory:map50"
    ) in markdown


def test_cross_run_report_confirms_contribution_with_repeated_seeds(tmp_path: Path) -> None:
    """Repeated-seed single-variable ablations should be marked confirmed."""
    runs_root = tmp_path / "runs"
    run1 = _write_run(
        runs_root,
        "exp001",
        "baseline",
        {"map50": 0.6, "latency_ms": 12},
    )
    store = EvidenceStore(runs_root)
    run2 = store.create_run("exp002")
    (run2 / "run_context.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "exp002",
                "run_root": str(runs_root),
                "task_path": "task.yaml",
                "data_yaml": "data.yaml",
                "dataset_version": "dataset-v1",
                "dataset_manifest_sha256": "sha-shared",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    nodes = [
        ExperimentNode(
            node_id="node_baseline",
            candidate_config=_candidate("baseline"),
            data_version="dataset-v1",
            command="yolo train ...",
            seed=1,
        ),
        ExperimentNode(
            node_id="node_nwd_seed1",
            candidate_config=_candidate("nwd_seed1", ["loss.bbox.nwd"]),
            data_version="dataset-v1",
            command="yolo train ...",
            parent_id="baseline",
            changed_variables={"bbox_loss": ["loss.bbox.nwd"]},
            seed=1,
        ),
        ExperimentNode(
            node_id="node_nwd_seed2",
            candidate_config=_candidate("nwd_seed2", ["loss.bbox.nwd"]),
            data_version="dataset-v1",
            command="yolo train ...",
            parent_id="baseline",
            changed_variables={"bbox_loss": ["loss.bbox.nwd"]},
            seed=2,
        ),
        ExperimentNode(
            node_id="node_nwd_seed3",
            candidate_config=_candidate("nwd_seed3", ["loss.bbox.nwd"]),
            data_version="dataset-v1",
            command="yolo train ...",
            parent_id="baseline",
            changed_variables={"bbox_loss": ["loss.bbox.nwd"]},
            seed=3,
        ),
    ]
    ExperimentPlan(plan_id="exp002_plan", nodes=nodes).to_yaml(run2 / "experiment_plan.yaml")
    store.log_candidate_metrics("exp002", "baseline", "node_baseline", {"map50": 0.6, "latency_ms": 12}, dataset_manifest_sha256="sha-shared", seed=1)
    store.log_candidate_metrics("exp002", "nwd_seed1", "node_nwd_seed1", {"map50": 0.66, "latency_ms": 12}, dataset_manifest_sha256="sha-shared", seed=1)
    store.log_candidate_metrics("exp002", "nwd_seed2", "node_nwd_seed2", {"map50": 0.67, "latency_ms": 12}, dataset_manifest_sha256="sha-shared", seed=2)
    store.log_candidate_metrics("exp002", "nwd_seed3", "node_nwd_seed3", {"map50": 0.665, "latency_ms": 12}, dataset_manifest_sha256="sha-shared", seed=3)
    store.log_metrics("exp002", {"map50": 0.67, "latency_ms": 12})
    (run2 / "evidence_status.json").write_text(
        json.dumps({"ok": True, "trusted": True, "statuses": [], "missing_required": [], "warning": None}),
        encoding="utf-8",
    )

    markdown = generate_cross_run_comparison_report([run1, run2], tmp_path / "comparison.md")

    assert "confirmed contribution from single-variable ablation `nwd_seed1`" in markdown
    assert "confidence=repeated_seeds:3" in markdown
