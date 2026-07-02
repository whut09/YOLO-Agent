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
    ExperimentPlan(
        plan_id=f"{run_id}_plan",
        nodes=[
            ExperimentNode(
                node_id=node_id,
                candidate_config=_candidate(candidate_id, components),
                data_version=dataset_version,
                command="yolo train ...",
                changed_variables=changed_variables or {},
            )
        ],
    ).to_yaml(run_dir / "experiment_plan.yaml")
    store.log_candidate_metrics(
        run_id,
        candidate_id=candidate_id,
        node_id=node_id,
        metrics=metrics,
        dataset_version=dataset_version,
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
    )

    markdown = generate_cross_run_comparison_report([run1, run2], tmp_path / "comparison.md")

    assert "# YOLO Agent Cross-Run Comparison" in markdown
    assert "Dataset version consistent: `True`" in markdown
    assert "Dataset manifest SHA consistent: `True`" in markdown
    assert "| exp001 -> exp002 | +0.08 | +0.06 | -2 |" in markdown
    assert "`bbox_loss=['loss.bbox.nwd']` may have contributed positively to map50, recall, latency_ms" in markdown
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
    )
    out_path = tmp_path / "comparison.md"

    assert main(["loop", "compare", "--runs", str(run1), str(run2), "--out", str(out_path)]) == 0

    text = out_path.read_text(encoding="utf-8")
    assert "YOLO Agent Cross-Run Comparison" in text
    assert "| exp001 -> exp002 | -0.02 | +0 | -4 |" in text
    assert "`assigner=['assigner.stal']` may have contributed positively to latency_ms" in text
