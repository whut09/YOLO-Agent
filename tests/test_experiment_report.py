"""Experiment report tests."""

from __future__ import annotations

from pathlib import Path

import json

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.cli import main
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import ExperimentNode, ExperimentPlan
from yolo_agent.reports.experiment_report import NO_EVIDENCE_WARNING, generate_experiment_report


def _candidate(candidate_id: str, risk: str = "low") -> CandidateConfig:
    return CandidateConfig(
        candidate_id=candidate_id,
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        components=[],
        risk=risk,  # type: ignore[arg-type]
    )


def _write_report_context(run_dir: Path) -> None:
    plan = ExperimentPlan(
        plan_id="plan-001",
        nodes=[
            ExperimentNode(
                node_id="report-run",
                candidate_config=_candidate("baseline"),
                data_version="dataset-v1",
                command="yolo train ...",
                metrics={"map50": 0.6, "precision": 0.7, "recall": 0.8, "latency_ms": 12, "model_size_mb": 5},
            ),
            ExperimentNode(
                node_id="candidate-missing",
                candidate_config=_candidate("p2_head", risk="medium"),
                data_version="dataset-v1",
                command="yolo train ...",
                changed_variables={"head_component": {"from": [], "to": ["head.p2_small_object"]}},
            ),
            ExperimentNode(
                node_id="fast-run",
                candidate_config=CandidateConfig(
                    candidate_id="fast",
                    base_model="yolo11n",
                    scale="n",
                    framework="ultralytics",
                    components=["assigner.stal"],
                ),
                data_version="dataset-v1",
                command="yolo train ...",
                metrics={"map50": 0.55, "precision": 0.8, "recall": 0.6, "latency_ms": 6, "model_size_mb": 3},
            ),
        ],
    )
    plan.to_yaml(run_dir / "experiment_plan.yaml")
    (run_dir / "dataset_report.json").write_text(
        json.dumps(
            {
                "image_count": 10,
                "label_count": 5,
                "object_size_ratio": {"small": 0.8},
                "empty_label_images": 2,
                "missing_label_files": 1,
                "recommendations": ["Enable the small-object recipe."],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "ablation_plan.yaml").write_text(
        "\n".join(
            [
                "baseline_id: baseline",
                "nodes:",
                "- node_id: ablate_p2_head",
                "  changed_variables:",
                "    head_component:",
                "      from: []",
                "      to: [head.p2_small_object]",
                "invalid_candidates: []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "evidence_status.json").write_text(
        json.dumps(
            {
                "ok": True,
                "trusted": True,
                "statuses": [],
                "missing_required": [],
                "warning": None,
            }
        ),
        encoding="utf-8",
    )


def test_experiment_report_marks_missing_evidence_and_recommends_best(tmp_path: Path) -> None:
    """Report should not trust candidates without evidence or invented metrics."""
    store = EvidenceStore(tmp_path / "runs")
    run_dir = store.create_run("report-run")
    store.log_config("report-run", {"task_spec": {"scene": "infrared_small_target", "task_type": "detect"}})
    store.log_metrics("report-run", {"map50": 0.6})
    _write_report_context(run_dir)

    markdown = generate_experiment_report(run_dir, tmp_path / "report.md")

    assert "# YOLO Agent Experiment Report" in markdown
    assert "## Task Profile" in markdown
    assert "Images: 10" in markdown
    assert "| baseline | 0.6 | 0.7 | 0.8 | 12 | 5 | ok |" in markdown
    assert f"| p2_head | unknown | unknown | unknown | unknown | unknown | {NO_EVIDENCE_WARNING} |" in markdown
    assert "## Pareto Front" in markdown
    assert "## Evidence Gate" in markdown
    assert "Trusted: `True`" in markdown
    assert "Recommend evaluating Pareto-front candidates" in markdown
    assert "`baseline`" in markdown
    assert "`fast`" in markdown
    assert NO_EVIDENCE_WARNING in markdown


def test_report_cli_writes_markdown(tmp_path: Path) -> None:
    """The report CLI should write a Markdown report from a run directory."""
    store = EvidenceStore(tmp_path / "runs")
    run_dir = store.create_run("cli-run")
    store.log_config("cli-run", {"model": "yolo11n"})
    store.log_metrics("cli-run", {"precision": 0.9})
    out_path = tmp_path / "report.md"

    assert main(["report", "--run", str(run_dir), "--out", str(out_path)]) == 0

    text = out_path.read_text(encoding="utf-8")
    assert "YOLO Agent Experiment Report" in text
    assert "| cli-run | unknown | 0.9 | unknown | unknown | unknown | ok |" in text
    assert NO_EVIDENCE_WARNING in text
