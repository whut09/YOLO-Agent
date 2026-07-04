"""Ultralytics training executor and importer tests."""

from __future__ import annotations

import types
from pathlib import Path

import yaml

from yolo_agent.adapters.ultralytics.training import (
    UltralyticsRunImporter,
    UltralyticsTrainingConfig,
    command_from_training_config,
    parse_results_csv,
    parse_ultralytics_run,
)
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.components.compatibility import BaseModelSpec, CompatibilityChecker
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.executor import UltralyticsTrainExecutor
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.task_spec import MetricPriority, TaskSpec
from yolo_agent.cli import main


def _node() -> ExperimentNode:
    candidate = CandidateConfig(
        candidate_id="yolo26s_coco_baseline",
        base_model="yolo26s.pt",
        scale="s",
        framework="ultralytics",
        train_overrides={"imgsz": 768},
    )
    return ExperimentNode(
        node_id="node_yolo26s_coco_baseline",
        candidate_config=candidate,
        data_version="coco2017",
        seed=1,
    )


def test_ultralytics_train_command_uses_typed_argv() -> None:
    """Training commands should be shell-free and include COCO execution args."""
    spec = CommandSpec.ultralytics_train(
        model="yolo26s.pt",
        data="configs/datasets/coco.yaml",
        project="runs/ultralytics",
        name="exp001_node",
        seed=1,
        epochs=100,
        imgsz=640,
        batch="auto",
        device=[0, 1],
        resume=True,
    )

    assert spec.command_type == "train"
    assert spec.shell is False
    assert spec.argv[:3] == ["yolo", "detect", "train"]
    assert "model=yolo26s.pt" in spec.argv
    assert "data=configs/datasets/coco.yaml" in spec.argv
    assert "device=0,1" in spec.argv
    assert "resume=True" in spec.argv
    assert spec.expected_artifacts["results_csv"] == Path("runs/ultralytics/exp001_node/results.csv")


def test_command_from_training_config_merges_candidate_overrides() -> None:
    """Candidate train_overrides should override recipe defaults."""
    config = UltralyticsTrainingConfig(
        model="yolo26s.pt",
        data=Path("configs/datasets/coco.yaml"),
        imgsz=640,
        device="0",
    )

    spec = command_from_training_config(_node(), config, run_id="exp001")

    assert "imgsz=768" in spec.argv
    assert "model=yolo26s.pt" in spec.argv
    assert "seed=1" in spec.argv
    assert spec.metadata["candidate_id"] == "yolo26s_coco_baseline"


def test_parse_ultralytics_results_csv_selects_best_row(tmp_path: Path) -> None:
    """Parser should map Ultralytics columns into harness metric names."""
    results_csv = tmp_path / "results.csv"
    results_csv.write_text(
        "\n".join(
            [
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),fitness",
                "0,0.40,0.50,0.55,0.30,0.31",
                "1,0.45,0.55,0.65,0.42,0.43",
                "",
            ]
        ),
        encoding="utf-8",
    )

    metrics = parse_results_csv(results_csv)

    assert metrics["precision"] == 0.45
    assert metrics["recall"] == 0.55
    assert metrics["map50"] == 0.65
    assert metrics["map50_95"] == 0.42
    assert metrics["best_epoch"] == 1


def test_ultralytics_run_importer_writes_node_evidence(tmp_path: Path) -> None:
    """Importer should persist metrics tied to candidate and node ids."""
    run_dir = tmp_path / "train_run"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    (run_dir / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "0,0.40,0.50,0.55,0.30\n",
        encoding="utf-8",
    )
    (run_dir / "args.yaml").write_text("imgsz: 640\nepochs: 100\n", encoding="utf-8")
    (weights_dir / "best.pt").write_bytes(b"0" * 4096)

    store = EvidenceStore(tmp_path / "runs")
    metrics = UltralyticsRunImporter(store).import_run("exp001", _node(), run_dir)
    evidence = store.load_run("exp001")

    assert metrics["map50_95"] == 0.3
    assert metrics["model_size_mb"] > 0
    assert {record.metric_name for record in evidence.metric_records} >= {"map50_95", "model_size_mb"}
    assert evidence.metric_records[0].candidate_id == "yolo26s_coco_baseline"
    assert evidence.metric_records[0].validator == "ultralytics_results_importer"


def test_ultralytics_train_executor_imports_metrics_after_success(monkeypatch, tmp_path: Path) -> None:
    """Executor should run a typed command and import completed Ultralytics artifacts."""
    import yolo_agent.core.executor as executor_mod
    from yolo_agent.adapters.ultralytics.adapter import UltralyticsAdapter

    run_dir = tmp_path / "ultra" / "exp001_node"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    (run_dir / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "0,0.40,0.50,0.55,0.30\n",
        encoding="utf-8",
    )
    (weights_dir / "best.pt").write_bytes(b"0" * 4096)
    command = CommandSpec.ultralytics_train(
        model="yolo26s.pt",
        data="configs/datasets/coco.yaml",
        project=tmp_path / "ultra",
        name="exp001_node",
    )

    class FakeCompletedProcess:
        returncode = 0
        stdout = "train ok"
        stderr = ""

    monkeypatch.setattr(UltralyticsAdapter, "is_available", lambda self: True)
    monkeypatch.setattr(executor_mod, "_resolve_executable", lambda command: command)
    monkeypatch.setattr(executor_mod.subprocess, "run", lambda *args, **kwargs: FakeCompletedProcess())

    store = EvidenceStore(tmp_path / "runs")
    result = UltralyticsTrainExecutor(evidence_store=store).execute(_node(), "exp001", command)
    evidence = store.load_run("exp001")

    assert result.status == "completed"
    assert result.metrics["map50_95"] == 0.3
    assert any(record.metric_name == "map50_95" for record in evidence.metric_records)


def test_yolo26_compatibility_warns_for_loss_patch() -> None:
    """YOLO26 should not silently accept old loss-patch assumptions."""
    registry = ComponentRegistry.from_path("configs/components")
    component = next(card for card in registry.cards if card.id == "loss.bbox.nwd")
    task = TaskSpec(
        task_type="detect",
        scene="generic",
        class_names=["object"],
        primary_metric=MetricPriority(name="map50_95"),
    )

    result = CompatibilityChecker().check(
        task,
        BaseModelSpec(name="yolo26s.pt", framework="ultralytics", model_family="yolo26"),
        [component],
    )

    assert result.ok is True
    assert any("YOLO26 is DFL-free" in warning for warning in result.warnings)
    assert result.estimated_risk == "medium"


def test_coco_yolo26_training_recipe_loads() -> None:
    """Bundled COCO YOLO26 recipe should validate as a training config."""
    raw = yaml.safe_load(Path("configs/training/yolo26_coco_goal.yaml").read_text(encoding="utf-8"))
    config = UltralyticsTrainingConfig.model_validate(raw["training"])
    metrics = parse_ultralytics_run(Path("definitely_missing"))

    assert config.model == "yolo26s.pt"
    assert config.data == Path("configs/datasets/coco.yaml")
    assert raw["goal"]["target_delta_points"] == 2.0
    assert metrics == {}


def test_loop_import_ultralytics_cli_writes_node_evidence(tmp_path: Path) -> None:
    """Loop CLI should import an existing Ultralytics run directory."""
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "task_type": "detect",
                "scene": "generic",
                "class_names": ["object"],
                "primary_metric": {"name": "map50_95"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "train").mkdir(parents=True)
    (dataset_root / "labels" / "train").mkdir(parents=True)
    (dataset_root / "images" / "train" / "img1.jpg").write_bytes(b"image")
    (dataset_root / "labels" / "train" / "img1.txt").write_text("", encoding="utf-8")
    data_yaml = dataset_root / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nnames:\n  0: object\n", encoding="utf-8")
    run_root = tmp_path / "runs"
    ultra_run = tmp_path / "ultralytics" / "exp"
    (ultra_run / "weights").mkdir(parents=True)
    (ultra_run / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "0,0.4,0.5,0.6,0.3\n",
        encoding="utf-8",
    )
    (ultra_run / "weights" / "best.pt").write_bytes(b"0" * 1024)

    assert main(
        [
            "loop",
            "init",
            "--run-id",
            "import-run",
            "--task",
            str(task_path),
            "--data",
            str(data_yaml),
            "--run-root",
            str(run_root),
            "--dataset-version",
            "coco2017",
        ]
    ) == 0
    assert main(
        [
            "loop",
            "import-ultralytics",
            "--run",
            str(run_root / "import-run"),
            "--ultralytics-run",
            str(ultra_run),
            "--candidate-id",
            "baseline",
            "--node-id",
            "node_baseline",
        ]
    ) == 0

    evidence = EvidenceStore(run_root).load_run("import-run")
    assert any(record.metric_name == "map50_95" for record in evidence.metric_records)
    assert any(record.node_id == "node_baseline" for record in evidence.metric_records)
