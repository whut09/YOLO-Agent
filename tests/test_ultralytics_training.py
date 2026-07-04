"""Ultralytics training executor and importer tests."""

from __future__ import annotations

import types
from pathlib import Path

import yaml

from yolo_agent.adapters.ultralytics.training import (
    TrainingBudgetProfile,
    UltralyticsRunImporter,
    UltralyticsTrainingConfig,
    command_from_training_config,
    default_training_budget_profiles,
    parse_results_csv,
    parse_ultralytics_run,
)
from yolo_agent.adapters.ultralytics.batch_tuner import (
    BatchTuner,
    BatchTuningConfig,
    BatchTuningResult,
    build_batch_trial_command,
)
from yolo_agent.adapters.ultralytics.runtime_profiler import RuntimeProfiler, RuntimeSample
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


def _plain_node() -> ExperimentNode:
    candidate = CandidateConfig(
        candidate_id="yolo26n_coco_debug",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
    )
    return ExperimentNode(
        node_id="node_yolo26n_coco_debug",
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
        imgsz=768,
        device="0",
    )

    spec = command_from_training_config(_node(), config, run_id="exp001")

    assert "imgsz=768" in spec.argv
    assert "model=yolo26s.pt" in spec.argv
    assert "seed=1" in spec.argv
    assert spec.metadata["candidate_id"] == "yolo26s_coco_baseline"


def test_command_from_training_config_blocks_imgsz_increase() -> None:
    """Training command construction should enforce fixed baseline input size."""
    config = UltralyticsTrainingConfig(
        model="yolo26s.pt",
        data=Path("configs/datasets/coco.yaml"),
        imgsz=640,
        device="0",
    )

    try:
        command_from_training_config(_node(), config, run_id="exp001")
    except ValueError as exc:
        assert "imgsz increase is blocked" in str(exc)
    else:  # pragma: no cover - explicit assertion path
        raise AssertionError("Expected imgsz increase guard to reject the command.")


def test_default_training_budget_profiles_define_staged_coco_budgets() -> None:
    """Default profiles should separate debug, pilot, and full COCO budgets."""
    profiles = default_training_budget_profiles()

    assert set(profiles) == {"debug", "pilot", "baseline_full", "candidate_full"}
    assert profiles["debug"].fraction == 0.01
    assert 1 <= profiles["debug"].epochs <= 3
    assert profiles["debug"].val is False
    assert profiles["pilot"].fraction == 0.1
    assert profiles["pilot"].epochs == 10
    assert isinstance(profiles["pilot"].batch, int)
    assert profiles["baseline_full"].fraction == 1.0
    assert profiles["baseline_full"].epochs == 100
    assert profiles["baseline_full"].seeds == [1, 2, 3]
    assert profiles["candidate_full"].requires_pilot_pass is True
    assert profiles["candidate_full"].confirms_contribution is True


def test_training_budget_profile_applies_to_ultralytics_command() -> None:
    """Selecting debug should create a fast COCO sanity command."""
    config = UltralyticsTrainingConfig(
        model="yolo26n.pt",
        data=Path("configs/datasets/coco.yaml"),
        imgsz=640,
        budget_profile="debug",
    )

    spec = command_from_training_config(_plain_node(), config, run_id="exp001")

    assert "epochs=3" in spec.argv
    assert "fraction=0.01" in spec.argv
    assert "val=False" in spec.argv
    assert "plots=False" in spec.argv
    assert "save_json=False" in spec.argv
    assert spec.metadata["training_budget_profile"] == "debug"
    assert spec.metadata["training_budget_fraction"] == 0.01
    assert spec.metadata["training_budget_epochs"] == 3


def test_training_budget_profile_from_yaml_can_select_pilot() -> None:
    """from_yaml should support selecting a different profile at loop init time."""
    config = UltralyticsTrainingConfig.from_yaml(
        Path("configs/training/yolo26_coco_goal.yaml"),
        budget_profile="pilot",
    )
    spec = command_from_training_config(_plain_node(), config, run_id="exp001")

    assert config.budget_profile == "pilot"
    assert "epochs=10" in spec.argv
    assert "fraction=0.1" in spec.argv
    assert "batch=64" in spec.argv
    assert spec.metadata["training_budget_profile"] == "pilot"


def test_batch_trial_command_preserves_imgsz_and_changes_only_batch_policy(tmp_path: Path) -> None:
    """Batch tuning trials must not change the input size used for fair comparison."""
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="configs/datasets/coco.yaml",
        project=tmp_path / "ultra",
        name="exp001_node",
        epochs=100,
        imgsz=640,
        batch="auto",
    )

    trial = build_batch_trial_command(command, 64, BatchTuningConfig(trial_fraction=0.01))

    assert "imgsz=640" in trial.argv
    assert "batch=64" in trial.argv
    assert "epochs=1" in trial.argv
    assert "fraction=0.01" in trial.argv
    assert "val=False" in trial.argv
    assert "name=exp001_node_batch_tune_b64" in trial.argv


def test_batch_tuner_selects_highest_throughput_and_records_oom(monkeypatch, tmp_path: Path) -> None:
    """BatchTuner should skip OOM trials and persist runtime tuning evidence."""
    import yolo_agent.adapters.ultralytics.batch_tuner as batch_tuner_mod

    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="configs/datasets/coco.yaml",
        project=tmp_path / "ultra",
        name="exp001_node",
        epochs=100,
        imgsz=640,
        batch="auto",
    )

    class FakeSampler:
        samples: list[RuntimeSample] = []

        def __enter__(self) -> "FakeSampler":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

    class FakeCompletedProcess:
        def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(argv: list[str], **kwargs: object) -> FakeCompletedProcess:
        text = " ".join(argv)
        if "batch=32" in text:
            return FakeCompletedProcess(0, "100/100 4.0it/s")
        if "batch=48" in text:
            return FakeCompletedProcess(0, "100/100 6.0it/s")
        if "batch=64" in text:
            return FakeCompletedProcess(1, "", "CUDA out of memory")
        return FakeCompletedProcess(1, "failed without throughput")

    monkeypatch.setattr(batch_tuner_mod, "RuntimeSampler", lambda interval_seconds: FakeSampler())
    monkeypatch.setattr(batch_tuner_mod.subprocess, "run", fake_run)

    store = EvidenceStore(tmp_path / "runs")
    result = BatchTuner(
        config=BatchTuningConfig(enabled=True, candidate_batches=[32, 48, 64, 96]),
        evidence_store=store,
    ).tune("exp001", _plain_node(), command)
    evidence = store.load_run("exp001")

    assert result.selected_batch == 48
    assert result.applied is True
    assert [trial.status for trial in result.trials] == ["completed", "completed", "oom", "failed"]
    assert "node_yolo26n_coco_debug_batch_tuning_result" in evidence.artifacts
    metric_values = {record.metric_name: record.value for record in evidence.metric_records}
    assert metric_values["batch_tuning_selected_batch"] == 48
    assert metric_values["batch_tuning_b64_oom"] is True
    assert metric_values["batch_tuning_b48_avg_it_per_sec"] == 6.0


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


def test_runtime_profiler_extracts_throughput_and_dataloader_facts(tmp_path: Path) -> None:
    """RuntimeProfiler should turn training logs into bottleneck evidence."""
    run_dir = tmp_path / "train_run"
    run_dir.mkdir()
    (run_dir / "args.yaml").write_text(
        "batch: auto\ncache: False\nworkers: 8\n",
        encoding="utf-8",
    )
    (run_dir / "results.csv").write_text(
        "epoch,time,metrics/mAP50-95(B)\n"
        "0,10,0.1\n"
        "1,22,0.2\n"
        "2,37,0.3\n",
        encoding="utf-8",
    )
    log_path = run_dir / "stdout.log"
    log_path.write_text(
        "\n".join(
            [
                "AutoBatch: Using batch-size 27",
                "WARNING Slow image access detected, training may be bottlenecked by storage.",
                "GPU_mem 4.2G 100/200 5.0it/s",
                "GPU_mem 4.8G 200/200 6.0it/s",
            ]
        ),
        encoding="utf-8",
    )

    profile = RuntimeProfiler().profile(run_dir, sample_gpu=False)
    metrics = profile.to_metrics()

    assert profile.batch_size == 27
    assert profile.cache_mode == "False"
    assert profile.dataloader_workers == 8
    assert profile.avg_it_per_sec == 5.5
    assert profile.max_it_per_sec == 6.0
    assert profile.epoch_time_seconds == 13.5
    assert profile.max_gpu_memory_used_mb == 4915.2
    assert profile.dataloader_wait_warning is True
    assert metrics["runtime_avg_it_per_sec"] == 5.5
    assert metrics["runtime_dataloader_wait_warning"] is True


def test_runtime_profiler_merges_executor_gpu_samples(tmp_path: Path) -> None:
    """Executor-collected GPU samples should become runtime metrics."""
    run_dir = tmp_path / "train_run"
    run_dir.mkdir()
    (run_dir / "args.yaml").write_text("batch: 16\ncache: ram\nworkers: 2\n", encoding="utf-8")
    sample = RuntimeSample(
        gpu_util_percent=72.0,
        gpu_memory_used_mb=8192.0,
        gpu_memory_total_mb=24576.0,
        gpu_memory_util_percent=33.0,
        power_w=280.0,
        source="test_sampler",
    )

    profile = RuntimeProfiler().profile(run_dir, samples=[sample], sample_gpu=False)

    assert profile.avg_gpu_util_percent == 72.0
    assert profile.max_gpu_memory_used_mb == 8192.0
    assert profile.samples[0].source == "test_sampler"


def test_ultralytics_run_importer_writes_node_evidence(tmp_path: Path) -> None:
    """Importer should persist metrics tied to candidate and node ids."""
    run_dir = tmp_path / "train_run"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    (run_dir / "results.csv").write_text(
        "epoch,time,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "0,12,0.40,0.50,0.55,0.30\n",
        encoding="utf-8",
    )
    (run_dir / "args.yaml").write_text("imgsz: 640\nepochs: 100\nbatch: 64\ncache: disk\nworkers: 4\n", encoding="utf-8")
    (weights_dir / "best.pt").write_bytes(b"0" * 4096)

    store = EvidenceStore(tmp_path / "runs")
    metrics = UltralyticsRunImporter(store).import_run(
        "exp001",
        _node(),
        run_dir,
        stdout="GPU_mem 2.0G 10/10 4.0it/s",
        sample_gpu=False,
    )
    evidence = store.load_run("exp001")

    assert metrics["map50_95"] == 0.3
    assert metrics["model_size_mb"] > 0
    assert metrics["runtime_avg_it_per_sec"] == 4.0
    assert {record.metric_name for record in evidence.metric_records} >= {
        "map50_95",
        "model_size_mb",
        "runtime_avg_it_per_sec",
        "runtime_batch_size",
    }
    assert evidence.metric_records[0].candidate_id == "yolo26s_coco_baseline"
    assert evidence.metric_records[0].validator == "ultralytics_results_importer"
    assert any(record.validator == "ultralytics_runtime_profiler" for record in evidence.metric_records)
    assert "node_yolo26s_coco_baseline_runtime_profile" in evidence.artifacts


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
    assert result.command.metadata["run_id"] == "exp001"
    assert result.command.metadata["candidate_id"] == "yolo26s_coco_baseline"
    assert result.command.metadata["node_id"] == "node_yolo26s_coco_baseline"
    assert result.metrics["map50_95"] == 0.3
    assert any(record.metric_name == "map50_95" for record in evidence.metric_records)


def test_ultralytics_train_executor_applies_batch_tuner_selection(monkeypatch, tmp_path: Path) -> None:
    """Executor should apply selected batch to the real train command before running."""
    import yolo_agent.adapters.ultralytics.batch_tuner as batch_tuner_mod
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
        model="yolo26n.pt",
        data="configs/datasets/coco.yaml",
        project=tmp_path / "ultra",
        name="exp001_node",
        batch="auto",
    )
    seen_argv: list[str] = []

    class FakeCompletedProcess:
        returncode = 0
        stdout = "train ok"
        stderr = ""

    def fake_tune(self: BatchTuner, run_id: str, node: ExperimentNode, command: CommandSpec) -> BatchTuningResult:
        return BatchTuningResult(selected_batch=48, selected_metric=6.0, applied=True, reason="test")

    def fake_run(argv: list[str], **kwargs: object) -> FakeCompletedProcess:
        if argv and argv[0] != "nvidia-smi":
            seen_argv[:] = argv
        return FakeCompletedProcess()

    monkeypatch.setattr(UltralyticsAdapter, "is_available", lambda self: True)
    monkeypatch.setattr(executor_mod, "_resolve_executable", lambda command: command)
    monkeypatch.setattr(batch_tuner_mod.BatchTuner, "tune", fake_tune)
    monkeypatch.setattr(executor_mod.subprocess, "run", fake_run)

    result = UltralyticsTrainExecutor(evidence_store=EvidenceStore(tmp_path / "runs")).execute(
        _plain_node(),
        "exp001",
        command,
    )

    assert result.status == "completed"
    assert "batch=48" in seen_argv
    assert "batch=auto" not in seen_argv
    assert result.command.metadata["batch_tuned"] is True
    assert result.command.metadata["batch_tuning_selected_batch"] == 48


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
    assert config.budget_profile == "baseline_full"
    assert config.batch_tuning.enabled is True
    assert config.batch_tuning.candidate_batches == [32, 48, 64, 96]
    assert isinstance(config.budget_profiles["debug"], TrainingBudgetProfile)
    assert config.budget_profiles["candidate_full"].requires_pilot_pass is True
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
