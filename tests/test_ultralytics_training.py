"""Ultralytics training executor and importer tests."""

from __future__ import annotations

import json
import types
from io import StringIO
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
from yolo_agent.adapters.ultralytics.data_cache_policy import (
    DataCachePolicy,
    DataCachePolicyConfig,
    MemorySnapshot,
    apply_cache_decision,
    estimate_yolo_image_bytes,
)
from yolo_agent.adapters.ultralytics.fast_baseline_gate import FastBaselineGate
from yolo_agent.adapters.ultralytics.runtime_profiler import RuntimeProfiler, RuntimeSample
from yolo_agent.adapters.ultralytics.stop_resume import StopResumeConfig, StopResumeGuard
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.components.compatibility import BaseModelSpec, CompatibilityChecker
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.error_facts import ErrorFactStore
from yolo_agent.core.event_log import EventLog
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


def _make_coco_data_yaml(root: Path) -> Path:
    images = root / "images" / "val2017"
    annotations = root / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir(parents=True)
    (images / "000000000001.jpg").write_bytes(b"image")
    (annotations / "instances_val2017.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 100, "height": 100}],
                "categories": [{"id": 1, "name": "bottle"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 8, 8],
                        "area": 64,
                        "iscrowd": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    data_yaml = root / "coco.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/val2017\nval: images/val2017\nnames:\n  1: bottle\n",
        encoding="utf-8",
    )
    return data_yaml


def _fake_popen_factory(
    lines: list[str] | None = None,
    returncode: int = 0,
    seen_argv: list[str] | None = None,
) -> type:
    class FakePopen:
        def __init__(self, argv: list[str], **kwargs: object) -> None:
            command_name = str(argv[0]) if argv else ""
            if seen_argv is not None and command_name not in {"nvidia-smi", "powershell"}:
                seen_argv[:] = list(argv)
            self.args = argv
            self.stdout = StringIO("".join(lines or ["train ok\n"]))
            self.returncode = returncode
            self.killed = False

        def __enter__(self) -> "FakePopen":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def communicate(self, input: object = None, timeout: int | float | None = None) -> tuple[str, str]:
            return self.stdout.read(), ""

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: int | float | None = None) -> int:
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    return FakePopen


def _hanging_popen_factory() -> type:
    class HangingPopen:
        def __init__(self, argv: list[str], **kwargs: object) -> None:
            self.args = argv
            self.stdout = StringIO("")
            command_name = str(argv[0]) if argv else ""
            self.returncode: int | None = 1 if command_name == "nvidia-smi" else None
            self.killed = False

        def __enter__(self) -> "HangingPopen":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def communicate(self, input: object = None, timeout: int | float | None = None) -> tuple[str, str]:
            return "", ""

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: int | float | None = None) -> int:
            return -9 if self.killed else 0

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    return HangingPopen


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

    assert set(profiles) == {"debug", "pilot", "baseline_full", "baseline_confirm", "candidate_full"}
    assert profiles["debug"].fraction == 0.01
    assert profiles["debug"].epochs == 1
    assert profiles["debug"].val is False
    assert profiles["debug"].timeout_seconds == 3600
    assert profiles["pilot"].fraction == 0.1
    assert profiles["pilot"].epochs == 10
    assert profiles["pilot"].batch == "auto"
    assert profiles["baseline_full"].batch == "auto"
    assert profiles["pilot"].timeout_seconds == 43200
    assert profiles["baseline_full"].fraction == 1.0
    assert profiles["baseline_full"].epochs == 100
    assert profiles["baseline_full"].seeds == [1]
    assert profiles["baseline_confirm"].seeds == [1, 2, 3]
    assert profiles["baseline_confirm"].confirms_contribution is True
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

    assert "epochs=1" in spec.argv
    assert "fraction=0.01" in spec.argv
    assert "val=False" in spec.argv
    assert "plots=False" in spec.argv
    assert "save_json=False" in spec.argv
    assert spec.metadata["training_budget_profile"] == "debug"
    assert spec.metadata["training_budget_fraction"] == 0.01
    assert spec.metadata["training_budget_epochs"] == 1
    assert spec.metadata["training_timeout_seconds"] == 3600
    assert spec.metadata["training_budget_seeds"] == "1"
    assert spec.metadata["training_budget_seed_count"] == 1
    assert spec.timeout_seconds == 3600


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
    assert "batch=-1" in spec.argv
    assert spec.metadata["training_batch_policy"] == "auto"
    assert spec.resource_requirements.requires_batch_tuning is True
    assert spec.metadata["training_budget_profile"] == "pilot"
    assert spec.metadata["training_timeout_seconds"] == 43200
    assert spec.timeout_seconds == 43200


def test_debug_profile_uses_ultralytics_auto_batch_without_batch_tuning() -> None:
    """Debug should stay quick while still emitting a valid Ultralytics auto-batch argument."""
    config = UltralyticsTrainingConfig.from_yaml(
        Path("configs/training/yolo26_coco_goal.yaml"),
        budget_profile="debug",
    )
    spec = command_from_training_config(_plain_node(), config, run_id="exp001")

    assert "batch=-1" in spec.argv
    assert "batch=auto" not in spec.argv
    assert spec.metadata["training_batch_policy"] == "auto"
    assert spec.resource_requirements.requires_batch_tuning is False


def test_fast_baseline_gate_enforces_sanity_pilot_full_confirmation(tmp_path: Path) -> None:
    """FastBaselineGate should prevent jumping directly to full COCO training."""
    store = EvidenceStore(tmp_path / "runs")
    store.create_run("exp001")
    node = _plain_node()
    gate = FastBaselineGate()

    assert gate.evaluate("debug", store.load_run("exp001"), node.candidate_config.candidate_id).ok is True
    pilot_gate = gate.evaluate("pilot", store.load_run("exp001"), node.candidate_config.candidate_id)
    assert pilot_gate.ok is False
    assert "fast_baseline_sanity_passed" in pilot_gate.blocked_by

    store.log_candidate_metrics(
        "exp001",
        node.candidate_config.candidate_id,
        node.node_id,
        gate.stage_metrics("debug", node, success=True),
        dataset_version=node.data_version,
        split="runtime",
        source="test",
    )
    assert gate.evaluate("pilot", store.load_run("exp001"), node.candidate_config.candidate_id).ok is True
    full_gate = gate.evaluate("baseline_full", store.load_run("exp001"), node.candidate_config.candidate_id)
    assert full_gate.ok is False
    assert "fast_baseline_pilot_passed" in full_gate.blocked_by

    store.log_candidate_metrics(
        "exp001",
        node.candidate_config.candidate_id,
        "node_pilot",
        gate.stage_metrics("pilot", node.model_copy(update={"node_id": "node_pilot"}), success=True),
        dataset_version=node.data_version,
        split="runtime",
        source="test",
    )
    assert gate.evaluate("baseline_full", store.load_run("exp001"), node.candidate_config.candidate_id).ok is True


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


def _make_cache_dataset(root: Path, image_sizes: list[int]) -> Path:
    images = root / "images" / "train"
    labels = root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index, size in enumerate(image_sizes):
        image_path = images / f"img{index}.jpg"
        with image_path.open("wb") as file:
            file.truncate(size)
        (labels / f"img{index}.txt").write_text("", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train\nnames:\n  0: object\n",
        encoding="utf-8",
    )
    return data_yaml


def test_data_cache_policy_chooses_ram_when_memory_is_sufficient(tmp_path: Path) -> None:
    """DataCachePolicy should use RAM cache only when safety margins are met."""
    data_yaml = _make_cache_dataset(tmp_path / "dataset", [1024, 2048])
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=data_yaml,
        project=tmp_path / "ultra",
        name="exp001_node",
        imgsz=640,
        workers=8,
        overrides={"cache": False},
    )
    memory = MemorySnapshot(
        total_bytes=64 * 1024**3,
        available_bytes=48 * 1024**3,
        source="test",
    )

    decision = DataCachePolicy().decide(data_yaml, command, memory=memory, storage_kind="nvme")
    updated = apply_cache_decision(command, decision)

    assert decision.selected_cache == "ram"
    assert decision.dataset_size_bytes == 3072
    assert "cache=ram" in updated.argv
    assert "workers=8" in updated.argv


def test_data_cache_policy_prefers_disk_when_ram_margin_is_unsafe(tmp_path: Path) -> None:
    """On a 64GB machine with about 17GB free, disk cache is safer than RAM for large data."""
    data_yaml = _make_cache_dataset(tmp_path / "dataset", [10 * 1024**3])
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=data_yaml,
        project=tmp_path / "ultra",
        name="exp001_node",
        imgsz=640,
        workers=8,
        overrides={"cache": False},
    )
    memory = MemorySnapshot(
        total_bytes=64 * 1024**3,
        available_bytes=17 * 1024**3,
        source="test",
    )

    decision = DataCachePolicy().decide(data_yaml, command, memory=memory, storage_kind="nvme")
    updated = apply_cache_decision(command, decision)

    assert decision.selected_cache == "disk"
    assert decision.estimated_ram_cache_bytes == 30 * 1024**3
    assert "cache=disk" in updated.argv
    assert "imgsz=640" in updated.argv


def test_data_cache_policy_raises_workers_when_cache_is_not_safe(tmp_path: Path) -> None:
    """If RAM and disk cache are unsafe, the policy should raise workers and recommend preheat."""
    data_yaml = _make_cache_dataset(tmp_path / "dataset", [10 * 1024])
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=data_yaml,
        project=tmp_path / "ultra",
        name="exp001_node",
        workers=8,
        overrides={"cache": False},
    )
    memory = MemorySnapshot(
        total_bytes=16 * 1024**3,
        available_bytes=1 * 1024**3,
        source="test",
    )

    decision = DataCachePolicy().decide(data_yaml, command, memory=memory, storage_kind="unknown")
    updated = apply_cache_decision(command, decision)

    assert decision.selected_cache == "False"
    assert decision.selected_workers == 12
    assert decision.preheat_recommended is True
    assert "cache=False" in updated.argv
    assert "workers=12" in updated.argv


def test_data_cache_policy_requires_nvme_for_disk_cache_by_default(tmp_path: Path) -> None:
    """SSD-like storage should not get disk cache when the policy requires NVMe."""
    data_yaml = _make_cache_dataset(tmp_path / "dataset", [10 * 1024])
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=data_yaml,
        project=tmp_path / "ultra",
        name="exp001_node",
        workers=8,
        overrides={"cache": False},
    )
    memory = MemorySnapshot(
        total_bytes=16 * 1024**3,
        available_bytes=1 * 1024**3,
        source="test",
    )

    decision = DataCachePolicy().decide(data_yaml, command, memory=memory, storage_kind="ssd")

    assert decision.selected_cache == "False"
    assert decision.selected_workers == 12
    assert decision.preheat_recommended is True


def test_data_cache_policy_estimates_yolo_split_bytes(tmp_path: Path) -> None:
    """Dataset byte estimation should follow YOLO split entries."""
    data_yaml = _make_cache_dataset(tmp_path / "dataset", [123, 456])

    root, size_bytes = estimate_yolo_image_bytes(data_yaml)

    assert root == tmp_path / "dataset"
    assert size_bytes == 579


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


def test_ultralytics_run_importer_auto_imports_coco_eval_error_facts(tmp_path: Path) -> None:
    """Importer should automatically ingest COCO eval metrics and error facts when present."""
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
    (run_dir / "coco_eval.json").write_text(
        json.dumps(
            {
                "AP": 0.37,
                "AP_small": 0.18,
                "per_class_ap": {"bottle": 0.12},
                "per_class_ar": {"bottle": 0.20},
            }
        ),
        encoding="utf-8",
    )

    store = EvidenceStore(tmp_path / "runs")
    metrics = UltralyticsRunImporter(store).import_run("exp001", _node(), run_dir, sample_gpu=False)
    evidence = store.load_run("exp001")
    facts = ErrorFactStore(tmp_path / "runs").read("exp001")

    assert metrics["map50_95"] == 0.37
    assert metrics["ap_small"] == 0.18
    assert any(record.validator == "coco_error_importer" and record.metric_name == "ap_small" for record in evidence.metric_records)
    assert any(fact.fact_type == "class_low_ap" and fact.class_name == "bottle" for fact in facts)
    assert any(entry.name == "node_yolo26s_coco_baseline_coco_eval" for entry in evidence.artifact_manifest)


def test_ultralytics_run_importer_auto_mines_coco_predictions(tmp_path: Path) -> None:
    """Importer should mine lightweight COCO error facts from predictions plus dataset annotations."""
    dataset_yaml = _make_coco_data_yaml(tmp_path / "coco")
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
    (run_dir / "predictions.json").write_text("[]", encoding="utf-8")

    store = EvidenceStore(tmp_path / "runs")
    UltralyticsRunImporter(store).import_run("exp001", _node(), run_dir, sample_gpu=False, data_path=dataset_yaml)
    evidence = store.load_run("exp001")
    facts = ErrorFactStore(tmp_path / "runs").read("exp001")

    assert any(fact.fact_type == "false_negative_heavy_class" and fact.class_name == "bottle" for fact in facts)
    assert any(entry.name == "node_yolo26s_coco_baseline_coco_predictions" for entry in evidence.artifact_manifest)
    assert any(entry.name == "node_yolo26s_coco_baseline_coco_error_report" for entry in evidence.artifact_manifest)


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
    (run_dir / "coco_eval.json").write_text(
        json.dumps({"AP": 0.31, "AP_small": 0.14, "per_class_ap": {"bottle": 0.10}}),
        encoding="utf-8",
    )
    (weights_dir / "best.pt").write_bytes(b"0" * 4096)
    command = CommandSpec.ultralytics_train(
        model="yolo26s.pt",
        data="configs/datasets/coco.yaml",
        project=tmp_path / "ultra",
        name="exp001_node",
    )

    monkeypatch.setattr(UltralyticsAdapter, "is_available", lambda self: True)
    monkeypatch.setattr(executor_mod, "_resolve_executable", lambda command: command)
    monkeypatch.setattr(executor_mod.subprocess, "Popen", _fake_popen_factory(["train ok\n"]))

    store = EvidenceStore(tmp_path / "runs")
    result = UltralyticsTrainExecutor(evidence_store=store).execute(_node(), "exp001", command)
    evidence = store.load_run("exp001")

    assert result.status == "completed"
    assert result.command.metadata["run_id"] == "exp001"
    assert result.command.metadata["candidate_id"] == "yolo26s_coco_baseline"
    assert result.command.metadata["node_id"] == "node_yolo26s_coco_baseline"
    assert result.metrics["map50_95"] == 0.31
    assert result.metrics["ap_small"] == 0.14
    assert any(record.metric_name == "map50_95" for record in evidence.metric_records)
    assert any(fact.fact_type == "class_low_ap" for fact in ErrorFactStore(tmp_path / "runs").read("exp001"))


def test_ultralytics_train_executor_streams_logs_and_live_metrics(monkeypatch, tmp_path: Path) -> None:
    """Executor should stream train logs to events, metric evidence, and runtime JSONL."""
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
    lines = [
        "Epoch GPU_mem box_loss cls_loss Instances Size\n",
        "1/1 1.25G 0.1 0.2 12 640: 100%|##########| 1/1 [00:01<00:00, 7.50it/s]\n",
    ]

    monkeypatch.setattr(UltralyticsAdapter, "is_available", lambda self: True)
    monkeypatch.setattr(executor_mod, "_resolve_executable", lambda command: command)
    monkeypatch.setattr(executor_mod.subprocess, "Popen", _fake_popen_factory(lines))

    store = EvidenceStore(tmp_path / "runs")
    result = UltralyticsTrainExecutor(evidence_store=store).execute(_node(), "exp001", command)
    evidence = store.load_run("exp001")
    events = EventLog(tmp_path / "runs" / "exp001" / "events.jsonl").read()
    runtime_jsonl = tmp_path / "runs" / "exp001" / "artifacts" / "node_yolo26s_coco_baseline_runtime_profile.jsonl"

    assert result.status == "completed"
    assert any(event.event_type == "executor_log" and "7.50it/s" in event.message for event in events)
    assert any(record.metric_name == "runtime_stream_it_per_sec" for record in evidence.metric_records)
    assert runtime_jsonl.is_file()
    assert "runtime_stream_gpu_memory_used_mb" in runtime_jsonl.read_text(encoding="utf-8")


def test_ultralytics_train_executor_persists_timeout_evidence(monkeypatch, tmp_path: Path) -> None:
    """Timed-out training should produce node-level evidence instead of disappearing."""
    import yolo_agent.core.executor as executor_mod
    from yolo_agent.adapters.ultralytics.adapter import UltralyticsAdapter

    command = CommandSpec.ultralytics_train(
        model="yolo26s.pt",
        data="configs/datasets/coco.yaml",
        project=tmp_path / "ultra",
        name="exp001_node",
        timeout_seconds=0,
    )

    monkeypatch.setattr(UltralyticsAdapter, "is_available", lambda self: True)
    monkeypatch.setattr(executor_mod, "_resolve_executable", lambda command: command)
    monkeypatch.setattr(executor_mod.subprocess, "Popen", _hanging_popen_factory())

    store = EvidenceStore(tmp_path / "runs")
    timeout_training_config = UltralyticsTrainingConfig(
        model="yolo26s.pt",
        data=Path("configs/datasets/coco.yaml"),
        data_cache_policy=DataCachePolicyConfig(enabled=False),
    )
    result = UltralyticsTrainExecutor(
        evidence_store=store,
        training_config=timeout_training_config,
    ).execute(_node(), "exp001", command)
    evidence = store.load_run("exp001")
    events = EventLog(tmp_path / "runs" / "exp001" / "events.jsonl").read()

    assert result.status == "failed"
    assert result.metrics["execution_timed_out"] is True
    assert result.metrics["execution_timeout_seconds"] == 0
    assert "timed out" in result.message
    assert any(event.event_type == "executor_timeout" for event in events)
    assert any(record.metric_name == "execution_timed_out" and record.value is True for record in evidence.metric_records)
    assert any(record.metric_name == "execution_timeout_seconds" and record.value == 0 for record in evidence.metric_records)


def test_stop_resume_guard_flags_runtime_bottleneck() -> None:
    """StopResumeGuard should flag sustained low GPU utilization without stopping by default."""
    guard = StopResumeGuard(
        StopResumeConfig(
            low_gpu_util_threshold=40.0,
            low_gpu_duration_seconds=0.0,
            low_gpu_min_samples=2,
        )
    )

    assert guard.observe_sample(RuntimeSample(gpu_util_percent=20.0)) is None
    decision = guard.observe_sample(RuntimeSample(gpu_util_percent=25.0))

    assert decision is not None
    assert decision.kind == "runtime_bottleneck"
    assert decision.should_stop is False
    assert "increase_workers_or_enable_cache_disk" in decision.recommendations


def test_stop_resume_guard_flags_early_map_drop(tmp_path: Path) -> None:
    """StopResumeGuard should flag early mAP collapse from results.csv."""
    results_csv = tmp_path / "results.csv"
    guard = StopResumeGuard(StopResumeConfig(early_map_drop_threshold=0.05))

    results_csv.write_text("epoch,metrics/mAP50-95(B)\n0,0.30\n", encoding="utf-8")
    assert guard.observe_results_csv(results_csv) is None
    results_csv.write_text("epoch,metrics/mAP50-95(B)\n0,0.30\n1,0.20\n", encoding="utf-8")
    decision = guard.observe_results_csv(results_csv)

    assert decision is not None
    assert decision.kind == "training_failure"
    assert decision.evidence["early_map_drop"] == 0.1
    assert "resume_from_last_checkpoint_with_safer_lr" in decision.recommendations


def test_ultralytics_train_executor_persists_stop_resume_runtime_bottleneck(monkeypatch, tmp_path: Path) -> None:
    """Executor should persist Stop/Resume guard decisions as node evidence."""
    import yolo_agent.adapters.ultralytics.runtime_profiler as runtime_profiler_mod
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
    config = UltralyticsTrainingConfig(
        model="yolo26s.pt",
        data=Path("configs/datasets/coco.yaml"),
        stop_resume=StopResumeConfig(
            low_gpu_duration_seconds=0.0,
            low_gpu_min_samples=2,
        ),
    )

    class FakeSampler:
        def __init__(self, sample_callback: object | None = None, **kwargs: object) -> None:
            self.sample_callback = sample_callback
            self.samples: list[RuntimeSample] = []

        def __enter__(self) -> "FakeSampler":
            for sample in [RuntimeSample(gpu_util_percent=10.0), RuntimeSample(gpu_util_percent=15.0)]:
                self.samples.append(sample)
                if self.sample_callback is not None:
                    self.sample_callback(sample)
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

    monkeypatch.setattr(UltralyticsAdapter, "is_available", lambda self: True)
    monkeypatch.setattr(executor_mod, "_resolve_executable", lambda command: command)
    monkeypatch.setattr(executor_mod.subprocess, "Popen", _fake_popen_factory(["train ok\n"]))
    monkeypatch.setattr(runtime_profiler_mod, "RuntimeSampler", FakeSampler)

    store = EvidenceStore(tmp_path / "runs")
    result = UltralyticsTrainExecutor(evidence_store=store, training_config=config).execute(_node(), "exp001", command)
    evidence = store.load_run("exp001")
    events = EventLog(tmp_path / "runs" / "exp001" / "events.jsonl").read()
    runtime_jsonl = tmp_path / "runs" / "exp001" / "artifacts" / "node_yolo26s_coco_baseline_runtime_profile.jsonl"

    assert result.status == "completed"
    assert any(record.metric_name == "runtime_bottleneck" and record.value is True for record in evidence.metric_records)
    assert any("Stop/Resume guard flagged runtime_bottleneck" in event.message for event in events)
    assert "stop_resume_decision" in runtime_jsonl.read_text(encoding="utf-8")


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

    def fake_tune(self: BatchTuner, run_id: str, node: ExperimentNode, command: CommandSpec) -> BatchTuningResult:
        return BatchTuningResult(selected_batch=48, selected_metric=6.0, applied=True, reason="test")

    monkeypatch.setattr(UltralyticsAdapter, "is_available", lambda self: True)
    monkeypatch.setattr(executor_mod, "_resolve_executable", lambda command: command)
    monkeypatch.setattr(batch_tuner_mod.BatchTuner, "tune", fake_tune)
    monkeypatch.setattr(executor_mod.subprocess, "Popen", _fake_popen_factory(["train ok\n"], seen_argv=seen_argv))

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


def test_ultralytics_train_executor_blocks_full_baseline_without_pilot(monkeypatch, tmp_path: Path) -> None:
    """FastBaselineGate should stop full baseline execution until pilot evidence exists."""
    import yolo_agent.core.executor as executor_mod
    from yolo_agent.adapters.ultralytics.adapter import UltralyticsAdapter

    config = UltralyticsTrainingConfig(
        model="yolo26n.pt",
        data=Path("configs/datasets/coco.yaml"),
        imgsz=640,
        budget_profile="baseline_full",
    )
    command = command_from_training_config(_plain_node(), config, run_id="exp001")
    subprocess_called = False

    def fake_run(*args: object, **kwargs: object) -> object:
        nonlocal subprocess_called
        subprocess_called = True
        raise AssertionError("full baseline should be blocked before subprocess.run")

    monkeypatch.setattr(UltralyticsAdapter, "is_available", lambda self: True)
    monkeypatch.setattr(executor_mod, "_resolve_executable", lambda command: command)
    monkeypatch.setattr(executor_mod.subprocess, "run", fake_run)

    store = EvidenceStore(tmp_path / "runs")
    result = UltralyticsTrainExecutor(evidence_store=store, training_config=config).execute(
        _plain_node(),
        "exp001",
        command,
    )
    evidence = store.load_run("exp001")

    assert result.status == "skipped"
    assert subprocess_called is False
    assert "Fast Baseline Gate blocked" in result.message
    assert any(record.metric_name == "fast_baseline_gate_ok" and record.value is False for record in evidence.metric_records)


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
    assert config.budget_profile == "debug"
    assert config.data_cache_policy.enabled is True
    assert config.data_cache_policy.candidate_cache_modes == ["ram", "disk", "False"]
    assert config.batch_tuning.enabled is True
    assert config.batch_tuning.candidate_batches == [32, 48, 64, 96]
    assert config.stop_resume.enabled is True
    assert config.stop_resume.stop_on_trigger is False
    assert config.stop_resume.low_gpu_util_threshold == 40.0
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
