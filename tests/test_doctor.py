"""Doctor/preflight command tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.cli import main
from yolo_agent.core.llm_config import LLMDecisionConfig
from yolo_agent.tools import doctor
from yolo_agent.tools.doctor import DoctorCheck, run_doctor


def _make_coco(root: Path) -> Path:
    for relative in [
        Path("images") / "train2017",
        Path("images") / "val2017",
        Path("images") / "test2017",
        Path("annotations"),
    ]:
        (root / relative).mkdir(parents=True)
    for filename in [
        "instances_train2017.json",
        "instances_val2017.json",
        "image_info_test2017.json",
    ]:
        (root / "annotations" / filename).write_text("{}", encoding="utf-8")
    data_yaml = root / "coco.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train2017",
                "val: images/val2017",
                "test: images/test2017",
                "names:",
                "  0: person",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def _patch_runtime_ok(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        doctor,
        "_package_check",
        lambda package, fix: DoctorCheck(name=package, ok=True, level="info", message="installed", fix=fix),
    )
    monkeypatch.setattr(
        doctor,
        "_gpu_status",
        lambda: {"ok": True, "message": "NVIDIA RTX", "free_vram_gb": 12.0, "total_vram_gb": 16.0},
    )
    monkeypatch.setattr(
        doctor,
        "_torch_cuda_status",
        lambda: {"ok": True, "message": "torch CUDA ready"},
    )


def test_doctor_passes_for_fake_coco(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Doctor should validate a COCO-like layout without requiring real GPU packages in tests."""
    _patch_runtime_ok(monkeypatch)
    data_yaml = _make_coco(tmp_path / "coco")
    model = tmp_path / "yolo26n.pt"
    model.write_bytes(b"weights")

    report = run_doctor(data_yaml=data_yaml, model=str(model), run_root=tmp_path / "runs")

    assert report.ok is True
    assert report.error_count == 0
    assert report.batch_estimate is not None
    assert report.batch_estimate.selected_batch == 96
    assert report.batch_estimate.confidence == "medium"
    assert {check.name for check in report.checks} >= {
        "cuda_driver",
        "torch_cuda",
        "train_split",
        "val_split",
        "test_split",
        "annotation_instances_train2017.json",
        "annotation_instances_val2017.json",
    }


def test_doctor_cli_reports_fix_for_missing_data(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    """CLI doctor should fail loudly with a concrete fix when data.yaml is missing."""
    _patch_runtime_ok(monkeypatch)

    code = main(
        [
            "doctor",
            "--data",
            str(tmp_path / "missing.yaml"),
            "--model",
            "yolo26n.pt",
            "--run-root",
            str(tmp_path / "runs"),
        ]
    )

    output = capsys.readouterr().out
    assert code == 1
    assert "doctor status=failed" in output
    assert "data_yaml: error" in output
    assert "fix:" in output


def test_doctor_cli_prints_batch_estimate(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    """CLI doctor should show the conservative batch estimate during preflight."""
    _patch_runtime_ok(monkeypatch)
    data_yaml = _make_coco(tmp_path / "coco")
    model = tmp_path / "yolo26s.pt"
    model.write_bytes(b"weights")

    code = main(
        [
            "doctor",
            "--data",
            str(data_yaml),
            "--model",
            str(model),
            "--run-root",
            str(tmp_path / "runs"),
            "--imgsz",
            "640",
            "--batch-candidates",
            "32,48,64,96",
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "batch_estimate=64" in output
    assert "candidates=32,48,64,96" in output
    assert "batch_note=Preflight estimate only." in output


def test_doctor_llm_only_reports_missing_key_without_failing(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    """LLM-only doctor should guide setup and fall back to rule proposals when the key is absent."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "yolo_agent.cli.load_llm_decision_config",
        lambda: LLMDecisionConfig(
            enabled=True,
            provider="openai",
            model="gpt-5.5",
            model_alias="codex-decision",
            api_key_env="OPENAI_API_KEY",
            use_by_default=True,
            require_api_key=True,
        ),
    )

    code = main(["doctor", "--llm"])

    output = capsys.readouterr().out
    assert code == 0
    assert "llm status=missing_key" in output
    assert "llm fallback=rule_engine" in output
    assert "OPENAI_API_KEY" in output


def test_doctor_llm_only_reports_ready_when_key_exists(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    """LLM-only doctor should report ready when local config and API key are available."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "yolo_agent.cli.load_llm_decision_config",
        lambda: LLMDecisionConfig(
            enabled=True,
            provider="openai",
            model="gpt-5.5",
            api_key_env="OPENAI_API_KEY",
            use_by_default=True,
            require_api_key=True,
        ),
    )

    code = main(["doctor", "--llm"])

    output = capsys.readouterr().out
    assert code == 0
    assert "llm status=ready" in output
    assert "llm fallback=rule_engine" not in output
