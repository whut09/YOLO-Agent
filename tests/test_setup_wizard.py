"""Setup wizard tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.cli import main
from yolo_agent.tools.doctor import DoctorCheck, DoctorReport
from yolo_agent.tools.setup_wizard import run_setup_wizard, setup_result_to_text


def _doctor(data_yaml: Path, model: str, run_root: Path) -> DoctorReport:
    return DoctorReport(data_yaml=data_yaml, model=model, run_root=run_root, kind="coco", checks=[])


def _doctor_with_missing_data(data_yaml: Path, model: str, run_root: Path) -> DoctorReport:
    return DoctorReport(
        data_yaml=data_yaml,
        model=model,
        run_root=run_root,
        kind="coco",
        checks=[
            DoctorCheck(
                name="data_yaml",
                ok=False,
                level="error",
                message=f"missing: {data_yaml}",
                fix=f"Create or pass the correct data yaml, for example: yolo-agent doctor --data {data_yaml}",
            )
        ],
    )


def test_setup_wizard_writes_local_files_and_report(tmp_path: Path, monkeypatch) -> None:
    """Setup should create ignored local config, env file, and a COCO path report."""
    data_yaml = tmp_path / "coco.yaml"
    data_yaml.write_text("path: .\ntrain: images/train2017\nval: images/val2017\nnames: []\n", encoding="utf-8")
    monkeypatch.setattr("yolo_agent.tools.setup_wizard.run_doctor", lambda **kwargs: _doctor(kwargs["data_yaml"], kwargs["model"], kwargs["run_root"]))

    result = run_setup_wizard(
        kind="coco",
        data_yaml=data_yaml,
        model="yolo26n.pt",
        run_root=tmp_path / "runs",
        env_file=tmp_path / ".env.local",
        llm_config_path=tmp_path / "configs" / "local" / "llm_decision.local.yaml",
    )

    assert result.ok is True
    assert result.run_id == "coco-yolo26n"
    assert result.env_file.is_file()
    assert result.llm_config_path.is_file()
    assert result.setup_report_path.is_file()
    llm_config = yaml.safe_load(result.llm_config_path.read_text(encoding="utf-8"))
    report = yaml.safe_load(result.setup_report_path.read_text(encoding="utf-8"))
    assert llm_config["model"] == "gpt-5.5"
    assert "api_key" in llm_config
    assert llm_config["use_by_default"] is True
    assert "OPENAI_API_KEY=PUT_YOUR_OPENAI_API_KEY_HERE" in result.env_file.read_text(encoding="utf-8")
    assert report["openai_key_detected"] is False
    assert report["next_command"] == result.next_command
    assert "--profile debug" not in result.next_command


def test_setup_wizard_detects_key_from_existing_local_llm_config(tmp_path: Path, monkeypatch) -> None:
    """Rerunning setup should not claim the key is missing when local ignored YAML contains it."""
    data_yaml = tmp_path / "coco.yaml"
    data_yaml.write_text("path: .\ntrain: images/train2017\nval: images/val2017\nnames: []\n", encoding="utf-8")
    llm_config_path = tmp_path / "configs" / "local" / "llm_decision.local.yaml"
    llm_config_path.parent.mkdir(parents=True)
    llm_config_path.write_text(
        yaml.safe_dump(
            {
                "enabled": True,
                "provider": "openai",
                "model": "gpt-5.5",
                "api_key_env": "sk-local-secret",
                "base_url_env": "https://deepkey.top/v1",
                "decision_role": "proposal_generator_only",
                "use_by_default": True,
                "require_api_key": True,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("yolo_agent.tools.setup_wizard.run_doctor", lambda **kwargs: _doctor(kwargs["data_yaml"], kwargs["model"], kwargs["run_root"]))

    result = run_setup_wizard(
        kind="coco",
        data_yaml=data_yaml,
        model="yolo26n.pt",
        run_root=tmp_path / "runs",
        env_file=tmp_path / ".env.local",
        llm_config_path=llm_config_path,
    )

    report = yaml.safe_load(result.setup_report_path.read_text(encoding="utf-8"))
    assert result.openai_key_detected is True
    assert report["openai_key_detected"] is True
    assert report["llm_key_source"] == "local_config:api_key_env_direct_value"
    assert report["llm_base_url_source"] == "local_config:base_url_env_direct_value"
    assert not any("API key was not detected" in note for note in result.notes)


def test_setup_coco_cli_prints_next_command(tmp_path: Path, monkeypatch, capsys) -> None:
    """CLI setup should be a one-command onboarding entrypoint."""
    data_yaml = tmp_path / "coco.yaml"
    data_yaml.write_text("path: .\ntrain: images/train2017\nval: images/val2017\nnames: []\n", encoding="utf-8")
    monkeypatch.setattr("yolo_agent.tools.setup_wizard.run_doctor", lambda **kwargs: _doctor(kwargs["data_yaml"], kwargs["model"], kwargs["run_root"]))

    code = main(
        [
            "setup",
            "coco",
            "--data",
            str(data_yaml),
            "--model",
            "yolo26n.pt",
            "--run-root",
            str(tmp_path / "runs"),
            "--env-file",
            str(tmp_path / ".env.local"),
            "--llm-config",
            str(tmp_path / "configs" / "local" / "llm_decision.local.yaml"),
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "setup status=ok" in output
    assert "next: yolo-agent train --model" in output
    assert "status: yolo-agent status" in output


def test_setup_coco_cli_prints_doctor_errors_and_fixes(tmp_path: Path, monkeypatch, capsys) -> None:
    """Setup should show the actual failing doctor check, not only an error count."""
    data_yaml = tmp_path / "missing.yaml"
    monkeypatch.setattr(
        "yolo_agent.tools.setup_wizard.run_doctor",
        lambda **kwargs: _doctor_with_missing_data(kwargs["data_yaml"], kwargs["model"], kwargs["run_root"]),
    )

    code = main(
        [
            "setup",
            "coco",
            "--data",
            str(data_yaml),
            "--model",
            "yolo26n.pt",
            "--run-root",
            str(tmp_path / "runs"),
            "--env-file",
            str(tmp_path / ".env.local"),
            "--llm-config",
            str(tmp_path / "configs" / "local" / "llm_decision.local.yaml"),
        ]
    )

    output = capsys.readouterr().out
    assert code == 1
    assert "setup status=needs_fix errors=1 warnings=0" in output
    assert "error[1].data_yaml: missing:" in output
    assert "fix: Create or pass the correct data yaml" in output


def test_setup_result_text_prints_warning_checks(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Warning checks should also be visible in setup output."""
    data_yaml = tmp_path / "data.yaml"
    monkeypatch.setattr("yolo_agent.tools.setup_wizard.run_doctor", lambda **kwargs: _doctor(kwargs["data_yaml"], kwargs["model"], kwargs["run_root"]))
    result = run_setup_wizard(
        kind="coco",
        data_yaml=data_yaml,
        model="yolo26n.pt",
        run_root=tmp_path / "runs",
        env_file=tmp_path / ".env.local",
        llm_config_path=tmp_path / "configs" / "local" / "llm_decision.local.yaml",
    )
    result.warning_checks.append(
        DoctorCheck(
            name="annotation_test2017_image_info",
            ok=False,
            level="warning",
            message="missing image_info_test2017.json",
            fix="Extract image_info_test2017.zip.",
        )
    )

    output = setup_result_to_text(result)

    assert "warning[1].annotation_test2017_image_info: missing image_info_test2017.json" in output
    assert "fix: Extract image_info_test2017.zip." in output
