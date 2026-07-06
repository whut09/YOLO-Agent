"""Setup wizard tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.cli import main
from yolo_agent.tools.doctor import DoctorReport
from yolo_agent.tools.setup_wizard import run_setup_wizard


def _doctor(data_yaml: Path, model: str, run_root: Path) -> DoctorReport:
    return DoctorReport(data_yaml=data_yaml, model=model, run_root=run_root, kind="coco", checks=[])


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
    assert "--profile debug --execute" in result.next_command


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
    assert "next: yolo-agent optimize coco" in output
    assert "status: yolo-agent loop status" in output
