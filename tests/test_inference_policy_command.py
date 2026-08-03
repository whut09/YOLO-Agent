from pathlib import Path

from yolo_agent.core.command_spec import CommandSpec


def test_inference_policy_command_is_typed_and_explicit(tmp_path: Path) -> None:
    config = tmp_path / "policy.yaml"
    command = CommandSpec.inference_policy(
        workdir=tmp_path / "run",
        model="yolo26n.pt",
        images=tmp_path / "images",
        annotations=tmp_path / "instances.json",
        config=config,
        standard_metrics=tmp_path / "standard.json",
        execute=True,
    )

    assert command.command_type == "inference_policy"
    assert command.argv[:3] == ["yolo-agent", "advanced", "certify-inference-policy"]
    assert "--execute" in command.argv
    assert command.shell is False
    assert command.metadata["training_attribution_allowed"] is False
    assert command.metadata["standard_imgsz"] == 640
    assert command.expected_artifacts["inference_policy_report"].name == "inference_policy_certification_report.yaml"
