"""Interface contract tests for UltralyticsAdapter and executors."""

from __future__ import annotations

import sys
import types
import time
from pathlib import Path

import pytest

from yolo_agent.adapters.ultralytics.adapter import UltralyticsAdapter
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.executor import (
    CommandSpec,
    ExecutionResult,
    DryRunExecutor,
    ShellExecutor,
    UltralyticsExecutor,
)
from yolo_agent.core.experiment_graph import ExperimentNode


def _candidate() -> CandidateConfig:
    return CandidateConfig(
        candidate_id="baseline",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
    )


def _node(command: str = "yolo train model=baseline.yaml") -> ExperimentNode:
    return ExperimentNode(
        node_id="node-baseline",
        candidate_config=_candidate(),
        data_version="dataset-v1",
        seed=42,
        command=command,
    )


@pytest.fixture()
def adapter() -> UltralyticsAdapter:
    return UltralyticsAdapter()


# ---------------------------------------------------------------- Adapter contracts
class TestUltralyticsAdapterContracts:
    def test_available_losses_returns_sorted_string_list(self, adapter: UltralyticsAdapter) -> None:
        """Adapter should expose a sorted list of registered loss names."""
        names = adapter.available_losses()
        assert isinstance(names, list)
        assert names == sorted(names)
        assert all(isinstance(name, str) for name in names)
        assert names == ["ciou", "mpdiou", "nwd", "wiou"]

    def test_get_loss_returns_object_with_interface(self, adapter: UltralyticsAdapter) -> None:
        """Loss adapter objects should implement name, supports_head and build."""
        loss = adapter.get_loss("ciou")
        assert hasattr(loss, "name")
        assert hasattr(loss, "supports_head")
        assert hasattr(loss, "build")
        assert loss.name == "ciou"

    def test_get_loss_raises_for_unknown_name(self, adapter: UltralyticsAdapter) -> None:
        """Registry get should raise KeyError with consistent message for unknown losses."""
        with pytest.raises(KeyError) as exc_info:
            adapter.get_loss("unknown_loss")
        assert "Unknown bbox loss adapter: unknown_loss" in str(exc_info.value)

    def test_build_train_command_defaults_to_generated_models_path(self, adapter: UltralyticsAdapter) -> None:
        """Without args, build_train_command should use generated_models/<candidate>.yaml."""
        node = _node()
        cmd = adapter.build_train_command(node)
        assert cmd == "yolo train model=generated_models/baseline.yaml"
        assert " " not in cmd.split("=", 1)[1]

    def test_build_train_command_honors_command_override(self, adapter: UltralyticsAdapter) -> None:
        """When explicit command is given, it should be returned unchanged."""
        command = "yolo train model=/tmp/custom.yaml epochs=50 batch=8"
        result = adapter.build_train_command(_node(), command=command)
        assert result == command

    def test_build_train_command_honors_model_yaml_path(self, adapter: UltralyticsAdapter) -> None:
        """model_yaml_path should be normalized to posix style."""
        path = Path("custom_models") / "my-model.yaml"
        result = adapter.build_train_command(_node(), model_yaml_path=path)
        assert result == "yolo train model=custom_models/my-model.yaml"

    def test_smoke_check_returns_false_when_package_missing(self, adapter: UltralyticsAdapter) -> None:
        """smoke_check should return False when ultralytics is not installed."""
        assert adapter.smoke_check("nonexistent.yaml") is False

    def test_smoke_check_returns_false_when_path_missing(self, adapter: UltralyticsAdapter, monkeypatch: pytest.MonkeyPatch) -> None:
        """smoke_check should return False when YAML file does not exist."""
        import importlib

        fake_module = types.SimpleNamespace()
        monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
        try:
            assert adapter.smoke_check(Path("does") / "not" / "exist.yaml") is False
        finally:
            sys.modules.pop("ultralytics", None)

    def test_generate_model_yaml_default_output_dir_when_none(
        self, adapter: UltralyticsAdapter
    ) -> None:
        """generate_model_yaml should default output_dir to 'generated_models'."""
        from yolo_agent.resources import ResourcePaths

        result = adapter.generate_model_yaml(
            candidate=_candidate(),
            base_template=ResourcePaths.ULTRALYTICS_BASE_TEMPLATE,
            nc=2,
            dry_run=True,
        )
        assert result.output_path == Path("generated_models/baseline.yaml")
        assert result.output_path.exists() is False


# ---------------------------------------------------------------- Execution contracts
class TestExecutionContracts:
    def test_execution_result_default_fields(self) -> None:
        """ExecutionResult defaults should match documented contract."""
        from yolo_agent.core.experiment_graph import CandidateConfig

        node = _node()
        result = DryRunExecutor().execute(node, run_id="contract-run")

        assert isinstance(result, ExecutionResult)
        assert result.run_id == "contract-run"
        assert result.node_id == "node-baseline"
        assert result.candidate_id == "baseline"
        assert result.status == "dry_run"
        assert result.return_code is None
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.duration_seconds == 0.0
        assert result.artifacts == {}
        assert result.metrics == {}
        assert isinstance(result.started_at, type(result.started_at))
        assert result.ended_at is not None

    def test_commandspec_from_experiment_node_is_shell_true(self) -> None:
        """CommandSpec built from a node should use shell=True by default."""
        spec = CommandSpec.from_experiment_node(_node())
        assert spec.command == "yolo train model=baseline.yaml"
        assert spec.shell is True
        assert spec.args == []
        assert spec.timeout_seconds is None
        assert spec.metadata["node_id"] == "node-baseline"
        assert spec.metadata["candidate_id"] == "baseline"
        assert spec.metadata["dataset_version"] == "dataset-v1"
        assert spec.metadata["seed"] == 42

    def test_commandspec_as_subprocess_args_shell(self) -> None:
        """Shell=True should return a single joined command string."""
        spec = CommandSpec(command="yolo train", args=["model=a.yaml"], shell=True)
        assert spec.as_subprocess_args() == "yolo train model=a.yaml"

    def test_commandspec_as_subprocess_args_list(self) -> None:
        """Shell=False should return a list of args."""
        spec = CommandSpec(command="python", args=["-m", "pytest"], shell=False)
        assert spec.as_subprocess_args() == ["python", "-m", "pytest"]

    def test_shell_executor_success_contract(self) -> None:
        """ShellExecutor should populate all result fields on success."""
        command = CommandSpec(
            command=sys.executable,
            args=["-c", "print('contract-ok')"],
            shell=False,
        )
        result = ShellExecutor().execute(_node(), run_id="shell-contract", command=command)

        assert result.status == "completed"
        assert result.return_code == 0
        assert "contract-ok" in result.stdout
        assert result.stderr == ""
        assert result.duration_seconds is not None and result.duration_seconds >= 0.0
        assert result.started_at <= result.ended_at

    def test_shell_executor_failure_contract(self) -> None:
        """ShellExecutor should mark failed status and include stderr on non-zero exit."""
        command = CommandSpec(
            command=sys.executable,
            args=["-c", "import sys; sys.stderr.write('fail'); sys.exit(1)"],
            shell=False,
        )
        result = ShellExecutor().execute(_node(), run_id="shell-fail", command=command)

        assert result.status == "failed"
        assert result.return_code == 1
        assert result.stdout == "" or "fail" in result.stderr

    def test_ultralytics_executor_rejects_non_adapter(self) -> None:
        """UltralyticsExecutor should raise TypeError for non-UltralyticsAdapter."""
        executor = UltralyticsExecutor(adapter=object())  # type: ignore[arg-type]

        with pytest.raises(TypeError) as exc_info:
            executor.execute(_node(), run_id="bad-adapter")

        assert "adapter must be an UltralyticsAdapter instance" in str(exc_info.value)

    def test_ultralytics_executor_returns_failed_on_prepare_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the adapter raises during preparation, UltralyticsExecutor should fail closed."""
        bad_adapter = UltralyticsAdapter()
        monkeypatch.setattr(bad_adapter, "is_available", lambda: True)
        monkeypatch.setattr(
            bad_adapter,
            "generate_model_yaml",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("yaml boom")),
        )

        result = UltralyticsExecutor(adapter=bad_adapter, try_forward=False).execute(_node(), run_id="prepare-fail")

        assert result.status == "failed"
        assert "failed to prepare experimental artifacts" in result.message
        assert "yaml boom" in result.message

    def test_ultralytics_executor_skipped_message_mentions_installation(self) -> None:
        """When unavailable, the skipped result should mention the installation hint."""
        result = UltralyticsExecutor().execute(_node(), run_id="skip-run")
        assert result.status == "skipped"
        assert "ultralytics" in result.message.lower() or "not installed" in result.message.lower()

    def test_dry_run_executor_command_from_node(self) -> None:
        """When no explicit command is provided, DryRunExecutor should use node.command."""
        result = DryRunExecutor().execute(_node(), run_id="dry-node")
        assert result.command.command == "yolo train model=baseline.yaml"
        assert result.command.shell is True


class TestCommandSpecSerialization:
    def test_commandspec_json_roundtrip(self, tmp_path: Path) -> None:
        """CommandSpec model_dump/validate should roundtrip."""
        spec = CommandSpec(
            command="yolo",
            args=["train", "model=a.yaml"],
            cwd=tmp_path,
            env={"VAR": "1"},
            timeout_seconds=600,
            shell=True,
            metadata={"candidate_id": "baseline"},
        )
        dumped = spec.model_dump(mode="json")
        restored = CommandSpec.model_validate(dumped)
        assert restored.command == spec.command
        assert restored.args == spec.args
        assert restored.cwd == spec.cwd
        assert restored.env == spec.env
        assert restored.timeout_seconds == spec.timeout_seconds
        assert restored.shell == spec.shell
        assert restored.metadata == spec.metadata
