"""Typed command specifications for controlled experiment execution."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_serializer, model_validator


CommandType = Literal["smoke", "train", "benchmark", "import_metrics", "custom"]


class CommandSpec(BaseModel):
    """A structured command prepared from an experiment node."""

    command_type: CommandType = "custom"
    command: str = ""
    args: list[str] = Field(default_factory=list)
    argv: list[str] = Field(default_factory=list)
    cwd: Path | None = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int | None = None
    shell: bool = False
    expected_artifacts: dict[str, Path] = Field(default_factory=dict)
    expected_metrics: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_serializer("cwd")
    def serialize_cwd(self, value: Path | None) -> str | None:
        """Serialize cwd portably."""
        return value.as_posix() if value is not None else None

    @field_serializer("expected_artifacts")
    def serialize_expected_artifacts(self, value: dict[str, Path]) -> dict[str, str]:
        """Serialize expected artifact paths portably."""
        return {key: path.as_posix() for key, path in value.items()}

    @model_validator(mode="after")
    def fill_argv(self) -> "CommandSpec":
        """Populate argv for non-shell commands when legacy fields are used."""
        if not self.argv and self.command:
            self.argv = [self.command, *self.args]
        if not self.command and self.argv:
            self.command = self.argv[0]
            self.args = self.argv[1:]
        return self

    @classmethod
    def ultralytics_train(
        cls,
        model: str | Path,
        data: str | Path,
        project: str | Path,
        name: str,
        seed: int = 42,
        task: str = "detect",
        mode: str = "train",
        epochs: int | None = None,
        imgsz: int | None = None,
        batch: int | str | None = None,
        device: int | str | list[int] | None = None,
        resume: bool | str | Path | None = None,
        workers: int | None = None,
        optimizer: str | None = None,
        patience: int | None = None,
        amp: bool | None = None,
        exist_ok: bool = True,
        timeout_seconds: int | None = None,
        overrides: dict[str, str | int | float | bool | Path] | None = None,
        env: dict[str, str] | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> "CommandSpec":
        """Build a typed Ultralytics ``yolo detect train`` command."""
        run_dir = Path(project) / name
        args: list[str] = [
            task,
            mode,
            f"model={_pathish(model)}",
            f"data={_pathish(data)}",
            f"project={_pathish(project)}",
            f"name={name}",
            f"seed={seed}",
            f"exist_ok={_bool_text(exist_ok)}",
        ]
        optional_values: dict[str, str | int | float | bool | Path] = {}
        if epochs is not None:
            optional_values["epochs"] = epochs
        if imgsz is not None:
            optional_values["imgsz"] = imgsz
        if batch is not None:
            optional_values["batch"] = batch
        if device is not None:
            optional_values["device"] = _device_text(device)
        if resume is not None:
            optional_values["resume"] = _bool_text(resume) if isinstance(resume, bool) else _pathish(resume)
        if workers is not None:
            optional_values["workers"] = workers
        if optimizer is not None:
            optional_values["optimizer"] = optimizer
        if patience is not None:
            optional_values["patience"] = patience
        if amp is not None:
            optional_values["amp"] = _bool_text(amp)
        optional_values.update(overrides or {})
        args.extend(f"{key}={_pathish(value)}" for key, value in optional_values.items())
        argv = ["yolo", *args]
        return cls(
            command_type="train",
            command=argv[0],
            args=argv[1:],
            argv=argv,
            shell=False,
            env=env or {},
            timeout_seconds=timeout_seconds,
            expected_artifacts={
                "results_csv": run_dir / "results.csv",
                "args_yaml": run_dir / "args.yaml",
                "best_pt": run_dir / "weights" / "best.pt",
                "last_pt": run_dir / "weights" / "last.pt",
            },
            expected_metrics=["map50_95", "map50", "precision", "recall", "model_size_mb"],
            metadata=metadata or {},
        )

    @classmethod
    def smoke(
        cls,
        plan_path: Path | str,
        data_path: Path | str,
        run_id: str,
        expected_artifacts: dict[str, Path] | None = None,
        expected_metrics: list[str] | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> "CommandSpec":
        """Build a typed smoke command."""
        argv = [
            "yolo-agent",
            "smoke",
            "--plan",
            Path(plan_path).as_posix(),
            "--data",
            Path(data_path).as_posix(),
            "--run-id",
            run_id,
        ]
        return cls(
            command_type="smoke",
            command=argv[0],
            args=argv[1:],
            argv=argv,
            shell=False,
            expected_artifacts=expected_artifacts or {},
            expected_metrics=expected_metrics or [
                "smoke_passed",
                "yaml_generated",
                "ultralytics_imported",
                "forward_checked",
            ],
            metadata=metadata or {},
        )

    @classmethod
    def from_experiment_node(cls, node: object) -> "CommandSpec":
        """Build a command spec from an experiment node.

        Newer plans carry node.command_spec. Legacy plans only have a command
        string, which remains executable as an explicit shell command for
        backwards compatibility.
        """
        spec = getattr(node, "command_spec", None)
        metadata = _node_metadata(node)
        if isinstance(spec, cls):
            return spec.model_copy(update={"metadata": {**spec.metadata, **metadata}})
        if isinstance(spec, dict):
            restored = cls.model_validate(spec)
            return restored.model_copy(update={"metadata": {**restored.metadata, **metadata}})
        command = str(getattr(node, "command", ""))
        return cls(
            command=command,
            shell=True,
            metadata=metadata,
        )

    def as_subprocess_args(self) -> str | list[str]:
        """Return subprocess command representation."""
        if self.shell:
            return " ".join(self.argv or [self.command, *self.args]).strip()
        return list(self.argv or [self.command, *self.args])

    def display(self) -> str:
        """Return a human-readable command string."""
        return " ".join(self.argv or [self.command, *self.args]).strip()


def _node_metadata(node: object) -> dict[str, str | int | float | bool]:
    candidate_config = getattr(node, "candidate_config", None)
    return {
        "node_id": str(getattr(node, "node_id", "")),
        "candidate_id": str(getattr(candidate_config, "candidate_id", "")),
        "dataset_version": str(getattr(node, "data_version", "")),
        "seed": int(getattr(node, "seed", 0)),
    }


def _pathish(value: str | int | float | bool | Path) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bool):
        return _bool_text(value)
    return str(value)


def _bool_text(value: bool) -> str:
    return "True" if value else "False"


def _device_text(value: int | str | list[int]) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)
