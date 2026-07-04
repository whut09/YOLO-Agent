"""Run initialization and dataset-manifest attachment for loop harness runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from yolo_agent.agents.loop_io import read_yaml
from yolo_agent.core.artifact_manifest import sha256_file
from yolo_agent.core.dataset_versioning import DatasetVersionStore
from yolo_agent.core.loop_state import LoopState
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.stage_contract import LoopStageContracts
from yolo_agent.resources import ResourcePaths


DatasetManifestMode = Literal["sha256", "metadata"]


class RunInitialization(BaseModel):
    """Initialized loop context and state."""

    context: RunContext
    state: LoopState
    dataset_manifest_path: Path


class RunInitializer:
    """Build the durable files that define a loop run."""

    def initialize(
        self,
        run_id: str,
        task_path: Path | str,
        data_yaml: Path | str,
        run_root: Path | str = "runs",
        component_path: Path | str = ResourcePaths.COMPONENTS_DIR,
        search_space_path: Path | str = ResourcePaths.SEARCH_SPACE,
        loop_policy_path: Path | str = ResourcePaths.LOOP_POLICY,
        predictions_path: Path | str | None = None,
        detection_errors_path: Path | str | None = None,
        metrics_input_path: Path | str | None = None,
        training_config_path: Path | str | None = None,
        dataset_version: str = "unversioned",
        dataset_manifest_mode: DatasetManifestMode = "sha256",
        seed: int = 42,
    ) -> RunInitialization:
        """Create a run context, initial loop state, and dataset manifest."""
        context = RunContext(
            run_id=run_id,
            run_root=Path(run_root),
            task_path=Path(task_path),
            data_yaml=Path(data_yaml),
            component_path=Path(component_path),
            search_space_path=Path(search_space_path),
            loop_policy_path=Path(loop_policy_path),
            predictions_path=Path(predictions_path) if predictions_path is not None else None,
            detection_errors_path=Path(detection_errors_path) if detection_errors_path is not None else None,
            metrics_input_path=Path(metrics_input_path) if metrics_input_path is not None else None,
            dataset_version=dataset_version,
            seed=seed,
        )
        if training_config_path is not None:
            context.metadata["training_config_path"] = Path(training_config_path).as_posix()
        context.metadata["dataset_manifest_mode"] = dataset_manifest_mode
        context.ensure_dirs()
        dataset_manifest_path = attach_dataset_manifest_to_context(context, dataset_manifest_mode)
        context.to_yaml()
        context.to_json()
        policy = LoopStageContracts.from_yaml(loop_policy_path)
        state = LoopState.create(
            run_id,
            policy.stage_order,
            dataset_version=dataset_version,
            task_spec=Path(task_path),
        )
        state.mark(
            "init",
            "completed",
            "Run context initialized.",
            {
                "run_context": context.run_dir / "run_context.yaml",
                "dataset_manifest": dataset_manifest_path,
            },
        )
        state.to_yaml(context.run_dir / "loop_state.yaml")
        return RunInitialization(context=context, state=state, dataset_manifest_path=dataset_manifest_path)


def attach_dataset_manifest_to_context(
    context: RunContext,
    mode: DatasetManifestMode | None = None,
) -> Path:
    """Create a dataset manifest for the run and attach its hash to context."""
    dataset_root = resolve_yolo_dataset_root(context.data_yaml)
    store_path = context.run_dir / "dataset_versions"
    manifest_mode = mode or str(context.metadata.get("dataset_manifest_mode", "sha256"))
    hash_files = manifest_mode != "metadata"
    DatasetVersionStore(store_path).create_version(
        dataset_root=dataset_root,
        version=context.dataset_version,
        notes=[
            f"run_id={context.run_id}",
            f"data_yaml={context.data_yaml.as_posix()}",
            "created_by=RunInitializer",
            f"manifest_mode={manifest_mode}",
        ],
        copy_data=False,
        hash_files=hash_files,
    )
    manifest_path = store_path / context.dataset_version / "manifest.json"
    context.dataset_root = dataset_root
    context.dataset_version_store_path = store_path
    context.dataset_manifest_path = manifest_path
    context.dataset_manifest_sha256 = sha256_file(manifest_path)
    return manifest_path


def inherit_dataset_manifest_to_context(parent: RunContext, child: RunContext) -> Path:
    """Copy the parent's dataset manifest into the child run when available."""
    parent_manifest_path = parent.dataset_manifest_path
    if parent_manifest_path is None or not parent_manifest_path.is_file():
        return attach_dataset_manifest_to_context(child)
    store_path = child.run_dir / "dataset_versions"
    manifest_path = store_path / child.dataset_version / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(parent_manifest_path, manifest_path)
    child.dataset_root = parent.dataset_root or resolve_yolo_dataset_root(child.data_yaml)
    child.dataset_version_store_path = store_path
    child.dataset_manifest_path = manifest_path
    child.dataset_manifest_sha256 = sha256_file(manifest_path)
    return manifest_path


def resolve_yolo_dataset_root(data_yaml: Path) -> Path:
    """Resolve the dataset root represented by a YOLO data.yaml."""
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data_yaml does not exist: {data_yaml}")
    raw = read_yaml(data_yaml)
    configured_path = raw.get("path")
    if configured_path is None:
        return data_yaml.parent
    dataset_root = Path(str(configured_path))
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root
    return dataset_root
