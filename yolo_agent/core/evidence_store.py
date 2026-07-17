"""Local filesystem evidence store for experiment runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

import yaml

from yolo_agent.core.artifact_manifest import ArtifactManifest, ArtifactManifestEntry
from yolo_agent.core.experiment_graph import Evidence, MetricEvidence, MetricValue


class EvidenceStore:
    """Persist run configuration, metrics, and artifact paths locally."""

    def __init__(self, root: Path | str = "runs") -> None:
        self.root = Path(root)

    def create_run(self, run_id: str) -> Path:
        """Create the local run directory structure."""
        run_dir = self._run_dir(run_id)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        return run_dir

    def log_config(self, run_id: str, config: dict[str, Any]) -> Path:
        """Write run configuration to config.yaml."""
        run_dir = self.create_run(run_id)
        config_path = run_dir / "config.yaml"
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)
        return config_path

    def log_metrics(self, run_id: str, metrics: dict[str, MetricValue]) -> Path:
        """Write run metrics to metrics.json."""
        run_dir = self.create_run(run_id)
        metrics_path = run_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, sort_keys=True)
        return metrics_path

    def log_metric_records(self, run_id: str, records: list[MetricEvidence]) -> Path:
        """Append candidate/node-level metric records to metrics_by_node.jsonl."""
        run_dir = self.create_run(run_id)
        records_path = run_dir / "metrics_by_node.jsonl"
        with records_path.open("a", encoding="utf-8") as file:
            for record in records:
                stored = _record_for_run(record, run_id)
                file.write(json.dumps(stored.model_dump(mode="json"), sort_keys=True) + "\n")
        return records_path

    def log_candidate_metrics(
        self,
        run_id: str,
        candidate_id: str,
        node_id: str,
        metrics: dict[str, MetricValue],
        dataset_version: str = "unversioned",
        split: str = "val",
        source: str = "manual",
        verified: bool = True,
        validator: str = "manual",
        source_artifact: Path | str | None = None,
        metric_schema_version: str = "1.0",
        protocol_hash: str | None = None,
        dataset_manifest_sha256: str | None = None,
        subset_manifest_sha256: str | None = None,
        eval_protocol_hash: str | None = None,
        seed: int | str | None = None,
        fidelity: str | None = None,
        epochs: int | None = None,
        batch_policy_hash: str | None = None,
        ultralytics_version: str | None = None,
        imgsz: int | None = None,
        evidence_role: Literal["current_observation", "inherited_context", "baseline_reference"] = "current_observation",
    ) -> Path:
        """Append multiple metrics for one candidate/node pair."""
        artifact = Path(source_artifact) if source_artifact is not None else None
        return self.log_metric_records(
            run_id,
            [
                MetricEvidence(
                    candidate_id=candidate_id,
                    node_id=node_id,
                    run_id=run_id,
                    origin_run_id=run_id,
                    evidence_role=evidence_role,
                    dataset_version=dataset_version,
                    dataset_manifest_sha256=dataset_manifest_sha256,
                    subset_manifest_sha256=subset_manifest_sha256,
                    split=split,
                    protocol_hash=protocol_hash,
                    eval_protocol_hash=eval_protocol_hash,
                    seed=seed,
                    fidelity=fidelity,
                    epochs=epochs,
                    batch_policy_hash=batch_policy_hash,
                    ultralytics_version=ultralytics_version,
                    imgsz=imgsz,
                    metric_name=metric_name,
                    value=value,
                    source=source,
                    verified=verified,
                    validator=validator,
                    source_artifact=artifact,
                    metric_schema_version=metric_schema_version,
                )
                for metric_name, value in metrics.items()
            ],
        )

    def log_artifact(self, run_id: str, artifact_path: Path | str, name: str | None = None) -> Path:
        """Copy an artifact into runs/{run_id}/artifacts/."""
        run_dir = self.create_run(run_id)
        source = Path(artifact_path)
        if not source.is_file():
            raise FileNotFoundError(f"Artifact does not exist or is not a file: {source}")
        destination = run_dir / "artifacts" / (name or source.name)
        shutil.copy2(source, destination)
        self.log_artifact_manifest(
            run_id=run_id,
            name=name or source.name,
            artifact_path=destination,
            producer_stage="evidence_store",
        )
        return destination

    def log_artifact_manifest(
        self,
        run_id: str,
        name: str,
        artifact_path: Path | str,
        producer_stage: str,
    ) -> ArtifactManifestEntry:
        """Record an artifact manifest entry without copying the artifact."""
        run_dir = self.create_run(run_id)
        entry = ArtifactManifestEntry.from_path(name=name, path=artifact_path, producer_stage=producer_stage)
        ArtifactManifest(run_dir / "artifacts" / "artifact_manifest.jsonl").append(entry)
        return entry

    def load_run(self, run_id: str) -> Evidence:
        """Load config, metrics, and artifact paths for a run."""
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run does not exist: {run_id}")

        config_path = run_dir / "config.yaml"
        metrics_path = run_dir / "metrics.json"
        metric_records_path = run_dir / "metrics_by_node.jsonl"
        artifacts_dir = run_dir / "artifacts"
        artifact_manifest_path = artifacts_dir / "artifact_manifest.jsonl"
        config = _read_yaml_mapping(config_path) if config_path.exists() else {}
        metrics = _read_json_mapping(metrics_path) if metrics_path.exists() else {}
        metric_records = _read_metric_records(metric_records_path, run_id) if metric_records_path.exists() else []
        legacy_run, run_protocol_hash = _run_protocol_state(run_dir)
        if legacy_run:
            metric_records = [_quarantine_legacy_record(record) for record in metric_records]
        artifact_manifest = ArtifactManifest(artifact_manifest_path).read() if artifact_manifest_path.exists() else []
        artifacts = {
            path.name: path
            for path in sorted(artifacts_dir.iterdir())
            if path.is_file()
        } if artifacts_dir.exists() else {}
        _apply_manifest_verification(artifacts, artifact_manifest)

        return Evidence(
            run_id=run_id,
            config_path=config_path if config_path.exists() else None,
            metrics_path=metrics_path if metrics_path.exists() else None,
            metric_records_path=metric_records_path if metric_records_path.exists() else None,
            artifact_manifest_path=artifact_manifest_path if artifact_manifest_path.exists() else None,
            artifacts_dir=artifacts_dir if artifacts_dir.exists() else None,
            config=config,
            metrics=metrics,
            metric_records=metric_records,
            artifact_manifest=artifact_manifest,
            artifacts=artifacts,
            run_protocol_hash=run_protocol_hash,
            legacy_run=legacy_run,
        )

    def _run_dir(self, run_id: str) -> Path:
        if not run_id or any(separator in run_id for separator in ("/", "\\")):
            raise ValueError("run_id must be a non-empty single path segment.")
        return self.root / run_id


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def _read_json_mapping(path: Path) -> dict[str, MetricValue]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain a mapping: {path}")
    return data


def _read_metric_records(path: Path, run_id: str) -> list[MetricEvidence]:
    records: list[MetricEvidence] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            text = line.strip()
            if text:
                records.append(_record_for_run(MetricEvidence.model_validate(json.loads(text)), run_id))
    return records


def _record_for_run(record: MetricEvidence, run_id: str) -> MetricEvidence:
    """Attach storage/origin provenance and migrate legacy inherited records."""
    inherited_source = str(record.source).startswith("inherited:")
    origin = record.origin_run_id or record.run_id
    if origin is None and inherited_source:
        parts = str(record.source).split(":", 2)
        origin = parts[1] if len(parts) > 1 and parts[1] else None
    origin = origin or run_id
    inherited = inherited_source or origin != run_id or record.inheritance_depth > 0
    role = record.evidence_role
    if inherited and role == "current_observation":
        role = "inherited_context"
    depth = record.inheritance_depth
    if inherited and depth == 0:
        depth = 1
    return record.model_copy(
        update={
            "run_id": run_id,
            "origin_run_id": origin,
            "evidence_role": role,
            "inheritance_depth": depth,
        }
    )


def _run_protocol_state(run_dir: Path) -> tuple[bool, str | None]:
    context_path = run_dir / "run_context.yaml"
    if not context_path.is_file():
        return False, None
    try:
        context = _read_yaml_mapping(context_path)
    except (OSError, ValueError):
        return True, None
    return bool(context.get("legacy_run", False)), str(context.get("run_protocol_hash") or "") or None


def _quarantine_legacy_record(record: MetricEvidence) -> MetricEvidence:
    """Prevent legacy candidate metrics from masquerading as current observations."""
    if record.evidence_role != "current_observation":
        return record
    return record.model_copy(
        update={
            "evidence_role": "inherited_context",
            "inheritance_depth": max(record.inheritance_depth, 1),
            "source": record.source if record.source.startswith("legacy:") else f"legacy:{record.source}",
        }
    )


def _apply_manifest_verification(
    artifacts: dict[str, Path],
    manifest: list[ArtifactManifestEntry],
) -> None:
    """Prefer manifest-verified artifacts and remove stale manifest entries."""
    for entry in manifest:
        aliases = {entry.name, entry.path.name, entry.path.stem}
        if entry.verify():
            for alias in aliases:
                artifacts[alias] = entry.path
            continue
        for alias in aliases:
            artifacts.pop(alias, None)
