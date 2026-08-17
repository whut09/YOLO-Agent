"""Automatic, candidate-scoped runtime readiness with identity-bound caching."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.agents.paper_recipe_materialization.runtime_identity import (
    validate_certified_runtime_node,
)
from yolo_agent.components.adapters import SmokeTestResult
from yolo_agent.components.execution_bridge import ComponentExecutionResult
from yolo_agent.components.maturity_registry import installed_ultralytics_version
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.yaml_io import YAMLModelMixin


RuntimeReadinessScope = Literal["component_smoke", "candidate_runtime"]
RuntimeReadinessStatus = Literal["ready", "cached", "failed"]


class RuntimeReadinessIdentity(BaseModel):
    """Inputs that must remain unchanged before readiness evidence is reused."""

    model_config = ConfigDict(extra="forbid")

    scope: RuntimeReadinessScope
    component_ids: list[str]
    adapter_hashes: dict[str, str]
    runtime_payload_hash: str
    protocol_hash: str
    python_version: str
    ultralytics_version: str

    @property
    def cache_key(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class RuntimeReadinessRecord(BaseModel, YAMLModelMixin):
    """One immutable CPU readiness result stored by execution identity."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "automatic_runtime_readiness.v1"
    identity: RuntimeReadinessIdentity
    passed: bool
    evidence_kind: str
    checks: dict[str, bool | str] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AutomaticRuntimeReadinessResult(BaseModel):
    """Candidate-scoped queue admission result; never an accuracy result."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    status: RuntimeReadinessStatus
    cache_key: str
    cache_hit: bool = False
    component_ids: list[str] = Field(default_factory=list)
    checks: dict[str, bool | str] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)
    artifact_path: Path | None = None
    optimization_metric_eligible: Literal[False] = False


class AutomaticRuntimeReadinessGate:
    """Run or reuse CPU readiness for each materialized runtime independently."""

    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def lookup_smoke(
        self,
        *,
        component_id: str,
        adapter_hash: str,
        runtime_payload_hash: str,
        protocol_hash: str,
    ) -> SmokeTestResult | None:
        identity = self._identity(
            scope="component_smoke",
            component_ids=[component_id],
            adapter_hashes={component_id: adapter_hash},
            runtime_payload_hash=runtime_payload_hash,
            protocol_hash=protocol_hash,
        )
        record = self._read(identity)
        if record is None or not record.passed or record.evidence_kind != "local":
            return None
        return SmokeTestResult(
            passed=True,
            evidence_kind="local",
            checks=dict(record.checks),
        )

    def record_smoke(
        self,
        *,
        component_id: str,
        adapter_hash: str,
        runtime_payload_hash: str,
        protocol_hash: str,
        result: SmokeTestResult,
    ) -> Path:
        identity = self._identity(
            scope="component_smoke",
            component_ids=[component_id],
            adapter_hashes={component_id: adapter_hash},
            runtime_payload_hash=runtime_payload_hash,
            protocol_hash=protocol_hash,
        )
        record = RuntimeReadinessRecord(
            identity=identity,
            passed=bool(result.passed and result.evidence_kind == "local"),
            evidence_kind=result.evidence_kind,
            checks=dict(result.checks),
            blockers=list(result.errors),
        )
        return self._write(record)

    def evaluate_node(self, node: ExperimentNode) -> AutomaticRuntimeReadinessResult:
        identity, identity_errors = self._node_identity(node)
        if identity is None:
            fallback_key = hashlib.sha256(node.node_id.encode("utf-8")).hexdigest()
            return AutomaticRuntimeReadinessResult(
                allowed=False,
                status="failed",
                cache_key=fallback_key,
                component_ids=list(node.candidate_config.components),
                blockers=identity_errors,
            )
        cached = self._read(identity)
        if cached is not None and cached.passed and cached.evidence_kind == "local":
            return AutomaticRuntimeReadinessResult(
                allowed=True,
                status="cached",
                cache_key=identity.cache_key,
                cache_hit=True,
                component_ids=list(identity.component_ids),
                checks=dict(cached.checks),
                artifact_path=self._path(identity),
            )

        blockers = [*identity_errors, *validate_certified_runtime_node(node)]
        checks: dict[str, bool | str] = {
            "runtime_payload_identity_valid": not blockers,
            "protocol_hash_bound": bool(identity.protocol_hash),
        }
        evidence = self._component_execution(node)
        if evidence is None:
            blockers.append("component_execution_evidence_missing")
            checks["component_execution_evidence"] = False
        else:
            smoke_ready = bool(evidence.adapters) and all(
                adapter.smoke_test.passed
                and adapter.smoke_test.evidence_kind == "local"
                for adapter in evidence.adapters
            )
            checks["component_contract_shape_forward_smoke"] = smoke_ready
            if not smoke_ready:
                blockers.append("component_local_smoke_incomplete")
            if evidence.runtime_payload_hash != identity.runtime_payload_hash:
                blockers.append("component_execution_payload_hash_mismatch")
            if evidence.protocol_hash != identity.protocol_hash:
                blockers.append("component_execution_protocol_hash_mismatch")
        blockers = list(dict.fromkeys(blockers))
        record = RuntimeReadinessRecord(
            identity=identity,
            passed=not blockers,
            evidence_kind="local",
            checks=checks,
            blockers=blockers,
        )
        path = self._write(record)
        return AutomaticRuntimeReadinessResult(
            allowed=not blockers,
            status="ready" if not blockers else "failed",
            cache_key=identity.cache_key,
            component_ids=list(identity.component_ids),
            checks=checks,
            blockers=blockers,
            artifact_path=path,
        )

    def _node_identity(
        self,
        node: ExperimentNode,
    ) -> tuple[RuntimeReadinessIdentity | None, list[str]]:
        command = node.command_spec
        if command is None:
            return None, ["runtime_readiness_command_missing"]
        metadata = command.metadata
        adapter_hashes = _metadata_mapping(metadata.get("adapter_hashes"))
        payload_hash = str(metadata.get("adapter_runtime_payload_hash") or "")
        protocol_hash = str(metadata.get("adapter_runtime_protocol_hash") or "")
        components = sorted(set(node.candidate_config.components))
        errors: list[str] = []
        if set(adapter_hashes) != set(components):
            errors.append("runtime_readiness_adapter_hashes_missing")
        if not payload_hash:
            errors.append("runtime_readiness_payload_hash_missing")
        if not protocol_hash:
            errors.append("runtime_readiness_protocol_hash_missing")
        if errors:
            return None, errors
        return self._identity(
            scope="candidate_runtime",
            component_ids=components,
            adapter_hashes=adapter_hashes,
            runtime_payload_hash=payload_hash,
            protocol_hash=protocol_hash,
        ), []

    @staticmethod
    def _component_execution(node: ExperimentNode) -> ComponentExecutionResult | None:
        command = node.command_spec
        if command is None:
            return None
        path = Path(str(command.metadata.get("adapter_evidence_path") or ""))
        if not path.is_file():
            return None
        try:
            return ComponentExecutionResult.from_yaml(path)
        except (OSError, TypeError, ValueError):
            return None

    @staticmethod
    def _identity(
        *,
        scope: RuntimeReadinessScope,
        component_ids: list[str],
        adapter_hashes: dict[str, str],
        runtime_payload_hash: str,
        protocol_hash: str,
    ) -> RuntimeReadinessIdentity:
        return RuntimeReadinessIdentity(
            scope=scope,
            component_ids=sorted(component_ids),
            adapter_hashes=dict(sorted(adapter_hashes.items())),
            runtime_payload_hash=runtime_payload_hash,
            protocol_hash=protocol_hash,
            python_version=platform.python_version(),
            ultralytics_version=installed_ultralytics_version(),
        )

    def _path(self, identity: RuntimeReadinessIdentity) -> Path:
        return self.cache_dir / f"{identity.scope}-{identity.cache_key}.yaml"

    def _read(self, identity: RuntimeReadinessIdentity) -> RuntimeReadinessRecord | None:
        path = self._path(identity)
        if not path.is_file():
            return None
        try:
            record = RuntimeReadinessRecord.from_yaml(path)
        except (OSError, TypeError, ValueError):
            return None
        return record if record.identity == identity else None

    def _write(self, record: RuntimeReadinessRecord) -> Path:
        path = self._path(record.identity)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        record.to_yaml(temporary, exclude_none=True, sort_keys=False)
        temporary.replace(path)
        return path


def _metadata_mapping(value: object) -> dict[str, str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(item)
        for key, item in value.items()
        if str(key) and str(item)
    }


__all__ = [
    "AutomaticRuntimeReadinessGate",
    "AutomaticRuntimeReadinessResult",
    "RuntimeReadinessIdentity",
    "RuntimeReadinessRecord",
]
