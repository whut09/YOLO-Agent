"""Locked machine-local registry for artifact-backed component maturity."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from typing import Any, Iterator

import yaml

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity import (
    ComponentMaturityArtifact,
    maturity_rank,
    record_maturity_artifact,
    transition_maturity,
)
from yolo_agent.components.maturity_registry_schemas import (
    ComponentEvidenceOverlay,
    ComponentMaturityRegistryDocument,
    ComponentOverlayResolution,
)


_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class ComponentMaturityRegistry:
    """Persist and resolve local maturity overlays without editing source YAML."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")

    def load(self) -> ComponentMaturityRegistryDocument:
        """Load a consistent registry snapshot under the file lock."""
        with _file_lock(self.lock_path):
            return self._load_unlocked()

    def upsert(self, overlay: ComponentEvidenceOverlay) -> ComponentEvidenceOverlay:
        """Atomically merge one overlay; repeated identical writes are no-ops."""
        with _file_lock(self.lock_path):
            document = self._load_unlocked()
            existing = next(
                (
                    item
                    for item in document.overlays
                    if item.identity_key == overlay.identity_key
                ),
                None,
            )
            if existing is not None:
                merged = _merge_overlay(existing, overlay)
                if merged.evidence_hash == existing.evidence_hash:
                    return existing
                overlays = [
                    merged if item.identity_key == overlay.identity_key else item
                    for item in document.overlays
                ]
                stored = merged
            else:
                overlays = [*document.overlays, overlay]
                stored = overlay
            updated = document.model_copy(
                update={"overlays": sorted(overlays, key=_overlay_sort_key)}
            )
            self._write_unlocked(updated)
            return stored

    def record_contract(
        self,
        contract: ComponentContract,
        *,
        adapter_hash: str,
        code_commit: str,
        ultralytics_version: str,
        protocol_hash: str,
    ) -> ComponentEvidenceOverlay:
        """Persist all current evidence from a validated component contract."""
        overlay = ComponentEvidenceOverlay(
            component_id=contract.component_id,
            adapter_hash=adapter_hash,
            code_commit=code_commit,
            ultralytics_version=ultralytics_version,
            protocol_hash=protocol_hash,
            artifacts=list(contract.maturity_artifacts),
        )
        return self.upsert(overlay)

    def apply(
        self,
        contract: ComponentContract,
        *,
        adapter_hash: str,
        ultralytics_version: str,
        protocol_hash: str | None = None,
    ) -> tuple[ComponentContract, ComponentOverlayResolution]:
        """Merge the newest valid matching overlay into a conservative contract."""
        effective, resolution, _ = self.resolve(
            contract,
            adapter_hash=adapter_hash,
            ultralytics_version=ultralytics_version,
            protocol_hash=protocol_hash,
        )
        return effective, resolution

    def resolve(
        self,
        contract: ComponentContract,
        *,
        adapter_hash: str,
        ultralytics_version: str,
        protocol_hash: str | None = None,
    ) -> tuple[ComponentContract, ComponentOverlayResolution, ComponentEvidenceOverlay | None]:
        """Return the effective contract, audit result, and selected overlay."""
        candidates = [
            item
            for item in self.load().overlays
            if item.component_id == contract.component_id
            and item.adapter_hash == adapter_hash
            and item.ultralytics_version == ultralytics_version
            and (protocol_hash is None or item.protocol_hash == protocol_hash)
        ]
        if not candidates:
            return contract, ComponentOverlayResolution(
                status="no_match",
                component_id=contract.component_id,
                source_maturity=contract.maturity,
                effective_maturity=contract.maturity,
                reasons=["matching_component_runtime_overlay_not_found"],
            ), None
        overlay = max(candidates, key=lambda item: item.updated_at)
        effective, resolution = _apply_overlay(contract, overlay)
        return effective, resolution, overlay

    def _load_unlocked(self) -> ComponentMaturityRegistryDocument:
        if not self.path.is_file():
            return ComponentMaturityRegistryDocument()
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8-sig")) or {}
        return ComponentMaturityRegistryDocument.model_validate(raw)

    def _write_unlocked(self, document: ComponentMaturityRegistryDocument) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                yaml.safe_dump(
                    document.model_dump(mode="json", exclude_none=True),
                    stream,
                    sort_keys=False,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def adapter_source_hash(
    contract: ComponentContract,
    *,
    adapter: Any | None = None,
) -> str:
    """Hash the concrete adapter source file used by a contract."""
    implementation = adapter
    if implementation is None:
        if not contract.implementation_path or not contract.adapter_class:
            raise ValueError(f"component has no adapter implementation: {contract.component_id}")
        module = importlib.import_module(contract.implementation_path)
        implementation = getattr(module, contract.adapter_class, None)
    implementation_type = implementation if isinstance(implementation, type) else type(implementation)
    source = inspect.getsourcefile(implementation_type)
    if not source or not Path(source).is_file():
        raise ValueError(f"adapter source file is unavailable: {contract.component_id}")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def current_code_commit(root: Path | str | None = None) -> str:
    """Return the current Git commit for provenance, or an explicit fallback."""
    workdir = Path(root or Path(__file__).resolve().parents[2])
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return completed.stdout.strip() or "unavailable"


def installed_ultralytics_version() -> str:
    """Return the installed runtime version without importing training code."""
    try:
        return importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _apply_overlay(
    contract: ComponentContract,
    overlay: ComponentEvidenceOverlay,
) -> tuple[ComponentContract, ComponentOverlayResolution]:
    updated = contract
    applied: list[str] = []
    retained: list[str] = []
    invalid: list[str] = []
    artifacts = sorted(
        overlay.artifacts,
        key=lambda item: (
            maturity_rank(item.target_maturity),
            item.status != "failed",
            item.artifact_sha256,
        ),
    )
    for artifact in artifacts:
        reason = _artifact_invalid_reason(artifact, overlay)
        if reason is not None:
            invalid.append(reason)
            continue
        if _artifact_recorded(updated, artifact):
            continue
        adjacent = maturity_rank(artifact.target_maturity) == maturity_rank(updated.maturity) + 1
        if artifact.status == "passed" and not artifact.mock and adjacent:
            updated = transition_maturity(
                updated,
                artifact.target_maturity,
                reason=f"valid local maturity overlay {overlay.identity_key}",
                artifact=artifact,
            )
            applied.append(artifact.artifact_sha256)
        else:
            updated = record_maturity_artifact(updated, artifact)
            retained.append(artifact.artifact_sha256)
    status = "applied" if applied or retained else "invalid" if invalid else "no_match"
    reasons: list[str] = []
    if invalid:
        reasons.append("invalid_artifacts_were_excluded")
    if updated.maturity == contract.maturity and not retained:
        reasons.append("no_adjacent_valid_promotion_artifact")
    return updated, ComponentOverlayResolution(
        status=status,
        component_id=contract.component_id,
        source_maturity=contract.maturity,
        effective_maturity=updated.maturity,
        overlay_identity_key=overlay.identity_key,
        applied_artifact_hashes=applied,
        retained_artifact_hashes=retained,
        invalid_artifacts=invalid,
        reasons=reasons,
    )


def _artifact_invalid_reason(
    artifact: ComponentMaturityArtifact,
    overlay: ComponentEvidenceOverlay,
) -> str | None:
    if artifact.protocol_hash not in {None, overlay.protocol_hash}:
        return f"artifact_protocol_mismatch:{artifact.target_maturity}"
    try:
        artifact.verify()
    except ValueError as exc:
        return f"artifact_invalid:{artifact.target_maturity}:{exc}"
    return None


def _merge_overlay(
    existing: ComponentEvidenceOverlay,
    incoming: ComponentEvidenceOverlay,
) -> ComponentEvidenceOverlay:
    artifacts = list(existing.artifacts)
    known = {
        (item.target_maturity, item.artifact_sha256, item.status, item.mock)
        for item in artifacts
    }
    for artifact in incoming.artifacts:
        key = (
            artifact.target_maturity,
            artifact.artifact_sha256,
            artifact.status,
            artifact.mock,
        )
        if key not in known:
            artifacts.append(artifact)
            known.add(key)
    return existing.model_copy(
        update={
            "artifacts": sorted(
                artifacts,
                key=lambda item: (
                    maturity_rank(item.target_maturity),
                    item.artifact_sha256,
                ),
            ),
            "updated_at": datetime.now(timezone.utc),
        }
    )


def _artifact_recorded(
    contract: ComponentContract,
    artifact: ComponentMaturityArtifact,
) -> bool:
    return any(
        item.target_maturity == artifact.target_maturity
        and item.artifact_sha256 == artifact.artifact_sha256
        and item.status == artifact.status
        and item.mock == artifact.mock
        for item in contract.maturity_artifacts
    )


def _overlay_sort_key(overlay: ComponentEvidenceOverlay) -> tuple[str, str]:
    return overlay.component_id, overlay.identity_key


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    key = str(path.resolve()).casefold()
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(key, threading.RLock())
    with thread_lock:
        with path.open("a+b") as stream:
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            _lock_stream(stream)
            try:
                yield
            finally:
                stream.seek(0)
                _unlock_stream(stream)


def _lock_stream(stream: Any) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)


def _unlock_stream(stream: Any) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


__all__ = [
    "ComponentMaturityRegistry",
    "adapter_source_hash",
    "current_code_commit",
    "installed_ultralytics_version",
]
