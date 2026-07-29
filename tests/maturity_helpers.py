"""Artifact-backed component maturity fixtures for downstream gate tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity import ComponentMaturityArtifact


def with_smoke_artifact(
    contract: ComponentContract,
    artifact_path: Path | None = None,
) -> ComponentContract:
    """Return a smoke-ready contract for tests that exercise later gates."""
    path = artifact_path or Path(__file__)
    artifact = ComponentMaturityArtifact(
        component_id=contract.component_id,
        target_maturity="smoke_passed",
        artifact_type="smoke_report",
        artifact_path=path,
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        status="passed",
        producer="pytest_fixture",
    )
    return contract.model_copy(
        update={
            "maturity": "smoke_passed",
            "maturity_artifacts": [*contract.maturity_artifacts, artifact],
        }
    )


__all__ = ["with_smoke_artifact"]
