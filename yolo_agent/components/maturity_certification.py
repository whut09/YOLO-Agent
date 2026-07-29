"""Apply verified GPU certification reports to component maturity contracts."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.certification.schemas import CertificationReport
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity import (
    ComponentMaturityArtifact,
    MaturityName,
    maturity_artifact,
    maturity_rank,
    record_maturity_artifact,
    transition_maturity,
)


class CertificationMaturityResult(BaseModel):
    """Outcome of applying one terminal certification report."""

    model_config = ConfigDict(extra="forbid")

    contract: ComponentContract
    report_status: str
    promoted_to: list[MaturityName] = Field(default_factory=list)
    retained_without_promotion: list[MaturityName] = Field(default_factory=list)


def apply_certification_report(
    contract: ComponentContract,
    report_path: Path | str,
    *,
    expected_recipe_id: str,
) -> CertificationMaturityResult:
    """Retain a verified report and advance only adjacent maturity states."""
    path = Path(report_path)
    report = CertificationReport.load_verified(path)
    if report.status not in {"passed", "failed"}:
        raise ValueError("only terminal passed or failed certification reports can be applied")
    if report.executed_recipe_id != expected_recipe_id:
        raise ValueError(
            "certification recipe mismatch: "
            f"expected {expected_recipe_id}, got {report.executed_recipe_id}"
        )

    targets: list[MaturityName] = (
        ["gpu_certified"]
        if report.level == "mini_gpu_pilot"
        else ["full_reproduced", "confirmed_multi_seed"]
    )
    updated = contract
    promoted: list[MaturityName] = []
    retained: list[MaturityName] = []
    for target in targets:
        artifact = _certification_artifact(
            updated.component_id,
            target,
            path,
            report,
        )
        if _already_recorded(updated, artifact):
            continue
        adjacent = maturity_rank(target) == maturity_rank(updated.maturity) + 1
        if report.status == "passed" and adjacent:
            updated = transition_maturity(
                updated,
                target,
                reason=f"verified {report.level} certification {report.certification_id}",
                artifact=artifact,
            )
            promoted.append(target)
        else:
            updated = record_maturity_artifact(updated, artifact)
            retained.append(target)
    return CertificationMaturityResult(
        contract=updated,
        report_status=report.status,
        promoted_to=promoted,
        retained_without_promotion=retained,
    )


def _certification_artifact(
    component_id: str,
    target: MaturityName,
    path: Path,
    report: CertificationReport,
) -> ComponentMaturityArtifact:
    return maturity_artifact(
        component_id=component_id,
        target_maturity=target,
        artifact_path=path,
        status="passed" if report.status == "passed" else "failed",
        producer="RealGpuAcceptanceSuite",
        protocol_hash=report.protocol_hash,
        metadata={
            "certification_id": report.certification_id,
            "certification_level": report.level,
            "executed_recipe_id": report.executed_recipe_id,
            "report_hash": report.report_hash,
        },
    )


def _already_recorded(
    contract: ComponentContract,
    artifact: ComponentMaturityArtifact,
) -> bool:
    return any(
        item.target_maturity == artifact.target_maturity
        and item.artifact_sha256 == artifact.artifact_sha256
        for item in contract.maturity_artifacts
    )


__all__ = ["CertificationMaturityResult", "apply_certification_report"]
