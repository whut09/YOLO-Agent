"""Component-specific certification gate before automatic queue admission."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.certification.code_identity import certification_code_hash
from yolo_agent.certification.schemas import CertificationReport


class ComponentQueueCertificationResult(BaseModel):
    """Auditable component certification decision for one recipe."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    component_ids: list[str]
    report_path: Path | None = None
    report_hash: str | None = None
    required_capabilities: list[str] = Field(default_factory=list)
    observed_capabilities: list[str] = Field(default_factory=list)
    checks: dict[str, bool] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)


class ComponentQueueCertificationGate:
    """Require the sampling golden path before ASHA can own its pilot budget."""

    sampling_component = "sampling.small_object"
    sampling_capability = "small_object_sampling_runtime"
    sampling_recipe = "small_object_sampling"
    sampling_stages = {
        "component_runtime_certification",
        "runtime_adapter",
        "post_eval",
        "error_facts",
        "paired_delta",
        "paired_bootstrap",
        "asha_decision",
        "pilot_10",
        "promotion_gate",
    }

    def evaluate(
        self,
        *,
        component_ids: list[str],
        report_path: Path | str | None,
    ) -> ComponentQueueCertificationResult:
        components = sorted(set(component_ids))
        if self.sampling_component not in components:
            return ComponentQueueCertificationResult(
                allowed=True,
                component_ids=components,
            )
        required = [self.sampling_capability]
        if report_path is None:
            return ComponentQueueCertificationResult(
                allowed=False,
                component_ids=components,
                required_capabilities=required,
                blockers=["sampling_end_to_end_certification_report_missing"],
            )
        path = Path(report_path)
        try:
            report = CertificationReport.load_verified(path)
        except (OSError, TypeError, ValueError) as exc:
            return ComponentQueueCertificationResult(
                allowed=False,
                component_ids=components,
                report_path=path,
                required_capabilities=required,
                blockers=[f"sampling_end_to_end_certification_report_invalid:{exc}"],
            )

        passed_stages = {
            stage.stage_id for stage in report.stages if stage.status == "passed"
        }
        observed = sorted({claim.capability_id for claim in report.capability_claims})
        capability_matched = any(
            claim.capability_id == self.sampling_capability
            and claim.recipe_id == self.sampling_recipe
            and claim.local_reproduction == "locally_pilot_reproduced"
            for claim in report.capability_claims
        )
        promotions = {item.stage_id: item.passed for item in report.promotion_results}
        objective = report.objective
        checks = {
            "report_passed": report.status == "passed",
            "fixed_imgsz_640": report.fixed_imgsz == 640,
            "code_hash_matched": report.certified_code_hash
            == certification_code_hash(),
            "sampling_recipe_executed": report.executed_recipe_id
            == self.sampling_recipe,
            "sampling_capability_claimed": capability_matched,
            "sampling_stages_complete": self.sampling_stages.issubset(passed_stages),
            "pilot_3_promoted": promotions.get("pilot_3") is True,
            "pilot_10_promoted": promotions.get("pilot_10") is True,
            "objective_passed": bool(objective is not None and objective.passed),
            "ap_small_objective": bool(
                objective is not None and objective.primary_metric == "ap_small"
            ),
            "target_error_delta_present": bool(
                objective is not None and objective.target_error_fact_deltas
            ),
        }
        blockers = [
            f"sampling_end_to_end_certification_failed:{name}"
            for name, passed in checks.items()
            if not passed
        ]
        return ComponentQueueCertificationResult(
            allowed=not blockers,
            component_ids=components,
            report_path=path,
            report_hash=report.report_hash,
            required_capabilities=required,
            observed_capabilities=observed,
            checks=checks,
            blockers=blockers,
        )


__all__ = [
    "ComponentQueueCertificationGate",
    "ComponentQueueCertificationResult",
]
