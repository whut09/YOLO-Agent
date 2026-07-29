"""Generate bounded implementation requests without generating adapter code."""

from __future__ import annotations

from yolo_agent.agents.paper_adapter_planning.schemas import (
    AdapterImplementationEstimate,
    PaperAdapterImplementationRequest,
)
from yolo_agent.research.component_aliases import ResolvedComponentAlias


def build_implementation_request(
    *,
    fingerprint: str,
    mapping: ResolvedComponentAlias,
    paper_ids: list[str],
    estimate: AdapterImplementationEstimate,
) -> PaperAdapterImplementationRequest:
    tests = [
        "adapter_payload_serialization",
        "adapter_dry_run",
        "fixed_imgsz_640",
        "rollback_plan",
        "offline_smoke_test",
    ]
    if mapping.category in {"backbone", "neck", "detection_head", "feature_pyramid", "attention"}:
        tests.extend(["shape", "backward", "amp", "export_dry_run", "latency_guard", "model_size_guard"])
    if mapping.category in {"assigner", "matching", "positive_sample_selection"}:
        tests.extend(["shadow_assignment_diff", "positive_ratio_evidence", "conflict_rate_evidence"])
    if mapping.category in {"sampling", "augmentation"}:
        tests.extend(["train_dataloader_only", "deterministic_resume", "validation_unchanged"])
    if mapping.category in {"slicing", "tta", "nms"}:
        tests.extend(["standard_metric_namespace_unchanged", "inference_pareto_isolated"])
    return PaperAdapterImplementationRequest(
        request_id=f"adapter-{fingerprint[:16]}",
        component_id=mapping.canonical_component_id,
        paper_ids=sorted(set(paper_ids)),
        insertion_point=mapping.insertion_point,
        required_runtime_hook=estimate.required_runtime_hook,
        reason=(
            "Implement and verify the canonical component through the declared runtime hook; "
            "catalog applicability and paper claims do not authorize execution."
        ),
        acceptance_tests=list(dict.fromkeys(tests)),
        generated_code_allowed=False,
    )


__all__ = ["build_implementation_request"]
