"""CPU golden-path certification for YOLO26 assignment shadow plugins."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from yolo_agent.adapters.ultralytics.plugin_bridge import (
    PluginCriterionWrapper,
    UltralyticsTrainerPluginBridge,
)
from yolo_agent.adapters.ultralytics.plugin_context import runtime_evidence_path
from yolo_agent.certification.graph_assignment_schemas import (
    AssignmentShadowCpuReport,
)
from yolo_agent.components.adapters.assigners.yolo26_assignment import (
    AssignmentActivationGate,
    AssignmentShadowEvidence,
)
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.recipes.schemas import AtomicRecipe, recipe_from_mapping


ASSIGNMENT_RECIPE_IDS = {
    "assigner.task_aligned": "yolo26_tood_tal_assignment_shadow",
    "assigner.optimal_transport": "yolo26_ota_assignment_shadow",
    "assigner.dynamic_smooth_label": "yolo26_dsla_assignment_shadow",
    "assigner.task_aligned_weighting": "yolo26_task_aligned_weighting_shadow",
    "assigner.dynamic_topk": "yolo26_dynamic_topk_assignment_shadow",
    "assigner.quality_aware": "yolo26_quality_aware_assignment_shadow",
    "assigner.soft_label": "yolo26_soft_label_assignment_shadow",
    "assigner.dual_path": "yolo26_dual_path_assignment_shadow",
    "assigner.conflict_aware": "yolo26_conflict_aware_assignment_shadow",
}


def run_assignment_shadow_cpu_fixture(
    *,
    runtime_payload_path: Path | str,
    workspace: Path | str,
) -> AssignmentShadowCpuReport:
    """Compare native and candidate assignment while returning native loss unchanged."""
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload_path = Path(runtime_payload_path).resolve()
    payload = AdapterRuntimePayload.read(payload_path, verify_imports=True)
    component_id = _assignment_component(payload)
    reference = payload.assigner_plugin[0]
    method = str(reference.options["method"])
    recipe_id = ASSIGNMENT_RECIPE_IDS[component_id]
    evidence_path = payload_path.parent / f"assignment_{method}_shadow_evidence.json"
    report_path = root / f"assignment_{method}_shadow_cpu_golden_path.yaml"
    # Isolated certification must not inherit counters or aggregates from a
    # previous invocation that reused the same payload and work directory.
    runtime_evidence_path(payload_path).unlink(missing_ok=True)
    evidence_path.unlink(missing_ok=True)
    checks: dict[str, bool | str | int | float] = {}
    metrics: dict[str, float] = {}
    errors: list[str] = []
    try:
        import torch
        from ultralytics.cfg import get_cfg
        from ultralytics.nn.tasks import DetectionModel

        checks["atomic_recipe_verified"] = _atomic_recipe_verified(
            component_id,
            recipe_id,
        )
        checks["matched_control_required"] = _matched_control_required(recipe_id)
        checks["shadow_mode_only"] = reference.options.get("mode") == "shadow"
        torch.manual_seed(37)
        model = DetectionModel("yolo26n.yaml", ch=3, nc=3, verbose=False)
        model.args = get_cfg(overrides={"imgsz": 640})
        model.train()
        trainer = SimpleNamespace(args=get_cfg(overrides={"imgsz": 640}))
        bridge = UltralyticsTrainerPluginBridge(payload_path)
        bridge.install_model_hooks(model, trainer=trainer)
        wrapped = model.init_criterion()
        if not isinstance(wrapped, PluginCriterionWrapper):
            raise TypeError("assignment bridge did not install criterion wrapper")
        native_criterion = wrapped.criterion
        native_one_to_many = native_criterion.one2many.assigner
        native_one_to_one = native_criterion.one2one.assigner
        image = torch.rand(1, 3, 64, 64)
        batch = {
            "img": image,
            "batch_idx": torch.tensor([0]),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
        }
        predictions = model(image)
        native_loss, native_items = native_criterion(predictions, batch)
        shadow_loss, shadow_items = wrapped(predictions, batch)
        checks["native_loss_equivalent"] = bool(
            torch.equal(shadow_loss, native_loss)
            and torch.equal(shadow_items, native_items)
        )
        checks["native_one_to_one_preserved"] = bool(
            native_criterion.one2many.assigner is native_one_to_many
            and native_criterion.one2one.assigner is native_one_to_one
        )
        evidence = AssignmentShadowEvidence.model_validate_json(
            evidence_path.read_text(encoding="utf-8-sig")
        )
        aggregate = evidence.aggregate
        metrics = {
            "baseline_positive_ratio": aggregate.baseline_positive_ratio,
            "candidate_positive_ratio": aggregate.candidate_positive_ratio,
            "conflict_rate": aggregate.conflict_rate,
            "matching_stability": aggregate.matching_stability,
        }
        for path, path_aggregate in evidence.path_aggregates.items():
            metrics[f"{path}.baseline_positive_ratio"] = (
                path_aggregate.baseline_positive_ratio
            )
            metrics[f"{path}.candidate_positive_ratio"] = (
                path_aggregate.candidate_positive_ratio
            )
            metrics[f"{path}.conflict_rate"] = path_aggregate.conflict_rate
            metrics[f"{path}.matching_stability"] = (
                path_aggregate.matching_stability
            )
        checks["native_audit_verified"] = evidence.native_audit.verified
        checks["positive_ratio_recorded"] = bool(
            aggregate.baseline_positive_count > 0
            and aggregate.candidate_positive_count > 0
            and 0.0 <= aggregate.baseline_positive_ratio <= 1.0
            and 0.0 <= aggregate.candidate_positive_ratio <= 1.0
        )
        checks["conflict_rate_recorded"] = bool(
            0.0 <= aggregate.conflict_rate <= 1.0
        )
        checks["matching_stability_recorded"] = bool(
            0.0 <= aggregate.matching_stability <= 1.0
        )
        expected_paths = (
            {"one_to_many", "one_to_one"}
            if reference.options.get("assignment_path") == "both"
            else {str(reference.options.get("assignment_path", "one_to_many"))}
        )
        checks["per_path_metrics_recorded"] = bool(
            set(evidence.path_aggregates) == expected_paths
            and all(
                item.batches > 0 and 0.0 <= item.matching_stability <= 1.0
                for item in evidence.path_aggregates.values()
            )
        )
        checks["shadow_passed"] = evidence.shadow_passed
        checks["paper_claim_not_local_evidence"] = bool(
            evidence.paper_prior.evidence_level == "paper_prior"
            and evidence.paper_prior.exact_reproduction is False
            and not evidence.paper_prior.reported_delta
        )
        missing_active = AssignmentActivationGate().evaluate(
            root / "missing-shadow-evidence.json",
            component_id=component_id,
            method=method,  # type: ignore[arg-type]
            assignment_path=str(reference.options.get("assignment_path", "one_to_many")),
            minimum_batches=1,
            maximum_conflict_rate=1.0,
        )
        checks["active_pilot_blocked_until_explicit_gate"] = bool(
            not missing_active.allowed
            and missing_active.blocked_by == ["shadow_evidence_missing"]
        )
        checks["trainer_bridge_called"] = bool(
            sum(
                hooks.get("compute_loss", 0)
                for hooks in bridge.context.evidence.hook_call_counts.values()
            )
            == 1
        )
        failed = sorted(key for key, value in checks.items() if value is not True)
        if failed:
            errors.append("failed assignment shadow CPU checks: " + ", ".join(failed))
    except (
        AttributeError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(str(exc))

    report = AssignmentShadowCpuReport(
        component_id=component_id,
        method=method,  # type: ignore[arg-type]
        recipe_id=recipe_id,
        status="failed" if errors else "passed",
        protocol_hash=payload.protocol_hash,
        runtime_payload_hash=payload.payload_hash,
        runtime_payload_path=payload_path,
        shadow_evidence_path=evidence_path if evidence_path.is_file() else None,
        checks=checks,
        metrics=metrics,
        errors=errors,
    )
    report.to_yaml(report_path, exclude_none=True, sort_keys=False)
    return report


def _assignment_component(payload: AdapterRuntimePayload) -> str:
    if (
        len(payload.component_ids) != 1
        or payload.component_ids[0] not in ASSIGNMENT_RECIPE_IDS
        or len(payload.assigner_plugin) != 1
    ):
        raise ValueError("assignment fixture requires one supported assigner plugin")
    if payload.assigner_plugin[0].options.get("mode") != "shadow":
        raise ValueError("assignment certification requires shadow mode")
    return payload.component_ids[0]


def _recipes() -> list[AtomicRecipe]:
    raw = yaml.safe_load(
        Path("configs/recipes/yolo26_assignment_shadow.yaml").read_text(
            encoding="utf-8"
        )
    )
    return [recipe_from_mapping(item) for item in raw["recipes"]]  # type: ignore[return-value]


def _atomic_recipe_verified(component_id: str, recipe_id: str) -> bool:
    recipe = next((item for item in _recipes() if item.recipe_id == recipe_id), None)
    return bool(
        isinstance(recipe, AtomicRecipe)
        and recipe.component_ids == [component_id]
        and recipe.train_overrides.get("imgsz") == 640
        and recipe.train_overrides.get(recipe.primary_changed_variable) == "shadow"
    )


def _matched_control_required(recipe_id: str) -> bool:
    recipe = next(item for item in _recipes() if item.recipe_id == recipe_id)
    return "matched_control" in recipe.compatibility_requirements


__all__ = ["ASSIGNMENT_RECIPE_IDS", "run_assignment_shadow_cpu_fixture"]
