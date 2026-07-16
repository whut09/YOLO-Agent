"""YOLO26-specific compatibility and experiment-discipline checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import BaseModel, Field

from yolo_agent.components.contracts import ComponentContract, contract_from_card
from yolo_agent.components.schema import ComponentCard
from yolo_agent.resources import ResourcePaths


class YOLO26CompatibilityResult(BaseModel):
    """Structured result used by policy and execution gates."""

    compatible: bool
    incompatible: bool
    research_adapter_required: bool = False
    metadata_only: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    required_adapters: list[str] = Field(default_factory=list)
    changed_variables: list[str] = Field(default_factory=list)


class YOLO26CompatibilityChecker:
    """Validate YOLO26's NMS-free, DFL-free, one-to-one execution path."""

    def __init__(self, rules_path: Path | str | None = None) -> None:
        self.rules_path = Path(rules_path) if rules_path else ResourcePaths.YOLO26_COMPATIBILITY
        self.rules = _load_rules(self.rules_path)

    def check(
        self,
        *,
        components: Iterable[ComponentCard | ComponentContract] = (),
        train_overrides: dict[str, Any] | None = None,
        changed_variables: dict[str, Any] | Iterable[str] | None = None,
        single_variable: bool = False,
        head_mode: str | None = None,
        checkpoint: str | None = None,
        amp: bool | None = None,
        ddp: bool | None = None,
        export_format: str = "none",
        execution_requested: bool = True,
    ) -> YOLO26CompatibilityResult:
        overrides = train_overrides or {}
        contracts = [_as_contract(item) for item in components]
        component_ids = [item.component_id for item in contracts]
        blocked: list[str] = []
        warnings: list[str] = []
        required_adapters: list[str] = []
        metadata = [item.component_id for item in contracts if item.maturity == "metadata_only"]
        variables = _changed_variables(contracts, overrides, changed_variables)
        active_head = str(head_mode or overrides.get("head_mode") or self.rules["defaults"]["head_mode"])

        imgsz = overrides.get("imgsz", self.rules["fixed_imgsz"])
        if int(imgsz) != int(self.rules["fixed_imgsz"]):
            blocked.append(f"fixed_imgsz_violation:{imgsz}")
        if overrides.get("allow_imgsz_increase") is True:
            blocked.append("automatic_imgsz_increase_forbidden")

        nms_prefixes = tuple(self.rules["component_rules"]["nms"]["prefixes"])
        postprocess = str(overrides.get("postprocess", overrides.get("postprocess_action", ""))).lower()
        uses_nms = any(item.startswith(nms_prefixes) for item in component_ids) or (
            "nms" in postprocess and postprocess not in {"", "nms_free", "none"}
        )
        if active_head == "one_to_one" and uses_nms:
            blocked.append("one_to_one_head_uses_nms_recipe")

        dfl_constraints = self.rules["component_rules"]["dfl_dependent"]["constraints"]
        for item in contracts:
            raw = _source_constraints(item)
            if any(bool(raw.get(name)) for name in dfl_constraints):
                blocked.append(f"dfl_dependent_loss_on_dfl_free_regression:{item.component_id}")

        anchor_constraints = self.rules["component_rules"]["anchor_based_assigner"]["constraints"]
        for item in contracts:
            raw = _source_constraints(item)
            if item.category == "assigner" and any(bool(raw.get(name)) for name in anchor_constraints):
                adapter = item.adapter_class or f"YOLO26{_adapter_name(item.component_id)}Adapter"
                required_adapters.append(adapter)
                if not item.adapter_class or not item.can_execute:
                    blocked.append(f"anchor_based_assigner_requires_adapter:{item.component_id}")

        for rule_name in ("stal", "musgd", "progressive_loss"):
            rule = self.rules["component_rules"][rule_name]
            if any(component_id in rule["ids"] for component_id in component_ids):
                matching = [item for item in contracts if item.component_id in rule["ids"]]
                if not matching or any(not item.can_execute for item in matching):
                    required_adapters.append(str(rule["adapter"]))

        if single_variable and len(variables) > 1:
            blocked.append("multi_variable_candidate_marked_single_variable")
        structural = {name for name in variables if name in {"assigner", "head", "bbox_loss", "loss"}}
        if single_variable and len(structural) > 1:
            blocked.append("assigner_head_loss_replaced_in_single_variable_ablation")

        if execution_requested and metadata:
            blocked.extend(f"metadata_only_component:{component_id}" for component_id in metadata)

        for item in contracts:
            if checkpoint and item.changes_model_graph is True:
                if item.checkpoint_compatibility in self.rules["checkpoint"]["blocked_values"]:
                    blocked.append(f"checkpoint_incompatible:{item.component_id}")
                elif item.checkpoint_compatibility not in self.rules["checkpoint"]["compatible_values"]:
                    warnings.append(f"Checkpoint compatibility is unverified for {item.component_id}.")
                    required_adapters.append(item.adapter_class or f"{_adapter_name(item.component_id)}Adapter")
            if amp is True and item.supports_amp is False:
                blocked.append(f"amp_unsupported:{item.component_id}")
            if ddp is True and item.supports_ddp is False:
                blocked.append(f"ddp_unsupported:{item.component_id}")
            if export_format == "onnx" and item.supports_onnx is False:
                blocked.append(f"onnx_export_unsupported:{item.component_id}")
            if export_format == "tensorrt" and item.supports_tensorrt is False:
                blocked.append(f"tensorrt_export_unsupported:{item.component_id}")
            if item.fixed_imgsz_compatible is False:
                blocked.append(f"fixed_imgsz_component_incompatible:{item.component_id}")

        required_adapters = sorted(set(required_adapters))
        blocked = sorted(set(blocked))
        return YOLO26CompatibilityResult(
            compatible=not blocked,
            incompatible=bool(blocked),
            research_adapter_required=bool(required_adapters),
            metadata_only=sorted(metadata),
            warnings=sorted(set(warnings)),
            blocked_by=blocked,
            required_adapters=required_adapters,
            changed_variables=sorted(variables),
        )


def _as_contract(item: ComponentCard | ComponentContract) -> ComponentContract:
    return item if isinstance(item, ComponentContract) else contract_from_card(item)


def _source_constraints(contract: ComponentContract) -> dict[str, Any]:
    return contract.tensor_input_contract.get("compatibility_constraints", {})


def _changed_variables(
    contracts: list[ComponentContract],
    overrides: dict[str, Any],
    explicit: dict[str, Any] | Iterable[str] | None,
) -> set[str]:
    if isinstance(explicit, dict):
        variables = set(explicit)
    elif explicit is not None:
        variables = {str(item) for item in explicit}
    else:
        variables = set()
    categories = {
        "detection_head": "head", "head": "head", "assigner": "assigner",
        "bbox_regression_loss": "bbox_loss", "bbox_loss": "bbox_loss",
        "classification_loss": "cls_loss", "cls_loss": "cls_loss",
    }
    variables.update(categories.get(item.category, item.category) for item in contracts)
    variables.update(key for key in overrides if key not in {"seed", "profile", "budget_profile", "imgsz", "allow_imgsz_increase", "gpu_hours", "epochs", "batch", "workers", "device", "val", "fraction"})
    return variables


def _adapter_name(component_id: str) -> str:
    return "".join(part.capitalize() for part in component_id.replace("-", ".").split("."))


def _load_rules(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YOLO26 compatibility YAML must contain a mapping: {path}")
    return data


__all__ = ["YOLO26CompatibilityChecker", "YOLO26CompatibilityResult"]
