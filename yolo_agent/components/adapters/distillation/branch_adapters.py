"""Independent adapter classes for the distillation branches.

Each branch runs its own mechanism, payload, and evidence schema.  The
classes are identity carriers: the shared runtime plugin executes the branch
loss, while the branch class pins the branch identity into every runtime
payload so one branch is never certified through another branch's adapter.
"""

from __future__ import annotations

from typing import Any

from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.distillation.method_registry import (
    BRANCH_TO_MECHANISM,
    DistillationBranchId,
    DistillationBranchSpec,
    build_branch,
)
from yolo_agent.components.adapters.distillation.yolo26_distillation import (
    YOLO26DistillationAdapter,
)
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.components.distillation.mechanisms import DISTILLATION_MECHANISMS


REQUIRED_BRANCH_ADAPTERS: tuple[DistillationBranchId, ...] = (
    "logits_distillation",
    "feature_distillation",
    "relation_distillation",
    "localization_distillation",
    "attention_distillation",
    "masked_feature_distillation",
    "quality_aware_distillation",
    "teacher_ensemble",
)

BRANCH_ADAPTERS: dict[
    DistillationBranchId, type[YOLO26DistillationAdapter]
] = {}


def branch_adapter_class_name(branch_id: DistillationBranchId) -> str:
    camel = "".join(part[:1].upper() + part[1:] for part in branch_id.split("_") if part)
    return f"{camel}Adapter"


def make_branch_adapter(
    branch_id: DistillationBranchId,
) -> type[YOLO26DistillationAdapter]:
    """Create (or return) the independent adapter class for one branch."""
    if branch_id not in BRANCH_TO_MECHANISM:
        raise ValueError(f"unknown distillation branch: {branch_id}")
    existing = BRANCH_ADAPTERS.get(branch_id)
    if existing is not None:
        return existing
    mechanism = BRANCH_TO_MECHANISM[branch_id]
    spec = DISTILLATION_MECHANISMS[mechanism]
    class_name = branch_adapter_class_name(branch_id)

    class BranchDistillationAdapter(YOLO26DistillationAdapter):
        branch_id: DistillationBranchId
        branch_mechanism: str
        branch_component_id: str
        branch_changed_variable: str

        def branch_spec(self) -> DistillationBranchSpec:
            return build_branch(self.branch_id)

        def build_runtime_payload(
            self,
            context: AdapterContext,
            *,
            protocol_hash: str,
            base_command: list[str],
            generated_config: dict[str, Any],
        ) -> AdapterRuntimePayload:
            options = dict(context.options)
            options.setdefault("component_id", self.branch_component_id)
            options.setdefault("mechanism", self.branch_mechanism)
            options.setdefault("branch_id", self.branch_id)
            options.setdefault("changed_variable", self.branch_changed_variable)
            context.options = options
            payload = super().build_runtime_payload(
                context,
                protocol_hash=protocol_hash,
                base_command=base_command,
                generated_config=generated_config,
            )
            payload.component_ids = [self.branch_component_id]
            payload.adapter_classes = [type(self).__name__]
            payload.adapter_versions = {self.branch_component_id: self.adapter_version}
            payload.source_commits = {self.branch_component_id: self.source_commit}
            return payload

    BranchDistillationAdapter.__name__ = class_name
    BranchDistillationAdapter.__qualname__ = class_name
    BranchDistillationAdapter.adapter_version = "yolo26_distillation_branch.v1"
    BranchDistillationAdapter.source_commit = f"yolo-agent:distillation-branch:{branch_id}"
    BranchDistillationAdapter.branch_id = branch_id
    BranchDistillationAdapter.branch_mechanism = mechanism
    BranchDistillationAdapter.branch_component_id = spec.component_id
    BranchDistillationAdapter.branch_changed_variable = spec.changed_variable
    globals()[class_name] = BranchDistillationAdapter
    BRANCH_ADAPTERS[branch_id] = BranchDistillationAdapter
    return BranchDistillationAdapter


def branch_adapter(branch_id: DistillationBranchId) -> type[YOLO26DistillationAdapter]:
    return make_branch_adapter(branch_id)


def _init_branch_adapters() -> None:
    for branch_id in BRANCH_TO_MECHANISM:
        make_branch_adapter(branch_id)


_init_branch_adapters()


__all__ = [
    "BRANCH_ADAPTERS",
    "REQUIRED_BRANCH_ADAPTERS",
    "branch_adapter",
    "branch_adapter_class_name",
    "make_branch_adapter",
]
