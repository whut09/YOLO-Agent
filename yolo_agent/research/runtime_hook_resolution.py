"""Resolution of canonical :class:`RuntimeHookIdentity` bindings.

Prompt-18G: every paper runtime path resolves to a real callable.  The
resolver covers the four runtime families the preflight executes:

* distillation mechanism losses (phase ``loss``);
* domain-adaptation branch plugins (phase ``loss``);
* component-contract adapters — assigners resolve to the ``assignment``
  phase, loss plugins to ``loss``, and graph-shaping adapters (heads,
  necks, feature pyramids) to ``model_graph``.

Data-side insertion points (``train_dataloader_sampler`` etc.) map to the
``data`` phase and inference protocols to ``inference`` — a data-side paper
is never forced into a model-forward shape it does not have (Prompt-18G
step 4).

Every identity carries a paper-specific ``paper_binding_id``, so a shared
adapter reused by two papers yields two independent audited bindings.
"""

from __future__ import annotations

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.research.runtime_hook_identity import (
    RuntimeHookIdentity,
    RuntimeHookIdentityError,
    RuntimeHookPhase,
)

# insertion_point (then category) -> canonical runtime phase.  Deliberately
# domain-aware: sampling/augmentation/annotation/postprocess surfaces map to
# data/preprocess/postprocess rather than a model-forward phase.
_INSERTION_POINT_PHASES: dict[str, RuntimeHookPhase] = {
    "train_dataloader_sampler": "data",
    "dataloader": "data",
    "dataset": "data",
    "preprocess": "preprocess",
    "mosaic_augmentation": "preprocess",
    "augmentation": "preprocess",
    "one_to_many_assignment": "assignment",
    "trainer_loss": "loss",
    "trainer_loss_after_native_criterion": "loss",
    "response_quality_loss": "loss",
    "model_graph": "model_graph",
    "detection_head": "model_graph",
    "neck": "model_graph",
    "before_detect_p3_p4_p5": "model_graph",
    "feature_pyramid": "model_graph",
    "backbone": "model_graph",
    "optimizer": "optimizer",
    "train_step": "train_step",
    "postprocess": "postprocess",
    "nms": "postprocess",
    "inference_protocol": "inference",
    "inference": "inference",
    "evaluation": "evaluation",
    "coco_evaluator": "evaluation",
}

_CATEGORY_PHASES: dict[str, RuntimeHookPhase] = {
    "assigner": "assignment",
    "loss": "loss",
    "quality_estimation": "loss",
    "distillation": "loss",
    "domain_adaptation": "loss",
    "detection_head": "model_graph",
    "neck": "model_graph",
    "feature_pyramid": "model_graph",
    "backbone": "model_graph",
    "sampling": "data",
    "augmentation": "preprocess",
    "data_pipeline": "data",
    "annotation": "data",
    "postprocess": "postprocess",
    "inference": "inference",
    "slicing": "inference",
    "evaluation": "evaluation",
    "optimizer": "optimizer",
}


def phase_for_contract(contract: ComponentContract) -> RuntimeHookPhase:
    """Domain-aware canonical phase for one component contract."""
    point = (contract.insertion_point or "").strip()
    if point in _INSERTION_POINT_PHASES:
        return _INSERTION_POINT_PHASES[point]
    category = (contract.category or "").strip()
    if category in _CATEGORY_PHASES:
        return _CATEGORY_PHASES[category]
    raise RuntimeHookIdentityError(
        f"cannot resolve a canonical runtime phase for {contract.component_id}: "
        f"unknown insertion_point {point!r} and category {category!r}"
    )


def hook_id_for_contract(contract: ComponentContract, phase: RuntimeHookPhase) -> str:
    """Canonical hook id: ``hook.<phase>.<component subtree>``."""
    parts = contract.component_id.split(".")
    subtree = ".".join(parts[1:]) if len(parts) > 1 else contract.component_id
    return f"hook.{phase}.{subtree}"


def identity_for_contract(
    contract: ComponentContract,
    *,
    paper_id: str,
    method_name: str = "smoke_test",
) -> RuntimeHookIdentity:
    """Resolve one component-contract adapter to its audited hook identity."""
    phase = phase_for_contract(contract)
    implementation_path = contract.implementation_path
    class_name = contract.adapter_class
    if not implementation_path or not class_name:
        raise RuntimeHookIdentityError(
            f"contract {contract.component_id} has no resolvable implementation "
            f"(implementation_path={implementation_path!r}, adapter_class={class_name!r})"
        )
    insertion_point = (contract.insertion_point or "").strip()
    if not insertion_point or insertion_point == "unknown":
        insertion_point = f"phase:{phase}"
    return RuntimeHookIdentity.resolve(
        hook_id=hook_id_for_contract(contract, phase),
        phase=phase,
        implementation_path=implementation_path,
        class_name=class_name,
        method_name=method_name,
        insertion_point=insertion_point,
        component_id=contract.component_id,
        paper_id=paper_id,
        paper_binding_id=f"{paper_id}/{contract.component_id}",
    )


def identity_for_distillation_mechanism(
    loss_object: object, *, paper_id: str, mechanism: str
) -> RuntimeHookIdentity:
    """Resolve a built distillation mechanism loss to its audited identity."""
    cls_obj = type(loss_object)
    module = cls_obj.__module__
    return RuntimeHookIdentity.resolve(
        hook_id="loss.distillation",
        phase="loss",
        implementation_path=module,
        class_name=cls_obj.__name__,
        method_name="compute",
        insertion_point="trainer_loss",
        component_id=f"distillation.{mechanism}",
        paper_id=paper_id,
        paper_binding_id=f"{paper_id}/distillation.{mechanism}",
    )


def identity_for_domain_adaptation_branch(
    *, paper_id: str, branch_id: str
) -> RuntimeHookIdentity:
    """Resolve a DA branch plugin to its audited identity."""
    return RuntimeHookIdentity.resolve(
        hook_id="loss.domain_adaptation",
        phase="loss",
        implementation_path="yolo_agent.components.adapters.domain_adaptation.branch_runtime",
        class_name="DomainAdaptationBranchPlugin",
        method_name="compute_loss",
        insertion_point="trainer_loss",
        component_id=f"domain_adaptation.{branch_id}",
        paper_id=paper_id,
        paper_binding_id=f"{paper_id}/domain_adaptation.{branch_id}",
    )
