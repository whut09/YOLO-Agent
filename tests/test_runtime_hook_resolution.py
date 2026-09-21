"""Prompt-18G: resolver tests — contracts/mechanisms/branches to identities."""

from __future__ import annotations

import pytest

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.research.paper_runtime_preflight import _load_component_contracts
from yolo_agent.research.runtime_hook_identity import RuntimeHookIdentityError
from yolo_agent.research.runtime_hook_resolution import (
    hook_id_for_contract,
    identity_for_contract,
    identity_for_domain_adaptation_branch,
    identity_for_distillation_mechanism,
    phase_for_contract,
)


@pytest.fixture(scope="module")
def contracts() -> dict:
    return _load_component_contracts()


def test_previously_unknown_papers_resolve(contracts: dict) -> None:
    """The 11 component-adapter papers now resolve to real identities."""
    for component_id in (
        "assigner.optimal_transport",
        "assigner.task_aligned",
        "assigner.dynamic_smooth_label",
        "loss.quality.pseudo_iou",
        "loss.quality.correlation",
        "loss.calibration.bpc",
        "loss.2109_05986",
        "detection_head.task_aligned",
        "neck.rtmdet_large_kernel",
        "feature_pyramid.multi_scale",
    ):
        contract = contracts[component_id]
        identity = identity_for_contract(contract, paper_id="arxiv:probe")
        assert identity.phase in {"assignment", "loss", "model_graph"}
        assert identity.class_name == contract.adapter_class
        assert identity.source_sha256, component_id
        assert "unknown" not in identity.hook_id


def test_assigner_resolves_to_assignment_phase(contracts: dict) -> None:
    identity = identity_for_contract(
        contracts["assigner.optimal_transport"], paper_id="arxiv:2103.14259"
    )
    assert identity.phase == "assignment"
    assert identity.hook_id == "hook.assignment.optimal_transport"
    assert identity.method_name == "smoke_test"


def test_head_and_neck_resolve_to_model_graph_phase(contracts: dict) -> None:
    assert identity_for_contract(
        contracts["detection_head.task_aligned"], paper_id="arxiv:2108.07755"
    ).phase == "model_graph"
    assert identity_for_contract(
        contracts["neck.rtmdet_large_kernel"], paper_id="arxiv:2212.07784"
    ).phase == "model_graph"


def test_shared_loss_adapter_two_papers_two_bindings(contracts: dict) -> None:
    first = identity_for_contract(
        contracts["loss.quality.pseudo_iou"], paper_id="arxiv:2104.14082"
    )
    second = identity_for_contract(
        contracts["loss.quality.correlation"], paper_id="arxiv:2301.01019"
    )
    assert first.binding_key != second.binding_key
    assert first.paper_binding_id == "arxiv:2104.14082/loss.quality.pseudo_iou"
    assert second.paper_binding_id == "arxiv:2301.01019/loss.quality.correlation"


def test_distillation_mechanism_resolves_to_concrete_class() -> None:
    from yolo_agent.components.distillation.mechanism_losses import (
        build_distillation_mechanism_loss,
    )

    loss_object = build_distillation_mechanism_loss("logits")
    identity = identity_for_distillation_mechanism(
        loss_object, paper_id="arxiv:probe", mechanism="logits"
    )
    assert identity.hook_id == "loss.distillation"
    assert identity.class_name == "LogitsDistillationLoss"
    assert identity.method_name == "compute"
    assert identity.phase == "loss"


def test_domain_adaptation_branch_resolves() -> None:
    identity = identity_for_domain_adaptation_branch(
        paper_id="arxiv:2210.11539", branch_id="adversarial_alignment"
    )
    assert identity.hook_id == "loss.domain_adaptation"
    assert identity.class_name == "DomainAdaptationBranchPlugin"
    assert identity.method_name == "compute_loss"
    assert identity.source_sha256


def test_domain_aware_phases_not_forced_into_model_forward() -> None:
    """Sampling/augmentation/postprocess papers map to their true phases."""
    assert phase_for_contract(
        ComponentContract(
            component_id="sampling.x",
            display_name="X",
            category="sampling",
            insertion_point="train_dataloader_sampler",
        )
    ) == "data"
    assert phase_for_contract(
        ComponentContract(
            component_id="inference.sahi",
            display_name="S",
            category="slicing",
            insertion_point="inference_protocol",
        )
    ) == "inference"
    assert phase_for_contract(
        ComponentContract(
            component_id="postprocess.nms",
            display_name="N",
            category="postprocess",
            insertion_point="nms",
        )
    ) == "postprocess"


def test_unresolvable_contract_fails_closed(contracts: dict) -> None:
    with pytest.raises(RuntimeHookIdentityError, match="no resolvable implementation"):
        identity_for_contract(
            ComponentContract(
                component_id="ghost.component",
                display_name="Ghost",
                category="loss",
            ),
            paper_id="arxiv:ghost",
        )


def test_unknown_phase_contract_fails_closed() -> None:
    with pytest.raises(RuntimeHookIdentityError, match="cannot resolve a canonical runtime phase"):
        phase_for_contract(
            ComponentContract(
                component_id="mystery.x",
                display_name="M",
                category="mystery",
                insertion_point="somewhere_else",
            )
        )


def test_hook_id_builder() -> None:
    contract = ComponentContract(
        component_id="assigner.dynamic_smooth_label",
        display_name="D",
        category="assigner",
    )
    assert hook_id_for_contract(contract, "assignment") == "hook.assignment.dynamic_smooth_label"
