from pathlib import Path

from yolo_agent.components.contracts import load_contracts


REUSABLE_ASSIGNERS = {
    "assigner.task_aligned_weighting",
    "assigner.dynamic_topk",
    "assigner.quality_aware",
    "assigner.soft_label",
    "assigner.dual_path",
    "assigner.conflict_aware",
}


def test_reusable_assignment_contracts_remain_shadow_gated() -> None:
    contracts = {
        item.component_id: item
        for item in load_contracts(
            Path("configs/components/assigner/yolo26_assignment.yaml")
        )
    }

    for component_id in REUSABLE_ASSIGNERS:
        contract = contracts[component_id]
        assert contract.adapter_class == "YOLO26AssignmentAdapter"
        assert contract.maturity == "adapter_implemented"
        assert contract.fixed_imgsz_compatible is True
        assert contract.training_only is True
        assert contract.inference_only is False
        assert contract.changes_model_graph is False
        assert contract.tensor_input_contract["anchor_representation"] == "point"
        assert contract.tensor_input_contract["default_mode"] == "shadow"
        assert contract.can_execute is False


def test_dual_path_contract_is_the_only_reusable_both_path_contract() -> None:
    contracts = {
        item.component_id: item
        for item in load_contracts(
            Path("configs/components/assigner/yolo26_assignment.yaml")
        )
    }
    both_path = {
        component_id
        for component_id in REUSABLE_ASSIGNERS
        if contracts[component_id].tensor_input_contract["assignment_path"] == "both"
    }

    assert both_path == {"assigner.dual_path"}
