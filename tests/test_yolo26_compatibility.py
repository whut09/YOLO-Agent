from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluator
from yolo_agent.agents.strategy_policy import CandidatePolicy, PolicyConstraint, PolicyEvaluator
from yolo_agent.components.compatibility import BaseModelSpec, CompatibilityChecker
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.registry import ComponentRegistry
from yolo_agent.components.schema import ComponentCard
from yolo_agent.components.yolo26_compatibility import YOLO26CompatibilityChecker
from yolo_agent.core.task_spec import MetricPriority, TaskSpec


def _contract(component_id: str, category: str, **updates: object) -> ComponentContract:
    data = {
        "component_id": component_id,
        "display_name": component_id,
        "category": category,
        "implementation_path": "tests",
        "adapter_class": "TestAdapter",
        "maturity": "smoke_passed",
        "fixed_imgsz_compatible": True,
    }
    data.update(updates)
    return ComponentContract.model_validate(data)


def _task() -> TaskSpec:
    return TaskSpec(task_type="detect", scene="generic", class_names=["person"], primary_metric=MetricPriority(name="map50_95"))


def test_default_one_to_one_path_is_nms_free_and_compatible() -> None:
    result = YOLO26CompatibilityChecker().check(train_overrides={"imgsz": 640})
    assert result.compatible
    assert not result.incompatible


def test_one_to_one_head_blocks_nms_but_one_to_many_can_request_it() -> None:
    checker = YOLO26CompatibilityChecker()
    blocked = checker.check(train_overrides={"postprocess": "soft_nms"}, head_mode="one_to_one")
    allowed = checker.check(train_overrides={"postprocess": "soft_nms"}, head_mode="one_to_many")
    assert "one_to_one_head_uses_nms_recipe" in blocked.blocked_by
    assert allowed.compatible


def test_dfl_dependent_loss_is_blocked_on_dfl_free_regression() -> None:
    loss = _contract(
        "loss.bbox.dfl_variant",
        "bbox_loss",
        tensor_input_contract={"compatibility_constraints": {"requires_dfl": True}},
    )
    result = YOLO26CompatibilityChecker().check(components=[loss])
    assert any(item.startswith("dfl_dependent_loss") for item in result.blocked_by)


def test_anchor_assigner_requires_a_real_adapter() -> None:
    assigner = _contract(
        "assigner.anchor",
        "assigner",
        adapter_class=None,
        maturity="metadata_only",
        tensor_input_contract={"compatibility_constraints": {"anchor_based": True}},
    )
    result = YOLO26CompatibilityChecker().check(components=[assigner])
    assert "anchor_based_assigner_requires_adapter:assigner.anchor" in result.blocked_by
    assert result.research_adapter_required


def test_stal_musgd_and_progressive_loss_report_required_adapters() -> None:
    components = [
        _contract("assigner.stal", "assigner", maturity="metadata_only", adapter_class=None),
        _contract("optimizer.musgd", "optimizer", maturity="metadata_only", adapter_class=None),
        _contract("loss.progressive", "loss_schedule", maturity="metadata_only", adapter_class=None),
    ]
    result = YOLO26CompatibilityChecker().check(components=components)
    assert set(result.required_adapters) == {
        "YOLO26STALAdapter", "YOLO26MuSGDAdapter", "YOLO26ProgressiveLossAdapter"
    }
    assert set(result.metadata_only) == {item.component_id for item in components}


def test_multi_variable_candidate_cannot_claim_single_variable() -> None:
    components = [
        _contract("head.new", "head"),
        _contract("assigner.new", "assigner"),
        _contract("loss.bbox.new", "bbox_loss"),
    ]
    result = YOLO26CompatibilityChecker().check(components=components, single_variable=True)
    assert "multi_variable_candidate_marked_single_variable" in result.blocked_by
    assert "assigner_head_loss_replaced_in_single_variable_ablation" in result.blocked_by
    assert result.changed_variables == ["assigner", "bbox_loss", "head"]


def test_fixed_imgsz_and_automatic_increase_are_blocked() -> None:
    result = YOLO26CompatibilityChecker().check(
        train_overrides={"imgsz": 1280, "allow_imgsz_increase": True}
    )
    assert "fixed_imgsz_violation:1280" in result.blocked_by
    assert "automatic_imgsz_increase_forbidden" in result.blocked_by


def test_metadata_only_cannot_enter_execution_but_can_be_researched() -> None:
    component = _contract("paper.idea", "attention", maturity="metadata_only", adapter_class=None)
    blocked = YOLO26CompatibilityChecker().check(components=[component], execution_requested=True)
    research = YOLO26CompatibilityChecker().check(components=[component], execution_requested=False)
    assert "metadata_only_component:paper.idea" in blocked.blocked_by
    assert research.compatible
    assert research.metadata_only == ["paper.idea"]


def test_checkpoint_amp_ddp_and_export_are_checked() -> None:
    component = _contract(
        "head.custom", "head", changes_model_graph=True, checkpoint_compatibility="incompatible",
        supports_amp=False, supports_ddp=False, supports_onnx=False, supports_tensorrt=False,
    )
    result = YOLO26CompatibilityChecker().check(
        components=[component], checkpoint="yolo26n.pt", amp=True, ddp=True, export_format="onnx"
    )
    assert "checkpoint_incompatible:head.custom" in result.blocked_by
    assert "amp_unsupported:head.custom" in result.blocked_by
    assert "ddp_unsupported:head.custom" in result.blocked_by
    assert "onnx_export_unsupported:head.custom" in result.blocked_by


def test_generic_checker_exposes_yolo26_result() -> None:
    checker = CompatibilityChecker()
    result = checker.check(
        _task(), BaseModelSpec(name="yolo26n.pt", framework="ultralytics", model_family="yolo26"), [],
        train_overrides={"imgsz": 1280},
    )
    assert not result.ok
    assert result.yolo26 is not None
    assert "fixed_imgsz_violation:1280" in result.errors


def test_policy_evaluator_blocks_yolo26_metadata_component() -> None:
    card = ComponentCard(
        id="head.paper_only", name="Paper only", type="head",
        compatible_frameworks=["ultralytics"], compatible_model_families=["yolo26"],
    )
    evaluator = PolicyEvaluator(ComponentRegistry([card]))
    policy = CandidatePolicy(
        policy_id="candidate", base_model="yolo26n.pt", scale="n", framework="ultralytics",
        components=[card.id], train_overrides={"imgsz": 640},
    )
    result = evaluator.evaluate_one(policy, _task())
    assert not result.accepted
    assert "metadata_only_component:head.paper_only" in result.errors


def test_policy_evaluator_blocks_multi_variable_single_ablation() -> None:
    cards = [
        ComponentCard(id="head.a", name="Head", type="head", compatible_model_families=["yolo26"]),
        ComponentCard(id="assigner.a", name="Assigner", type="assigner", compatible_model_families=["yolo26"]),
        ComponentCard(id="loss.bbox.a", name="Loss", type="bbox_loss", compatible_model_families=["yolo26"]),
    ]
    policy = CandidatePolicy(
        policy_id="bad_ablation", base_model="yolo26n.pt", scale="n", framework="generic",
        components=[item.id for item in cards], constraints=[PolicyConstraint(name="single_variable", value=True)],
    )
    result = PolicyEvaluator(ComponentRegistry(cards)).evaluate_one(policy, _task())
    assert "assigner_head_loss_replaced_in_single_variable_ablation" in result.errors


def test_loop_policy_evaluator_cannot_materialize_metadata_component() -> None:
    card = ComponentCard(id="head.paper_only", name="Paper only", type="head", compatible_frameworks=["ultralytics"], compatible_model_families=["yolo26"])
    proposal = CandidatePolicy(policy_id="loop_candidate", base_model="yolo26n.pt", scale="n", framework="ultralytics", components=[card.id], train_overrides={"imgsz": 640})
    result = LoopPolicyEvaluator(ComponentRegistry([card])).evaluate_one(proposal, _task())
    assert result.decision == "rejected"
    assert result.experiment_node is None
    assert "metadata_only_component:head.paper_only" in result.errors
