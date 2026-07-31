from __future__ import annotations

import json
from pathlib import Path

import yaml

from yolo_agent.agents.asha_scheduler import ASHAScheduler
from yolo_agent.agents.auto_optimization_loop import _register_guarded_pilot_trials
from yolo_agent.agents.optimize_runner import OptimizeRunner
from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.components.adapters import ComponentAdapterRegistry, DummyAdapter
from yolo_agent.components.execution_bridge import ComponentExecutionBridge
from yolo_agent.core.round_execution_plan import RoundExecutionPlan
from yolo_agent.recipes.recipe_materializer import RecipeMaterializer
from tests.paper_materialization_fixtures import contract, node, prior


def _dataset(root: Path) -> Path:
    images = root / "images" / "train"
    labels = root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "sample.jpg").write_bytes(b"image")
    (labels / "sample.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    data = root / "data.yaml"
    data.write_text(
        "path: .\ntrain: images/train\nnames:\n  0: object\n",
        encoding="utf-8",
    )
    return data


def test_auto_loop_registers_only_certified_component_runtime(tmp_path: Path) -> None:
    base = OptimizeRunner().run(
        kind="coco",
        model="yolo26n.pt",
        data_yaml=_dataset(tmp_path / "dataset"),
        run_id="paper-auto",
        run_root=tmp_path / "runs",
        profile="pilot",
        execute=False,
    )
    child = LoopOrchestrator.from_run_dir(base.run_dir)
    component_contract = contract()
    snapshot_dir = tmp_path / "research" / "snapshot"
    snapshot_dir.mkdir(parents=True)
    snapshot_contract = component_contract.model_dump(
        mode="json",
        exclude={"component_id"},
    )
    (snapshot_dir / "component_contracts.yaml").write_text(
        yaml.safe_dump(
            {"components": {component_contract.component_id: snapshot_contract}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    child.context.metadata["research_snapshot_path"] = snapshot_dir.as_posix()
    recipe = RecipeMaterializer().materialize(
        prior(),
        component_contracts={"dummy.component": component_contract},
    ).recipe
    assert recipe is not None
    source = node("paper-candidate")
    source.candidate_config = source.candidate_config.model_copy(update={
        "action_id": recipe.recipe_id,
        "target_error_facts": recipe.target_error_facts,
    })
    registry = ComponentAdapterRegistry()
    registry.register("dummy.component", DummyAdapter)
    runtime = ComponentExecutionBridge(adapter_registry=registry).prepare(
        recipe=recipe,
        node=source,
        contracts={"dummy.component": component_contract},
        workspace=child.context.artifact_path("paper_runtime_test"),
        protocol_hash="paper-protocol-640",
    )
    assert runtime.status == "executable"
    plan = RoundExecutionPlan(
        run_id=child.context.run_id,
        round_id="paper-registration",
        deferred_nodes=[runtime.node, node("matched-control", control=True)],
    )
    plan.to_yaml(child.context.artifact_path("round_execution_plan.yaml"))
    scheduler = ASHAScheduler.create(child.context.run_id)

    registered = _register_guarded_pilot_trials(
        scheduler,
        child,
        [runtime.node],
    )

    assert registered == 1
    assert len(scheduler.study.trials) == 1
    trial = scheduler.study.trials[0]
    assert trial.source_node.command_spec is not None
    assert "--payload" in trial.source_node.command_spec.argv
    ledger = [
        json.loads(line)
        for line in child.context.artifact_path("decision_ledger.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    entry = ledger[-1]
    assert entry["decision_type"] == "paper_recipe_asha_registration"
    assert entry["proposal"]["adapter_ids"] == ["dummy.component"]
    assert entry["proposal"]["adapter_patch_hash"]
    assert entry["proposal"]["adapter_runtime_payload_hash"]
    assert entry["proposal"]["scalar_hpo_enabled"] is False
