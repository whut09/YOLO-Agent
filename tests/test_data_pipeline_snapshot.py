from __future__ import annotations

from pathlib import Path

from yolo_agent.components.contracts import load_contracts
from yolo_agent.recipes.registry import RecipeRegistry
from yolo_agent.research.maturity_snapshot import EffectiveComponentMaturityManifest
from yolo_agent.research.production_pipeline import ResearchProductionPipeline


DATA_COMPONENTS = {
    "sampling.small_object_weighted",
    "sampling.class_balanced",
    "sampling.repeat_factor",
    "sampling.hard_negative_replay",
    "sampling.false_negative_class_boost",
    "augmentation.copy_paste_rare_classes",
    "augmentation.scale_aware_crop",
    "augmentation.object_centric_crop",
    "augmentation.multi_image_sampling_schedule",
}


def test_snapshot_freezes_data_implementations_without_granting_execution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "research"
    metadata_only = ResearchProductionPipeline(root).run(
        include_local_implementations=False
    )
    implemented = ResearchProductionPipeline(root).run(
        include_local_implementations=True
    )

    assert implemented.snapshot_hash != metadata_only.snapshot_hash
    snapshot = Path(implemented.snapshot_path or "")
    all_contracts = load_contracts(snapshot / "component_contracts.yaml")
    contracts = {
        item.component_id: item
        for item in all_contracts
        if item.component_id in DATA_COMPONENTS
    }
    recipes = RecipeRegistry.from_path(
        snapshot / "recipes.yaml",
        component_contracts=all_contracts,
    ).list()
    data_recipes = [
        item for item in recipes if set(item.component_ids) & DATA_COMPONENTS
    ]
    maturity = EffectiveComponentMaturityManifest.from_yaml(
        snapshot / "effective_component_maturity.yaml"
    ).by_component()

    assert set(contracts) == DATA_COMPONENTS
    assert all(item.maturity == "adapter_implemented" for item in contracts.values())
    assert all(not item.can_execute for item in contracts.values())
    assert len(data_recipes) == 9
    assert all(not item.is_executable for item in data_recipes)
    assert all(
        maturity[component_id].runtime_execution_ready is False
        for component_id in DATA_COMPONENTS
    )
