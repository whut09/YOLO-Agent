from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from yolo_agent.agents.active_learning import (
    ActiveLearningMiner,
    MiningConfig,
    PredictionSummary,
)
from yolo_agent.components.adapters.data_pipeline import (
    DataPipelineDataset,
    DataSampleRecord,
    DataTransformConfig,
    DistributedExposureSampler,
    ExposureConfig,
    blend_multi_image_samples,
    copy_paste_sample,
    compute_exposure_details,
    crop_sample,
)
from yolo_agent.components.adapters.data_pipeline.annotation import (
    AnnotationFilterConfig,
    filter_detection_sample,
)
from yolo_agent.components.adapters.data_pipeline.preprocessing import (
    PreprocessingConfig,
    preprocess_sample,
)
from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest
from yolo_agent.research.paper_83_engineering_plan import (
    Paper83EngineeringPlan,
)
from yolo_agent.research.paper_data_side import (
    PaperDataSideAuditBuilder,
    data_side_routes,
    render_paper_83_data_side_status,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "research" / "paper_83_manifest.yaml"
PLAN_PATH = ROOT / "configs" / "research" / "paper_83_engineering_plan.yaml"


def _sample(
    boxes: list[list[float]],
    classes: list[int],
) -> dict[str, torch.Tensor]:
    return {
        "img": torch.zeros((3, 16, 16), dtype=torch.uint8),
        "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "cls": torch.tensor(classes, dtype=torch.float32).reshape(-1, 1),
        "batch_idx": torch.zeros(len(boxes), dtype=torch.int64),
    }


def test_annotation_filter_clips_geometry_and_keeps_class_alignment() -> None:
    sample = _sample(
        [[0.05, 0.5, 0.2, 0.2], [0.5, 0.5, 0.0, 0.2], [0.8, 0.8, 0.1, 0.1]],
        [4, 5, 6],
    )

    result = filter_detection_sample(sample, AnnotationFilterConfig())

    assert result.kept_indices == [0, 2]
    assert result.removed_indices == [1]
    assert result.sample["cls"].reshape(-1).tolist() == [4.0, 6.0]
    assert result.sample["bboxes"][0].tolist() == pytest.approx(
        [0.075, 0.5, 0.15, 0.2]
    )
    assert len(result.sample["bboxes"]) == len(result.sample["cls"])


def test_annotation_filter_removes_outside_and_empty_boxes() -> None:
    sample = _sample(
        [[1.4, 0.5, 0.2, 0.2], [0.5, 0.5, 0.0, 0.2]],
        [1, 2],
    )

    result = filter_detection_sample(
        sample,
        AnnotationFilterConfig(clip_boxes=False),
    )

    assert result.kept_indices == []
    assert result.removed_indices == [0, 1]
    assert result.sample["bboxes"].shape == (0, 4)
    assert result.sample["cls"].shape == (0, 1)


@pytest.mark.parametrize("field", ["masks", "segments", "obb", "keypoints"])
def test_annotation_filter_rejects_unsupported_annotation_formats(field: str) -> None:
    sample = _sample([[0.5, 0.5, 0.2, 0.2]], [1])
    sample[field] = torch.ones(1)  # type: ignore[literal-required]

    with pytest.raises(ValueError, match="support detect boxes only"):
        filter_detection_sample(sample, AnnotationFilterConfig())


def test_preprocessing_changes_pixels_but_preserves_labels() -> None:
    sample = _sample([[0.5, 0.5, 0.2, 0.2]], [3])
    sample["img"] = torch.full((3, 4, 4), 255, dtype=torch.uint8)

    output = preprocess_sample(
        sample,
        PreprocessingConfig(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    )

    assert output["img"].dtype == torch.float32
    assert not torch.equal(output["img"], sample["img"].float())
    assert torch.equal(output["bboxes"], sample["bboxes"])
    assert torch.equal(output["cls"], sample["cls"])


def test_preprocessing_disabled_is_pixel_only_and_imgsz_is_fixed() -> None:
    sample = _sample([[0.5, 0.5, 0.2, 0.2]], [3])
    sample["img"] = torch.full((3, 4, 4), 255, dtype=torch.uint8)

    output = preprocess_sample(
        sample,
        PreprocessingConfig(normalize=False),
    )

    assert torch.equal(output["img"], sample["img"].float())
    with pytest.raises(ValueError, match="fixed imgsz=640"):
        PreprocessingConfig(imgsz=320)


def test_sampling_policy_changes_exposure_distribution() -> None:
    records = [
        DataSampleRecord(image_path="common-a.jpg", class_ids=[0]),
        DataSampleRecord(image_path="common-b.jpg", class_ids=[0]),
        DataSampleRecord(image_path="rare.jpg", class_ids=[1]),
    ]

    _, exposure, _ = compute_exposure_details(
        records,
        ExposureConfig(mechanism="class_balanced_sampling"),
    )

    assert len(set(exposure)) > 1
    assert exposure[2] > exposure[0]


def test_sampler_selection_distribution_follows_exposure_weights() -> None:
    sampler = DistributedExposureSampler(
        [1.0, 1.0, 4.0],
        sample_count=4000,
        seed=17,
        rank=0,
        world_size=1,
        dataset_manifest="train-hash",
        adapter_hash="adapter-hash",
        mechanism_id="class_balanced_sampling",
    )

    selected = sampler.global_indices()

    assert selected.count(2) > selected.count(0)
    assert selected.count(2) > selected.count(1)


class _TransformDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.samples = [
            _sample([[0.2, 0.2, 0.05, 0.05]], [1]),
            _sample([[0.7, 0.7, 0.05, 0.05]], [3]),
        ]
        self.samples[1]["img"] = torch.full((3, 16, 16), 100, dtype=torch.uint8)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self.samples[index].items()}


def test_fixed_seed_augmentation_changes_geometry_and_is_reproducible() -> None:
    dataset = _TransformDataset()
    config = DataTransformConfig(
        mechanism="scale_aware_crop",
        crop_scale=0.5,
        small_area_threshold=0.02,
        seed=19,
    )

    first = DataPipelineDataset(dataset, config)[0]
    second = DataPipelineDataset(dataset, config)[0]

    assert not torch.equal(first["bboxes"], dataset[0]["bboxes"])
    assert torch.equal(first["img"], second["img"])
    assert torch.equal(first["bboxes"], second["bboxes"])
    assert len(first["bboxes"]) == len(first["cls"])


def test_copy_paste_and_multi_image_routes_preserve_annotation_alignment() -> None:
    target = _sample([[0.2, 0.2, 0.1, 0.1]], [1])
    donor = _sample([[0.7, 0.7, 0.2, 0.2]], [3])
    donor["img"] = torch.full((3, 16, 16), 100, dtype=torch.uint8)

    pasted = copy_paste_sample(target, donor, rare_class_ids={3})
    blended = blend_multi_image_samples([target, donor])

    assert pasted["bboxes"].shape == (2, 4)
    assert pasted["cls"].reshape(-1).tolist() == [1.0, 3.0]
    assert len(pasted["bboxes"]) == len(pasted["cls"])
    assert not torch.equal(blended["img"], target["img"])
    assert len(blended["bboxes"]) == len(blended["cls"])


def test_crop_transforms_partial_intersection_and_removes_empty_boxes() -> None:
    sample = _sample(
        [[0.5, 0.5, 0.6, 0.6], [0.95, 0.95, 0.05, 0.05]],
        [1, 2],
    )

    output = crop_sample(
        sample,
        center_x=0.25,
        center_y=0.25,
        scale=0.5,
    )

    assert output["bboxes"].shape == (1, 4)
    assert output["cls"].reshape(-1).tolist() == [1.0]
    assert len(output["bboxes"]) == len(output["cls"])
    assert torch.all(output["bboxes"] >= 0) and torch.all(output["bboxes"] <= 1)


@pytest.mark.parametrize("mechanism", ["scale_aware_crop", "object_centric_crop"])
def test_geometric_routes_keep_empty_background_samples_valid(mechanism: str) -> None:
    dataset = [_sample([], [])]
    config = DataTransformConfig(
        mechanism=mechanism,  # type: ignore[arg-type]
        probability=1.0,
        crop_scale=0.5,
        seed=7,
    )

    output = DataPipelineDataset(dataset, config)[0]

    assert output["bboxes"].shape == (0, 4)
    assert output["cls"].shape == (0, 1)
    assert torch.equal(output["img"], dataset[0]["img"])


def test_active_learning_acquisition_ranking_differs_from_input_order() -> None:
    miner = ActiveLearningMiner(MiningConfig(max_samples=2))
    predictions = [
        PredictionSummary(image_path="easy.jpg", max_confidence=0.98),
        PredictionSummary(image_path="uncertain.jpg", max_confidence=0.1),
    ]

    plan = miner.mine(predictions, dataset_version="v1")

    assert [item.image_path.name for item in plan.mined_samples] == ["uncertain.jpg"]
    assert plan.mined_samples[0].score > 0


def test_all_declared_data_routes_use_callable_runtime_hooks(tmp_path: Path) -> None:
    builder = PaperDataSideAuditBuilder(workspace=tmp_path)

    for mechanism_id, route in data_side_routes().items():
        result = builder._probe_route(route)
        if mechanism_id == "hard_negative_replay":
            assert any("hard-negative replay runtime payload" in error for error in result.errors)
            continue
        assert result.passed, (mechanism_id, result.errors, result.checks)
        assert result.checks["adapter_callable"] is True
        assert result.checks["typed_runtime_payload"] is True
        assert result.checks["runtime_hook_registered"] is True
        assert result.checks["runtime_hook_invoked"] is True
        assert result.checks["non_mock_smoke"] is True


def test_frozen_data_side_audit_keeps_all_83_out_of_scope() -> None:
    audit = PaperDataSideAuditBuilder(
        manifest_path=MANIFEST_PATH,
        plan_path=PLAN_PATH,
        workspace=ROOT,
    ).build()

    assert audit.paper_count == 83
    assert audit.data_side_paper_count == 0
    assert audit.summary == {
        "ready": 0,
        "blocked_missing_evidence": 0,
        "out_of_scope": 83,
    }
    assert len({item.paper_id for item in audit.records}) == 83
    assert all(item.status == "out_of_scope" for item in audit.records)
    assert "not attributed to any of the frozen 83 papers" in render_paper_83_data_side_status(audit)


def test_explicit_data_side_plan_is_audited_per_paper() -> None:
    manifest = Paper83Manifest.from_yaml(MANIFEST_PATH)
    source_plan = Paper83EngineeringPlan.from_yaml(PLAN_PATH)
    first = source_plan.papers[0].model_copy(
        update={
            "implementation_domain": "sampling",
            "secondary_domains": [],
            "paper_specific_mechanism_ids": ["class_balanced_sampling"],
            "paper_specific_config": {"strength": 1.0},
            "evidence_status": "sufficient_for_planning",
            "source_locations": ["paper:explicit-data-evidence"],
            "required_evidence": ["class_distribution_change"],
            "blockers": [],
            "unknown_facts": [],
        }
    )
    plan = source_plan.model_copy(
        update={
            "papers": [first, *source_plan.papers[1:]],
            "plan_identity_hash": "",
        }
    )

    audit = PaperDataSideAuditBuilder(workspace=ROOT).build(
        manifest=manifest,
        plan=plan,
    )
    record = audit.records[0]

    assert record.in_scope is True
    assert record.status == "ready"
    assert record.paper_specific_mechanism_ids == ["class_balanced_sampling"]
    assert record.resolved_route_ids == ["class_balanced_sampling"]
    assert record.behavior.checks["class_balanced_sampling.runtime_hook_invoked"] is True


def test_unknown_data_mechanism_is_blocked_without_fuzzy_mapping() -> None:
    manifest = Paper83Manifest.from_yaml(MANIFEST_PATH)
    source_plan = Paper83EngineeringPlan.from_yaml(PLAN_PATH)
    first = source_plan.papers[0].model_copy(
        update={
            "implementation_domain": "sampling",
            "secondary_domains": [],
            "paper_specific_mechanism_ids": ["unseen_sampling_method"],
            "paper_specific_config": {"paper_id": manifest.papers[0].paper_id},
            "evidence_status": "sufficient_for_planning",
            "source_locations": ["paper:explicit-data-evidence"],
            "required_evidence": ["paper_data_evidence"],
            "blockers": [],
            "unknown_facts": [],
        }
    )
    plan = source_plan.model_copy(
        update={
            "papers": [first, *source_plan.papers[1:]],
            "plan_identity_hash": "",
        }
    )

    audit = PaperDataSideAuditBuilder(workspace=ROOT).build(
        manifest=manifest,
        plan=plan,
    )
    record = audit.records[0]

    assert record.resolved_route_ids == []
    assert record.status == "blocked_missing_evidence"
    assert "blocked_missing_evidence:data_side_route_unresolved:unseen_sampling_method" in record.blockers
