from __future__ import annotations

import pytest
from pydantic import ValidationError

from yolo_agent.components.adapters.data_pipeline import (
    DataPipelineIdentity,
    DataPipelineManifest,
)


def _identity() -> DataPipelineIdentity:
    return DataPipelineIdentity(
        mechanism_id="class_balanced_sampling",
        component_id="sampling.class_balanced",
        adapter_family="data.sampling.class_balanced",
        mechanism_kind="weighted_sampler",
        changed_variable="data.class_balanced_sampling",
    )


def test_data_pipeline_identity_requires_exact_changed_variable() -> None:
    with pytest.raises(ValidationError, match="identify one mechanism exactly"):
        DataPipelineIdentity(
            mechanism_id="class_balanced_sampling",
            component_id="sampling.class_balanced",
            adapter_family="data.sampling",
            mechanism_kind="weighted_sampler",
            changed_variable="data.sampling_policy",
        )


def test_manifest_is_hash_stable_and_keeps_eval_splits_unchanged() -> None:
    manifest = DataPipelineManifest(
        identity=_identity(),
        dataset_manifest="dataset-v1",
        adapter_hash="adapter-v1",
        plugin_version="plugin-v1",
        image_paths=["a.jpg", "b.jpg"],
        raw_exposure=[1.0, 2.0],
        final_exposure=[1.0, 2.0],
        sample_count=2,
    )

    assert manifest.with_hash().manifest_hash == manifest.with_hash().manifest_hash
    assert manifest.val_unchanged and manifest.test_unchanged


def test_exact_reproduction_requires_explicit_method_profile() -> None:
    with pytest.raises(ValidationError, match="MethodProfile identity"):
        DataPipelineManifest(
            identity=_identity(),
            dataset_manifest="dataset-v1",
            adapter_hash="adapter-v1",
            plugin_version="plugin-v1",
            exact_reproduction=True,
        )
