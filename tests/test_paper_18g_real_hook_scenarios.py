"""Prompt-18G: real per-phase hook scenarios (CPU-only, training-free).

Pins that the four canonical runtime surfaces resolve to audited
identities and execute real (non-mock) adapter smoke tests:

* ``data``       — sampling adapter (train_dataloader_sampler);
* ``loss``       — calibration loss plugin (trainer_loss);
* ``model_graph``— neck adapter (before_detect_p3_p4_p5);
* ``inference``  — SAHI slicing protocol (inference_protocol).

No epoch, no trainer, no GPU is touched anywhere in this file.
"""

from __future__ import annotations

import pytest

from yolo_agent.research.paper_runtime_preflight import (
    _load_component_contracts,
    _run_adapter_component,
)

pytestmark = pytest.mark.run_slow


@pytest.fixture(scope="module")
def contracts() -> dict:
    return _load_component_contracts()


def test_real_data_hook_resolves_and_executes(contracts: dict) -> None:
    record, hook_id = _run_adapter_component(
        "sampling.small_object", paper_id="arxiv:probe-data"
    )
    assert record.status == "PASS"
    assert hook_id == "hook.data.small_object"
    identity = record.runtime_hook_identities[0]
    assert identity["phase"] == "data"
    assert identity["class_name"] == "SmallObjectSamplingAdapter"
    assert identity["method_name"] == "smoke_test"
    assert identity["insertion_point"] == "train_dataloader_sampler"
    assert identity["paper_binding_id"] == "arxiv:probe-data/sampling.small_object"
    assert identity["source_sha256"]
    assert record.materialized and record.behavior_changed


def test_real_loss_hook_resolves_and_executes(contracts: dict) -> None:
    record, hook_id = _run_adapter_component(
        "loss.calibration.bpc", paper_id="arxiv:2303.14404"
    )
    assert record.status == "PASS"
    assert hook_id == "hook.loss.calibration.bpc"
    identity = record.runtime_hook_identities[0]
    assert identity["phase"] == "loss"
    assert identity["class_name"] == "QualityAlignmentAuxiliaryLossAdapter"
    assert identity["insertion_point"] == "trainer_loss"
    assert record.synthetic_forward and record.synthetic_backward


def test_real_model_graph_hook_resolves_and_executes(contracts: dict) -> None:
    record, hook_id = _run_adapter_component(
        "neck.rtmdet_large_kernel", paper_id="arxiv:2212.07784"
    )
    assert record.status == "PASS"
    assert hook_id == "hook.model_graph.rtmdet_large_kernel"
    identity = record.runtime_hook_identities[0]
    assert identity["phase"] == "model_graph"
    assert identity["insertion_point"] == "before_detect_p3_p4_p5"
    assert record.materialized and record.behavior_changed


def test_real_inference_hook_resolves_and_executes(contracts: dict) -> None:
    record, hook_id = _run_adapter_component(
        "inference.sahi_slicing", paper_id="arxiv:probe-inference"
    )
    assert record.status == "PASS"
    assert hook_id == "hook.inference.sahi_slicing"
    identity = record.runtime_hook_identities[0]
    assert identity["phase"] == "inference"
    assert identity["class_name"] == "SlicingInferenceAdapter"
    assert identity["insertion_point"] == "inference_protocol"
    assert record.materialized


def test_all_four_identities_are_independent_bindings(contracts: dict) -> None:
    hooks = [
        _run_adapter_component(component_id, paper_id=f"arxiv:binding-{index}")
        for index, component_id in enumerate(
            (
                "sampling.small_object",
                "loss.calibration.bpc",
                "neck.rtmdet_large_kernel",
                "inference.sahi_slicing",
            )
        )
    ]
    binding_keys = {record.runtime_hook_identities[0]["paper_binding_id"] for record, _ in hooks}
    hook_ids = {hook_id for _, hook_id in hooks}
    assert len(binding_keys) == 4
    assert hook_ids == {
        "hook.data.small_object",
        "hook.loss.calibration.bpc",
        "hook.model_graph.rtmdet_large_kernel",
        "hook.inference.sahi_slicing",
    }
