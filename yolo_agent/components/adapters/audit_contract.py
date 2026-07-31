"""Canonical runtime expectations for the audited paper adapters."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from yolo_agent.components.adapters.runtime import AdapterRuntimePayload


PluginKind = Literal[
    "dataloader_plugin",
    "model_graph_plugin",
    "loss_plugin",
    "assigner_plugin",
    "inference_plugin",
]


class RuntimeAdapterExpectation(BaseModel):
    plugin_kind: PluginKind
    required_hook: str
    changed_variable: str


EXPECTED_RUNTIME_ADAPTERS: dict[str, RuntimeAdapterExpectation] = {
    "sampling.small_object": RuntimeAdapterExpectation(plugin_kind="dataloader_plugin", required_hook="build_train_dataloader", changed_variable="data.sampling_policy"),
    "head.p2_small_object": RuntimeAdapterExpectation(plugin_kind="model_graph_plugin", required_hook="build_model", changed_variable="model.p2_head"),
    "loss.quality.correlation": RuntimeAdapterExpectation(plugin_kind="loss_plugin", required_hook="compute_loss", changed_variable="loss.correlation.weight"),
    "loss.calibration.bpc": RuntimeAdapterExpectation(plugin_kind="loss_plugin", required_hook="compute_loss", changed_variable="loss.bpc_calibration.weight"),
    "loss.quality.pseudo_iou": RuntimeAdapterExpectation(plugin_kind="loss_plugin", required_hook="compute_loss", changed_variable="loss.pseudo_iou.weight"),
    "distillation.yolo26_teacher_student": RuntimeAdapterExpectation(plugin_kind="loss_plugin", required_hook="compute_loss", changed_variable="loss.distillation"),
    "assigner.task_aligned": RuntimeAdapterExpectation(plugin_kind="assigner_plugin", required_hook="compute_loss", changed_variable="assignment.one_to_many.tood_tal.mode"),
    "assigner.optimal_transport": RuntimeAdapterExpectation(plugin_kind="assigner_plugin", required_hook="compute_loss", changed_variable="assignment.one_to_many.ota.mode"),
    "assigner.dynamic_smooth_label": RuntimeAdapterExpectation(plugin_kind="assigner_plugin", required_hook="compute_loss", changed_variable="assignment.one_to_many.dsla.mode"),
    "neck.multi_scale_fusion": RuntimeAdapterExpectation(plugin_kind="model_graph_plugin", required_hook="build_model", changed_variable="model.neck_plugin"),
    "neck.gold_gather_distribute": RuntimeAdapterExpectation(plugin_kind="model_graph_plugin", required_hook="build_model", changed_variable="model.neck_plugin"),
    "neck.rtmdet_large_kernel": RuntimeAdapterExpectation(plugin_kind="model_graph_plugin", required_hook="build_model", changed_variable="model.neck_plugin"),
    "inference.sahi_slicing": RuntimeAdapterExpectation(plugin_kind="inference_plugin", required_hook="prepare_command", changed_variable="inference.slicing_policy"),
}


def validate_audited_runtime_payload(
    payload: AdapterRuntimePayload,
    component_id: str,
) -> dict[str, bool | str | int | float]:
    """Fail unless one audited adapter uses its canonical runtime boundary."""
    expectation = EXPECTED_RUNTIME_ADAPTERS.get(component_id)
    if expectation is None:
        return {"audited_runtime_component": False}
    if payload.component_ids != [component_id]:
        raise ValueError("audited runtime payload must contain exactly its component")
    references = getattr(payload, expectation.plugin_kind)
    if len(references) != 1:
        raise ValueError(
            f"audited runtime payload requires one {expectation.plugin_kind}"
        )
    other_references = [
        item
        for kind in (
            "dataloader_plugin",
            "trainer_plugin",
            "model_graph_plugin",
            "loss_plugin",
            "assigner_plugin",
            "inference_plugin",
        )
        if kind != expectation.plugin_kind
        for item in getattr(payload, kind)
    ]
    if other_references:
        raise ValueError("audited atomic adapter payload contains unexpected plugin kinds")
    if expectation.required_hook not in references[0].required_hooks:
        raise ValueError(
            f"audited runtime payload is missing required hook {expectation.required_hook}"
        )
    if set(payload.changed_variables) != {expectation.changed_variable}:
        raise ValueError(
            "audited runtime payload changed variables do not match canonical contract"
        )
    return {
        "audited_runtime_component": True,
        "audited_plugin_kind": expectation.plugin_kind,
        "audited_required_hook": expectation.required_hook,
        "audited_changed_variable": expectation.changed_variable,
    }


__all__ = [
    "EXPECTED_RUNTIME_ADAPTERS",
    "RuntimeAdapterExpectation",
    "validate_audited_runtime_payload",
]
