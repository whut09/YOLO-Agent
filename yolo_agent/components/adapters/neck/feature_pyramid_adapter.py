"""Adapter for the independent multi-scale feature-pyramid candidate."""

from typing import Any

from yolo_agent.components.adapters.neck.runtime import GuardedYOLO26NeckAdapter
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload


class FeaturePyramidMultiScaleAdapter(GuardedYOLO26NeckAdapter):
    component_id = "feature_pyramid.multi_scale"
    neck_kind = "feature_pyramid_multi_scale"
    adapter_version = "feature_pyramid_multi_scale_adapter.v1"
    source_commit = "yolo-agent:feature-pyramid-multi-scale-v1"

    def build_runtime_payload(
        self,
        context: Any,
        *,
        protocol_hash: str,
        base_command: list[str],
        generated_config: dict[str, Any],
    ) -> AdapterRuntimePayload:
        payload = super().build_runtime_payload(
            context,
            protocol_hash=protocol_hash,
            base_command=base_command,
            generated_config=generated_config,
        )
        config = payload.changed_variables.pop("model.neck_plugin")
        payload.changed_variables["model.feature_pyramid"] = config
        return payload


__all__ = ["FeaturePyramidMultiScaleAdapter"]
