"""Executable adapter for generic YOLO26 multi-scale feature fusion."""

from yolo_agent.components.adapters.neck.runtime import GuardedYOLO26NeckAdapter


class MultiScaleFusionAdapter(GuardedYOLO26NeckAdapter):
    """Attach the generic bidirectional feature-pyramid plugin."""

    component_id = "neck.multi_scale_fusion"
    neck_kind = "multi_scale_fusion"
    adapter_version = "multi_scale_fusion_adapter.v1"
    source_commit = "yolo-agent:multi-scale-fusion-v1"


__all__ = ["MultiScaleFusionAdapter"]
