"""Executable adapter for isolated RTMDet-style large-kernel neck blocks."""

from yolo_agent.components.adapters.neck.runtime import GuardedYOLO26NeckAdapter


class RTMDetLargeKernelNeckAdapter(GuardedYOLO26NeckAdapter):
    """Attach 5x5 depthwise neck blocks without copying RTMDet."""

    component_id = "neck.rtmdet_large_kernel"
    neck_kind = "rtmdet_large_kernel"
    adapter_version = "rtmdet_large_kernel_adapter.v1"
    source_commit = "yolo-agent:rtmdet-large-kernel-adaptation-v1"


__all__ = ["RTMDetLargeKernelNeckAdapter"]
