"""Executable adapter for the isolated Gold-YOLO gather-distribute component."""

from yolo_agent.components.adapters.neck.runtime import GuardedYOLO26NeckAdapter


class GoldGatherDistributeAdapter(GuardedYOLO26NeckAdapter):
    """Attach gather-distribute without copying the Gold-YOLO detector."""

    component_id = "neck.gold_gather_distribute"
    neck_kind = "gold_gather_distribute"
    adapter_version = "gold_gather_distribute_adapter.v1"
    source_commit = "yolo-agent:gold-gd-adaptation-v1"


__all__ = ["GoldGatherDistributeAdapter"]
