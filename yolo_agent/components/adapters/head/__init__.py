"""Detection-head adapters."""

from yolo_agent.components.adapters.head.p2_head import (
    P2Head,
    P2HeadAdapter,
    P2HeadConfig,
    P2HeadCheckpointReport,
)

__all__ = ["P2Head", "P2HeadAdapter", "P2HeadConfig", "P2HeadCheckpointReport"]
