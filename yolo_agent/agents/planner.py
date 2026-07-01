"""Planning scaffold for future YOLO experiment recommendations."""

from __future__ import annotations

from yolo_agent.core.schemas import AgentConfig


class OptimizationPlanner:
    """Create experiment plans from task and deployment constraints."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def create_plan(self) -> dict[str, str]:
        """Return a placeholder plan until real planning logic is implemented."""
        return {"status": "scaffold", "project_name": self.config.project_name}

