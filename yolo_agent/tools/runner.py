"""Experiment runner scaffold."""

from __future__ import annotations

from yolo_agent.core.executor import CommandSpec, DryRunExecutor, ExecutionResult
from yolo_agent.core.experiment_graph import ExperimentNode


class ExperimentRunner:
    """Thin compatibility wrapper around the dry-run executor."""

    def __init__(self, executor: DryRunExecutor | None = None) -> None:
        self.executor = executor or DryRunExecutor()

    def dry_run(self) -> bool:
        """Return success for a scaffold dry run."""
        return True

    def run_node(
        self,
        node: ExperimentNode,
        run_id: str,
        command: CommandSpec | None = None,
    ) -> ExecutionResult:
        """Dry-run one experiment node."""
        return self.executor.execute(node=node, run_id=run_id, command=command)
