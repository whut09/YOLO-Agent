"""CLI and experiment utility tools."""

__all__ = ["DatasetProfiler", "DatasetReport", "SmokeRunner", "SmokeRunResult", "profile_dataset"]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazily expose tool classes without creating import cycles."""
    if name in {"DatasetProfiler", "DatasetReport", "profile_dataset"}:
        from yolo_agent.tools import dataset_stats

        return getattr(dataset_stats, name)
    if name in {"SmokeRunner", "SmokeRunResult"}:
        from yolo_agent.tools import smoke_runner

        return getattr(smoke_runner, name)
    raise AttributeError(name)
