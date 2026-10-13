"""fix 39: forks inherit the base run's current on-disk snapshot binding.

An external preflight can rebind the base run to a refreshed research
snapshot mid-campaign by rewriting only the base run's run_context.yaml.
In-memory round contexts never observe that rebind, so forking from memory
pinned every child to the snapshot that was current when the driver started;
a stale snapshot fails component contract validation and blocks candidates
at runtime readiness.  Forks (and resumed rounds) must follow the base run's
current disk binding instead.

fix 41: the paper execution inventory binding is campaign-level state too.
Forks that lose it seed a coverage ledger with expected_paper_count=0 and
report paper_inventory_count: 0, so the planner believes the paper registry
is empty.  The same disk-following inheritance covers those three keys.
"""

from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.next_round_forker import (
    NextRoundForker,
    apply_base_snapshot_binding,
    base_run_snapshot_binding,
)
from yolo_agent.core.run_context import RunContext

NEW_HASH = "a" * 64
OLD_HASH = "b" * 64


def _persisted_context(
    run_root: Path,
    run_id: str,
    *,
    parent_dir: Path | None,
    snapshot_hash: str,
    extra_metadata: dict[str, object] | None = None,
) -> RunContext:
    """Write a run context to disk the way a real round would persist it."""
    metadata: dict[str, object] = {
        "research_snapshot_hash": snapshot_hash,
        "research_snapshot_path": f"E:/snapshots/{snapshot_hash}",
        "research_snapshot_verified": True,
        "snapshot_status": "current",
        "stale_reasons": [],
    }
    if parent_dir is not None:
        metadata["parent_run_id"] = parent_dir.name
        metadata["parent_run_dir"] = parent_dir.as_posix()
    if extra_metadata:
        metadata.update(extra_metadata)
    context = RunContext(
        run_id=run_id,
        run_root=run_root,
        task_path=run_root / "task.yaml",
        data_yaml=run_root / "data.yaml",
        metadata=metadata,
    )
    context.run_dir.mkdir(parents=True, exist_ok=True)
    context.to_yaml()
    return context


def test_fork_follows_base_run_disk_binding_after_external_rebind(tmp_path: Path) -> None:
    """A rebind of the base run on disk must reach the next forked child."""
    base = _persisted_context(tmp_path, "campaign", parent_dir=None, snapshot_hash=NEW_HASH)
    # The driver's in-memory round context still carries the stale binding.
    round_context = _persisted_context(
        tmp_path,
        "campaign-r105",
        parent_dir=base.run_dir,
        snapshot_hash=OLD_HASH,
    )

    forker = NextRoundForker(round_context, None)  # type: ignore[arg-type]
    child = forker._child_context("campaign-r106", tmp_path / "next_round.yaml", {}, [])

    assert child.metadata["research_snapshot_hash"] == NEW_HASH
    assert child.metadata["research_snapshot_path"] == f"E:/snapshots/{NEW_HASH}"
    assert child.metadata["research_snapshot_verified"] is True
    assert child.metadata["snapshot_status"] == "current"


def test_fork_walks_multi_level_parent_chain(tmp_path: Path) -> None:
    """The walk must reach the base run through intermediate round contexts."""
    base = _persisted_context(tmp_path, "campaign", parent_dir=None, snapshot_hash=NEW_HASH)
    first = _persisted_context(
        tmp_path, "campaign-r1", parent_dir=base.run_dir, snapshot_hash=OLD_HASH
    )
    second = _persisted_context(
        tmp_path, "campaign-r2", parent_dir=first.run_dir, snapshot_hash=OLD_HASH
    )

    forker = NextRoundForker(second, None)  # type: ignore[arg-type]
    child = forker._child_context("campaign-r3", tmp_path / "next_round.yaml", {}, [])

    assert child.metadata["research_snapshot_hash"] == NEW_HASH
    assert child.metadata["research_snapshot_path"] == f"E:/snapshots/{NEW_HASH}"


def test_fork_falls_back_to_memory_binding_without_base_disk(tmp_path: Path) -> None:
    """No readable parent chain: keep the in-memory binding (prior behavior)."""
    context = RunContext(
        run_id="solo",
        run_root=tmp_path,
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
        metadata={
            "research_snapshot_hash": OLD_HASH,
            "research_snapshot_path": f"E:/snapshots/{OLD_HASH}",
            "research_snapshot_verified": True,
            "snapshot_status": "current",
        },
    )

    forker = NextRoundForker(context, None)  # type: ignore[arg-type]
    child = forker._child_context("solo-r2", tmp_path / "next_round.yaml", {}, [])

    assert child.metadata["research_snapshot_hash"] == OLD_HASH
    assert child.metadata["research_snapshot_verified"] is True
    assert child.metadata["snapshot_status"] == "current"


def test_binding_keeps_memory_values_for_keys_missing_on_base_disk(tmp_path: Path) -> None:
    """A base run without snapshot keys must not clobber the in-memory binding."""
    base = RunContext(
        run_id="campaign",
        run_root=tmp_path,
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "data.yaml",
        metadata={"paper_method_coverage_version": "not_available"},
    )
    base.run_dir.mkdir(parents=True, exist_ok=True)
    base.to_yaml()
    round_context = _persisted_context(
        tmp_path,
        "campaign-r1",
        parent_dir=base.run_dir,
        snapshot_hash=OLD_HASH,
    )

    binding = base_run_snapshot_binding(round_context)

    assert binding["research_snapshot_hash"] == OLD_HASH
    assert binding["snapshot_status"] == "current"
    assert binding["paper_method_coverage_version"] == "not_available"


def test_fork_context_payload_reports_base_binding(tmp_path: Path) -> None:
    """The fork audit artifact must record the binding the child actually got."""
    base = _persisted_context(tmp_path, "campaign", parent_dir=None, snapshot_hash=NEW_HASH)
    round_context = _persisted_context(
        tmp_path,
        "campaign-r105",
        parent_dir=base.run_dir,
        snapshot_hash=OLD_HASH,
    )

    forker = NextRoundForker(round_context, None)  # type: ignore[arg-type]
    child = forker._child_context("campaign-r106", tmp_path / "next_round.yaml", {}, [])
    payload = forker._fork_context_payload(
        child=child,
        next_round={},
        next_round_path=tmp_path / "next_round.yaml",
        inherited_next_round_path=tmp_path / "parent_next_round.yaml",
        missing_evidence=[],
    )

    assert payload["research_snapshot_hash"] == NEW_HASH
    assert payload["research_snapshot_path"] == f"E:/snapshots/{NEW_HASH}"


def test_apply_base_snapshot_binding_refreshes_resumed_round(tmp_path: Path) -> None:
    """A resumed round directory follows the base run's current disk binding."""
    base = _persisted_context(tmp_path, "campaign", parent_dir=None, snapshot_hash=NEW_HASH)
    resumed = _persisted_context(
        tmp_path,
        "campaign-r105",
        parent_dir=base.run_dir,
        snapshot_hash=OLD_HASH,
    )

    assert apply_base_snapshot_binding(resumed) is True
    assert resumed.metadata["research_snapshot_hash"] == NEW_HASH
    assert resumed.metadata["research_snapshot_path"] == f"E:/snapshots/{NEW_HASH}"
    # Idempotent: a second call reports no change.
    assert apply_base_snapshot_binding(resumed) is False


def test_apply_base_snapshot_binding_noop_without_rebind(tmp_path: Path) -> None:
    """When the base run was never rebound the resumed round is untouched."""
    _persisted_context(tmp_path, "campaign", parent_dir=None, snapshot_hash=OLD_HASH)
    round_context = _persisted_context(
        tmp_path,
        "campaign-r1",
        parent_dir=tmp_path / "campaign",
        snapshot_hash=OLD_HASH,
    )

    assert apply_base_snapshot_binding(round_context) is False
    assert round_context.metadata["research_snapshot_hash"] == OLD_HASH


INVENTORY_PATH = "E:/codex/YOLO-Agent/runs/coverage-audit/paper_execution_inventory.yaml"
INVENTORY_HASH = "61d9f6f299cce13a1f91a8877e930e913167d865d26c0b9d1f5d3010d60d937f"


def test_fork_inherits_paper_execution_inventory_binding(tmp_path: Path) -> None:
    """fix 41: forks inherit the base run's paper execution inventory binding.

    Without it the child writes a coverage ledger with expected_paper_count=0
    and every round summary reports paper_inventory_count: 0, even though the
    base run is bound to the compatible-paper inventory.
    """
    base = _persisted_context(
        tmp_path,
        "campaign",
        parent_dir=None,
        snapshot_hash=OLD_HASH,
        extra_metadata={
            "paper_execution_inventory_path": INVENTORY_PATH,
            "paper_execution_inventory_hash": INVENTORY_HASH,
            "compatible_paper_count": 83,
        },
    )
    # The round context was forked before the inventory was bound and does
    # not carry the three keys at all.
    round_context = _persisted_context(
        tmp_path,
        "campaign-r105",
        parent_dir=base.run_dir,
        snapshot_hash=OLD_HASH,
    )

    forker = NextRoundForker(round_context, None)  # type: ignore[arg-type]
    child = forker._child_context("campaign-r106", tmp_path / "next_round.yaml", {}, [])
    payload = forker._fork_context_payload(
        child=child,
        next_round={},
        next_round_path=tmp_path / "next_round.yaml",
        inherited_next_round_path=tmp_path / "parent_next_round.yaml",
        missing_evidence=[],
    )

    assert child.metadata["paper_execution_inventory_path"] == INVENTORY_PATH
    assert child.metadata["paper_execution_inventory_hash"] == INVENTORY_HASH
    assert child.metadata["compatible_paper_count"] == 83
    assert payload["paper_execution_inventory_path"] == INVENTORY_PATH
    assert payload["paper_execution_inventory_hash"] == INVENTORY_HASH
    assert payload["compatible_paper_count"] == 83


def test_fork_without_inventory_binding_keys_absent_on_base(tmp_path: Path) -> None:
    """A base run without the inventory binding must not fabricate one."""
    base = _persisted_context(tmp_path, "campaign", parent_dir=None, snapshot_hash=OLD_HASH)
    round_context = _persisted_context(
        tmp_path,
        "campaign-r1",
        parent_dir=base.run_dir,
        snapshot_hash=OLD_HASH,
    )

    forker = NextRoundForker(round_context, None)  # type: ignore[arg-type]
    child = forker._child_context("campaign-r2", tmp_path / "next_round.yaml", {}, [])

    assert child.metadata.get("paper_execution_inventory_path") is None
    assert child.metadata.get("paper_execution_inventory_hash") is None
    assert child.metadata.get("compatible_paper_count") is None


def test_apply_base_snapshot_binding_refreshes_paper_execution_inventory(
    tmp_path: Path,
) -> None:
    """A resumed round follows the base run's current inventory binding."""
    base = _persisted_context(tmp_path, "campaign", parent_dir=None, snapshot_hash=OLD_HASH)
    resumed = _persisted_context(
        tmp_path,
        "campaign-r105",
        parent_dir=base.run_dir,
        snapshot_hash=OLD_HASH,
    )
    # The base run was rebound to the inventory after the round was forked.
    base.metadata.update(
        {
            "paper_execution_inventory_path": INVENTORY_PATH,
            "paper_execution_inventory_hash": INVENTORY_HASH,
            "compatible_paper_count": 83,
        }
    )
    base.to_yaml()

    assert apply_base_snapshot_binding(resumed) is True
    assert resumed.metadata["paper_execution_inventory_path"] == INVENTORY_PATH
    assert resumed.metadata["paper_execution_inventory_hash"] == INVENTORY_HASH
    assert resumed.metadata["compatible_paper_count"] == 83
    # Idempotent: a second call reports no change.
    assert apply_base_snapshot_binding(resumed) is False
