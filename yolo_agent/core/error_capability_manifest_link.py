"""Prompt-18I-v2 §8: promotion record -> capability-matrix authorization.

The capability matrix (``configs/capability_maturity.yaml``) is generated
from audited data; README text is never edited by hand.  This module is
the *only* bridge that lets a real run's
:class:`ErrorLoopPromotionRecord` authorize promoting the two error-loop
capabilities (``candidate_coco_error_facts``, ``error_delta_next_round``)
from ``incomplete``/``partial`` to ``executable`` in the manifest:

* the promotion record must be complete (all four loop conditions true),
* the record must reference the run it evaluated,
* the manifest generator accepts the record as evidence and rewrites the
  two capability entries' status through the normal generation path.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.core.error_capability_promotion import (
    DECISION_CAPABILITY,
    FACTS_CAPABILITY,
    ErrorLoopPromotionRecord,
    manifest_upgrade_allowed,
)

LINK_SCHEMA_VERSION = "error_capability_manifest_link.v1"

#: Status values the matrix may move to when the record authorizes it.
PROMOTED_STATUS = "executable"


class ManifestLinkResult(BaseModel):
    """Outcome of applying a promotion record to the manifest."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = LINK_SCHEMA_VERSION
    run_id: str
    upgraded: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    reason: str = ""


def apply_promotion_to_manifest(
    record: ErrorLoopPromotionRecord,
    manifest_path: Path | str,
) -> ManifestLinkResult:
    """Apply a real run's promotion record to the capability manifest.

    Without a complete record nothing changes — the function is the gate,
    not a rubber stamp.  Statuses of all other capabilities are untouched.
    """
    path = Path(manifest_path)
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = {item["capability_id"]: item for item in manifest.get("capabilities", [])}

    result = ManifestLinkResult(run_id=record.run_id)
    if not manifest_upgrade_allowed(record):
        result.rejected = [FACTS_CAPABILITY, DECISION_CAPABILITY]
        result.reason = "promotion record is incomplete: " + "; ".join(record.reasons)
        return result

    for capability_id in (FACTS_CAPABILITY, DECISION_CAPABILITY):
        entry = entries.get(capability_id)
        if entry is None:
            result.rejected.append(capability_id)
            continue
        if entry.get("status") == PROMOTED_STATUS:
            continue
        entry["status"] = PROMOTED_STATUS
        result.upgraded.append(capability_id)

    if result.upgraded:
        manifest["capabilities"] = [entries[cid] for cid in entries]
        path.write_text(
            yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return result


__all__ = [
    "DECISION_CAPABILITY",
    "FACTS_CAPABILITY",
    "PROMOTED_STATUS",
    "ManifestLinkResult",
    "apply_promotion_to_manifest",
]
