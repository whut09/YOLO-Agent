"""Strict paper-83 implementation gate for the first real training run.

This gate only reads the frozen campaign manifest and the paper-level
implementation registry.  It deliberately does not run a trainer, inspect a
GPU, or treat component maturity as paper implementation.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest
from yolo_agent.research.paper_implementation_registry import (
    PaperImplementationRegistry,
)


class PaperImplementationCampaignDecision(BaseModel):
    """Read-only authorization result for the frozen paper campaign."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool = False
    expected_paper_count: int = 83
    manifest_path: str
    registry_path: str
    membership_hash: str | None = None
    paper_count: int = 0
    implementation_ready_count: int = 0
    blockers: list[str] = Field(default_factory=list)


class PaperImplementationCampaignGate:
    """Allow real training only after all frozen papers are implementation-ready."""

    expected_paper_count = 83

    def evaluate(
        self,
        *,
        repository_root: Path | str = ".",
        manifest_path: Path | str | None = None,
        registry_path: Path | str | None = None,
    ) -> PaperImplementationCampaignDecision:
        root = Path(repository_root).resolve()
        manifest = Path(manifest_path) if manifest_path is not None else (
            root / "configs" / "research" / "paper_83_manifest.yaml"
        )
        registry = Path(registry_path) if registry_path is not None else (
            root / "runs" / "paper-readiness" / "paper_implementation_registry.yaml"
        )
        manifest = manifest.resolve()
        registry = registry.resolve()
        blockers: list[str] = []
        membership_hash: str | None = None
        paper_count = 0
        ready_count = 0

        campaign: Paper83Manifest | None = None
        if not manifest.is_file():
            blockers.append("paper_83_manifest_missing")
        else:
            try:
                campaign = Paper83Manifest.from_yaml(manifest)
            except (OSError, TypeError, ValueError) as exc:
                blockers.append(f"paper_83_manifest_invalid:{type(exc).__name__}")
            else:
                membership_hash = campaign.campaign.membership_hash
                paper_count = campaign.paper_count
                if paper_count != self.expected_paper_count:
                    blockers.append(
                        f"paper_83_manifest_count_mismatch:{paper_count}"
                    )

        implementation: PaperImplementationRegistry | None = None
        if not registry.is_file():
            blockers.append("paper_implementation_registry_missing")
        else:
            try:
                implementation = PaperImplementationRegistry.from_yaml(registry)
            except (OSError, TypeError, ValueError) as exc:
                blockers.append(
                    f"paper_implementation_registry_invalid:{type(exc).__name__}"
                )

        if campaign is not None and implementation is not None:
            campaign_ids = {item.paper_id for item in campaign.papers}
            implementation_ids = {item.paper_id for item in implementation.records}
            if implementation.paper_count != self.expected_paper_count:
                blockers.append(
                    f"paper_implementation_registry_count_mismatch:{implementation.paper_count}"
                )
            if implementation.manifest_membership_hash != membership_hash:
                blockers.append("paper_implementation_membership_hash_mismatch")
            if implementation_ids != campaign_ids:
                missing = ",".join(sorted(campaign_ids - implementation_ids))
                extra = ",".join(sorted(implementation_ids - campaign_ids))
                blockers.append(
                    f"paper_implementation_membership_mismatch:missing={missing or 'none'}:extra={extra or 'none'}"
                )
            ready_count = implementation.implementation_ready_count
            for record in implementation.records:
                if not record.is_implementation_ready:
                    blockers.append(
                        f"paper_not_implementation_ready:{record.paper_id}:{record.readiness}"
                    )

        return PaperImplementationCampaignDecision(
            allowed=not blockers
            and paper_count == self.expected_paper_count
            and ready_count == self.expected_paper_count,
            manifest_path=str(manifest),
            registry_path=str(registry),
            membership_hash=membership_hash,
            paper_count=paper_count,
            implementation_ready_count=ready_count,
            blockers=sorted(set(blockers)),
        )


__all__ = [
    "PaperImplementationCampaignDecision",
    "PaperImplementationCampaignGate",
]
