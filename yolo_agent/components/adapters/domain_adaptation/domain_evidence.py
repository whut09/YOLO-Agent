"""Typed, split-safe evidence for domain-adaptation execution.

The adapter can be imported without these assets, but a paper candidate cannot
be authorized until a complete :class:`DomainProtocolResolution` exists.  The
objects in this module are deliberately independent from mAP results: they
only prove that the requested source/target protocol is well-defined.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DomainRole = Literal["source", "target"]
LabelAvailability = Literal["labeled", "unlabeled", "partial", "pseudo"]
AdaptationMode = Literal[
    "unsupervised",
    "semi_supervised",
    "supervised",
    "source_free",
    "active",
]


class DomainEvidenceError(ValueError):
    """Raised when domain evidence would authorize an unsafe protocol."""


class DomainDatasetManifest(BaseModel):
    """Identity and split contract for one domain dataset manifest."""

    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    dataset_hash: str
    domain_id: str
    domain_name: str
    role: DomainRole
    split: str
    label_availability: LabelAvailability
    sample_index_hash: str = ""
    is_coco_supervised: bool = False
    source_run_id: str | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "DomainDatasetManifest":
        for field in ("path", "sha256", "dataset_hash", "domain_id", "domain_name", "split"):
            if not getattr(self, field).strip():
                raise DomainEvidenceError(f"domain manifest requires {field}")
        if len(self.sha256) < 8:
            raise DomainEvidenceError("domain manifest sha256 is too short")
        if self.is_coco_supervised:
            raise DomainEvidenceError("COCO supervised data cannot be a paper domain")
        return self


class DomainPairIdentity(BaseModel):
    """Stable identity for the two-domain adaptation protocol."""

    model_config = ConfigDict(extra="forbid")

    source_domain_id: str
    target_domain_id: str
    source_dataset_hash: str
    target_dataset_hash: str
    source_split: str
    target_split: str
    domain_pair_id: str
    protocol_hash: str = ""

    @model_validator(mode="after")
    def validate_pair(self) -> "DomainPairIdentity":
        if self.source_domain_id == self.target_domain_id:
            raise DomainEvidenceError("source and target domain IDs must differ")
        if self.source_dataset_hash == self.target_dataset_hash:
            raise DomainEvidenceError("source and target dataset hashes must differ")
        if self.source_split == self.target_split and self.source_dataset_hash == self.target_dataset_hash:
            raise DomainEvidenceError("source and target manifests are not independent")
        digest = _hash_payload(self.model_dump(mode="json", exclude={"protocol_hash"}))
        if self.protocol_hash and self.protocol_hash != digest:
            raise DomainEvidenceError("domain pair protocol hash mismatch")
        self.protocol_hash = digest
        return self


class DomainProtocolResolution(BaseModel):
    """Result of resolving domain assets; it never claims model improvement."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "domain_protocol_resolution.v1"
    ok: bool
    source: DomainDatasetManifest | None = None
    target: DomainDatasetManifest | None = None
    pair: DomainPairIdentity | None = None
    adaptation_mode: AdaptationMode
    source_free: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    recovery_action: str = ""
    evidence_artifact: str = "domain_protocol_evidence.json"

    @model_validator(mode="after")
    def validate_resolution(self) -> "DomainProtocolResolution":
        if self.ok and self.pair is None:
            raise DomainEvidenceError("valid domain resolution requires a domain pair")
        if not self.ok and not self.reason_codes:
            raise DomainEvidenceError("failed domain resolution requires reason codes")
        if not self.recovery_action.strip():
            raise DomainEvidenceError("domain resolution requires a recovery action")
        return self

    @property
    def protocol_hash(self) -> str:
        return self.pair.protocol_hash if self.pair else ""

    def runtime_payload(self) -> dict[str, Any]:
        """Return only evidence metadata required by a runtime plugin."""
        if not self.ok or self.source is None or self.target is None or self.pair is None:
            raise DomainEvidenceError("incomplete domain evidence cannot build runtime payload")
        return {
            "source_manifest": self.source.path,
            "source_manifest_sha256": self.source.sha256,
            "source_dataset_hash": self.source.dataset_hash,
            "source_domain_id": self.source.domain_id,
            "source_split": self.source.split,
            "source_label_availability": self.source.label_availability,
            "target_manifest": self.target.path,
            "target_manifest_sha256": self.target.sha256,
            "target_dataset_hash": self.target.dataset_hash,
            "target_domain_id": self.target.domain_id,
            "target_split": self.target.split,
            "target_label_availability": self.target.label_availability,
            "domain_pair_id": self.pair.domain_pair_id,
            "domain_protocol_hash": self.pair.protocol_hash,
            "adaptation_mode": self.adaptation_mode,
            "source_free": self.source_free,
            "evidence_artifact": self.evidence_artifact,
        }


def resolve_domain_protocol(
    *,
    source: DomainDatasetManifest | None,
    target: DomainDatasetManifest | None,
    adaptation_mode: AdaptationMode,
    domain_pair_id: str | None = None,
    source_free: bool = False,
) -> DomainProtocolResolution:
    """Resolve and validate a paper-specific source/target protocol."""
    missing: list[str] = []
    if not source_free and source is None:
        missing.append("source_domain_manifest_missing")
    if target is None:
        missing.append("target_domain_manifest_missing")
    if source is not None and source.role != "source":
        missing.append("source_manifest_role_invalid")
    if target is not None and target.role != "target":
        missing.append("target_manifest_role_invalid")
    if missing:
        return _failed_resolution(missing, adaptation_mode, source_free)
    assert target is not None
    if source is None:
        # Source-free adaptation still needs the source-trained model evidence;
        # it does not silently manufacture a source dataset from COCO.
        return DomainProtocolResolution(
            ok=False,
            target=target,
            adaptation_mode=adaptation_mode,
            source_free=True,
            reason_codes=["source_free_source_model_evidence_missing"],
            required_evidence=["source_trained_checkpoint", "source_model_protocol_hash"],
            recovery_action="provide the frozen source-trained checkpoint and protocol evidence",
        )
    if source.is_coco_supervised or target.is_coco_supervised:
        return _failed_resolution(
            ["coco_supervised_data_cannot_be_domain_pair"], adaptation_mode, source_free, source, target
        )
    if source.dataset_hash == target.dataset_hash or source.sha256 == target.sha256:
        return _failed_resolution(
            ["source_target_manifest_identity_collision"], adaptation_mode, source_free, source, target
        )
    pair = DomainPairIdentity(
        source_domain_id=source.domain_id,
        target_domain_id=target.domain_id,
        source_dataset_hash=source.dataset_hash,
        target_dataset_hash=target.dataset_hash,
        source_split=source.split,
        target_split=target.split,
        domain_pair_id=domain_pair_id or f"{source.domain_id}->{target.domain_id}",
    )
    return DomainProtocolResolution(
        ok=True,
        source=source,
        target=target,
        pair=pair,
        adaptation_mode=adaptation_mode,
        source_free=source_free,
        recovery_action="matched source/target domain evidence is ready for paired evaluation",
    )


def manifest_from_file(
    path: Path | str,
    *,
    role: DomainRole,
    dataset_hash: str,
    domain_id: str,
    domain_name: str,
    split: str,
    label_availability: LabelAvailability,
) -> DomainDatasetManifest:
    """Create a manifest identity from an existing file without GPU access."""
    resolved = Path(path)
    if not resolved.is_file():
        raise DomainEvidenceError(f"domain manifest file missing: {resolved}")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return DomainDatasetManifest(
        path=str(resolved),
        sha256=digest,
        dataset_hash=dataset_hash,
        domain_id=domain_id,
        domain_name=domain_name,
        role=role,
        split=split,
        label_availability=label_availability,
    )


def _failed_resolution(
    reasons: list[str],
    adaptation_mode: AdaptationMode,
    source_free: bool,
    source: DomainDatasetManifest | None = None,
    target: DomainDatasetManifest | None = None,
) -> DomainProtocolResolution:
    return DomainProtocolResolution(
        ok=False,
        source=source,
        target=target,
        adaptation_mode=adaptation_mode,
        source_free=source_free,
        reason_codes=list(dict.fromkeys(reasons)),
        required_evidence=["source_dataset_manifest", "target_dataset_manifest", "domain_pair_protocol"],
        recovery_action="provide distinct, hashed source and target manifests with explicit splits and labels",
    )


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "AdaptationMode",
    "DomainDatasetManifest",
    "DomainEvidenceError",
    "DomainPairIdentity",
    "DomainProtocolResolution",
    "LabelAvailability",
    "manifest_from_file",
    "resolve_domain_protocol",
]
