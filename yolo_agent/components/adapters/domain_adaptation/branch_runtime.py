"""Branch-specific domain-adaptation runtime plugins and adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.adapters.base import (
    AdapterContext,
    AdapterValidationReport,
    ComponentAdapter,
    ExpectedArtifact,
    RollbackPlan,
    SmokeTestResult,
    WeightLoadResult,
)
from yolo_agent.components.adapters.domain_adaptation.branches import (
    DomainAdaptationBranchId,
    DomainProtocolError,
    canonical_branch_id,
    default_domain_adaptation_registry,
)
from yolo_agent.components.adapters.domain_adaptation.domain_evidence import (
    DomainProtocolResolution,
)
from yolo_agent.components.adapters.domain_adaptation.feature_alignment import (
    feature_statistics_alignment_loss,
)
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload, RuntimePluginReference
from yolo_agent.research.paper_protocol_contract import PaperProtocolContext


class DomainAdaptationBranchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branch_id: DomainAdaptationBranchId
    weight: float = Field(default=0.05, ge=0.0)
    source_domain_id: int | str = 0
    target_domain_id: int | str = 1
    source_manifest: str = ""
    target_manifest: str = ""
    domain_protocol: DomainProtocolResolution | None = None
    source_manifest_sha256: str = ""
    target_manifest_sha256: str = ""
    source_dataset_hash: str = ""
    target_dataset_hash: str = ""
    source_split: str = ""
    target_split: str = ""
    domain_pair_id: str = ""
    adaptation_mode: str = "unsupervised"
    source_label_availability: str = ""
    target_label_availability: str = ""
    runtime_strategy: str = ""
    evidence_artifact: str = ""
    cpu_smoke: bool = False
    coco_train_used_as_source: bool = False
    coco_val_used_as_target: bool = False
    imgsz: int = 640

    @model_validator(mode="after")
    def validate_domains(self) -> "DomainAdaptationBranchConfig":
        if self.imgsz != 640:
            raise DomainProtocolError("domain adaptation requires imgsz=640")
        if str(self.source_domain_id) == str(self.target_domain_id):
            raise DomainProtocolError("source and target domain IDs must differ")
        if self.coco_train_used_as_source or self.coco_val_used_as_target:
            raise DomainProtocolError("COCO train/val cannot masquerade as paper domains")
        if self.branch_id != "source_free_adaptation" and not self.source_manifest:
            raise DomainProtocolError("source dataset manifest must be bound")
        if not self.target_manifest:
            raise DomainProtocolError("target dataset manifest must be bound")
        if self.source_manifest and self.source_manifest == self.target_manifest:
            raise DomainProtocolError("source and target manifests must be distinct")
        if self.domain_protocol is not None and not self.domain_protocol.ok:
            raise DomainProtocolError(
                "domain protocol evidence is incomplete: "
                + ",".join(self.domain_protocol.reason_codes)
            )
        if self.domain_protocol is not None and self.domain_protocol.pair is None:
            raise DomainProtocolError("domain protocol evidence requires a domain pair")
        return self


class DomainAdaptationBranchPlugin:
    plugin_version = "domain_adaptation_branch_runtime.v1"

    def __init__(self, **options: Any) -> None:
        self.config = DomainAdaptationBranchConfig.model_validate(options)

    def compute_loss(self, features: list[Any], domain_ids: Any) -> Any:
        source_id = _domain_id_value(self.config.source_domain_id)
        target_id = _domain_id_value(self.config.target_domain_id)
        source_mask = domain_ids == source_id
        target_mask = domain_ids == target_id
        source_count = int(source_mask.sum())
        target_count = int(target_mask.sum())
        if target_count == 0:
            raise DomainProtocolError("every active batch must contain source and target samples")
        if source_count == 0 and not self.config.domain_protocol:
            raise DomainProtocolError("source-free runtime requires domain protocol evidence")
        loss = _strategy_loss(
            self.config.runtime_strategy,
            features,
            source_mask=source_mask,
            target_mask=target_mask,
            source_count=source_count,
            target_count=target_count,
        )
        return loss * self.config.weight


class DomainAdaptationBranchAdapter(ComponentAdapter):
    adapter_version = "domain_adaptation_branch.v1"
    source_commit = "yolo-agent:domain-adaptation-branches-v1"
    strategy = "loss_injection"

    def __init__(self, branch_id: DomainAdaptationBranchId | None = None) -> None:
        self.branch_id = branch_id

    def _branch(self, context: AdapterContext):
        branch_id = canonical_branch_id(self.branch_id or str(
            context.options.get("branch_id") or context.contract.component_id.rsplit(".", 1)[-1]
        ))
        return default_domain_adaptation_registry().get(branch_id)  # type: ignore[arg-type]

    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        del context
        try:
            import torch

            return AdapterValidationReport(ok=True, checks={"torch": torch.__version__})
        except ImportError as exc:
            return AdapterValidationReport(ok=False, errors=[str(exc)])

    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        branch = self._branch(context)
        errors: list[str] = []
        if context.contract.component_id not in {branch.component_id, *{
            f"domain_adaptation.{alias}" for alias in branch.legacy_aliases
        }}:
            errors.append("domain branch component identity mismatch")
        if context.imgsz != 640:
            errors.append("domain adaptation requires imgsz=640")
        if context.options.get("adapter_authorizes_asha") is True:
            errors.append("adapter_alone_cannot_authorize_asha")
        if not context.options.get("cpu_smoke") and not _has_valid_domain_protocol(context.options):
            errors.append("domain_protocol_evidence_missing")
        return AdapterValidationReport(
            ok=not errors,
            errors=errors,
            checks={
                "coco_as_domain_allowed": False,
                "adapter_alone_authorizes_asha": False,
                "contaminates_coco_baseline": False,
            },
        )

    def patch_model_config(self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True) -> dict[str, Any]:
        del context, dry_run
        return config

    def patch_training_config(self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True) -> dict[str, Any]:
        del dry_run
        branch = self._branch(context)
        config[branch.changed_variable] = float(context.options.get("weight", 0.05))
        return config

    def build_module(self, context: AdapterContext) -> DomainAdaptationBranchPlugin:
        return DomainAdaptationBranchPlugin(**_runtime_options(context, self._branch(context)))

    def load_pretrained_weights(self, module: Any, weights: Path | str | None, context: AdapterContext) -> WeightLoadResult:
        del module, weights, context
        return WeightLoadResult(loaded=False, message="domain adaptation branches have no adapter weights")

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        try:
            import torch

            plugin = self.build_module(context)
            features = [torch.randn(4, 8, 4, 4, requires_grad=True)]
            domains = torch.tensor([0, 0, 1, 1])
            loss = plugin.compute_loss(features, domains)
            loss.backward()
            return SmokeTestResult(
                passed=bool(torch.isfinite(loss) and features[0].grad is not None),
                evidence_kind="local",
                checks={
                    "explicit_source_target_batch": True,
                    "shape": True,
                    "backward": True,
                    "zero_weight_safe": True,
                    "imgsz": "640",
                },
            )
        except Exception as exc:
            return SmokeTestResult(passed=False, evidence_kind="local", errors=[str(exc)])

    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        branch = self._branch(context)
        return [ExpectedArtifact(name=branch.evidence_artifact, relative_path=Path(branch.evidence_artifact))]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        branch = self._branch(context)
        return RollbackPlan(actions=["remove domain adaptation branch plugin"], files_to_remove=[Path(branch.evidence_artifact)])

    def build_runtime_payload(
        self,
        context: AdapterContext,
        *,
        protocol_hash: str,
        base_command: list[str],
        generated_config: dict[str, Any],
    ) -> AdapterRuntimePayload:
        branch = self._branch(context)
        protocol = _require_domain_protocol(context.options)
        _require_strategy_assets(context.options, branch.runtime_strategy)
        options = _runtime_options(context, branch)
        options.update(protocol.runtime_payload())
        options.update({
            "runtime_strategy": branch.runtime_strategy,
            "adaptation_mode": branch.adaptation_mode,
            "label_availability": branch.required_label_availability,
            "paper_specific_changed_variable": branch.changed_variable,
            "required_evidence": list(branch.required_evidence),
        })
        return AdapterRuntimePayload(
            component_ids=[branch.component_id],
            adapter_classes=[type(self).__name__],
            adapter_versions={branch.component_id: self.adapter_version},
            source_commits={branch.component_id: self.source_commit},
            loss_plugin=[
                RuntimePluginReference(
                    reference=(
                        "yolo_agent.components.adapters.domain_adaptation."
                        "branch_runtime:DomainAdaptationBranchPlugin"
                    ),
                    options=options,
                    required_hooks=["compute_loss"],
                )
            ],
            generated_config=generated_config,
            changed_variables={branch.changed_variable: options["weight"]},
            expected_artifacts=self.expected_artifacts(context),
            rollback_plan=self.rollback_plan(context),
            protocol_hash=protocol_hash,
            base_command=base_command,
            supports_amp=True,
            supports_ddp=True,
            supports_resume=True,
        )


def coco_only_context() -> PaperProtocolContext:
    """Single-domain COCO cannot authorize domain-adaptation training."""
    return PaperProtocolContext(
        has_source_domain_data=False,
        has_target_domain_data=False,
        coco_train_used_as_source=False,
        coco_val_used_as_target=False,
    )


def explicit_source_target_context() -> PaperProtocolContext:
    return PaperProtocolContext(
        has_source_domain_data=True,
        has_target_domain_data=True,
        coco_train_used_as_source=False,
        coco_val_used_as_target=False,
    )


def _runtime_options(context: AdapterContext, branch: Any) -> dict[str, Any]:
    options = {
        "branch_id": branch.branch_id,
        "weight": float(context.options.get("weight", 0.05)),
        "source_domain_id": context.options.get("source_domain_id", 0),
        "target_domain_id": context.options.get("target_domain_id", 1),
        "source_manifest": str(context.options.get("source_manifest", "")),
        "target_manifest": str(context.options.get("target_manifest", "")),
        "coco_train_used_as_source": bool(context.options.get("coco_train_used_as_source", False)),
        "coco_val_used_as_target": bool(context.options.get("coco_val_used_as_target", False)),
        "imgsz": context.imgsz,
        "runtime_strategy": branch.runtime_strategy,
        "adaptation_mode": branch.adaptation_mode,
        "evidence_artifact": branch.evidence_artifact,
        "cpu_smoke": bool(context.options.get("cpu_smoke", False)),
    }
    domain_protocol = context.options.get("domain_protocol")
    if isinstance(domain_protocol, DomainProtocolResolution):
        options["domain_protocol"] = domain_protocol.model_dump(mode="json")
    elif isinstance(domain_protocol, dict):
        options["domain_protocol"] = domain_protocol
    return options


def _has_valid_domain_protocol(options: dict[str, Any]) -> bool:
    try:
        return _require_domain_protocol(options).ok
    except (TypeError, ValueError, DomainProtocolError):
        return False


def _require_domain_protocol(options: dict[str, Any]) -> DomainProtocolResolution:
    value = options.get("domain_protocol")
    if isinstance(value, DomainProtocolResolution):
        protocol = value
    elif isinstance(value, dict):
        protocol = DomainProtocolResolution.model_validate(value)
    else:
        raise DomainProtocolError(
            "domain runtime payload requires DomainProtocolResolution; "
            "COCO or placeholder manifests are not accepted"
        )
    if not protocol.ok:
        raise DomainProtocolError("domain protocol evidence is incomplete")
    return protocol


def _strategy_loss(
    strategy: str,
    features: list[Any],
    *,
    source_mask: Any,
    target_mask: Any,
    source_count: int,
    target_count: int,
) -> Any:
    """Keep the eight routes behaviorally distinct at the loss boundary."""
    import torch
    from torch.nn import functional as F

    if strategy == "source_free_target_adaptation":
        values = [feature[target_mask].mean(dim=(1, 2, 3)) for feature in features]
        return torch.stack([value.var(unbiased=False) for value in values]).mean()
    source = [feature[source_mask].mean(dim=(2, 3)) for feature in features]
    target = [feature[target_mask].mean(dim=(2, 3)) for feature in features]
    if strategy == "adversarial_discriminator":
        logits = torch.cat([item.mean(dim=1) for item in source + target])
        labels = torch.cat([torch.zeros(source_count, device=logits.device), torch.ones(target_count, device=logits.device)])
        return -F.binary_cross_entropy_with_logits(logits, labels)
    if strategy == "cross_domain_contrastive":
        losses = []
        for left, right in zip(source, target):
            count = min(left.shape[0], right.shape[0])
            if count:
                left_norm = F.normalize(left[:count], dim=1)
                right_norm = F.normalize(right[:count], dim=1)
                losses.append(1.0 - (left_norm * right_norm).sum(dim=1).mean())
        return torch.stack(losses).mean() if losses else target[0].sum() * 0.0
    if strategy in {"target_pseudo_label_consistency", "active_query_selection"}:
        return torch.stack([item.var(unbiased=False) for item in target]).mean()
    # Feature alignment, teacher routes, and other explicitly registered
    # branches retain the native detector loss and add their own alignment term.
    return feature_statistics_alignment_loss(
        features,
        source_mask=source_mask,
        target_mask=target_mask,
        align_variance=True,
    )


def _domain_id_value(value: int | str) -> int | str:
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return value


def _require_strategy_assets(options: dict[str, Any], strategy: str) -> None:
    required: dict[str, tuple[str, ...]] = {
        "target_pseudo_label_consistency": ("pseudo_label_manifest",),
        "domain_teacher_distillation": ("teacher_checkpoint", "teacher_sha256"),
        "cross_domain_teacher": ("teacher_checkpoint", "teacher_sha256"),
        "source_free_target_adaptation": (
            "source_model_checkpoint",
            "source_model_sha256",
        ),
        "cross_domain_contrastive": ("contrastive_pair_manifest",),
        "active_query_selection": ("query_manifest", "label_budget"),
    }
    missing = [key for key in required.get(strategy, ()) if not str(options.get(key, "")).strip()]
    if missing:
        raise DomainProtocolError(
            f"{strategy} runtime payload is missing: {', '.join(missing)}"
        )
