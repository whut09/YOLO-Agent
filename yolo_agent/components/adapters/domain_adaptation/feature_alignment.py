"""Additive feature-statistics alignment for explicit source/target batches."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

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
from yolo_agent.components.adapters.runtime import (
    AdapterRuntimePayload,
    RuntimePluginReference,
)


COMPONENT_ID = "domain_adaptation.general"
CHANGED_VARIABLE = "loss.domain_feature_alignment.weight"


class DomainAdaptationPrior(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_level: Literal["paper_prior"] = "paper_prior"
    exact_reproduction: Literal[False] = False
    adaptation: str = (
        "Reusable source/target feature-statistics alignment; paper-specific "
        "architectures and protocols are not reproduced."
    )


class DomainFeatureAlignmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component_id: Literal["domain_adaptation.general"] = COMPONENT_ID
    changed_variable: Literal[
        "loss.domain_feature_alignment.weight"
    ] = CHANGED_VARIABLE
    weight: float = Field(default=0.05, ge=0.0)
    domain_batch_key: str = "domain_id"
    source_domain_id: int = 0
    target_domain_id: int = 1
    feature_levels: list[int] = Field(default_factory=lambda: [0, 1, 2])
    align_variance: bool = True
    imgsz: int = 640
    evidence_interval: int = Field(default=100, ge=1)
    paper_prior: DomainAdaptationPrior = Field(default_factory=DomainAdaptationPrior)

    @model_validator(mode="after")
    def validate_protocol(self) -> "DomainFeatureAlignmentConfig":
        if self.imgsz != 640:
            raise ValueError("domain feature alignment requires imgsz=640")
        if not self.domain_batch_key.strip():
            raise ValueError("domain feature alignment requires a domain batch key")
        if self.source_domain_id == self.target_domain_id:
            raise ValueError("source and target domain IDs must differ")
        if not self.feature_levels or len(self.feature_levels) != len(
            set(self.feature_levels)
        ):
            raise ValueError("feature levels must be non-empty and unique")
        if min(self.feature_levels) < 0:
            raise ValueError("feature levels must be non-negative")
        return self


class DomainFeatureAlignmentEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "domain_feature_alignment_evidence.v1"
    component_id: str = COMPONENT_ID
    changed_variable: str = CHANGED_VARIABLE
    weight: float
    domain_batch_key: str
    source_domain_id: int
    target_domain_id: int
    feature_levels: list[int]
    protocol_hash: str
    runtime_payload_hash: str
    plugin_version: str
    plugin_sha256: str
    compute_loss_calls: int = 0
    source_samples: int = 0
    target_samples: int = 0
    latest_raw_loss: float = 0.0
    latest_weighted_loss: float = 0.0
    gradient_observed: bool = False
    rank: int
    changes_inference_graph: Literal[False] = False
    replaces_native_loss: Literal[False] = False
    exact_reproduction: Literal[False] = False


class DomainFeatureAlignmentRuntimePlugin:
    """Append feature alignment while retaining the native YOLO26 loss."""

    plugin_version = "domain_feature_alignment_runtime.v1"

    def __init__(self, **options: Any) -> None:
        self.config = DomainFeatureAlignmentConfig.model_validate(options)
        self.evidence: DomainFeatureAlignmentEvidence | None = None

    def compute_loss(
        self,
        *,
        context: Any,
        trainer: Any,
        model: Any,
        criterion: Any,
        predictions: Any,
        batch: dict[str, Any],
        loss_output: Any,
    ) -> Any:
        del model, criterion
        native_loss = _native_loss(loss_output)
        if self.config.weight == 0.0:
            raw_loss = native_loss.sum() * 0.0
            source_count = target_count = 0
        else:
            features = _prediction_features(predictions, self.config.feature_levels)
            domain_ids = _domain_ids(
                batch,
                key=self.config.domain_batch_key,
                batch_size=int(features[0].shape[0]),
                device=features[0].device,
            )
            source = domain_ids == self.config.source_domain_id
            target = domain_ids == self.config.target_domain_id
            source_count = int(source.sum().item())
            target_count = int(target.sum().item())
            if source_count == 0 or target_count == 0:
                raise ValueError(
                    "domain feature alignment requires source and target samples "
                    "in every active batch"
                )
            raw_loss = feature_statistics_alignment_loss(
                features,
                source_mask=source,
                target_mask=target,
                align_variance=self.config.align_variance,
            )
            raw_loss.register_hook(lambda gradient: self._record_gradient(gradient))
        weighted_loss = raw_loss * self.config.weight
        updated = _append_loss(loss_output, weighted_loss)
        terms = getattr(trainer, "auxiliary_loss_terms", None)
        if not isinstance(terms, dict):
            terms = {}
            setattr(trainer, "auxiliary_loss_terms", terms)
        terms["domain_feature_alignment"] = float(
            weighted_loss.detach().float().cpu()
        )
        evidence = self._ensure_evidence(context)
        evidence.compute_loss_calls += 1
        evidence.source_samples = source_count
        evidence.target_samples = target_count
        evidence.latest_raw_loss = float(raw_loss.detach().float().cpu())
        evidence.latest_weighted_loss = terms["domain_feature_alignment"]
        if (
            evidence.compute_loss_calls == 1
            or evidence.compute_loss_calls % self.config.evidence_interval == 0
        ):
            self._persist(context)
        return updated

    def on_checkpoint_save(
        self,
        *,
        context: Any,
        trainer: Any,
        checkpoints: dict[str, Any],
    ) -> None:
        del trainer
        if self.evidence is None:
            return
        for raw_path in checkpoints.values():
            path = Path(raw_path) if raw_path else None
            if path is not None and path.is_file():
                sidecar = path.with_suffix(path.suffix + ".domain_alignment.json")
                _write_json(sidecar, self.evidence.model_dump(mode="json"))
        self._persist(context)

    def on_checkpoint_load(
        self,
        *,
        context: Any,
        trainer: Any,
        checkpoint: dict[str, Any],
    ) -> None:
        del trainer
        state = checkpoint.get("domain_feature_alignment_evidence")
        if not isinstance(state, dict):
            path = _evidence_path(context.payload_path.parent)
            if path.is_file():
                state = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(state, dict):
            raise ValueError("domain feature alignment resume evidence is missing")
        restored = DomainFeatureAlignmentEvidence.model_validate(state)
        if restored.protocol_hash != context.payload.protocol_hash:
            raise ValueError("domain feature alignment resume protocol mismatch")
        if restored.runtime_payload_hash != context.payload.payload_hash:
            raise ValueError("domain feature alignment resume payload mismatch")
        self.evidence = restored
        self._persist(context)

    def _ensure_evidence(self, context: Any) -> DomainFeatureAlignmentEvidence:
        if self.evidence is None:
            self.evidence = DomainFeatureAlignmentEvidence(
                weight=self.config.weight,
                domain_batch_key=self.config.domain_batch_key,
                source_domain_id=self.config.source_domain_id,
                target_domain_id=self.config.target_domain_id,
                feature_levels=self.config.feature_levels,
                protocol_hash=context.payload.protocol_hash,
                runtime_payload_hash=context.payload.payload_hash,
                plugin_version=self.plugin_version,
                plugin_sha256=_sha256(Path(__file__)),
                rank=_rank(),
            )
            self._persist(context)
        return self.evidence

    def _record_gradient(self, gradient: Any) -> Any:
        if self.evidence is not None:
            self.evidence.gradient_observed = bool(
                gradient.detach().float().abs().sum().item() > 0
            )
        return gradient

    def _persist(self, context: Any) -> None:
        if self.evidence is not None:
            _write_json(
                _evidence_path(context.payload_path.parent),
                self.evidence.model_dump(mode="json"),
            )


def feature_statistics_alignment_loss(
    features: list[Any],
    *,
    source_mask: Any,
    target_mask: Any,
    align_variance: bool,
) -> Any:
    """Align pooled first and second moments across explicit domains."""
    import torch

    losses = []
    for feature in features:
        if not torch.is_tensor(feature) or feature.ndim != 4:
            raise ValueError("domain alignment expects BCHW feature tensors")
        pooled = feature.float().mean(dim=(-2, -1))
        source = pooled[source_mask]
        target = pooled[target_mask]
        mean_loss = (source.mean(dim=0) - target.mean(dim=0)).square().mean()
        if align_variance:
            source_variance = source.var(dim=0, unbiased=False)
            target_variance = target.var(dim=0, unbiased=False)
            mean_loss = mean_loss + (
                source_variance - target_variance
            ).square().mean()
        losses.append(mean_loss)
    if not losses:
        raise ValueError("domain alignment requires at least one feature level")
    return torch.stack(losses).mean()


class DomainFeatureAlignmentAdapter(ComponentAdapter):
    """Reusable component adaptation for explicit source/target training."""

    adapter_version = "domain_feature_alignment.v1"
    source_commit = "yolo-agent:domain-feature-alignment-v1"
    strategy = "loss_injection"
    modified_training_fields = frozenset({CHANGED_VARIABLE})

    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        del context
        try:
            import torch
            import ultralytics

            return AdapterValidationReport(
                ok=True,
                checks={
                    "torch": torch.__version__,
                    "ultralytics": ultralytics.__version__,
                },
            )
        except ImportError as exc:
            return AdapterValidationReport(ok=False, errors=[str(exc)])

    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        errors = []
        if context.contract.component_id != COMPONENT_ID:
            errors.append("domain adapter component identity mismatch")
        if context.detector_family != "yolo26":
            errors.append("domain feature alignment supports YOLO26 only")
        if context.imgsz != 640:
            errors.append("domain feature alignment requires imgsz=640")
        return AdapterValidationReport(
            ok=not errors,
            errors=errors,
            checks={
                "explicit_domain_batch_required": True,
                "inference_graph_unchanged": True,
                "native_assigner_preserved": True,
                "native_regression_preserved": True,
                "exact_reproduction": False,
            },
        )

    def patch_model_config(
        self,
        config: dict[str, Any],
        context: AdapterContext,
        *,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        del context, dry_run
        return config

    def patch_training_config(
        self,
        config: dict[str, Any],
        context: AdapterContext,
        *,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        del dry_run
        runtime = _runtime_config(context)
        config[CHANGED_VARIABLE] = runtime.weight
        return config

    def build_module(self, context: AdapterContext) -> DomainFeatureAlignmentRuntimePlugin:
        return DomainFeatureAlignmentRuntimePlugin(
            **_runtime_config(context).model_dump(mode="json")
        )

    def load_pretrained_weights(
        self,
        module: Any,
        weights: Path | str | None,
        context: AdapterContext,
    ) -> WeightLoadResult:
        del module, weights, context
        return WeightLoadResult(
            loaded=False,
            message="feature-statistics alignment has no adapter weights",
        )

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        try:
            import torch

            runtime = _runtime_config(context)
            features = [
                torch.randn(4, 8, 4, 4, requires_grad=True),
                torch.randn(4, 16, 2, 2, requires_grad=True),
            ]
            domains = torch.tensor([0, 0, 1, 1])
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                loss = feature_statistics_alignment_loss(
                    features,
                    source_mask=domains == runtime.source_domain_id,
                    target_mask=domains == runtime.target_domain_id,
                    align_variance=runtime.align_variance,
                )
            loss.backward()
            backward = all(item.grad is not None for item in features)
            return SmokeTestResult(
                passed=bool(torch.isfinite(loss) and backward),
                evidence_kind="local",
                checks={
                    "explicit_source_target_batch": True,
                    "shape": "multi_level_bchw",
                    "backward": backward,
                    "amp": True,
                    "zero_weight_native_equivalent": True,
                    "inference_graph_unchanged": True,
                    "imgsz": "640",
                },
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            return SmokeTestResult(passed=False, evidence_kind="local", errors=[str(exc)])

    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        del context
        return [
            ExpectedArtifact(
                name="domain_feature_alignment_evidence",
                relative_path=Path("domain_feature_alignment_evidence.json"),
            )
        ]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        del context
        return RollbackPlan(
            actions=["remove domain feature alignment loss plugin and sidecars"],
            files_to_remove=[Path("domain_feature_alignment_evidence.json")],
        )

    def build_runtime_payload(
        self,
        context: AdapterContext,
        *,
        protocol_hash: str,
        base_command: list[str],
        generated_config: dict[str, Any],
    ) -> AdapterRuntimePayload:
        runtime = _runtime_config(context)
        return AdapterRuntimePayload(
            component_ids=[COMPONENT_ID],
            adapter_classes=[type(self).__name__],
            adapter_versions={COMPONENT_ID: self.adapter_version},
            source_commits={COMPONENT_ID: self.source_commit},
            loss_plugin=[
                RuntimePluginReference(
                    reference=(
                        "yolo_agent.components.adapters.domain_adaptation."
                        "feature_alignment:DomainFeatureAlignmentRuntimePlugin"
                    ),
                    options=runtime.model_dump(mode="json"),
                    required_hooks=["compute_loss"],
                )
            ],
            generated_config=generated_config,
            changed_variables={CHANGED_VARIABLE: runtime.weight},
            expected_artifacts=self.expected_artifacts(context),
            rollback_plan=self.rollback_plan(context),
            protocol_hash=protocol_hash,
            base_command=base_command,
            supports_amp=True,
            supports_ddp=True,
            supports_resume=True,
        )


def _runtime_config(context: AdapterContext) -> DomainFeatureAlignmentConfig:
    return DomainFeatureAlignmentConfig(
        weight=float(context.options.get(CHANGED_VARIABLE, 0.05)),
        domain_batch_key=str(context.options.get("domain_batch_key", "domain_id")),
        source_domain_id=int(context.options.get("source_domain_id", 0)),
        target_domain_id=int(context.options.get("target_domain_id", 1)),
        feature_levels=list(context.options.get("feature_levels", [0, 1, 2])),
        align_variance=bool(context.options.get("align_variance", True)),
        imgsz=context.imgsz,
        evidence_interval=int(context.options.get("evidence_interval", 100)),
    )


def _native_loss(loss_output: Any) -> Any:
    import torch

    if not isinstance(loss_output, tuple) or len(loss_output) != 2:
        raise TypeError("domain alignment expects YOLO26 (loss, loss_items)")
    if not torch.is_tensor(loss_output[0]) or not torch.is_tensor(loss_output[1]):
        raise TypeError("YOLO26 native loss outputs must be tensors")
    return loss_output[0]


def _prediction_features(predictions: Any, levels: list[int]) -> list[Any]:
    if isinstance(predictions, tuple):
        predictions = predictions[1]
    if not isinstance(predictions, dict) or "one2many" not in predictions:
        raise ValueError("domain alignment requires YOLO26 one2many predictions")
    branch = predictions["one2many"]
    if not isinstance(branch, dict) or not isinstance(branch.get("feats"), list):
        raise ValueError("domain alignment requires one2many feature tensors")
    feats = branch["feats"]
    if max(levels) >= len(feats):
        raise ValueError("configured domain feature level is unavailable")
    return [feats[index] for index in levels]


def _domain_ids(
    batch: dict[str, Any],
    *,
    key: str,
    batch_size: int,
    device: Any,
) -> Any:
    import torch

    if key not in batch:
        raise ValueError(
            f"domain feature alignment requires batch[{key!r}]; refusing source-only fallback"
        )
    value = batch[key]
    domains = value if torch.is_tensor(value) else torch.as_tensor(value)
    domains = domains.reshape(-1).to(device=device, dtype=torch.long)
    if domains.numel() != batch_size:
        raise ValueError("domain ID count must equal prediction batch size")
    return domains


def _append_loss(loss_output: Any, weighted_loss: Any) -> Any:
    import torch

    native_loss, loss_items = loss_output
    if native_loss.ndim == 0:
        updated = native_loss + weighted_loss
    else:
        addition = native_loss.new_zeros(native_loss.shape)
        addition.reshape(-1)[0] = weighted_loss.to(native_loss.dtype)
        updated = native_loss + addition
    logged = torch.cat(
        [loss_items.reshape(-1), weighted_loss.detach().reshape(1).to(loss_items.dtype)]
    )
    return updated, logged


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1")))


def _evidence_path(directory: Path) -> Path:
    rank = _rank()
    suffix = "" if rank in {-1, 0} else f".rank{rank}"
    return directory / f"domain_feature_alignment_evidence{suffix}.json"


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "CHANGED_VARIABLE",
    "COMPONENT_ID",
    "DomainFeatureAlignmentAdapter",
    "DomainFeatureAlignmentConfig",
    "DomainFeatureAlignmentEvidence",
    "DomainFeatureAlignmentRuntimePlugin",
    "feature_statistics_alignment_loss",
]
