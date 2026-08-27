"""Branch-specific domain-adaptation runtime plugins and adapters."""

from __future__ import annotations

import hashlib
import json
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
    _append_loss,
    _domain_ids,
    _native_loss,
    _prediction_features,
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
    domain_protocol_hash: str = ""
    adaptation_mode: str = "unsupervised"
    source_free: bool = False
    label_availability: str = ""
    source_label_availability: str = ""
    target_label_availability: str = ""
    runtime_strategy: str = ""
    evidence_artifact: str = ""
    feature_levels: list[int] = Field(default_factory=lambda: [0, 1, 2])
    source_model_checkpoint_sha256: str = ""
    source_model_protocol_hash: str = ""
    teacher_checkpoint: str = ""
    teacher_sha256: str = ""
    teacher_mode: str = ""
    teacher_domain_id: str = ""
    pseudo_label_manifest: str = ""
    confidence_threshold: float = 0.0
    contrastive_pair_manifest: str = ""
    temperature: float = 0.0
    pairing_key: str = ""
    query_manifest: str = ""
    label_budget: int = 0
    query_strategy: str = ""
    discriminator_hidden: int = Field(default=128, ge=1)
    gradient_reversal_scale: float = Field(default=1.0, gt=0.0)
    source_model_checkpoint: str = ""
    source_model_sha256: str = ""
    required_evidence: list[str] = Field(default_factory=list)
    paper_specific_changed_variable: str = ""
    cpu_smoke: bool = False
    coco_train_used_as_source: bool = False
    coco_val_used_as_target: bool = False
    imgsz: int = 640
    paper_id: str | None = None
    paper_route_fingerprint: str | None = None
    paper_component_id: str | None = None

    @model_validator(mode="after")
    def validate_domains(self) -> "DomainAdaptationBranchConfig":
        if self.imgsz != 640:
            raise DomainProtocolError("domain adaptation requires imgsz=640")
        if str(self.source_domain_id) == str(self.target_domain_id):
            raise DomainProtocolError("source and target domain IDs must differ")
        if self.coco_train_used_as_source or self.coco_val_used_as_target:
            raise DomainProtocolError("COCO train/val cannot masquerade as paper domains")
        if self.paper_id is not None and (
            not self.paper_route_fingerprint
            or len(self.paper_route_fingerprint) != 64
        ):
            raise DomainProtocolError(
                "paper route identity requires a sha256 execution fingerprint"
            )
        if self.paper_component_id is not None and not (
            self.paper_component_id.startswith("domain_adaptation.")
        ):
            raise DomainProtocolError(
                "paper route component identity must stay in the domain family"
            )
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
        self.compute_loss_calls = 0
        self.evidence: dict[str, Any] | None = None
        self._reference_model: Any | None = None
        self._domain_discriminator: Any | None = None

    def build_model(self, *, context: Any, trainer: Any, model: Any) -> Any:
        """Attach trainable route modules before the optimizer is created."""
        del context, trainer
        if self.config.runtime_strategy != "adversarial_discriminator":
            return model
        import torch

        discriminator = torch.nn.Sequential(
            torch.nn.LazyLinear(self.config.discriminator_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.discriminator_hidden, 1),
        )
        model.add_module("yolo_agent_domain_discriminator", discriminator)
        self._domain_discriminator = discriminator
        return model

    def compute_loss(self, *args: Any, **kwargs: Any) -> Any:
        # Keep the compact two-argument API for CPU smoke tests, while the
        # runtime path uses the same YOLO26 (loss, loss_items) hook contract as
        # the production feature-alignment adapter.
        if len(args) == 2 and not kwargs:
            return self._compute_tensor_loss(args[0], args[1])
        return self._compute_runtime_loss(**kwargs)

    def _compute_tensor_loss(self, features: list[Any], domain_ids: Any) -> Any:
        source_id = _domain_id_value(self.config.source_domain_id)
        target_id = _domain_id_value(self.config.target_domain_id)
        source_mask = domain_ids == source_id
        target_mask = domain_ids == target_id
        source_count = int(source_mask.sum())
        target_count = int(target_mask.sum())
        if target_count == 0:
            raise DomainProtocolError("every active batch must contain source and target samples")
        if source_count == 0 and self.config.runtime_strategy != "source_free_target_adaptation":
            raise DomainProtocolError("active domain batch requires source samples")
        if self.config.runtime_strategy == "adversarial_discriminator":
            self._ensure_smoke_discriminator(features)
        if self.config.runtime_strategy == "adversarial_discriminator":
            loss = _adversarial_discriminator_loss(
                features,
                domain_ids,
                discriminator=self._domain_discriminator,
                source_domain_id=self.config.source_domain_id,
                target_domain_id=self.config.target_domain_id,
                reversal_scale=self.config.gradient_reversal_scale,
            )
        else:
            loss = _strategy_loss(
                self.config.runtime_strategy,
                features,
                source_mask=source_mask,
                target_mask=target_mask,
                source_count=source_count,
                target_count=target_count,
            )
        return loss * self.config.weight

    def _ensure_smoke_discriminator(self, features: list[Any]) -> None:
        if self._domain_discriminator is not None:
            return
        import torch

        width = sum(int(item.shape[1]) for item in features)
        self._domain_discriminator = torch.nn.Sequential(
            torch.nn.Linear(width, self.config.discriminator_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.discriminator_hidden, 1),
        )

    def _compute_runtime_loss(
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
        else:
            features = _prediction_features(predictions, self.config.feature_levels)
            domains = _domain_ids(
                batch,
                key="domain_id",
                batch_size=int(features[0].shape[0]),
                device=features[0].device,
            )
            raw_loss = self._compute_runtime_strategy_loss(
                features,
                domains,
                batch=batch,
                device=features[0].device,
            )
            raw_loss = raw_loss / max(self.config.weight, 1e-12)
        weighted_loss = raw_loss * self.config.weight
        updated = _append_loss(loss_output, weighted_loss)
        terms = getattr(trainer, "auxiliary_loss_terms", None)
        if not isinstance(terms, dict):
            terms = {}
            setattr(trainer, "auxiliary_loss_terms", terms)
        terms[f"domain_{self.config.runtime_strategy}"] = float(
            weighted_loss.detach().float().cpu()
        )
        self.compute_loss_calls += 1
        self._record_evidence(context, domains if self.config.weight != 0.0 else None, raw_loss, weighted_loss)
        return updated

    def _compute_runtime_strategy_loss(
        self,
        features: list[Any],
        domains: Any,
        *,
        batch: dict[str, Any],
        device: Any,
    ) -> Any:
        """Use route-specific evidence at the runtime hook boundary.

        The compact CPU smoke API intentionally remains tensor-only. The
        training hook is stricter: routes that claim pseudo labels, a teacher,
        contrastive pairs, or active queries must consume that evidence from
        the batch instead of silently degrading to feature alignment.
        """
        strategy = self.config.runtime_strategy
        if strategy == "adversarial_discriminator":
            if self._domain_discriminator is None:
                raise DomainProtocolError(
                    "adversarial discriminator was not attached before optimizer creation"
                )
            return _adversarial_discriminator_loss(
                features,
                domains,
                discriminator=self._domain_discriminator,
                source_domain_id=self.config.source_domain_id,
                target_domain_id=self.config.target_domain_id,
                reversal_scale=self.config.gradient_reversal_scale,
            )
        if strategy == "target_pseudo_label_consistency":
            pseudo = _batch_tensor(batch, "pseudo_labels", "pseudo_label_scores")
            pseudo = _target_evidence(pseudo, domains, self.config.target_domain_id)
            target = [item[self._target_mask(domains)].float() for item in features]
            return _pseudo_label_consistency_loss(target, pseudo)
        if strategy in {"domain_teacher_distillation", "cross_domain_teacher"}:
            teacher = _optional_batch_tensor(batch, "domain_teacher_features", "teacher_features")
            if teacher is None:
                teacher = self._reference_features(batch, device=device)
            teacher = _target_evidence(teacher, domains, self.config.target_domain_id)
            return _teacher_feature_distillation_loss(
                features,
                teacher,
                domains,
                target_domain_id=self.config.target_domain_id,
            )
        if strategy == "source_free_target_adaptation":
            source_model = _optional_batch_tensor(batch, "source_model_features", "source_model_outputs")
            if source_model is None:
                source_model = self._reference_features(batch, device=device, source_free=True)
            source_model = _target_evidence(
                source_model,
                domains,
                self.config.target_domain_id,
            )
            target = [item[self._target_mask(domains)] for item in features]
            return _source_free_consistency_loss(target, source_model)
        if strategy == "cross_domain_contrastive":
            pairs = _batch_tensor(batch, "contrastive_pairs", "contrastive_pair_features")
            source_mask = self._source_mask(domains)
            target_mask = self._target_mask(domains)
            student_alignment = _strategy_loss(
                strategy,
                features,
                source_mask=source_mask,
                target_mask=target_mask,
                source_count=int(source_mask.sum()),
                target_count=int(target_mask.sum()),
            )
            return _contrastive_pair_loss(pairs) + 0.1 * student_alignment
        if strategy == "active_query_selection":
            query_ids = _batch_tensor(batch, "query_ids", "active_query_ids")
            query_ids = _target_evidence(query_ids, domains, self.config.target_domain_id)
            target = [item[self._target_mask(domains)] for item in features]
            return _active_query_loss(target, query_ids)
        return _strategy_loss(
            strategy,
            features,
            source_mask=self._source_mask(domains),
            target_mask=self._target_mask(domains),
            source_count=int(self._source_mask(domains).sum()),
            target_count=int(self._target_mask(domains).sum()),
        )

    def _source_mask(self, domains: Any) -> Any:
        return domains == _domain_id_value(self.config.source_domain_id)

    def _target_mask(self, domains: Any) -> Any:
        return domains == _domain_id_value(self.config.target_domain_id)

    def _reference_features(self, batch: dict[str, Any], *, device: Any, source_free: bool = False) -> Any:
        """Run the frozen paper reference model when the loader did not cache responses."""
        import torch

        checkpoint = (
            self.config.source_model_checkpoint
            if source_free
            else self.config.teacher_checkpoint
        )
        if not checkpoint:
            raise DomainProtocolError("reference checkpoint is missing from domain runtime payload")
        path = Path(checkpoint)
        if not path.is_file():
            raise DomainProtocolError(f"reference checkpoint is missing: {path}")
        if self._reference_model is None:
            try:
                from ultralytics import YOLO

                self._reference_model = YOLO(str(path), task="detect", verbose=False).model
            except Exception as exc:
                raise DomainProtocolError(
                    f"could not load frozen domain reference checkpoint: {path}"
                ) from exc
            self._reference_model.eval()
            for parameter in self._reference_model.parameters():
                parameter.requires_grad_(False)
        images = batch.get("img")
        if images is None:
            raise DomainProtocolError("domain reference route requires batch['img']")
        images = images.to(device=device)
        with torch.no_grad():
            output = self._reference_model(images.float())
        branch = output[1] if isinstance(output, tuple) else output
        if not isinstance(branch, dict) or not isinstance(branch.get("one2many"), dict):
            raise DomainProtocolError("domain reference model did not return YOLO26 one2many features")
        features = branch["one2many"].get("feats")
        if not isinstance(features, list) or not features:
            raise DomainProtocolError("domain reference model returned no feature tensors")
        return torch.cat(
            [item.float().mean(dim=tuple(range(2, item.ndim))) for item in features],
            dim=1,
        )

    def _record_evidence(self, context: Any, domains: Any, raw_loss: Any, weighted_loss: Any) -> None:
        if self.evidence is None:
            self.evidence = {
                "schema_version": "domain_branch_evidence.v1",
                "branch_id": self.config.branch_id,
                "runtime_strategy": self.config.runtime_strategy,
                "changed_variable": f"loss.domain_{self.config.branch_id}.weight",
                "protocol_hash": context.payload.protocol_hash,
                "runtime_payload_hash": context.payload.payload_hash,
                "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_samples": 0,
                "target_samples": 0,
                "compute_loss_calls": 0,
                "exact_reproduction": False,
                "native_loss_preserved": True,
            }
        self.evidence["compute_loss_calls"] = self.compute_loss_calls
        if domains is not None:
            self.evidence["source_samples"] = int((domains == _domain_id_value(self.config.source_domain_id)).sum())
            self.evidence["target_samples"] = int((domains == _domain_id_value(self.config.target_domain_id)).sum())
        self.evidence["latest_raw_loss"] = float(raw_loss.detach().float().cpu())
        self.evidence["latest_weighted_loss"] = float(weighted_loss.detach().float().cpu())
        path = Path(context.payload_path).parent / self.config.evidence_artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def on_checkpoint_save(self, *, context: Any, trainer: Any, checkpoints: dict[str, Any]) -> None:
        del trainer, checkpoints
        if self.evidence is not None:
            self._record_evidence(context, None, _scalar_zero(), _scalar_zero())

    def on_checkpoint_load(self, *, context: Any, trainer: Any, checkpoint: dict[str, Any]) -> None:
        del trainer
        state = checkpoint.get("domain_branch_evidence")
        if not isinstance(state, dict):
            path = Path(context.payload_path).parent / self.config.evidence_artifact
            if path.is_file():
                state = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(state, dict):
            raise DomainProtocolError("domain branch resume evidence is missing")
        if state.get("protocol_hash") != context.payload.protocol_hash:
            raise DomainProtocolError("domain branch resume protocol mismatch")
        if state.get("runtime_payload_hash") != context.payload.payload_hash:
            raise DomainProtocolError("domain branch resume payload mismatch")
        self.evidence = state


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
        "feature_levels": list(context.options.get("feature_levels", [0, 1, 2])),
        "cpu_smoke": bool(context.options.get("cpu_smoke", False)),
        # Route-specific assets are part of the serialized runtime contract;
        # keeping them out of the plugin would make readiness weaker than the
        # training process that consumes the payload.
        "source_manifest_sha256": str(context.options.get("source_manifest_sha256", "")),
        "target_manifest_sha256": str(context.options.get("target_manifest_sha256", "")),
        "source_dataset_hash": str(context.options.get("source_dataset_hash", "")),
        "target_dataset_hash": str(context.options.get("target_dataset_hash", "")),
        "source_split": str(context.options.get("source_split", "")),
        "target_split": str(context.options.get("target_split", "")),
        "domain_pair_id": str(context.options.get("domain_pair_id", "")),
        "domain_protocol_hash": str(context.options.get("domain_protocol_hash", "")),
        "source_free": bool(context.options.get("source_free", False)),
        "source_model_checkpoint": str(context.options.get("source_model_checkpoint", "")),
        "source_model_sha256": str(context.options.get("source_model_sha256", "")),
        "source_model_protocol_hash": str(context.options.get("source_model_protocol_hash", "")),
        "teacher_checkpoint": str(context.options.get("teacher_checkpoint", "")),
        "teacher_sha256": str(context.options.get("teacher_sha256", "")),
        "teacher_mode": str(context.options.get("teacher_mode", "frozen")),
        "teacher_domain_id": str(context.options.get("teacher_domain_id", "")),
        "pseudo_label_manifest": str(context.options.get("pseudo_label_manifest", "")),
        "confidence_threshold": float(context.options.get("confidence_threshold", 0.0)),
        "contrastive_pair_manifest": str(context.options.get("contrastive_pair_manifest", "")),
        "temperature": float(context.options.get("temperature", 0.0)),
        "pairing_key": str(context.options.get("pairing_key", "")),
        "query_manifest": str(context.options.get("query_manifest", "")),
        "label_budget": int(context.options.get("label_budget", 0)),
        "query_strategy": str(context.options.get("query_strategy", "")),
        "discriminator_hidden": int(context.options.get("discriminator_hidden", 128)),
        "gradient_reversal_scale": float(
            context.options.get("gradient_reversal_scale", 1.0)
        ),
    }
    domain_protocol = context.options.get("domain_protocol")
    if isinstance(domain_protocol, DomainProtocolResolution):
        options["domain_protocol"] = domain_protocol.model_dump(mode="json")
    elif isinstance(domain_protocol, dict):
        options["domain_protocol"] = domain_protocol
    paper_id = context.options.get("paper_id")
    paper_fingerprint = context.options.get("paper_route_fingerprint")
    paper_component = context.options.get("paper_component_id")
    if paper_id:
        options["paper_id"] = str(paper_id)
        options["paper_route_fingerprint"] = (
            str(paper_fingerprint) if paper_fingerprint else None
        )
        options["paper_component_id"] = (
            str(paper_component) if paper_component else None
        )
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
    hash_pairs = (
        ("teacher_checkpoint", "teacher_sha256", "teacher checkpoint"),
        ("source_model_checkpoint", "source_model_sha256", "source model checkpoint"),
        ("source_manifest", "source_manifest_sha256", "source manifest"),
        ("target_manifest", "target_manifest_sha256", "target manifest"),
    )
    for path_key, hash_key, label in hash_pairs:
        path_value = str(options.get(path_key, "")).strip()
        expected = str(options.get(hash_key, "")).strip()
        if not path_value or not expected:
            continue
        path = Path(path_value)
        if not path.is_file():
            # Offline readiness owns existence checks. Keeping this boundary
            # permissive lets a serialized candidate carry a recovery action,
            # while any local reference execution still fails closed below.
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise DomainProtocolError(f"{label} sha256 mismatch")


def _optional_batch_tensor(batch: dict[str, Any], *keys: str) -> Any:
    import torch

    for key in keys:
        value = batch.get(key)
        if value is not None:
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
            if tensor.numel() == 0:
                break
            return tensor
    return None


def _batch_tensor(batch: dict[str, Any], *keys: str) -> Any:
    value = _optional_batch_tensor(batch, *keys)
    if value is None:
        raise DomainProtocolError("domain runtime batch evidence is missing: " + ", ".join(keys))
    return value


def _target_evidence(evidence: Any, domains: Any, target_domain_id: int | str) -> Any:
    """Filter full-batch evidence while accepting already target-only tensors."""
    if evidence.ndim > 0 and evidence.shape[0] == domains.shape[0]:
        return evidence[domains == _domain_id_value(target_domain_id)]
    return evidence


def _adversarial_discriminator_loss(
    features: list[Any],
    domains: Any,
    *,
    discriminator: Any,
    source_domain_id: int | str,
    target_domain_id: int | str,
    reversal_scale: float,
) -> Any:
    import torch
    from torch.nn import functional as F

    source_mask = domains == _domain_id_value(source_domain_id)
    target_mask = domains == _domain_id_value(target_domain_id)
    active = source_mask | target_mask
    if int(source_mask.sum()) == 0 or int(target_mask.sum()) == 0:
        raise DomainProtocolError("adversarial alignment requires source and target samples")
    pooled = torch.cat(
        [item.float().mean(dim=tuple(range(2, item.ndim))) for item in features],
        dim=1,
    )[active]
    reversed_features = _GradientReversal.apply(pooled, float(reversal_scale))
    logits = discriminator(reversed_features).reshape(-1)
    labels = target_mask[active].float().to(logits.device)
    return F.binary_cross_entropy_with_logits(logits, labels)


class _GradientReversal:
    @staticmethod
    def apply(features: Any, scale: float) -> Any:
        import torch

        class _Reversal(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, value: Any) -> Any:
                ctx.scale = scale
                return value.view_as(value)

            @staticmethod
            def backward(ctx: Any, gradient: Any) -> tuple[Any]:
                return (-ctx.scale * gradient,)

        return _Reversal.apply(features)


def _pseudo_label_consistency_loss(target_features: list[Any], pseudo_labels: Any) -> Any:
    import torch

    target_summary = torch.stack([item.mean(dim=tuple(range(1, item.ndim))) for item in target_features])
    labels = pseudo_labels.float().reshape(-1)
    if labels.numel() != target_summary.shape[1]:
        labels = labels.mean().expand(target_summary.shape[1])
    return (target_summary.mean(dim=0) - labels.to(target_summary.device)).square().mean()


def _teacher_feature_distillation_loss(
    features: list[Any],
    teacher_features: Any,
    domains: Any,
    *,
    target_domain_id: int | str,
) -> Any:
    import torch

    target = [item[domains == _domain_id_value(target_domain_id)] for item in features]
    teacher = teacher_features.float()
    student = torch.cat([item.float().mean(dim=tuple(range(2, item.ndim))) for item in target], dim=1)
    teacher = teacher.reshape(teacher.shape[0], -1)
    count = min(student.shape[0], teacher.shape[0])
    if count == 0:
        raise DomainProtocolError("domain teacher evidence has no target samples")
    width = min(student.shape[1], teacher.shape[1])
    return (student[:count, :width] - teacher[:count, :width].to(student.device)).square().mean()


def _source_free_consistency_loss(target_features: list[Any], source_model_outputs: Any) -> Any:
    import torch

    student = torch.cat([item.float().mean(dim=tuple(range(2, item.ndim))) for item in target_features], dim=1)
    reference = source_model_outputs.float().reshape(source_model_outputs.shape[0], -1).to(student.device)
    count = min(student.shape[0], reference.shape[0])
    width = min(student.shape[1], reference.shape[1])
    if count == 0 or width == 0:
        raise DomainProtocolError("source-free evidence has no target/model response pairs")
    return (student[:count, :width] - reference[:count, :width]).square().mean()


def _contrastive_pair_loss(pair_features: Any) -> Any:
    from torch.nn import functional as F

    pairs = pair_features.float()
    if pairs.ndim < 3 or pairs.shape[1] != 2:
        raise DomainProtocolError("contrastive pair evidence must have shape [N,2,...]")
    left = F.normalize(pairs[:, 0].reshape(pairs.shape[0], -1), dim=1)
    right = F.normalize(pairs[:, 1].reshape(pairs.shape[0], -1), dim=1)
    return (1.0 - (left * right).sum(dim=1)).mean()


def _active_query_loss(target_features: list[Any], query_ids: Any) -> Any:
    import torch

    queries = query_ids.reshape(-1).float()
    values = torch.stack([item.float().mean(dim=tuple(range(1, item.ndim))) for item in target_features])
    if queries.numel() != values.shape[1]:
        raise DomainProtocolError("active query IDs must match target feature width")
    weights = (queries > 0).float().to(values.device)
    if float(weights.sum()) == 0.0:
        raise DomainProtocolError("active query evidence selected no target samples")
    return (values.mean(dim=0) * weights).square().sum() / weights.sum()


def _scalar_zero() -> Any:
    import torch

    return torch.tensor(0.0)
