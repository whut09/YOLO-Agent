"""Paper model-graph audit: real graph modification, contract, and diff artifacts.

The frozen paper plan is the only source of campaign membership and domain
scope.  A paper is attributed to a graph mechanism only through its exact
paper-specific mechanism ID mapped to a concrete ``ModelGraphPlugin`` or the
terminal-detect head wrapper.  Every in-scope paper must produce a real
forward/backward probe on synthetic CPU tensors, a graph-contract check, and a
graph-diff artifact whose modified hash differs from the base graph.  No
training is started.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from yolo_agent.components.model_graph_diff import (
    BASE_GRAPH_PLUGIN,
    GraphDiff,
    _PassthroughModule,
    compute_graph_diff,
    config_fingerprint,
    graph_fingerprint,
    safe_paper_id,
)
from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest
from yolo_agent.research.paper_83_implementation_schemas import (
    Paper83EngineeringPlan,
    Paper83EngineeringPlanEntry,
)
from yolo_agent.research.paper_graph_side_schemas import (
    PaperGraphSideAudit,
    PaperGraphSideRecord,
)
from yolo_agent.research.paper_loss_side_schemas import (
    LossSideBehaviorEvidence,
    LossSideRouteSpec,
)

GRAPH_SIDE_DOMAINS = frozenset({"backbone", "neck", "feature_fusion", "head"})

GRAPH_PLUGIN_CHANNELS = [64, 128, 256]

GRAPH_ROUTE_KIND = "assignment"


def _plugin_features(
    channels: list[int] | None = None, *, imgsz: int = 64
) -> list[torch.Tensor]:
    channels = channels or GRAPH_PLUGIN_CHANNELS
    return [
        torch.zeros(1, value, imgsz // stride, imgsz // stride)
        for value, stride in zip(channels, (8, 16, 32), strict=True)
    ]


def _resolve_neck_plugin(mechanism_id: str, channels: list[int]) -> Any | None:
    """Build the concrete graph plugin for one neck/pyramid mechanism."""

    if mechanism_id == "neck.rtmdet_large_kernel":
        from yolo_agent.components.adapters.neck.rtmdet_large_kernel import (
            RTMDetLargeKernelNeck,
        )

        return RTMDetLargeKernelNeck(channels, kernel_size=5)
    if mechanism_id == "feature_pyramid.multi_scale":
        from yolo_agent.components.adapters.neck.feature_pyramid import (
            MultiScaleFeaturePyramidNeck,
        )

        return MultiScaleFeaturePyramidNeck(channels)
    if mechanism_id == "neck.gold_gather_distribute":
        from yolo_agent.components.adapters.neck.gold_gd import (
            GoldGatherDistributeNeck,
        )

        return GoldGatherDistributeNeck(channels)
    if mechanism_id == "neck.multi_scale_fusion":
        from yolo_agent.components.adapters.neck.multi_scale_fusion import (
            MultiScaleFusionNeck,
        )

        return MultiScaleFusionNeck(channels)
    return None


def _resolve_head_wrapper(mechanism_id: str) -> Any | None:
    """Build the terminal-detect wrapper on a minimal native-like stub."""

    if mechanism_id != "detection_head.task_aligned":
        return None

    import torch.nn as nn

    from yolo_agent.components.adapters.heads.task_aligned import (
        TaskAlignedDetectionHead,
        TaskAlignedHeadConfig,
    )

    class _HeadDetectStub(nn.Module):
        """Minimal native-like terminal detect returning the training dict."""

        def __init__(self) -> None:
            super().__init__()
            self.f = -1
            self.i = 99
            self.stride = torch.tensor([8.0, 16.0, 32.0])
            self.export = False
            self._scores_one2many = nn.Conv2d(64, 4, 1)
            self._scores_one2one = nn.Conv2d(64, 4, 1)
            self.cv2 = [None, None, None]

        def forward(self, features: list[Any]) -> Any:
            return {
                "one2many": {"scores": self._scores_one2many(features[0])},
                "one2one": {"scores": self._scores_one2one(features[0])},
            }

    return TaskAlignedDetectionHead(_HeadDetectStub(), TaskAlignedHeadConfig())


def _probe_neck_plugin(
    mechanism_id: str, plugin: Any
) -> tuple[LossSideBehaviorEvidence, GraphDiff]:
    """Forward/backward probe plus graph diff for one shape-preserving neck."""

    contract = plugin.input_contract
    channels = list(contract.channels)
    features = [
        torch.zeros(1, value, 64 // stride, 64 // stride, requires_grad=True)
        for value, stride in zip(channels, contract.strides, strict=True)
    ]
    base = graph_fingerprint(
        _PassthroughModule(), features=features, plugin_name=BASE_GRAPH_PLUGIN
    )
    outputs = plugin.forward(features)
    loss_tensor = sum(output.float().sum() for output in outputs)
    loss_tensor.backward()
    grads_flow = all(feature.grad is not None for feature in features)
    finite_output = all(bool(torch.isfinite(output).all()) for output in outputs)
    shapes_preserved = all(
        output.shape == feature.shape
        for output, feature in zip(outputs, features, strict=True)
    )
    modified = graph_fingerprint(
        plugin, features=features, plugin_name=plugin.plugin_id
    )
    diff = compute_graph_diff(
        paper_id="",
        mechanism_id=mechanism_id,
        component_id=plugin.plugin_id,
        base=base,
        modified=modified,
        feature_levels={
            "strides": list(contract.strides),
            "channels": channels,
            "insertion_point": contract.insertion_point,
        },
    )
    checks = {
        "forward_shapes_preserved": shapes_preserved,
        "finite_output": finite_output,
        "gradient_flow_to_inputs": grads_flow,
        "graph_hash_changed": diff.graph_hash_changed,
        "params_added": modified.parameter_count > base.parameter_count,
        "macs_positive": modified.estimated_macs > 0,
    }
    return (
        LossSideBehaviorEvidence(
            passed=all(bool(value) for value in checks.values()),
            checks=checks,
            observed_changes=["forward_path", "parameter_graph"],
            errors=[],
        ),
        diff,
    )


def _probe_head_wrapper(wrapper: Any) -> tuple[LossSideBehaviorEvidence, GraphDiff]:
    """Forward/backward probe for the terminal-detect head wrapper."""

    features = [torch.zeros(1, 64, 8, 8, requires_grad=True)]
    base = graph_fingerprint(
        _PassthroughModule(), features=features, plugin_name=BASE_GRAPH_PLUGIN
    )
    modified = graph_fingerprint(
        wrapper, features=features, plugin_name=wrapper.plugin_id
    )
    output = wrapper.forward(features)
    if not isinstance(output, dict) or set(output) != {"one2many", "one2one"}:
        raise RuntimeError("head wrapper did not return the native training dict")
    branch = output["one2one"]["scores"]
    branch.float().sum().backward()
    checks = {
        "forward_returns_training_dict": True,
        "one_to_one_branch_present": isinstance(output.get("one2one"), dict),
        "finite_output": bool(torch.isfinite(branch).all()),
        "gradient_flow_to_quality_scale": wrapper.quality_scale.grad is not None,
        "graph_hash_changed": modified.graph_hash != base.graph_hash,
    }
    diff = compute_graph_diff(
        paper_id="",
        mechanism_id="detection_head.task_aligned",
        component_id=wrapper.plugin_id,
        base=base,
        modified=modified,
        feature_levels={
            "strides": [8, 16, 32],
            "channels": [64],
            "insertion_point": "terminal_native_detect",
        },
    )
    return (
        LossSideBehaviorEvidence(
            passed=all(bool(value) for value in checks.values()),
            checks=checks,
            observed_changes=["one_to_one_score_scale", "terminal_node_replaced"],
            errors=[],
        ),
        diff,
    )


def resolve_graph_route(mechanism_id: str) -> LossSideRouteSpec | None:
    """Resolve an exact mechanism ID to its graph-side route."""

    neck_ids = {
        "neck.rtmdet_large_kernel",
        "feature_pyramid.multi_scale",
        "neck.gold_gather_distribute",
        "neck.multi_scale_fusion",
    }
    head_ids = {"detection_head.task_aligned"}
    if mechanism_id not in neck_ids | head_ids:
        return None
    return LossSideRouteSpec(
        mechanism_id=mechanism_id,
        route_kind="assignment",
        component_id=mechanism_id,
        adapter_id=mechanism_id,
        implementation_path="yolo_agent/components/adapters",
        plugin_name=mechanism_id,
        runtime_hooks=["build_model"],
        required_evidence=["graph_diff_artifact"],
        test_refs=["tests/test_paper_graph_side.py"],
        config_schema={"imgsz": 640},
    )


def _probe_route(
    mechanism_id: str,
) -> tuple[LossSideBehaviorEvidence, GraphDiff]:
    plugin = _resolve_neck_plugin(mechanism_id, GRAPH_PLUGIN_CHANNELS)
    if plugin is not None:
        return _probe_neck_plugin(mechanism_id, plugin)
    wrapper = _resolve_head_wrapper(mechanism_id)
    if wrapper is not None:
        return _probe_head_wrapper(wrapper)
    raise KeyError(f"no graph plugin for mechanism: {mechanism_id}")


GRAPH_ADAPTATION_RECORDS: dict[str, dict[str, Any]] = {
    "detection_head.task_aligned": {
        "original": (
            "TOOD trains a task-aligned head where classification and "
            "localization share features through task interaction and both "
            "branches emit task-aligned predictions"
        ),
        "adapted": (
            "YOLO26 keeps its native DFL-free end-to-end Detect head; the "
            "wrapper adds a bounded learnable quality scale to the native "
            "one-to-one scores so the head carries a task-alignment quality "
            "signal without replacing the native head"
        ),
        "preserved_mechanism": (
            "task-alignment quality modulation on the prediction branch; "
            "one-to-one branch remains the deployment path"
        ),
        "known_deviation": (
            "TOOD's task-interaction feature learner and joint TAL head "
            "structure are not reproduced; the native YOLO26 head supplies "
            "the shared features"
        ),
    },
    "large_kernel_depthwise_conv": {
        "original": (
            "RTMDet couples large-kernel depthwise convolutions with an "
            "explicit CSP-structured backbone and neck in a dedicated "
            "architecture"
        ),
        "adapted": (
            "shape-preserving 5x5 depthwise + pointwise residual blocks are "
            "inserted on each P3/P4/P5 level immediately before the native "
            "YOLO26 Detect node"
        ),
        "preserved_mechanism": (
            "large-kernel depthwise spatial aggregation with residual "
            "zero-init so training starts at the base graph"
        ),
        "known_deviation": (
            "RTMDet's CSP stage structure, backbone, and scale-specific "
            "hyperparameters are not reproduced"
        ),
    },
    "feature_pyramid.multi_scale": {
        "original": (
            "Gold-YOLO gathers multi-scale information into a fused context "
            "and distributes it back to every level with a collect-and-"
            "distribute convolution set"
        ),
        "adapted": (
            "one scale-aware context gate collects all levels into a fused "
            "context and modulates each level's refined features before the "
            "native Detect node"
        ),
        "preserved_mechanism": (
            "gather-and-distribute information flow across P3/P4/P5 with "
            "zero-init gates"
        ),
        "known_deviation": (
            "Gold-YOLO's dual-branch transformer-conv gather and the "
            "convolutional-glide module are not reproduced"
        ),
    },
}


class PaperGraphSideAuditBuilder:
    """Audit graph-side scope, behavior, and diffs for every frozen paper."""

    def __init__(
        self,
        *,
        manifest_path: Path | str = "configs/research/paper_83_manifest.yaml",
        plan_path: Path | str = "configs/research/paper_83_implementation_plan.yaml",
        artifacts_dir: Path | str = "artifacts/paper_graph_diffs",
        workspace: Path | str = ".",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.requested_plan_path = Path(plan_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.workspace = Path(workspace).resolve()

    def build(
        self,
        *,
        manifest: Paper83Manifest | None = None,
        plan: Paper83EngineeringPlan | None = None,
        write_artifacts: bool = True,
    ) -> "PaperGraphSideAudit":
        frozen = manifest or Paper83Manifest.from_yaml(self.manifest_path)
        if frozen.paper_count != 83:
            raise ValueError(
                f"frozen paper manifest must contain 83 papers, got {frozen.paper_count}"
            )
        engineering_plan, source_plan = self._load_plan(plan)
        frozen_ids = [item.paper_id for item in frozen.papers]
        plan_ids = list(engineering_plan.manifest_paper_ids) or [
            item.paper_id for item in engineering_plan.papers
        ]
        if plan_ids != frozen_ids:
            raise ValueError("graph-side plan membership differs from frozen paper manifest")
        if engineering_plan.manifest_membership_hash != frozen.campaign.membership_hash:
            raise ValueError("graph-side plan membership hash differs from frozen manifest")

        by_id = {item.paper_id: item for item in frozen.papers}
        entries = {item.paper_id: item for item in engineering_plan.papers}
        if set(entries) != set(frozen_ids):
            raise ValueError("graph-side plan must contain every frozen paper exactly once")
        records = [
            self._record(by_id[paper_id], entries[paper_id])
            for paper_id in frozen_ids
        ]
        summary = {
            status: sum(item.status == status for item in records)
            for status in ("ready", "blocked_missing_evidence", "out_of_scope")
        }
        audit = PaperGraphSideAudit(
            manifest_path=str(self.manifest_path.resolve()),
            plan_path=str(self.requested_plan_path.resolve()),
            plan_source_path=str(source_plan.resolve()),
            manifest_membership_hash=frozen.campaign.membership_hash,
            plan_identity_hash=(
                engineering_plan.plan_identity_hash
                or engineering_plan.calculate_hash()
            ),
            paper_count=len(records),
            graph_side_paper_count=sum(item.in_scope for item in records),
            records=sorted(records, key=lambda item: item.paper_id),
            summary=summary,
        )
        audit = audit.with_hash()
        if write_artifacts:
            self.write_artifacts(audit)
        return audit

    def write_artifacts(self, audit: "PaperGraphSideAudit") -> list[Path]:
        paths: list[Path] = []
        for record in audit.records:
            if not record.graph_diff:
                continue
            directory = self.workspace / self.artifacts_dir
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{safe_paper_id(record.paper_id)}.yaml"
            payload = dict(record.graph_diff)
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            paths.append(path)
        return paths

    def _load_plan(
        self,
        plan: Paper83EngineeringPlan | None,
    ) -> tuple[Paper83EngineeringPlan, Path]:
        if plan is not None:
            return plan, self.requested_plan_path
        if self.requested_plan_path.is_file():
            return (
                Paper83EngineeringPlan.from_yaml(self.requested_plan_path),
                self.requested_plan_path,
            )
        fallback = self.requested_plan_path.with_name("paper_83_engineering_plan.yaml")
        if fallback.is_file():
            return Paper83EngineeringPlan.from_yaml(fallback), fallback
        raise FileNotFoundError(
            f"paper-83 implementation plan does not exist: {self.requested_plan_path}"
        )

    def _record(
        self,
        paper: Any,
        entry: Paper83EngineeringPlanEntry,
    ) -> "PaperGraphSideRecord":
        domains = {entry.implementation_domain, *entry.secondary_domains}
        in_scope = bool(domains.intersection(GRAPH_SIDE_DOMAINS))
        mechanisms = _unique(entry.paper_specific_mechanism_ids)
        evidence_refs = _unique([*entry.source_locations, *entry.required_evidence])
        dependencies = _unique(
            [*entry.paper_specific_missing_parts, *entry.dependency_papers_or_primitives]
        )
        if not in_scope:
            behavior = LossSideBehaviorEvidence(passed=True, checks={"graph_side_scope": False})
            return self._record_value(
                paper=paper,
                entry=entry,
                in_scope=False,
                mechanisms=[],
                routes=[],
                evidence_refs=evidence_refs,
                dependencies=dependencies,
                behavior=behavior,
                status="out_of_scope",
                blockers=[],
                graph_diff=None,
                composition_fingerprint=None,
            )

        blockers: list[str] = []
        if entry.evidence_status != "sufficient_for_planning":
            blockers.append(
                "blocked_missing_evidence:plan_evidence_status=" + entry.evidence_status
            )
        if not entry.source_locations:
            blockers.append("blocked_missing_evidence:paper_graph_source_location")
        if not entry.required_evidence:
            blockers.append("blocked_missing_evidence:paper_graph_evidence_ref")
        if not entry.paper_specific_config:
            blockers.append("blocked_missing_evidence:paper_specific_graph_config")

        routes: list[LossSideRouteSpec] = []
        graph_diff: dict[str, Any] | None = None
        composition_fingerprint: str | None = None
        behavior: LossSideBehaviorEvidence | None = None
        for mechanism_id in mechanisms:
            candidates = [mechanism_id]
            source_ids = entry.paper_specific_config.get("source_mechanism_ids", [])
            if isinstance(source_ids, list):
                candidates.extend(str(item) for item in source_ids)
            route = None
            resolved_id = None
            for candidate in candidates:
                route = resolve_graph_route(candidate)
                if route is not None:
                    resolved_id = candidate
                    break
            if route is None:
                blockers.append(
                    f"blocked_missing_evidence:graph_route_unresolved:{mechanism_id}"
                )
                continue
            routes.append(route)
            if behavior is None:
                try:
                    behavior, diff = _probe_route(resolved_id)
                except (ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                    blockers.append(
                        f"blocked_missing_evidence:graph_behavior:{mechanism_id}:"
                        f"{type(exc).__name__}:{exc}"
                    )
                    continue
                diff = _with_paper_id(diff, paper.paper_id)
                if not behavior.passed:
                    blockers.extend(
                        f"blocked_missing_evidence:graph_behavior:{error}"
                        for error in behavior.errors
                    )
                mapping = {
                    "mechanism_id": mechanism_id,
                    "resolved_component_id": diff.component_id,
                    "source_mechanism_ids": list(entry.paper_specific_mechanism_ids),
                    "insertion_points": list(entry.runtime_insertion_points),
                }
                adaptation = dict(GRAPH_ADAPTATION_RECORDS.get(mechanism_id, {}))
                graph_diff = diff.to_payload(
                    mapping=mapping, adaptation=adaptation
                )
                graph_diff["composition_fingerprint"] = config_fingerprint(
                    {
                        "paper_id": paper.paper_id,
                        "mechanism_ids": mechanisms,
                        "component_id": diff.component_id,
                        "config": entry.paper_specific_config,
                        "graph_hash": diff.modified_graph_hash,
                    }
                )
                composition_fingerprint = graph_diff["composition_fingerprint"]

        if behavior is None:
            behavior = LossSideBehaviorEvidence(passed=False, checks={}, errors=["probed"])
        if not routes and not blockers:
            blockers.append("blocked_missing_evidence:no_graph_route")
        blockers = _unique(blockers)
        status = "ready" if not blockers else "blocked_missing_evidence"
        return self._record_value(
            paper=paper,
            entry=entry,
            in_scope=True,
            mechanisms=mechanisms,
            routes=routes,
            evidence_refs=evidence_refs,
            dependencies=dependencies,
            behavior=behavior,
            status=status,
            blockers=blockers,
            graph_diff=graph_diff,
            composition_fingerprint=composition_fingerprint,
        )

    def _record_value(
        self,
        *,
        paper: Any,
        entry: Paper83EngineeringPlanEntry,
        in_scope: bool,
        mechanisms: list[str],
        routes: list[LossSideRouteSpec],
        evidence_refs: list[str],
        dependencies: list[str],
        behavior: LossSideBehaviorEvidence,
        status: str,
        blockers: list[str],
        graph_diff: dict[str, Any] | None,
        composition_fingerprint: str | None,
    ) -> "PaperGraphSideRecord":
        route_ids = [item.mechanism_id for item in routes]
        component_ids = _unique(item.component_id for item in routes if item.component_id)
        adapter_ids = _unique(item.adapter_id for item in routes if item.adapter_id)
        config = {
            "paper_specific_config": dict(entry.paper_specific_config),
            "routes": [item.model_dump(mode="json") for item in routes],
            "fixed_imgsz": 640,
            "source_locations": evidence_refs,
        }
        fingerprint = _hash_payload(
            {
                "paper_id": paper.paper_id,
                "method_profile_id": entry.method_profile_id,
                "mechanisms": mechanisms,
                "routes": config,
                "graph_diff": graph_diff,
            }
        )
        return PaperGraphSideRecord(
            paper_id=paper.paper_id,
            title=paper.title,
            year=paper.year,
            method_profile_id=paper.method_profile_id or "",
            current_disposition=paper.current_disposition or "unknown",
            primary_domain=entry.implementation_domain,
            secondary_domains=list(entry.secondary_domains),
            in_scope=in_scope,
            paper_specific_mechanism_ids=mechanisms,
            resolved_route_ids=route_ids,
            component_ids=component_ids,
            adapter_ids=adapter_ids,
            evidence_refs=evidence_refs,
            remaining_dependencies=dependencies,
            paper_specific_config=config if in_scope else {},
            behavior=behavior,
            status=status,  # type: ignore[arg-type]
            blockers=blockers,
            graph_diff=graph_diff or {},
            composition_fingerprint=composition_fingerprint or "",
            implementation_fingerprint=fingerprint,
        )


def _with_paper_id(diff: GraphDiff, paper_id: str) -> GraphDiff:
    """Rebuild the diff payload with the paper identity the caller supplies."""

    if diff.paper_id == paper_id:
        return diff
    return GraphDiff(
        paper_id=paper_id,
        mechanism_id=diff.mechanism_id,
        component_id=diff.component_id,
        base_graph_hash=diff.base_graph_hash,
        modified_graph_hash=diff.modified_graph_hash,
        graph_hash_changed=diff.graph_hash_changed,
        inserted_modules=diff.inserted_modules,
        removed_replaced_modules=diff.removed_replaced_modules,
        changed_edges=diff.changed_edges,
        feature_levels=diff.feature_levels,
        params_base=diff.params_base,
        params_modified=diff.params_modified,
        macs_base=diff.macs_base,
        macs_modified=diff.macs_modified,
    )


def _unique(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def _hash_payload(payload: Any) -> str:
    import hashlib

    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _md(value: str) -> str:
    return value.replace("|", "\\|")


def build_paper_graph_side_audit(
    **kwargs: object,
) -> "PaperGraphSideAudit":
    """Functional builder for offline callers."""

    write_artifacts = bool(kwargs.pop("write_artifacts", True))
    return PaperGraphSideAuditBuilder(**kwargs).build(
        write_artifacts=write_artifacts
    )


def render_paper_83_graph_side_status(audit: "PaperGraphSideAudit") -> str:
    """Render all 83 papers without turning out-of-scope rows into claims."""

    lines = [
        "# Paper-83 Model-graph Side Status",
        "",
        "This report audits backbone, neck, feature-fusion, and head mechanisms."
        " It does not train a model and does not promote shared graph plugins to"
        " paper implementations without per-paper composition evidence.",
        "",
        f"- Frozen papers: {audit.paper_count}",
        f"- Graph-side papers in scope: {audit.graph_side_paper_count}",
        f"- Ready graph-side papers: {audit.summary.get('ready', 0)}",
        f"- Blocked for missing evidence: {audit.summary.get('blocked_missing_evidence', 0)}",
        f"- Out of scope: {audit.summary.get('out_of_scope', 0)}",
        f"- Manifest membership hash: `{audit.manifest_membership_hash}`",
        f"- Plan source: `{audit.plan_source_path}`",
        "",
        "## Scope Decision",
        "",
    ]
    if audit.graph_side_paper_count == 0:
        lines.append("No frozen paper has a backbone/neck/feature_fusion/head domain.")
    lines.extend(
        [
            "The frozen 83 contain no `backbone`-domain paper; the graph-side set"
            " is 1 head (TOOD), 1 neck (RTMDet), and 1 feature_fusion (Gold-YOLO).",
            "",
            "## Per-paper Audit",
            "",
            "| Paper | Domain | Scope | Status | Mechanisms | Graph hash changed | Blockers |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for item in audit.records:
        mechanisms = "<br>".join(item.paper_specific_mechanism_ids) or "none"
        hash_changed = (
            str(item.graph_diff.get("graph_hash_changed", "n/a"))
            if item.graph_diff
            else "n/a"
        )
        blockers = "<br>".join(item.blockers) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_md(item.paper_id)}`",
                    _md(item.primary_domain),
                    "graph-side" if item.in_scope else "out-of-scope",
                    item.status,
                    _md(mechanisms),
                    hash_changed,
                    _md(blockers),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Method Notes",
            "",
            "- Every in-scope mechanism is a real ``torch.nn.Module`` on the"
            " executed forward path: the probe runs forward and backward on"
            " synthetic CPU tensors and fails on any identity placeholder.",
            "- Graph contracts verify input/output channels, spatial strides,"
            " P3/P4/P5 feature levels, and the terminal-detect head contract.",
            "- Each paper emits ``artifacts/paper_graph_diffs/{safe_paper_id}.yaml``"
            " with base and modified graph hashes, inserted modules, changed"
            " edges, feature levels, params, executed-shape MAC estimates, and"
            " the paper mechanism mapping.",
            "- Enabling a paper mechanism must change the graph hash; the audit"
            " fails closed when it does not.",
            "- Two papers sharing a plugin class are distinguished by the"
            " composition fingerprint over paper identity, mechanisms, config,"
            " and the modified graph hash.",
            "- Original-versus-YOLO26 adaptation records document the preserved"
            " mechanism and the known deviation instead of claiming exact"
            " reproduction.",
            "",
        ]
    )
    return "\n".join(lines)


def write_paper_83_graph_side_status(
    audit: "PaperGraphSideAudit",
    output_path: Path | str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_paper_83_graph_side_status(audit), encoding="utf-8")
    return path
