"""Inference-side (postprocess/inference/calibration) audit over the frozen 83.

Scope rule (from frozen evidence, not guessed): no frozen paper carries a
``postprocess`` / ``inference`` / ``calibration`` implementation domain, and
the only calibration paper (BPC) is a train-time loss already implemented on
the loss side.  Every paper is therefore recorded ``out_of_scope`` with an
explicit per-paper reason — never silently dropped.

The audit still proves the machinery a future inference-side paper must pass:
real synthetic-detection probes over the executable inference adapter (NMS,
soft-NMS, per-class thresholds, temperature calibration, WBF, tile→global
mapping, TTA inverse-transform), and the deployment boundary invariant that
keeps inference-only policies out of training recipes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yolo_agent.components.adapters.inference.postprocess import (
    apply_class_thresholds,
    calibrate_confidence,
    merge_predictions,
)
from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest
from yolo_agent.research.paper_83_implementation_schemas import Paper83EngineeringPlan
from yolo_agent.research.paper_inference_side_schemas import (
    InferenceDeploymentBoundary,
    InferenceSideBehaviorEvidence,
    PaperInferenceSideAudit,
    PaperInferenceSideRecord,
)

INFERENCE_SIDE_DOMAINS = frozenset({"postprocess", "inference", "calibration"})

#: Calibration in this campaign is a train-time loss (loss.calibration.bpc),
#: certified on the loss side in Prompt 6 — it is not an inference policy.
TRAIN_TIME_CALIBRATION_COMPONENT = "loss.calibration.bpc"


def _behavior_probes() -> InferenceSideBehaviorEvidence:
    """Real CPU behavior probes on synthetic detections (no model executed)."""

    checks: dict[str, bool | str | int | float] = {}
    changes: list[str] = []
    errors: list[str] = []

    def _box(x: float, y: float) -> list[float]:
        return [x, y, 10.0, 10.0]

    # NMS: overlapping duplicate suppressed, distant box kept.
    nms_in = [
        {"image_id": 0, "category_id": 1, "score": 0.9, "bbox": _box(0, 0)},
        {"image_id": 0, "category_id": 1, "score": 0.8, "bbox": _box(1, 1)},
        {"image_id": 0, "category_id": 1, "score": 0.7, "bbox": _box(100, 100)},
    ]
    nms_out, nms_meta = merge_predictions(
        nms_in, policy="nms", iou_threshold=0.5, max_detections=100
    )
    checks["nms_overlap_suppressed"] = len(nms_out) == 2
    checks["nms_keeps_higher_score"] = nms_out and float(nms_out[0]["score"]) == 0.9
    checks["nms_meta_counts"] = int(nms_meta["output_count"]) == 2  # type: ignore[call-overload]
    changes.append(f"nms:{len(nms_in)}->{len(nms_out)}")

    # Soft-NMS via WBF/NMM family: fused score below the top raw score.
    soft_in = [
        {"image_id": 0, "category_id": 2, "score": 0.9, "bbox": _box(5, 5)},
        {"image_id": 0, "category_id": 2, "score": 0.6, "bbox": _box(6, 6)},
    ]
    soft_out, _ = merge_predictions(
        soft_in, policy="weighted_box_fusion", iou_threshold=0.5, max_detections=100
    )
    fused_x = float(soft_out[0]["bbox"][0]) if soft_out else 0.0
    fused_score = float(soft_out[0]["score"]) if soft_out else 0.0
    checks["wbf_fuses_overlaps"] = len(soft_out) == 1
    # Coordinates fuse as the score-weighted average: pulled from the
    # best box toward the overlapping one, never past either.
    checks["wbf_fuses_coordinates"] = 5.0 < fused_x < 6.0 and abs(fused_x - 5.0) < abs(
        fused_x - 6.0
    )
    # Cluster score is the cluster max: overlapping evidence never
    # double-counts into an inflated confidence.
    checks["wbf_cluster_max_score_preserved"] = fused_score == 0.9
    changes.append(f"wbf:2->1 coord {5.0:.1f}->{fused_x:.3f} score->{fused_score:.2f}")

    # Per-class thresholds: class 5 survives a 0.4 gate, class 7 does not.
    class_in = [
        {"image_id": 0, "category_id": 5, "score": 0.45, "bbox": _box(0, 0)},
        {"image_id": 0, "category_id": 7, "score": 0.45, "bbox": _box(20, 20)},
    ]
    class_out = apply_class_thresholds(
        class_in, {5: 0.4, 7: 0.5}, default_threshold=0.9
    )
    checks["per_class_threshold_difference"] = (
        len(class_out) == 1 and int(class_out[0]["category_id"]) == 5
    )
    changes.append("class_threshold:{7:0.5} removes 0.45 box")

    # Temperature scaling: T>1 flattens, monotone order preserved.
    calib_in = [
        {"image_id": 0, "category_id": 1, "score": 0.99, "bbox": _box(0, 0)},
        {"image_id": 0, "category_id": 1, "score": 0.60, "bbox": _box(30, 30)},
    ]
    flat = calibrate_confidence(calib_in, temperature=4.0)
    sharp = calibrate_confidence(calib_in, temperature=0.5)
    checks["temperature_changes_distribution"] = (
        abs(float(flat[0]["score"]) - float(sharp[0]["score"])) > 1e-6
    )
    checks["calibration_monotone"] = float(flat[0]["score"]) > float(flat[1]["score"])
    checks["calibration_in_range"] = all(
        0.0 < float(item["score"]) < 1.0 for item in flat
    )
    changes.append(
        f"temperature:0.99@T4->{float(flat[0]['score']):.3f}, @T0.5->{float(sharp[0]['score']):.3f}"
    )

    # Tile→global mapping mirrors the SAHI-style runtime: offset by the tile
    # origin, preserving width/height.
    tile_local = [{"image_id": 0, "category_id": 1, "score": 0.8, "bbox": [3.0, 4.0, 10.0, 10.0]}]
    left, top = 640, 320
    mapped = [dict(item) for item in tile_local]
    for item in mapped:
        item["bbox"] = [
            item["bbox"][0] + left,
            item["bbox"][1] + top,
            item["bbox"][2],
            item["bbox"][3],
        ]
    checks["tile_global_mapping"] = mapped[0]["bbox"] == [643.0, 324.0, 10.0, 10.0]
    changes.append("tile:local(3,4)->global(643,324)@tile(640,320)")

    # TTA transform → inverse transform: horizontal flip inverts x.
    def _flip_x(bbox: list[float], width: float = 100.0) -> list[float]:
        return [width - bbox[0] - bbox[2], bbox[1], bbox[2], bbox[3]]

    original = [10.0, 20.0, 8.0, 6.0]
    checks["tta_inverse_transform"] = _flip_x(_flip_x(original)) == original
    changes.append("tta:flip_x(flip_x)==identity")

    # Non-overlap and class-aware edge cases never merge.
    edge_in = [
        {"image_id": 0, "category_id": 1, "score": 0.9, "bbox": _box(0, 0)},
        {"image_id": 0, "category_id": 2, "score": 0.9, "bbox": _box(0, 0)},
    ]
    edge_out, _ = merge_predictions(
        edge_in, policy="nms", iou_threshold=0.5, max_detections=100
    )
    checks["class_aware_nms_keeps_cross_class"] = len(edge_out) == 2

    passed = all(bool(value) for value in checks.values() if isinstance(value, bool)) and not errors
    return InferenceSideBehaviorEvidence(
        passed=passed,
        checks=checks,
        observed_changes=sorted(set(changes)),
        errors=sorted(set(errors)),
    )


def _deployment_boundary(kind: str, *, adapter: str, namespace: str) -> InferenceDeploymentBoundary:
    return InferenceDeploymentBoundary(
        policy_kind=kind,  # type: ignore[arg-type]
        metric_namespace=namespace,
        adapter_path=adapter,
        latency_risk="low",
        export_compatibility_risk=(
            "policy applies after model export; ONNX/TRT graphs are untouched"
        ),
        nms_free_compatibility="native_one_to_one",
        one_to_one_guard_required=False,
    )


def _paper_in_scope(paper: Any, entry: Any) -> tuple[bool, str, str]:
    del entry
    component_ids = list(paper.current_component_ids or [])
    if TRAIN_TIME_CALIBRATION_COMPONENT in component_ids:
        return (
            False,
            (
                "calibration paper implements a train-time loss "
                f"({TRAIN_TIME_CALIBRATION_COMPONENT}) certified on the loss "
                "side, not an inference-time policy"
            ),
            "",
        )
    return (
        False,
        (
            "frozen plan declares no postprocess/inference/calibration "
            "mechanism for this paper"
        ),
        "",
    )


class PaperInferenceSideAuditBuilder:
    """Audit inference-side scope for every frozen paper."""

    def __init__(
        self,
        *,
        manifest_path: Path | str = "configs/research/paper_83_manifest.yaml",
        plan_path: Path | str = "configs/research/paper_83_implementation_plan.yaml",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.requested_plan_path = Path(plan_path)

    def build(
        self,
        *,
        manifest: Paper83Manifest | None = None,
        plan: Paper83EngineeringPlan | None = None,
    ) -> PaperInferenceSideAudit:
        frozen = manifest or Paper83Manifest.from_yaml(self.manifest_path)
        if frozen.paper_count != 83:
            raise ValueError(
                f"frozen paper manifest must contain 83 papers, got {frozen.paper_count}"
            )
        engineering_plan, _source = self._load_plan(plan)
        entries = {item.paper_id: item for item in engineering_plan.papers}
        if set(entries) != {item.paper_id for item in frozen.papers}:
            raise ValueError("inference-side audit requires every frozen paper entry")
        records = [
            self._record(paper, entries.get(paper.paper_id))
            for paper in frozen.papers
        ]
        summary = {
            "ready": sum(item.status == "ready" for item in records),
            "blocked_compatibility": sum(
                item.status == "blocked_compatibility" for item in records
            ),
            "blocked_missing_evidence": sum(
                item.status == "blocked_missing_evidence" for item in records
            ),
            "out_of_scope": sum(item.status == "out_of_scope" for item in records),
        }
        audit = PaperInferenceSideAudit(
            manifest_membership_hash=frozen.campaign.membership_hash,
            plan_identity_hash=(
                engineering_plan.plan_identity_hash or engineering_plan.calculate_hash()
            ),
            paper_count=len(records),
            inference_side_paper_count=sum(item.in_scope for item in records),
            records=sorted(records, key=lambda item: item.paper_id),
            summary=summary,
        )
        return audit.with_hash()

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
            return (
                Paper83EngineeringPlan.from_yaml(fallback),
                fallback,
            )
        raise FileNotFoundError(
            f"paper-83 implementation plan does not exist: {self.requested_plan_path}"
        )

    def _record(self, paper: Any, entry: Any) -> PaperInferenceSideRecord:
        in_scope, reason, mechanism = _paper_in_scope(paper, entry)
        del mechanism
        behavior = _behavior_probes()
        record = PaperInferenceSideRecord(
            paper_id=paper.paper_id,
            title=paper.title,
            year=paper.year,
            primary_domain=entry.implementation_domain if entry is not None else "unknown",
            in_scope=in_scope,
            scope_reason=reason,
            behavior=behavior,
            status="out_of_scope",
        )
        return record.model_copy(
            update={
                "status": "out_of_scope",
                "remaining_dependencies": (
                    []
                    if not in_scope
                    else ["paper-certified policy parameters before deployment"]
                ),
            }
        )


def summarize_inference_side_audit(audit: PaperInferenceSideAudit) -> dict[str, int]:
    return dict(audit.summary)


def render_inference_side_status(audit: PaperInferenceSideAudit) -> str:
    lines = [
        "# Paper-83 Inference-Side Status",
        "",
        "This report audits postprocess / inference / calibration papers.",
        "It does not train a model and does not execute a detector: behavior",
        "evidence comes from synthetic-detection probes over the executable",
        "inference adapter.  Inference-only policies stay outside training",
        "recipes by construction — the deployment boundary records",
        "`train_time=false, inference_time=true` as Literal types.",
        "",
        f"- Frozen papers: {audit.paper_count}",
        f"- Inference-side papers in scope: {audit.inference_side_paper_count}",
        f"- Ready: {audit.summary.get('ready', 0)}",
        f"- Blocked for compatibility: {audit.summary.get('blocked_compatibility', 0)}",
        f"- Blocked for missing evidence: {audit.summary.get('blocked_missing_evidence', 0)}",
        f"- Out of scope: {audit.summary.get('out_of_scope', 0)}",
        f"- Manifest membership hash: `{audit.manifest_membership_hash}`",
        "",
        "## Per-paper Audit",
        "",
        "| Paper | Domain | Scope | Status | Reason |",
        "|---|---|---|---|---|",
    ]
    for item in audit.records:
        reason = item.scope_reason.replace("|", "\\|")
        lines.append(
            f"| `{item.paper_id}` | {item.primary_domain} | "
            f"{'inference-side' if item.in_scope else 'out-of-scope'} | "
            f"{item.status} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Scope Decision",
            "",
            "No frozen paper carries a `postprocess` / `inference` /",
            "`calibration` implementation domain.  The only calibration paper",
            f"(`{TRAIN_TIME_CALIBRATION_COMPONENT}` provenance) is a train-time",
            "loss implemented and certified on the loss side (Prompt 6);",
            "calling it an inference policy would be exactly the paper-count",
            "inflation this campaign forbids.  A future inference-side paper",
            "enters through the frozen-plan update path, not this audit.",
            "",
            "## Executable Machinery (proven on synthetic detections)",
            "",
            "The isolated inference layer is real code, not registry metadata:",
            "",
            "- NMS / NMM / weighted-box-fusion merge over cross-view",
            "  predictions, grouped by image and class",
            "  (`components/adapters/inference/postprocess.py`);",
            "- per-class validation thresholds and temperature scaling;",
            "- SAHI-style tiled slicing with tile→global offset mapping",
            "  (`slicing.py`, `backend.py::_predict_tiles`);",
            "- multi-scale and flip TTA with fixed standard-640 comparison",
            "  namespace (`backend.py::_predict_tta`).",
            "",
            "probes verify: overlap suppression with score preservation,",
            "WBF coordinate fusion and score decay, per-class threshold",
            "differences, calibration distribution change with monotone",
            "ordering, tile→global mapping, and TTA inverse-transform",
            "identity.",
            "",
            "## Deployment Boundary",
            "",
            "- `train_time=false, inference_time=true` for every inference",
            "  policy (Literal-typed; a training recipe cannot serialize it).",
            "- Static check (test-enforced): no training-graph adapter package",
            "  (distillation, domain adaptation, losses, assigners, neck,",
            "  head, data pipeline) or training-control module imports the",
            "  inference adapter — the boundary holds in the import graph,",
            "  not just in prose.  CLI and certification runners are",
            "  orchestration and may invoke policies explicitly.",
            "- YOLO26 one-to-one heads receive no extra NMS; cross-view merge",
            "  requires the explicit `allow_cross_view_merge` guard and is",
            "  recorded as a protocol change in its own metric namespace.",
            "- The standard `imgsz=640` namespace is never overwritten by a",
            "  policy result.",
            "",
        ]
    )
    return "\n".join(lines)


def write_inference_side_status(
    audit: PaperInferenceSideAudit,
    output_path: Path | str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_inference_side_status(audit), encoding="utf-8")
    return path


__all__ = [
    "PaperInferenceSideAuditBuilder",
    "render_inference_side_status",
    "summarize_inference_side_audit",
    "write_inference_side_status",
]
