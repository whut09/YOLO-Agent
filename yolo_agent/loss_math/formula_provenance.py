"""Formula provenance for paper-specific detection losses.

Every paper-specific loss that this campaign implements must point at an
explicit mathematical definition: the paper, the section/equation anchor,
the parameter semantics, and the formula family.  Two formula families are
never aliases of each other: a mechanism whose family differs from the
native CIoU family must not be registered as a CIoU weight change, and the
registry fails closed when a family claims an incompatible signature.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


FormulaFamily = Literal[
    "pseudo_iou",
    "mutual_supervision",
    "concordance_correlation",
    "precision_confidence_quadrant",
    "task_aligned_matching",
    "entropic_optimal_transport",
    "dynamic_smooth_label",
]

# Families that are genuinely distinct CIoU-family variants.  A paper
# mechanism registered with one of these families must not be implemented as
# a plain IoU/CIoU with a different box weight; the distinct-term check in
# ``validate_no_loss_alias`` enforces the distinguishing terms.
DISTINCT_IOU_FAMILIES: frozenset[str] = frozenset(
    {"pseudo_iou", "mutual_supervision"}
)

# Families whose geometry must not be reduced to native CIoU geometry.
NON_CIOU_FAMILIES: frozenset[str] = frozenset(
    {
        "concordance_correlation",
        "precision_confidence_quadrant",
        "task_aligned_matching",
        "entropic_optimal_transport",
        "dynamic_smooth_label",
    }
)


class FormulaProvenance(BaseModel):
    """Explicit, auditable mathematical identity for one paper loss."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str
    mechanism_id: str
    formula_family: FormulaFamily
    formula: str
    source_anchor: str
    parameter_semantics: dict[str, str] = Field(default_factory=dict)
    shared_primitives: tuple[str, ...] = ()
    distinguishing_terms: tuple[str, ...] = ()
    adaptation_class: Literal["faithful_adaptation", "exact_reproduction"] = (
        "faithful_adaptation"
    )
    loss_plugin_name: str | None = None

    @model_validator(mode="after")
    def validate_provenance(self) -> "FormulaProvenance":
        if not self.formula.strip():
            raise ValueError(f"{self.mechanism_id}: formula text is required")
        if not self.source_anchor.strip():
            raise ValueError(f"{self.mechanism_id}: source anchor is required")
        if not self.parameter_semantics and self.formula_family in {
            "entropic_optimal_transport",
            "precision_confidence_quadrant",
        }:
            raise ValueError(
                f"{self.mechanism_id}: transport/quadrant losses must document "
                "their parameter semantics"
            )
        if self.formula_family in NON_CIOU_FAMILIES and not self.distinguishing_terms:
            raise ValueError(
                f"{self.mechanism_id}: family {self.formula_family} must declare "
                "its distinguishing terms so it cannot collapse into CIoU"
            )
        return self

    @property
    def implementation_name(self) -> str:
        """Concrete plugin implementing this mechanism (or the family name)."""

        return self.loss_plugin_name or self.formula_family


LOSS_FORMULA_PROVENANCE: dict[str, FormulaProvenance] = {
    item.mechanism_id: item
    for item in (
        FormulaProvenance(
            paper_id="arxiv:2104.14082",
            mechanism_id="loss.quality.pseudo_iou",
            formula_family="pseudo_iou",
            formula=(
                "B = anchor point p expanded to a pseudo box of the target "
                "size; L = BCEWithLogits(z, IoU(B, G)) with G the target box, "
                "z the true-class logit"
            ),
            source_anchor="arxiv:2104.14082 Sec. 3.1, pseudo-IoU definition",
            parameter_semantics={
                "pseudo_box": "anchor point treated as box center with target w/h",
                "z": "true-class logit of the matched location",
            },
            shared_primitives=("geometry.elementwise_iou",),
            distinguishing_terms=("anchor_pseudo_box_geometry",),
            loss_plugin_name="pseudo_iou",
        ),
        FormulaProvenance(
            paper_id="arxiv:2109.05986",
            mechanism_id="loss.2109_05986",
            formula_family="mutual_supervision",
            formula=(
                "L = 0.5 * ( BCEWithLogits(z, sg(IoU(P, G))) "
                "+ SmoothL1(IoU(P, G), sg(sigmoid(z))) ) with stop-gradient sg "
                "on the opposite branch in each term"
            ),
            source_anchor=(
                "arxiv:2109.05986 Sec. 3.2, mutual supervision between "
                "classification and localization branches"
            ),
            parameter_semantics={
                "z": "true-class logit of the matched location",
                "P": "predicted box decoded in xyxy",
                "G": "target box assigned by the native assigner",
            },
            shared_primitives=("geometry.elementwise_iou",),
            distinguishing_terms=("cross_branch_stop_gradient",),
            loss_plugin_name="mutual_supervision",
        ),
        FormulaProvenance(
            paper_id="arxiv:2301.01019",
            mechanism_id="loss.quality.correlation",
            formula_family="concordance_correlation",
            formula=(
                "L = 1 - CCC(s, q) where CCC = 2*Cov(s, q) / "
                "(Var(s) + Var(q) + (E[s]-E[q])^2 + eps), s = sigmoid(z), "
                "q = detached matched IoU"
            ),
            source_anchor=(
                "arxiv:2301.01019 Eq. 1-4, Lin concordance correlation "
                "coefficient over positives"
            ),
            parameter_semantics={
                "s": "sigmoid true-class confidence over positives",
                "q": "detached matched IoU serving as localization quality",
            },
            shared_primitives=("geometry.elementwise_iou",),
            distinguishing_terms=(
                "batch_moment_alignment",
                "no_box_regression_gradient",
            ),
            loss_plugin_name="correlation",
        ),
        FormulaProvenance(
            paper_id="arxiv:2303.14404",
            mechanism_id="loss.calibration.bpc",
            formula_family="precision_confidence_quadrant",
            formula=(
                "L = log1p(Bad / (Good + eps)); Good = sum[tanh(c) on "
                "accurate-confident + tanh(1-c) on inaccurate-unconfident]; "
                "Bad = sum[tanh(1-c) on accurate-unconfident + tanh(c) on "
                "inaccurate-confident]"
            ),
            source_anchor=(
                "arxiv:2303.14404 Eq. 5-7, precision/confidence quadrant "
                "counts with differentiable soft counting"
            ),
            parameter_semantics={
                "c": "max-class confidence of a candidate",
                "accurate": "predicted class correct and matched IoU >= tau_iou",
                "confident": "c >= tau_conf",
                "tau_conf": "confidence quadrant threshold (default 0.5)",
                "tau_iou": "localization accuracy threshold (default 0.5)",
            },
            shared_primitives=("geometry.elementwise_iou",),
            distinguishing_terms=("soft_quadrant_counts",),
            loss_plugin_name="bpc_calibration",
        ),
    )
}


class AssignmentFormulaProvenance(BaseModel):
    """Explicit mathematical identity for one paper assignment mechanism."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str
    mechanism_id: str
    formula_family: FormulaFamily
    formula: str
    source_anchor: str
    parameter_semantics: dict[str, str] = Field(default_factory=dict)
    shared_primitives: tuple[str, ...] = ()
    distinguishing_terms: tuple[str, ...] = ()
    adaptation_class: Literal["faithful_adaptation", "exact_reproduction"] = (
        "faithful_adaptation"
    )
    original_contract: str
    yolo26_contract: str
    adapter_transform: str
    information_preserved: tuple[str, ...] = ()
    information_approximated: tuple[str, ...] = ()
    assigner_plugin_method: str | None = None

    @model_validator(mode="after")
    def validate_assignment_provenance(self) -> "AssignmentFormulaProvenance":
        if not self.distinguishing_terms:
            raise ValueError(
                f"{self.mechanism_id}: assignment provenance must declare "
                "distinguishing terms"
            )
        if not self.original_contract.strip() or not self.yolo26_contract.strip():
            raise ValueError(
                f"{self.mechanism_id}: original and YOLO26 contracts are required"
            )
        if not self.adapter_transform.strip():
            raise ValueError(f"{self.mechanism_id}: adapter transform is required")
        if not self.information_preserved:
            raise ValueError(
                f"{self.mechanism_id}: preserved information must be recorded"
            )
        return self

    @property
    def implementation_method(self) -> str:
        """Concrete assigner method implementing this mechanism."""

        return self.assigner_plugin_method or self.formula_family


ASSIGNMENT_FORMULA_PROVENANCE: dict[str, AssignmentFormulaProvenance] = {
    item.mechanism_id: item
    for item in (
        AssignmentFormulaProvenance(
            paper_id="arxiv:2103.14259",
            mechanism_id="assigner.optimal_transport",
            formula_family="entropic_optimal_transport",
            formula=(
                "min_T <T, C> + eps * sum T log T s.t. T 1 = n (positive "
                "supply per GT), T^T 1 = m (background supply); C = -log p_cls "
                "+ lambda * (1 - IoU) + rho * outside-center penalty; "
                "positives are the top-n transport columns per GT row"
            ),
            source_anchor=(
                "arxiv:2103.14259 Eq. 1-3 and Sec. 3.2, Sinkhorn-Hamming "
                "transport with dynamic k"
            ),
            parameter_semantics={
                "lambda": "regression_weight applied to 1 - IoU cost",
                "eps": "sinkhorn_epsilon entropic regularization",
                "rho": "center_penalty for points outside the GT box",
                "n": "dynamic positive supply from IoU-ranked top-k",
            },
            shared_primitives=("geometry.pairwise_iou",),
            distinguishing_terms=(
                "global_transport_supply",
                "background_column_cost",
                "sinkhorn_row_scaling",
            ),
            assigner_plugin_method="ota",
            original_contract=(
                "OTA end-to-end detector: OT plan over anchor boxes shared by "
                "RPN-style heads with unified loss on the transport support"
            ),
            yolo26_contract=(
                "YOLO26 point anchors, per-point fixed positive supply, "
                "one_to_many path only; native loss and head unchanged"
            ),
            adapter_transform=(
                "Transport is recomputed per image over YOLO26 anchor points; "
                "the resulting matches are converted to the native "
                "(labels, boxes, scores, fg mask, gt indices) tuple"
            ),
            information_preserved=(
                "classification-regression joint cost",
                "dynamic positive supply per GT",
                "global competition with background column",
            ),
            information_approximated=(
                "end-to-end OTA head coupling is dropped (assignment only)",
                "loss reweighting by transport plan is replaced by native loss",
            ),
        ),
        AssignmentFormulaProvenance(
            paper_id="arxiv:2203.16250",
            mechanism_id="assigner.task_aligned",
            formula_family="task_aligned_matching",
            formula=(
                "t = s^alpha * u^beta; positives are the top-k points of the "
                "aligned metric t inside each GT; target scores equal the "
                "normalized alignment t / max(t)"
            ),
            source_anchor=(
                "arxiv:2203.16250 (PP-YOLOE) Sec. 3.2, TAL with alpha=1.0, "
                "beta=6.0, topk=13, alignment-normalized soft targets"
            ),
            parameter_semantics={
                "s": "predicted class score of the GT class",
                "u": "predicted IoU with the GT",
                "alpha": "classification exponent (1.0)",
                "beta": "localization exponent (6.0)",
                "k": "topk candidate count (13)",
            },
            shared_primitives=("geometry.pairwise_iou",),
            distinguishing_terms=(
                "alignment_metric_product",
                "alignment_normalized_soft_targets",
            ),
            assigner_plugin_method="tood_tal",
            original_contract=(
                "PP-YOLOE anchor-free head with TAL assignment and "
                "task-aligned soft classification targets"
            ),
            yolo26_contract=(
                "YOLO26 point anchors and native head; TAL used on the "
                "one_to_many path with PP-YOLOE hyperparameters"
            ),
            adapter_transform=(
                "TAL metric evaluated over YOLO26 anchor points; soft targets "
                "written into the native target-score tensor"
            ),
            information_preserved=(
                "alignment metric product form",
                "top-k positive selection",
                "PP-YOLOE alpha/beta/topk values",
            ),
            information_approximated=(
                "PP-YOLOE head architecture and auxiliary ATSS assigner are "
                "not reproduced",
            ),
        ),
        AssignmentFormulaProvenance(
            paper_id="arxiv:2208.00817",
            mechanism_id="assigner.dynamic_smooth_label",
            formula_family="dynamic_smooth_label",
            formula=(
                "q = 1[dist_l >= -delta*s] * centerness(d, s) * IoU_online; "
                "centerness uses the relaxed interval around the GT boundary "
                "at stride s; positives are argmax q per GT region"
            ),
            source_anchor=(
                "arxiv:2208.00817 Eq. 4-6, interval-based scale prior with "
                "online IoU quality targets"
            ),
            parameter_semantics={
                "delta": "interval_relaxation of the FCOS boundary (0.2)",
                "s": "per-point stride used by the interval score",
                "d": "point-to-GT-side distances",
            },
            shared_primitives=("geometry.pairwise_iou",),
            distinguishing_terms=(
                "stride_relaxed_interval_score",
                "online_iou_quality_target",
            ),
            assigner_plugin_method="dsla",
            original_contract=(
                "DSLA adapts FCOS-scale interval priors on anchor-free centers "
                "with dynamic smooth classification targets"
            ),
            yolo26_contract=(
                "YOLO26 point anchors with per-point strides; native head and "
                "loss unchanged, one_to_many path only"
            ),
            adapter_transform=(
                "FCOS box intervals are recomputed from YOLO26 point-stride "
                "priors; smooth quality becomes the native target score"
            ),
            information_preserved=(
                "stride-conditioned interval prior",
                "centerness factor",
                "online IoU quality target",
            ),
            information_approximated=(
                "DSLA's scale-varying classification head is not reproduced",
            ),
        ),
    )
}


def loss_formula_provenance() -> dict[str, FormulaProvenance]:
    """Return a copy of the loss provenance registry."""

    return dict(LOSS_FORMULA_PROVENANCE)


def assignment_formula_provenance() -> dict[str, AssignmentFormulaProvenance]:
    """Return a copy of the assignment provenance registry."""

    return dict(ASSIGNMENT_FORMULA_PROVENANCE)


def validate_no_loss_alias(mechanism_id: str, implementation: object) -> None:
    """Fail closed when a registered family is implemented as a CIoU alias.

    A CIoU alias here means: the implementation exposes no distinguishing
    attribute of its declared family and therefore only the box weight could
    differ.  Implementations expose their distinguishing structure via a
    ``DISTINGUISHING_TERMS`` class attribute; shared geometry helpers (IoU,
    smooth L1, BCE) are explicitly allowed and never count as aliases.
    """

    provenance = LOSS_FORMULA_PROVENANCE.get(mechanism_id)
    if provenance is None:
        raise KeyError(f"unregistered loss mechanism: {mechanism_id}")
    declared = set(provenance.distinguishing_terms)
    if not declared:
        raise ValueError(
            f"{mechanism_id}: provenance must declare distinguishing terms"
        )
    exposed = set(getattr(implementation, "DISTINGUISHING_TERMS", ()))
    if not declared & exposed:
        raise ValueError(
            f"{mechanism_id}: implementation exposes none of the distinguishing "
            f"terms {sorted(declared)}; refusing to treat it as a new loss"
        )


__all__ = [
    "ASSIGNMENT_FORMULA_PROVENANCE",
    "DISTINCT_IOU_FAMILIES",
    "LOSS_FORMULA_PROVENANCE",
    "NON_CIOU_FAMILIES",
    "AssignmentFormulaProvenance",
    "FormulaFamily",
    "FormulaProvenance",
    "assignment_formula_provenance",
    "loss_formula_provenance",
    "validate_no_loss_alias",
]
