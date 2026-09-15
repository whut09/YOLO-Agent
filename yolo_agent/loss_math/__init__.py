"""Loss and assignment mathematics with explicit formula provenance.

This package never trains a model.  It exists so that every paper-specific
loss and assignment mechanism in the frozen campaign carries an auditable
mathematical identity and can be property-tested against degenerate inputs.
"""

from yolo_agent.loss_math.formula_provenance import (
    ASSIGNMENT_FORMULA_PROVENANCE,
    LOSS_FORMULA_PROVENANCE,
    AssignmentFormulaProvenance,
    FormulaProvenance,
    assignment_formula_provenance,
    loss_formula_provenance,
    validate_no_loss_alias,
)
from yolo_agent.loss_math.loss_math_matrix import (
    LOSS_MATH_CASES,
    LossMathCase,
    evaluate_loss_math_matrix,
)
from yolo_agent.loss_math.tal_reference import (
    TALReferenceParams,
    tal_reference_assignment,
)

__all__ = [
    "ASSIGNMENT_FORMULA_PROVENANCE",
    "LOSS_FORMULA_PROVENANCE",
    "AssignmentFormulaProvenance",
    "FormulaProvenance",
    "LOSS_MATH_CASES",
    "LossMathCase",
    "TALReferenceParams",
    "assignment_formula_provenance",
    "evaluate_loss_math_matrix",
    "loss_formula_provenance",
    "tal_reference_assignment",
    "validate_no_loss_alias",
]
