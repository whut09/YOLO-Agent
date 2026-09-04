"""Research and paper-intelligence schemas for the YOLO Agent."""

from yolo_agent.research.schemas import (
    Applicability,
    BenchmarkEvidenceLevel,
    ComponentCategory,
    ComponentTaxonomy,
    EvidenceLevel,
    PaperBenchmark,
    PaperComponentClaim,
    PaperRecord,
    PaperProvenance,
)
from yolo_agent.research.awesome_catalog_importer import (
    AwesomeCatalogImporter,
    PaperImportResult,
    import_awesome_catalog,
)
from yolo_agent.research.awesome_snapshot_builder import (
    AwesomeSnapshotBuildResult,
    AwesomeSnapshotBuilder,
    AwesomeSourceManifest,
)
from yolo_agent.research.component_aliases import (
    ComponentAliasConfig,
    ComponentAliasResolution,
    ComponentAliasResolver,
    ResolvedComponentAlias,
)
from yolo_agent.research.component_coverage import ComponentCoverageAnalyzer, ComponentCoverageReport
from yolo_agent.research.harness_hint_parser import (
    HarnessHintParseResult,
    HarnessHintParser,
    PaperDiagnosticHint,
)
from yolo_agent.research.note_parser import (
    PaperAblationHint,
    PaperEvidenceClaim,
    PaperEvidenceSummary,
    PaperLimitation,
    PaperMethodClaim,
    PaperNoteParser,
)
from yolo_agent.research.method_profiles import (
    ImplementationDecisionKind,
    PaperImplementationDecision,
    PaperMethodCoverageReport,
    PaperMethodProfile,
    PaperMethodProfileBuilder,
)
from yolo_agent.research.paper_index import PaperIndex
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.paper_execution_inventory import (
    GENERIC_COMPONENT_IDS,
    PaperExecutionInventoryBuilder,
    render_paper_execution_inventory_markdown,
    write_paper_execution_inventory_artifacts,
)
from yolo_agent.research.paper_execution_schemas import (
    PaperExecutionDisposition,
    PaperExecutionInventory,
    PaperExecutionSpec,
)
from yolo_agent.research.paper_execution_requirement_schemas import (
    ExecutionRoute,
    PaperExecutionRequirement,
    PaperExecutionRequirementsMatrix,
)
from yolo_agent.research.paper_execution_requirements import (
    PaperExecutionRequirementsBuilder,
    build_paper_execution_requirements,
)
from yolo_agent.research.paper_training_cohort import (
    PaperTrainingCohortBuilder,
    build_paper_training_cohort,
)
from yolo_agent.research.paper_training_cohort_schemas import (
    COHORT_CATEGORIES,
    PaperTrainingCohort,
    PaperTrainingCohortCategory,
    PaperTrainingCohortRecord,
)
from yolo_agent.research.paper_mechanism_resolver import (
    GENERIC_MECHANISM_IDS,
    PaperMechanismExecutionGroup,
    PaperMechanismResolution,
    PaperMechanismResolutionSet,
    PaperMechanismResolver,
    merge_paper_mechanism_resolutions,
)
from yolo_agent.research.paper_evidence_requirements import (
    evidence_artifacts_for_family,
    missing_dataset_actions,
    required_metrics_for_family,
)
from yolo_agent.research.paper_protocol_catalog import (
    certified_paper_ids,
    load_certified_paper_protocols,
    requires_explicit_protocol,
)
from yolo_agent.research.paper_protocol_contract import (
    PaperProtocolContract,
    PaperProtocolEvaluation,
    PaperProtocolRegistry,
    authorize_paper_execution,
    evaluate_paper_protocol,
    missing_protocol_evaluation,
)
from yolo_agent.research.component_extractor import ComponentExtractionResult, ComponentExtractor
from yolo_agent.research.llm_paper_analyzer import LLMPaperAnalyzer
from yolo_agent.research.reproduction_pipeline import ReproductionPipeline, ReproductionTransitionError
from yolo_agent.research.reproduction_state import ReproductionContract, ReproductionState, ReproductionStatus
from yolo_agent.research.production_pipeline import ResearchProductionPipeline, ResearchProductionResult
from yolo_agent.research.snapshot import (
    ResearchMaturitySummary,
    ResearchRuntimeBinding,
    ResearchSnapshot,
    bind_research_snapshot,
    load_research_snapshot,
)

__all__ = [
    "Applicability",
    "BenchmarkEvidenceLevel",
    "ComponentCategory",
    "ComponentTaxonomy",
    "EvidenceLevel",
    "PaperBenchmark",
    "PaperComponentClaim",
    "PaperRecord",
    "PaperProvenance",
    "AwesomeCatalogImporter",
    "PaperImportResult",
    "import_awesome_catalog",
    "AwesomeSnapshotBuildResult",
    "AwesomeSnapshotBuilder",
    "AwesomeSourceManifest",
    "ComponentAliasConfig",
    "ComponentAliasResolution",
    "ComponentAliasResolver",
    "ResolvedComponentAlias",
    "ComponentCoverageAnalyzer",
    "ComponentCoverageReport",
    "HarnessHintParseResult",
    "HarnessHintParser",
    "PaperDiagnosticHint",
    "PaperAblationHint",
    "PaperEvidenceClaim",
    "PaperEvidenceSummary",
    "PaperLimitation",
    "PaperMethodClaim",
    "PaperNoteParser",
    "ImplementationDecisionKind",
    "PaperImplementationDecision",
    "PaperMethodCoverageReport",
    "PaperMethodProfile",
    "PaperMethodProfileBuilder",
    "PaperIndex",
    "PaperRegistry",
    "GENERIC_COMPONENT_IDS",
    "PaperExecutionDisposition",
    "PaperExecutionInventory",
    "PaperExecutionInventoryBuilder",
    "PaperExecutionSpec",
    "ExecutionRoute",
    "PaperExecutionRequirement",
    "PaperExecutionRequirementsMatrix",
    "PaperExecutionRequirementsBuilder",
    "build_paper_execution_requirements",
    "COHORT_CATEGORIES",
    "PaperTrainingCohort",
    "PaperTrainingCohortBuilder",
    "PaperTrainingCohortCategory",
    "PaperTrainingCohortRecord",
    "build_paper_training_cohort",
    "GENERIC_MECHANISM_IDS",
    "PaperMechanismExecutionGroup",
    "PaperMechanismResolution",
    "PaperMechanismResolutionSet",
    "PaperMechanismResolver",
    "merge_paper_mechanism_resolutions",
    "evidence_artifacts_for_family",
    "missing_dataset_actions",
    "required_metrics_for_family",
    "certified_paper_ids",
    "load_certified_paper_protocols",
    "requires_explicit_protocol",
    "PaperProtocolContract",
    "PaperProtocolEvaluation",
    "PaperProtocolRegistry",
    "authorize_paper_execution",
    "evaluate_paper_protocol",
    "missing_protocol_evaluation",
    "render_paper_execution_inventory_markdown",
    "write_paper_execution_inventory_artifacts",
    "ComponentExtractionResult",
    "ComponentExtractor",
    "LLMPaperAnalyzer",
    "ReproductionContract",
    "ReproductionPipeline",
    "ReproductionState",
    "ReproductionStatus",
    "ReproductionTransitionError",
    "ResearchProductionPipeline",
    "ResearchProductionResult",
    "ResearchMaturitySummary",
    "ResearchRuntimeBinding",
    "ResearchSnapshot",
    "bind_research_snapshot",
    "load_research_snapshot",
]
