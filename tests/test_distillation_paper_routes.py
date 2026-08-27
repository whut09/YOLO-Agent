"""CPU tests for paper-specific distillation runtime routes.

Covers the eight required branch routes, the 32 per-paper routes, payload
and changed-variable independence, teacher resolution failures, student-only
export protocol, and fail-closed certification without GPU training.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest
import yaml

from yolo_agent.certification.distillation_paper_routes import (
    DistillationPaperRouteCertificationSummary,
    certify_all_paper_routes,
    certify_distillation_paper_route,
)
from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.distillation import (
    BRANCH_ADAPTERS,
    REQUIRED_BRANCH_ADAPTERS,
    DistillationPaperRoute,
    build_paper_routes,
    default_paper_route_registry,
    paper_route_coverage,
)
from yolo_agent.components.adapters.distillation.paper_routes import (
    CERTIFIED_DISTILLATION_PAPERS,
    PAPER_ROUTE_ADAPTERS,
    create_paper_route_adapter,
)
from yolo_agent.components.adapters.distillation.yolo26_distillation import (
    YOLO26DistillationConfig,
)
from yolo_agent.components.contracts import ComponentContract, load_contracts


REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPES_PATH = REPO_ROOT / "configs" / "recipes" / "paper_specific_methods.yaml"
CONTRACTS_PATH = REPO_ROOT / "configs" / "components" / "distillation"

DAI_PAPER = "cvf:cvpr2021:Dai_General_Instance_Distillation_for_Object_Detection"
HU_PAPER = (
    "cvf:cvpr2021:Hu_Dense_Relation_Distillation_With_Context-Aware_Aggregation_"
    "for_Few-Shot_Object_Detection"
)
ZHENP_PAPER = "cvf:cvpr2022:Zheng_Localization_Distillation_for_Dense_Object_Detection"
ECVA_1356 = "ecva:eccv2022:1356"
NEURIPS_082 = "neurips:2021:082a8bbf2c357c09f26675f9cf5bcba3-Abstract"
DATASET_HASH = "d" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sidecar(
    checkpoint: Path,
    *,
    architecture: str,
    dataset_hash: str,
    split: str,
) -> None:
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".metadata.json")
    sidecar.write_text(
        json.dumps(
            {
                "architecture": architecture,
                "dataset_hash": dataset_hash,
                "split": split,
                "imgsz": 640,
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def routes() -> list[DistillationPaperRoute]:
    return build_paper_routes()


def test_all_32_papers_own_exactly_one_route(routes) -> None:
    assert len(routes) == 32
    assert {item.paper_id for item in routes} == set(CERTIFIED_DISTILLATION_PAPERS)
    assert len({item.component_id for item in routes}) == 32
    assert len({item.recipe_id for item in routes}) == 32
    assert len({item.paper_specific_mechanism_id for item in routes}) == 32
    assert len({item.method_profile_id for item in routes}) == 32


def test_route_split_is_18_branch_bound_and_14_recovery(routes) -> None:
    branch_bound = [
        item for item in routes if item.method_identity_status == "branch_bound"
    ]
    recovery = [
        item
        for item in routes
        if item.method_identity_status == "identity_recovery"
    ]
    assert len(branch_bound) == 18
    assert len(recovery) == 14
    for item in branch_bound:
        assert item.branch_id is not None
        assert item.branch_component_id
    for item in recovery:
        assert item.branch_id is None
        assert item.reason_codes


def test_route_fingerprints_and_adapters_are_independent_per_paper(routes) -> None:
    assert len({item.execution_fingerprint for item in routes}) == 32
    assert len({item.adapter_class for item in routes}) == 32
    assert len({item.adapter_version for item in routes}) == 32


def test_routes_match_paper_specific_recipes(routes) -> None:
    raw = yaml.safe_load(RECIPES_PATH.read_text(encoding="utf-8-sig"))
    recipes = {item["recipe_id"]: item for item in raw["recipes"]}
    for route in routes:
        recipe = recipes[route.recipe_id]
        assert (
            recipe["paper_specific_mechanism_id"]
            == route.paper_specific_mechanism_id
        )
        assert route.paper_id in recipe["paper_ids"]
        assert (
            recipe["train_overrides"]["distillation.method"]
            == route.paper_specific_mechanism_id
        )


def test_no_generic_adapter_collapse(routes) -> None:
    for route in routes:
        assert route.component_id != "distillation.yolo26_teacher_student"
        assert route.adapter_class != "YOLO26DistillationAdapter"
        assert route.component_id == route.paper_specific_mechanism_id
        assert route.component_id.startswith("distillation.")


def test_route_protocols_bind_teacher_and_student_identity(routes) -> None:
    for route in routes:
        teacher = route.teacher_protocol
        assert teacher["frozen"] is True
        assert teacher["checkpoint_required"] is True
        assert teacher["sha256_required"] is True
        assert teacher["architecture_required"] is True
        assert teacher["dataset_hash_required"] is True
        assert teacher["split_required"] is True
        assert teacher["imgsz"] == 640
        assert teacher["export_forbidden"] is True
        assert teacher["mock_forbidden"] is True
        student = route.student_protocol
        assert student["architecture"] == "yolo26n"
        assert student["checkpoint"] == "yolo26n.pt"
        assert student["imgsz"] == 640
        assert student["export_and_measure"] == "student_only"
        assert student["dataset_hash_required"] is True
        assert route.student_only_export is True
        assert route.matched_baseline_required is True
        assert set(route.student_only_metrics) == {"latency_ms", "model_size_mb"}


def test_required_8_branch_routes_have_independent_adapters() -> None:
    classes = [BRANCH_ADAPTERS[branch_id] for branch_id in REQUIRED_BRANCH_ADAPTERS]
    assert len({cls.__name__ for cls in classes}) == 8
    assert len({cls.branch_component_id for cls in classes}) == 8
    assert len({cls.branch_changed_variable for cls in classes}) == 8
    fingerprints = {cls().branch_spec().execution_fingerprint for cls in classes}
    assert len(fingerprints) == 8
    for cls in classes:
        assert cls.__name__ != "YOLO26DistillationAdapter"
        assert cls.branch_changed_variable.startswith("loss.distillation.")


def test_zheng_route_reuses_the_localization_branch_component() -> None:
    route = default_paper_route_registry().route(ZHENP_PAPER)
    assert route.branch_id == "localization_distillation"
    assert route.component_id == route.branch_component_id
    assert route.component_id == "distillation.localization"


def test_recovery_papers_keep_explicit_identity_routes() -> None:
    registry = default_paper_route_registry()
    for paper_id in (ECVA_1356, NEURIPS_082):
        route = registry.route(paper_id)
        assert route.method_identity_status == "identity_recovery"
        assert route.reason_codes == [
            "distillation_branch_unmapped",
            "paper_method_identity_missing",
        ]
        assert route.component_id.startswith("distillation.")
        assert route.recipe_id.startswith("paper_")


def test_paper_route_contracts_resolve_independent_adapters(routes) -> None:
    contracts = load_contracts(CONTRACTS_PATH)
    component_ids = [item.component_id for item in contracts]
    assert len(component_ids) == len(set(component_ids))
    paper_contracts = [
        item for item in contracts if item.implementation_family == "distillation.paper_route"
    ]
    assert len(paper_contracts) == 31
    route_components = {item.component_id for item in routes}
    for contract in paper_contracts:
        assert contract.component_id in route_components
        module = importlib.import_module(contract.implementation_path)
        assert getattr(module, contract.adapter_class) is not None
        assert contract.paper_specific_mechanism_ids == [contract.component_id]


def _paper_payload(route: DistillationPaperRoute):
    adapter_cls = create_paper_route_adapter(route)
    contract = ComponentContract(
        component_id=route.component_id,
        display_name=route.paper_id,
        category="distillation",
    )
    context = AdapterContext(
        contract=contract,
        detector_family="yolo26",
        workspace=REPO_ROOT,
        options={
            "teacher_data": "coco.yaml",
            "student_data": "coco.yaml",
            "teacher_split": "train",
            "student_split": "train",
        },
    )
    return adapter_cls().build_runtime_payload(
        context,
        protocol_hash="p" * 64,
        base_command=["python", "-m", "ultralytics", "train"],
        generated_config={},
    )


def test_per_paper_payloads_are_independent() -> None:
    registry = default_paper_route_registry()
    dai_route = registry.route(DAI_PAPER)
    hu_route = registry.route(HU_PAPER)
    dai = _paper_payload(dai_route)
    hu = _paper_payload(hu_route)

    assert dai.component_ids == [dai_route.component_id]
    assert hu.component_ids == [hu_route.component_id]
    assert dai.component_ids != hu.component_ids
    assert dai.adapter_classes == [dai_route.adapter_class]
    assert hu.adapter_classes == [hu_route.adapter_class]
    assert dai.changed_variables == dai_route.changed_variables
    assert hu.changed_variables == hu_route.changed_variables
    assert dai.changed_variables != hu.changed_variables
    assert dai.payload_hash != hu.payload_hash

    dai_options = dai.loss_plugin[0].options
    assert dai_options["paper_id"] == DAI_PAPER
    assert (
        dai_options["paper_route_fingerprint"] == dai_route.execution_fingerprint
    )
    assert dai_options["component_id"] == dai_route.component_id
    assert dai_options["mechanism"] == "relation"
    assert dai_options["branch_id"] == "relation_distillation"


def test_recovery_paper_payload_stays_paper_bound() -> None:
    registry = default_paper_route_registry()
    route = registry.route(ECVA_1356)
    payload = _paper_payload(route)
    options = payload.loss_plugin[0].options
    assert payload.component_ids == [route.component_id]
    assert payload.adapter_classes == [route.adapter_class]
    assert payload.changed_variables == route.changed_variables
    assert options["paper_id"] == ECVA_1356
    assert options["paper_route_fingerprint"] == route.execution_fingerprint
    # Unmapped papers have no known mechanism: the route stays in
    # multi-term recovery mode and must not claim a mechanism identity.
    assert options.get("mechanism") is None


def test_all_paper_route_adapter_classes_resolve_by_import() -> None:
    assert len(PAPER_ROUTE_ADAPTERS) == 32
    module = importlib.import_module(
        "yolo_agent.components.adapters.distillation.paper_routes"
    )
    for route in build_paper_routes():
        assert getattr(module, route.adapter_class) is not None


def test_paper_route_coverage_has_no_silent_drops() -> None:
    coverage = paper_route_coverage()
    assert coverage.papers_total == 32
    assert coverage.branch_bound == 18
    assert coverage.identity_recovery == 14
    assert coverage.silent_drops == []
    assert {item.paper_id for item in coverage.routes} == set(
        CERTIFIED_DISTILLATION_PAPERS
    )


def test_coverage_rejects_silent_drops() -> None:
    from yolo_agent.components.adapters.distillation import (
        DistillationPaperRouteCoverage,
    )

    routes = build_paper_routes()
    with pytest.raises(ValueError, match="silent drops"):
        DistillationPaperRouteCoverage.model_validate(
            {
                "papers_total": 32,
                "branch_bound": 18,
                "identity_recovery": 14,
                "routes": [item.model_dump() for item in routes],
                "silent_drops": [DAI_PAPER],
            }
        )


def test_route_payload_schema_rejects_missing_keys() -> None:
    route = build_paper_routes()[0]
    data = route.model_dump()
    data["execution_fingerprint"] = ""
    del data["runtime_payload_schema"]["matched_baseline"]
    with pytest.raises(ValueError, match="matched_baseline"):
        DistillationPaperRoute.model_validate(data)


def test_route_rejects_missing_changed_variables() -> None:
    route = build_paper_routes()[0]
    data = route.model_dump()
    data["execution_fingerprint"] = ""
    data["changed_variables"] = {}
    with pytest.raises(ValueError, match="changed variables"):
        DistillationPaperRoute.model_validate(data)


def test_route_rejects_mismatched_fingerprint() -> None:
    route = build_paper_routes()[0]
    data = route.model_dump()
    data["execution_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        DistillationPaperRoute.model_validate(data)


def test_config_rejects_paper_id_without_route_fingerprint() -> None:
    with pytest.raises(ValueError, match="sha256 execution fingerprint"):
        YOLO26DistillationConfig(
            mechanism="relation",
            component_id="distillation.general_instance",
            changed_variable="loss.distillation.relation.weight",
            branch_id="relation_distillation",
            paper_id=DAI_PAPER,
        )


def test_config_rejects_paper_component_outside_distillation_family() -> None:
    with pytest.raises(ValueError, match="distillation family"):
        YOLO26DistillationConfig(
            mechanism="relation",
            component_id="neck.gold",
            changed_variable="loss.distillation.relation.weight",
            branch_id="relation_distillation",
            paper_id=DAI_PAPER,
            paper_route_fingerprint="a" * 64,
        )


@pytest.fixture()
def ready_workspace(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    teacher = workspace / "yolo26s.pt"
    teacher.write_bytes(b"frozen-teacher-bytes")
    student = workspace / "yolo26n.pt"
    student.write_bytes(b"student-bytes")
    _write_sidecar(teacher, architecture="yolo26s", dataset_hash=DATASET_HASH, split="train")
    _write_sidecar(student, architecture="yolo26n", dataset_hash=DATASET_HASH, split="train")
    return workspace, teacher, student


def test_certify_missing_teacher_is_evidence_recovery(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    student = workspace / "yolo26n.pt"
    student.write_bytes(b"student-bytes")
    _write_sidecar(
        student, architecture="yolo26n", dataset_hash=DATASET_HASH, split="train"
    )
    report = certify_distillation_paper_route(
        ECVA_1356,
        workspace=workspace,
        teacher=str(workspace / "yolo26s.pt"),
        student=str(student),
        expected_student_sha256=_sha256(student),
        dataset_manifest_hash=DATASET_HASH,
    )
    assert report.disposition == "evidence_recovery"
    assert any("teacher_checkpoint_missing" in code for code in report.reason_codes)
    assert "distillation_branch_unmapped" in report.reason_codes
    assert report.paper_route_fingerprint == (
        default_paper_route_registry().route(ECVA_1356).execution_fingerprint
    )


def test_certify_full_protocol_is_runtime_ready(ready_workspace) -> None:
    workspace, teacher, student = ready_workspace
    report = certify_distillation_paper_route(
        DAI_PAPER,
        workspace=workspace,
        teacher=str(teacher),
        student=str(student),
        expected_teacher_sha256=_sha256(teacher),
        expected_student_sha256=_sha256(student),
        dataset_manifest_hash=DATASET_HASH,
        matched_baseline={"model": "yolo26n.pt", "mAP": 0.45},
    )
    assert report.disposition == "runtime_ready"
    assert report.reason_codes == []
    assert report.teacher_disposition == "runtime_ready"
    assert report.student_disposition == "runtime_ready"
    assert report.route_checks["paper_route_identity_bound"] is True
    assert report.route_checks["payload_schema_complete"] is True
    assert report.route_checks["student_only_export_protocol"] is True
    assert report.report_hash


def test_certify_teacher_hash_mismatch_blocks_runtime(ready_workspace) -> None:
    workspace, teacher, student = ready_workspace
    report = certify_distillation_paper_route(
        DAI_PAPER,
        workspace=workspace,
        teacher=str(teacher),
        student=str(student),
        expected_teacher_sha256="0" * 64,
        expected_student_sha256=_sha256(student),
        dataset_manifest_hash=DATASET_HASH,
        matched_baseline={"model": "yolo26n.pt"},
    )
    assert report.disposition == "blocked_runtime"
    assert "teacher_checkpoint_sha256_mismatch" in report.reason_codes


def test_certify_split_mismatch_blocks_runtime(ready_workspace) -> None:
    workspace, teacher, student = ready_workspace
    _write_sidecar(
        teacher, architecture="yolo26s", dataset_hash=DATASET_HASH, split="val"
    )
    report = certify_distillation_paper_route(
        DAI_PAPER,
        workspace=workspace,
        teacher=str(teacher),
        student=str(student),
        expected_teacher_sha256=_sha256(teacher),
        expected_student_sha256=_sha256(student),
        dataset_manifest_hash=DATASET_HASH,
        matched_baseline={"model": "yolo26n.pt"},
    )
    assert report.disposition == "blocked_runtime"
    assert "teacher_split_mismatch" in report.reason_codes


def test_certify_missing_matched_baseline_is_recovery(ready_workspace) -> None:
    workspace, teacher, student = ready_workspace
    report = certify_distillation_paper_route(
        DAI_PAPER,
        workspace=workspace,
        teacher=str(teacher),
        student=str(student),
        expected_teacher_sha256=_sha256(teacher),
        expected_student_sha256=_sha256(student),
        dataset_manifest_hash=DATASET_HASH,
    )
    assert report.disposition == "evidence_recovery"
    assert "matched_baseline_missing" in report.reason_codes
    assert report.route_checks["matched_baseline_present"] is False


def test_certify_all_32_without_assets_never_reports_ready(tmp_path) -> None:
    workspace = tmp_path / "empty"
    workspace.mkdir()
    summary = certify_all_paper_routes(
        output_path=tmp_path / "certification.yaml",
        workspace=workspace,
    )
    assert summary.papers_total == 32
    assert summary.runtime_ready == 0
    assert summary.silent_drops == []
    assert summary.runtime_ready + summary.evidence_recovery + summary.blocked_runtime == 32
    assert (tmp_path / "certification.yaml").is_file()
    reloaded = DistillationPaperRouteCertificationSummary.from_yaml(
        tmp_path / "certification.yaml"
    )
    assert reloaded.summary_hash == summary.summary_hash
    assert len(reloaded.reports) == 32
