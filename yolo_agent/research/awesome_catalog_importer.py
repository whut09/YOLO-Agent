"""Offline importer for the Awesome-object-detection paper catalog."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.provenance import ImportProvenance
from yolo_agent.research.schemas import PaperRecord


IMPORTER_VERSION = "awesome_object_detection.v1"
SOURCE_NAME = "awesome_object_detection"

_REQUIRED_FIELDS = {"paper_id", "title", "year"}
_KNOWN_FIELDS = {
    "paper_id",
    "title",
    "year",
    "publication",
    "category",
    "paper_url",
    "official_code_url",
    "institution",
    "summary",
    "abstract",
    "authors",
    "task_families",
    "detector_family",
    "component_ids",
    "applicability",
    "harness_hints",
    "datasets",
    "framework",
    "code_license",
    "note_path",
}

_CATEGORY_TASK_FAMILIES = {
    "Assignment, Loss, and Training": ["training_optimization"],
    "DETR and End-to-End Detection": ["end_to_end_detection"],
    "General Object Detection": ["object_detection"],
    "Open-World and Domain-Robust Detection": ["open_world_detection", "domain_adaptation"],
    "Open-Vocabulary and Grounded Detection": ["open_vocabulary_detection"],
    "Small, Aerial, and Oriented Detection": [
        "small_object_detection",
        "aerial_detection",
        "oriented_detection",
    ],
    "YOLO and Real-Time Detection": ["real_time_detection"],
}
_CATEGORY_COMPONENT_CATEGORIES = {
    "Assignment, Loss, and Training": [
        "assigner",
        "matching",
        "bbox_regression_loss",
        "classification_loss",
        "optimizer",
        "lr_schedule",
    ],
    "DETR and End-to-End Detection": ["attention", "matching", "detection_head"],
    "General Object Detection": [],
    "Open-World and Domain-Robust Detection": ["domain_adaptation"],
    "Open-Vocabulary and Grounded Detection": ["pretraining"],
    "Small, Aerial, and Oriented Detection": ["feature_pyramid", "sampling", "slicing"],
    "YOLO and Real-Time Detection": ["backbone", "neck", "detection_head"],
}
_ALLOWED_APPLICABILITY = {
    "direct_adapter_candidate",
    "recipe_idea_only",
    "separate_detector_family",
    "incompatible",
    "insufficient_information",
}


class PaperImportResult(BaseModel, YAMLModelMixin):
    """Auditable result of one catalog import attempt."""

    schema_version: str = "awesome_catalog_import.v1"
    source_repository: str
    source_path: str
    source_commit: str = "unknown"
    catalog_hash: str
    dry_run: bool = False
    records: list[PaperRecord] = Field(default_factory=list)
    would_import_count: int = 0
    imported_count: int = 0
    skipped_count: int = 0
    conflict_count: int = 0
    skipped: list[dict[str, str]] = Field(default_factory=list)
    conflicts: list[dict[str, str]] = Field(default_factory=list)
    unknown_fields: dict[str, list[str]] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AwesomeCatalogImporter:
    """Convert a local catalog checkout into normalized registry records."""

    def __init__(self, registry: PaperRegistry | None = None, *, importer_version: str = IMPORTER_VERSION) -> None:
        self.registry = registry
        self.importer_version = importer_version

    def import_source(
        self,
        source: Path | str,
        *,
        dry_run: bool = False,
        source_commit: str | None = None,
    ) -> PaperImportResult:
        catalog_path = _resolve_catalog_path(Path(source))
        raw_bytes = catalog_path.read_bytes()
        catalog_hash = hashlib.sha256(raw_bytes).hexdigest()
        source_root = _source_root(catalog_path)
        repository = _repository_identity(source_root)
        commit = source_commit or _git_commit(source_root)
        payload = json.loads(raw_bytes.decode("utf-8-sig"))
        if not isinstance(payload, list):
            raise ValueError(f"catalog must contain a JSON array: {catalog_path}")

        records: list[PaperRecord] = []
        skipped: list[dict[str, str]] = []
        unknown_fields: dict[str, list[str]] = {}
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                skipped.append({"row": str(index), "reason": "record_not_mapping"})
                continue
            missing = sorted(field for field in _REQUIRED_FIELDS if not item.get(field))
            if missing:
                skipped.append({"row": str(index), "reason": f"missing_fields:{','.join(missing)}"})
                continue
            paper_id = str(item["paper_id"]).strip()
            unknown = sorted(set(item) - _KNOWN_FIELDS)
            if unknown:
                unknown_fields[paper_id] = unknown
            try:
                records.append(
                    _to_paper_record(
                        item,
                        source_repository=repository,
                        source_commit=commit,
                        source_path=_record_source_path(catalog_path, source_root, paper_id),
                        source_record_hash=_record_hash(item),
                        importer_version=self.importer_version,
                    )
                )
            except Exception as exc:
                skipped.append({"row": str(index), "paper_id": paper_id, "reason": f"invalid_record:{exc}"})

        conflicts: list[dict[str, str]] = []
        accepted: list[PaperRecord] = []
        existing_records = self.registry.list() if self.registry is not None else []
        for record in records:
            previous = _matching_existing(record, existing_records)
            if previous is not None and previous.source != SOURCE_NAME:
                conflicts.append(
                    {
                        "paper_id": record.paper_id,
                        "reason": "paper_id_owned_by_other_source",
                        "existing_source": previous.source,
                    }
                )
                continue
            if previous is not None and _same_source_record(previous, record):
                skipped.append({"paper_id": record.paper_id, "reason": "unchanged"})
                continue
            accepted.append(_with_history(record, previous))

        if self.registry is not None and not dry_run and accepted:
            self.registry.upsert_many(accepted)

        return PaperImportResult(
            source_repository=repository,
            source_path=catalog_path.as_posix(),
            source_commit=commit,
            catalog_hash=catalog_hash,
            dry_run=dry_run,
            records=accepted,
            would_import_count=len(accepted),
            imported_count=0 if dry_run else len(accepted),
            skipped_count=len(skipped),
            conflict_count=len(conflicts),
            skipped=skipped,
            conflicts=conflicts,
            unknown_fields=unknown_fields,
        )


def import_awesome_catalog(
    source: Path | str,
    *,
    registry_root: Path | str | None = "research",
    dry_run: bool = False,
    source_commit: str | None = None,
) -> PaperImportResult:
    """Convenience API for importing one local catalog."""
    registry = PaperRegistry(registry_root) if registry_root is not None else None
    return AwesomeCatalogImporter(registry).import_source(
        source,
        dry_run=dry_run,
        source_commit=source_commit,
    )


def _to_paper_record(
    item: dict[str, Any],
    *,
    source_repository: str,
    source_commit: str,
    source_path: str,
    source_record_hash: str,
    importer_version: str,
) -> PaperRecord:
    category = _optional_text(item.get("category"))
    summary = _optional_text(item.get("summary")) or ""
    abstract = _optional_text(item.get("abstract")) or summary
    task_families = _string_list(item.get("task_families"))
    for family in _CATEGORY_TASK_FAMILIES.get(category or "", []):
        if family not in task_families:
            task_families.append(family)
    applicability = str(item.get("applicability") or "insufficient_information")
    provenance = ImportProvenance(
        source_repository=source_repository,
        source_commit=source_commit,
        source_path=source_path,
        source_record_hash=source_record_hash,
        importer_version=importer_version,
        original_category=category,
        original_applicability=applicability if applicability in _ALLOWED_APPLICABILITY else None,
        original_harness_hints=_string_list(item.get("harness_hints")),
        original_component_ids=_string_list(item.get("component_ids")),
        original_note_path=_optional_text(item.get("note_path")),
        abstract_source="abstract" if item.get("abstract") else "summary" if summary else "unknown",
    )
    return PaperRecord(
        paper_id=str(item["paper_id"]).strip(),
        title=str(item["title"]).strip(),
        abstract=abstract,
        year=int(item["year"]),
        authors=_string_list(item.get("authors")),
        task_families=task_families,
        detector_family=_optional_text(item.get("detector_family")),
        source_url=_optional_text(item.get("paper_url")),
        paper_url=_optional_text(item.get("paper_url")),
        official_code_url=_optional_text(item.get("official_code_url")),
        code_license=_optional_text(item.get("code_license")) or "unknown",
        framework=_optional_text(item.get("framework")),
        datasets=_string_list(item.get("datasets")),
        component_ids=_string_list(item.get("component_ids")),
        component_categories=_CATEGORY_COMPONENT_CATEGORIES.get(category or "", []),
        applicability=applicability,
        source=SOURCE_NAME,
        ingestion_version=importer_version,
        evidence_level="paper_prior",
        provenance=provenance,
    )


def _with_history(record: PaperRecord, previous: PaperRecord | None) -> PaperRecord:
    if previous is None or previous.provenance is None or record.provenance is None:
        return record
    history = list(previous.provenance.history)
    history.append(previous.provenance.model_dump(mode="json", exclude={"history"}))
    return record.model_copy(update={"provenance": record.provenance.model_copy(update={"history": history})})


def _same_source_record(previous: PaperRecord, current: PaperRecord) -> bool:
    return bool(
        previous.provenance
        and current.provenance
        and previous.provenance.source_record_hash == current.provenance.source_record_hash
        and previous.provenance.source_commit == current.provenance.source_commit
    )


def _matching_existing(record: PaperRecord, existing: list[PaperRecord]) -> PaperRecord | None:
    record_title = _normalize_title(record.title)
    record_doi = _normalize_doi(record.doi)
    for paper in existing:
        if paper.paper_id == record.paper_id:
            return paper
        if record_doi and _normalize_doi(paper.doi) == record_doi:
            return paper
        if record_title and _normalize_title(paper.title) == record_title:
            return paper
    return None


def _normalize_doi(value: str | None) -> str:
    return str(value or "").strip().lower().removeprefix("https://doi.org/").removeprefix("doi:")


def _normalize_title(value: str) -> str:
    ascii_title = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", ascii_title)).strip()


def _resolve_catalog_path(source: Path) -> Path:
    if source.is_dir():
        source = source / "data" / "papers.json"
    if not source.is_file():
        raise FileNotFoundError(f"Awesome catalog not found: {source}")
    return source.resolve()


def _source_root(catalog_path: Path) -> Path:
    return catalog_path.parent.parent if catalog_path.parent.name == "data" else catalog_path.parent


def _repository_identity(source_root: Path) -> str:
    remote = _run_git(source_root, "config", "--get", "remote.origin.url")
    return remote or SOURCE_NAME


def _git_commit(source_root: Path) -> str:
    return _run_git(source_root, "rev-parse", "HEAD") or "unknown"


def _run_git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def _record_source_path(catalog_path: Path, source_root: Path, paper_id: str) -> str:
    try:
        relative = catalog_path.relative_to(source_root).as_posix()
    except ValueError:
        relative = catalog_path.name
    return f"{relative}#{paper_id}"


def _record_hash(item: dict[str, Any]) -> str:
    payload = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        raise ValueError("expected a list of strings")
    return [str(item).strip() for item in value if str(item).strip()]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "AwesomeCatalogImporter",
    "IMPORTER_VERSION",
    "PaperImportResult",
    "import_awesome_catalog",
]
