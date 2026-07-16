"""Generate user-facing capability maturity documentation from one audited manifest."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Literal, Sequence

import yaml
from pydantic import BaseModel, Field, model_validator


CapabilityStatus = Literal[
    "executable",
    "incomplete",
    "partial",
    "artifact_only",
    "mixed",
    "supported_not_automatic",
    "not_guaranteed",
]
AutomationLevel = Literal["yes", "partial", "guarded", "mixed", "explicit_confirmation", "no"]
ReproductionLevel = Literal["run_dependent", "partial", "mixed", "not_claimed"]

README_START = "<!-- capability-maturity:start -->"
README_END = "<!-- capability-maturity:end -->"


class CapabilityEntry(BaseModel):
    """One audited capability with separate implementation and evidence dimensions."""

    capability_id: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    name_zh: str = Field(min_length=1)
    name_en: str = Field(min_length=1)
    status: CapabilityStatus
    code_present: bool
    automatic_execution: AutomationLevel
    local_reproduction: ReproductionLevel
    boundary_zh: str = Field(min_length=1)
    boundary_en: str = Field(min_length=1)
    source_paths: list[Path] = Field(min_length=1)


class CapabilityManifest(BaseModel):
    """Versioned source of truth for the generated maturity matrix."""

    schema_version: int = Field(ge=1)
    reviewed_at: date
    capabilities: list[CapabilityEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "CapabilityManifest":
        ids = [item.capability_id for item in self.capabilities]
        if len(ids) != len(set(ids)):
            raise ValueError("capability_id values must be unique")
        return self

    @classmethod
    def from_yaml(cls, path: Path | str) -> "CapabilityManifest":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)


def validate_source_paths(manifest: CapabilityManifest, *, root: Path | str = ".") -> list[Path]:
    """Return referenced implementation paths that do not exist."""
    base = Path(root)
    return [path for item in manifest.capabilities for path in item.source_paths if not (base / path).is_file()]


def render_readme_matrix(manifest: CapabilityManifest, *, language: Literal["zh", "en"]) -> str:
    """Render the compact matrix embedded in the two README files."""
    if language == "zh":
        lines = [
            "| 能力 | 当前状态 | 代码存在 | 自动执行 | 本地复现 | 现实边界 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    else:
        lines = [
            "| Capability | Current status | Code present | Automatic execution | Local reproduction | Boundary |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    for item in manifest.capabilities:
        name = item.name_zh if language == "zh" else item.name_en
        boundary = item.boundary_zh if language == "zh" else item.boundary_en
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(name),
                    f"`{_status_label(item.status)}`",
                    _boolean(item.code_present, language),
                    _automation(item.automatic_execution, language),
                    _reproduction(item.local_reproduction, language),
                    _cell(boundary),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_detail_document(manifest: CapabilityManifest) -> str:
    """Render the detailed Chinese document with implementation traceability."""
    lines = [
        "# 能力成熟度矩阵",
        "",
        "> 本页由 `configs/capability_maturity.yaml` 自动生成。请修改清单后运行以下命令：",
        "> `python -m yolo_agent.tools.capability_matrix`，不要直接编辑表格。",
        "",
        f"最近审计日期：`{manifest.reviewed_at.isoformat()}`；Schema：`v{manifest.schema_version}`。",
        "",
        "这里刻意拆开三个概念：代码存在不代表可以自动执行，可以执行也不代表已经在本地复现；任何一项都不等于保证指标提升。",
        "",
        render_readme_matrix(manifest, language="zh"),
        "",
        "## 状态含义",
        "",
        "- `executable`：已接入实际执行路径，但仍受环境、evidence 和 guard 约束。",
        "- `incomplete`：存在主要模块，但关键证据链或协议仍不完整。",
        "- `partial`：能覆盖一部分闭环，缺失条件下会降级或停止。",
        "- `artifact_only`：目前只生成计划或 artifact，尚不权威控制执行。",
        "- `mixed`：同一能力族包含不同成熟度的实现，必须逐项检查。",
        "- `supported_not_automatic`：具备实现和门禁，但默认流程不会端到端自动完成。",
        "- `not_guaranteed`：这是目标或期望结果，不是软件能力承诺。",
        "",
        "## 源码依据",
        "",
    ]
    for item in manifest.capabilities:
        refs = ", ".join(f"`{path.as_posix()}`" for path in item.source_paths)
        lines.append(f"- **{item.name_zh}**：{refs}")
    lines.append("")
    return "\n".join(lines)


def update_readme(text: str, matrix: str) -> str:
    """Replace one generated README block while preserving surrounding prose."""
    if README_START not in text or README_END not in text:
        raise ValueError("README capability maturity markers are missing")
    before, remainder = text.split(README_START, 1)
    _, after = remainder.split(README_END, 1)
    return f"{before}{README_START}\n{matrix}\n{README_END}{after}"


def generate(
    *,
    config_path: Path,
    document_path: Path,
    readme_path: Path,
    readme_en_path: Path,
    check: bool = False,
) -> bool:
    """Generate all maturity docs, returning whether they were already current."""
    manifest = CapabilityManifest.from_yaml(config_path)
    missing = validate_source_paths(manifest, root=config_path.parent.parent)
    if missing:
        raise ValueError(f"capability source paths are missing: {', '.join(path.as_posix() for path in missing)}")

    expected_doc = render_detail_document(manifest)
    expected_readme = update_readme(_read_bom_text(readme_path), render_readme_matrix(manifest, language="zh"))
    expected_readme_en = update_readme(_read_bom_text(readme_en_path), render_readme_matrix(manifest, language="en"))
    outputs = [
        (document_path, expected_doc),
        (readme_path, expected_readme),
        (readme_en_path, expected_readme_en),
    ]
    current = all(path.is_file() and _read_bom_text(path) == content for path, content in outputs)
    if check:
        return current
    for path, content in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8-sig")
    return current


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the audited capability maturity matrix.")
    parser.add_argument("--config", type=Path, default=Path("configs/capability_maturity.yaml"))
    parser.add_argument("--document", type=Path, default=Path("docs/capability-maturity.md"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--readme-en", type=Path, default=Path("README.en.md"))
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    current = generate(
        config_path=args.config,
        document_path=args.document,
        readme_path=args.readme,
        readme_en_path=args.readme_en,
        check=args.check,
    )
    if args.check and not current:
        print("Capability maturity documentation is stale.")
        return 1
    print("Capability maturity documentation is current." if current else "Generated capability maturity documentation.")
    return 0


def _read_bom_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _boolean(value: bool, language: Literal["zh", "en"]) -> str:
    return ("是" if value else "否") if language == "zh" else ("yes" if value else "no")


def _status_label(value: CapabilityStatus) -> str:
    return {
        "artifact_only": "artifact only",
        "supported_not_automatic": "supported, not automatic end-to-end",
        "not_guaranteed": "not guaranteed",
    }.get(value, value)


def _automation(value: AutomationLevel, language: Literal["zh", "en"]) -> str:
    labels = {
        "zh": {
            "yes": "是",
            "partial": "部分",
            "guarded": "有门禁",
            "mixed": "混合",
            "explicit_confirmation": "需显式确认",
            "no": "否",
        },
        "en": {
            "yes": "yes",
            "partial": "partial",
            "guarded": "guarded",
            "mixed": "mixed",
            "explicit_confirmation": "explicit confirmation",
            "no": "no",
        },
    }
    return labels[language][value]


def _reproduction(value: ReproductionLevel, language: Literal["zh", "en"]) -> str:
    labels = {
        "zh": {
            "run_dependent": "取决于本地 run",
            "partial": "部分",
            "mixed": "混合",
            "not_claimed": "未声明",
        },
        "en": {
            "run_dependent": "depends on local runs",
            "partial": "partial",
            "mixed": "mixed",
            "not_claimed": "not claimed",
        },
    }
    return labels[language][value]


if __name__ == "__main__":
    raise SystemExit(main())
