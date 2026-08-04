"""Contracts for the beginner-facing command and documentation surface."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.cli import USER_COMMANDS, build_parser


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8-sig")


def test_beginner_cli_exposes_only_four_commands() -> None:
    assert USER_COMMANDS == ("setup", "train", "status", "stop")
    help_text = build_parser().format_help()
    assert "{setup,train,status,stop}" in help_text
    assert " research " not in help_text
    assert " advanced " not in help_text


def test_readmes_keep_research_commands_out_of_quickstart() -> None:
    for path in ("README.md", "README.zh-CN.md"):
        text = _read(path)
        for command in USER_COMMANDS:
            assert f"yolo-agent {command}" in text
        assert "yolo-agent research import-awesome" not in text
        assert "yolo-agent research build-snapshot" not in text
        assert "yolo-agent advanced certify-gpu" not in text


def test_readmes_explain_problem_driven_optimization_without_promising_gain() -> None:
    english = _read("README.md")
    chinese = _read("README.zh-CN.md")

    for phrase in (
        "Too many false positives",
        "Performance dropped in a new scene",
        "Small objects are often missed",
        "Improve overall mAP",
        "does not guarantee",
    ):
        assert phrase in english

    for phrase in (
        "当前模型误检多",
        "换了场景后效果不好",
        "小目标检测不好",
        "提高整体 mAP",
        "不表示任何数据集或每次运行都保证提升 mAP",
    ):
        assert phrase in chinese


def test_advanced_docs_and_maturity_boundaries_are_explicit() -> None:
    cli = _read("docs/cli.md")
    awesome = _read("docs/awesome-object-detection.md")
    paper = _read("docs/paper-intelligence.md")
    maturity = _read("docs/capability-maturity.md")

    assert "yolo-agent research import-awesome" in cli
    assert "yolo-agent research build-snapshot" in cli
    assert "yolo-agent advanced certify-gpu" in cli

    combined = "\n".join((awesome, paper, maturity))
    for statement in (
        "论文库不是训练集",
        "recipe_idea_only",
        "有论文记录不代表",
        "有 adapter 不代表",
        "runtime_integrated",
        "gpu_certified",
        "GPU certification 失败",
        "smoke passed 不代表 pilot reproduced",
        "pilot reproduced 不代表 full COCO confirmed",
        "+2 mAP",
        "full COCO 必须显式确认",
    ):
        assert statement in combined
