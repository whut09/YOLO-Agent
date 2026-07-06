"""README encoding and quick-start documentation tests."""

from __future__ import annotations

from pathlib import Path


def test_chinese_readme_is_utf8_bom_for_windows_powershell() -> None:
    """Chinese README should display correctly with default Windows PowerShell Get-Content."""
    readme = Path("README.md")
    assert readme.read_bytes().startswith(b"\xef\xbb\xbf")
    text = readme.read_text(encoding="utf-8-sig")
    assert "3 条命令跑起来" in text
    assert "运行模式一句话" in text
    assert "下一步读哪个文档" in text
    assert "docs/install.md" in text
    assert "yolo-agent setup coco --data E:\\dataset\\coco.yaml --model yolo26n.pt" in text
    assert "yolo-agent doctor --data E:\\dataset\\coco.yaml --model yolo26n.pt" in text
    assert "yolo-agent loop status --run runs/coco-yolo26n" in text


def test_readme_points_to_new_user_docs() -> None:
    """The homepage should link to focused beginner documentation pages."""
    text = Path("README.md").read_text(encoding="utf-8-sig")
    for doc in [
        "docs/install.md",
        "docs/quickstart.md",
        "docs/training-modes.md",
        "docs/coco-yolo26.md",
        "docs/custom-dataset.md",
        "docs/llm-setup.md",
        "docs/troubleshooting.md",
    ]:
        assert doc in text


def test_chinese_docs_are_utf8_bom_for_windows_powershell() -> None:
    """Chinese docs should also be readable via default Windows PowerShell Get-Content."""
    for doc in [
        Path("docs/install.md"),
        Path("docs/quickstart.md"),
        Path("docs/training-modes.md"),
        Path("docs/coco-yolo26.md"),
        Path("docs/custom-dataset.md"),
        Path("docs/llm-setup.md"),
        Path("docs/concepts.md"),
        Path("docs/loop-engineering.md"),
        Path("docs/evidence.md"),
        Path("docs/cli.md"),
        Path("docs/troubleshooting.md"),
    ]:
        assert doc.read_bytes().startswith(b"\xef\xbb\xbf")
