"""README encoding and quick-start documentation tests."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree


def test_chinese_readme_is_utf8_bom_for_windows_powershell() -> None:
    """The Chinese README should display correctly in Windows PowerShell."""
    readme = Path("README.zh-CN.md")
    assert readme.read_bytes().startswith(b"\xef\xbb\xbf")
    text = readme.read_text(encoding="utf-8-sig")
    assert "证据驱动自动优化训练工具" in text
    assert "新人只需要四个命令" in text
    assert "能力边界" in text
    assert "yolo-agent train --model yolo26n.pt --data E:\\dataset\\coco.yaml --run-id coco-yolo26n" in text
    assert "yolo-agent status --run runs/coco-yolo26n" in text
    assert "yolo-agent stop --run runs/coco-yolo26n" in text
    assert "yolo-agent setup coco --data E:\\dataset\\coco.yaml --model yolo26n.pt" in text


def test_default_readme_is_english_and_links_to_chinese() -> None:
    text = Path("README.md").read_text(encoding="utf-8-sig")
    assert "evidence-driven optimization runner" in text
    assert "README.zh-CN.md" in text
    assert "README.en.md" not in text


def test_readmes_embed_a_valid_architecture_svg() -> None:
    asset_path = Path("docs/assets/yolo-agent-architecture.svg")
    root = ElementTree.parse(asset_path).getroot()
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert root.attrib["viewBox"] == "0 0 1600 1120"
    for readme in (Path("README.md"), Path("README.zh-CN.md")):
        assert asset_path.as_posix() in readme.read_text(encoding="utf-8-sig")


def test_readmes_point_to_new_user_docs() -> None:
    """Both homepages should link to the focused documentation pages."""
    for readme in (Path("README.md"), Path("README.zh-CN.md")):
        text = readme.read_text(encoding="utf-8-sig")
        for doc in [
            "docs/install.md",
            "docs/quickstart.md",
            "docs/training-modes.md",
            "docs/coco-yolo26.md",
            "docs/custom-dataset.md",
            "docs/llm-setup.md",
            "docs/troubleshooting.md",
            "docs/capability-maturity.md",
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
        Path("docs/capability-maturity.md"),
    ]:
        assert doc.read_bytes().startswith(b"\xef\xbb\xbf")
