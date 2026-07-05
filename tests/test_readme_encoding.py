"""README encoding and quick-start documentation tests."""

from __future__ import annotations

from pathlib import Path


def test_chinese_readme_is_utf8_bom_for_windows_powershell() -> None:
    """Chinese README should display correctly with default Windows PowerShell Get-Content."""
    readme = Path("README.md")
    assert readme.read_bytes().startswith(b"\xef\xbb\xbf")
    text = readme.read_text(encoding="utf-8-sig")
    assert "推荐把日常使用理解成三件事：启动按钮、仪表盘、导航路线。" in text
    assert "yolo-agent doctor --data E:\\dataset\\coco.yaml --model yolo26n.pt" in text
    assert "yolo-agent loop status --run runs/coco-yolo26n" in text
