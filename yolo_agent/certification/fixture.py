"""Generate a tiny deterministic COCO-compatible detection fixture."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import yaml


def create_mini_coco_fixture(root: Path | str) -> Path:
    output = Path(root)
    annotations: list[dict[str, object]] = []
    images: list[dict[str, object]] = []
    annotation_id = 1
    for split, count in (("train2017", 6), ("val2017", 4)):
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(1, count + 1):
            filename = f"{index:012d}.png"
            x = 8 + (index % 3) * 6
            y = 10 + (index % 2) * 8
            width = 24
            height = 20
            (image_dir / filename).write_bytes(_png_with_box(64, 64, x, y, width, height))
            (label_dir / f"{Path(filename).stem}.txt").write_text(
                f"0 {(x + width / 2) / 64:.6f} {(y + height / 2) / 64:.6f} {width / 64:.6f} {height / 64:.6f}\n",
                encoding="utf-8",
            )
            if split == "val2017":
                images.append({"id": index, "file_name": filename, "width": 64, "height": 64})
                annotations.append({
                    "id": annotation_id,
                    "image_id": index,
                    "category_id": 0,
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                })
                annotation_id += 1
    annotation_dir = output / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    (annotation_dir / "instances_val2017.json").write_text(
        json.dumps({
            "info": {"description": "YOLO Agent mini COCO GPU certification fixture"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 0, "name": "object", "supercategory": "object"}],
        }, indent=2),
        encoding="utf-8",
    )
    data_yaml = output / "coco.yaml"
    data_yaml.write_text(
        yaml.safe_dump({
            "path": output.resolve().as_posix(),
            "train": "images/train2017",
            "val": "images/val2017",
            "names": {0: "object"},
        }, sort_keys=False),
        encoding="utf-8",
    )
    return data_yaml


def _png_with_box(width: int, height: int, x: int, y: int, box_width: int, box_height: int) -> bytes:
    rows = []
    for row in range(height):
        pixels = bytearray()
        for column in range(width):
            inside = x <= column < x + box_width and y <= row < y + box_height
            pixels.extend((240, 240, 240) if inside else (20, 30, 40))
        rows.append(b"\x00" + bytes(pixels))
    raw = b"".join(rows)
    signature = b"\x89PNG\r\n\x1a\n"
    return signature + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + _chunk(b"IDAT", zlib.compress(raw, 9)) + _chunk(b"IEND", b"")


def _chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


__all__ = ["create_mini_coco_fixture"]
