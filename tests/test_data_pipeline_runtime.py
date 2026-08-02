from __future__ import annotations

from torch.utils.data import DataLoader, Dataset, SequentialSampler

from yolo_agent.components.adapters.data_pipeline.runtime import (
    dataset_manifest_hash,
    rebuild_dataloader,
    records_from_yolo_dataset,
)


class TinyDataset(Dataset[int]):
    im_files = ["a.jpg", "b.jpg"]
    labels = [
        {
            "normalized": True,
            "bbox_format": "xywh",
            "bboxes": [[0.5, 0.5, 0.1, 0.2]],
            "cls": [[1]],
        },
        {
            "normalized": True,
            "bbox_format": "xywh",
            "bboxes": [],
            "cls": [],
            "is_hard_negative": True,
            "false_negative_score": 0.7,
        },
    ]

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> int:
        return index


def test_yolo_records_preserve_local_error_evidence() -> None:
    records = records_from_yolo_dataset(TinyDataset())

    assert records[0].normalized_areas == [0.020000000000000004]
    assert records[0].class_ids == [1]
    assert records[1].is_hard_negative is True
    assert records[1].false_negative_score == 0.7
    assert dataset_manifest_hash(TinyDataset(), records) == dataset_manifest_hash(
        TinyDataset(), records
    )


def test_loader_rebuild_preserves_dataset_and_batch_contract() -> None:
    original = DataLoader(TinyDataset(), batch_size=2, num_workers=0)
    rebuilt = rebuild_dataloader(original, SequentialSampler(original.dataset))

    assert rebuilt.dataset is original.dataset
    assert rebuilt.batch_size == original.batch_size
    assert list(next(iter(rebuilt))) == [0, 1]
