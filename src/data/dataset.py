"""torchvision-detection Dataset over the RDD2022 Czech subset.

Reads the frozen split manifest (data/splits/czech_splits.json) and serves the
images for one partition. The split is the single source of truth: this class
never re-derives folds, it only *selects* by the assignment already committed.

Item contract (torchvision detection):
    image  -> FloatTensor[3, H, W] in [0, 1]
    target -> {
        "boxes":  FloatTensor[N, 4]  xyxy, float32   ((0, 4) for a negative)
        "labels": Int64Tensor[N]     1-indexed via CLASS_TO_ID, 0 = background
                                     ((0,) for a negative)
        "image_id": Int64Tensor[1]
    }
A negative image (no objects) yields an *empty* target, not a skipped sample —
torchvision's detection losses accept empty boxes/labels and learn background
from them.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset

from data.voc import CLASS_TO_ID, parse_voc_xml

_VALID_SPLITS = ("train", "val", "test")


def _select(assignments: dict, split: str, fold) -> set[str]:
    """Image filenames belonging to the requested partition.

    train -> every dev image whose fold != `fold`
    val   -> every dev image whose fold == `fold`
    test  -> the frozen test set (fold must be None)
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")
    if split == "test":
        if fold is not None:
            raise ValueError("fold must be None for split='test'")
        return {fn for fn, a in assignments.items() if a["split"] == "test"}
    if fold is None:
        raise ValueError(f"split={split!r} requires a fold index")

    keep = set()
    for fn, a in assignments.items():
        if a["split"] != "dev":
            continue
        if (split == "val") == (a["fold"] == fold):
            keep.add(fn)
    return keep


class RddDataset(Dataset):
    def __init__(self, ann_dir, img_dir, split_json, split, fold=None,
                 transforms=None):
        self.img_dir = str(img_dir)
        self.transforms = transforms

        with open(split_json) as f:
            assignments = json.load(f)["assignments"]
        wanted = _select(assignments, split, fold)

        # Build the sample index the same way the split was built: iterate the
        # XMLs in sorted order and keep those whose image is in this partition.
        # This guarantees the image/label identity matches the manifest exactly.
        self.samples = []
        for xp in sorted(pathlib.Path(ann_dir).glob("*.xml")):
            s = parse_voc_xml(str(xp), self.img_dir)
            if pathlib.Path(s.image_path).name in wanted:
                self.samples.append(s)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        from PIL import Image
        with Image.open(s.image_path) as im:
            arr = np.array(im.convert("RGB"))            # HWC uint8 (writable copy)
        image = (torch.from_numpy(arr).permute(2, 0, 1)  # CHW
                 .contiguous().float().div_(255.0))       # float32 in [0, 1]

        boxes = torch.as_tensor(s.boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor([CLASS_TO_ID[c] for c in s.labels],
                                 dtype=torch.int64)
        target = {
            "boxes": boxes,                    # (N, 4) or (0, 4)
            "labels": labels,                  # (N,)   or (0,)
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def detection_collate(batch):
    """Collate variable-count detection targets: keep images/targets as tuples.

    Detection models take a list/tuple of images and a matching list/tuple of
    target dicts (boxes differ per image, so they can't be stacked).
    """
    return tuple(zip(*batch))
