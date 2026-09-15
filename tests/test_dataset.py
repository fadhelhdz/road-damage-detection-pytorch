"""RddDataset contract tests on the committed mini_voc fixtures.

Asserts the torchvision-detection item contract: float32 CHW image, xyxy
float32 boxes, int64 1-indexed labels, and — the easy one to get wrong — a
negative image yields an empty target (boxes (0, 4), labels (0,)) rather than
being dropped.
"""
from __future__ import annotations

import json
import pathlib

import pytest
import torch

from data.dataset import RddDataset, detection_collate
from data.voc import CLASS_TO_ID

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "mini_voc"
ANN = FIX / "annotations"
IMG = FIX / "images"

STEMS = ["neg_0001", "corrupt_0002", "normal_0003", "clip_0004", "nosize_0005"]
_IMAGES = {f"{s}.jpg": (24, 24) for s in STEMS}


@pytest.fixture(scope="module", autouse=True)
def placeholder_images():
    """Regenerate the tiny placeholder images the fixtures reference."""
    from PIL import Image
    IMG.mkdir(parents=True, exist_ok=True)
    for name, (w, h) in _IMAGES.items():
        p = IMG / name
        if not p.exists():
            Image.new("RGB", (w, h), (127, 127, 127)).save(p)


def _make_split(tmp_path, assignments) -> str:
    p = tmp_path / "splits.json"
    p.write_text(json.dumps({"_meta": {}, "assignments": assignments}))
    return str(p)


@pytest.fixture
def all_in_val(tmp_path):
    """All five fixtures in dev fold 0, so split='val', fold=0 serves them all."""
    return _make_split(
        tmp_path, {f"{s}.jpg": {"split": "dev", "fold": 0} for s in STEMS})


def _ds(split_json, split="val", fold=0):
    return RddDataset(str(ANN), str(IMG), split_json, split=split, fold=fold)


def test_item_contract_dtypes_and_xyxy(all_in_val):
    ds = _ds(all_in_val)
    assert len(ds) == len(STEMS)
    for i in range(len(ds)):
        image, target = ds[i]
        # float32 CHW image
        assert image.dtype == torch.float32
        assert image.ndim == 3 and image.shape[0] == 3
        assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0

        boxes, labels = target["boxes"], target["labels"]
        assert boxes.dtype == torch.float32
        assert boxes.ndim == 2 and boxes.shape[1] == 4
        assert labels.dtype == torch.int64
        assert labels.ndim == 1
        assert boxes.shape[0] == labels.shape[0]
        if boxes.shape[0]:
            # xyxy: xmax > xmin, ymax > ymin
            assert torch.all(boxes[:, 2] > boxes[:, 0])
            assert torch.all(boxes[:, 3] > boxes[:, 1])
            # 1-indexed labels (0 is reserved for background)
            assert torch.all(labels >= 1)
            assert torch.all(labels <= len(CLASS_TO_ID))


def test_negative_image_yields_empty_target(all_in_val):
    ds = _ds(all_in_val)
    empties = []
    for i in range(len(ds)):
        _, t = ds[i]
        if t["labels"].numel() == 0:
            empties.append((t["boxes"], t["labels"]))
    # exactly one fixture (neg_0001) is a negative image
    assert len(empties) == 1
    boxes, labels = empties[0]
    assert boxes.shape == (0, 4)
    assert labels.shape == (0,)
    assert boxes.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_labels_map_through_class_to_id(all_in_val):
    ds = _ds(all_in_val)
    # normal_0003 is the only 2-object image: D00, D40 -> ids 1, 4
    two = [t for _, t in (ds[i] for i in range(len(ds)))
           if t["labels"].numel() == 2]
    assert len(two) == 1
    assert two[0]["labels"].tolist() == [CLASS_TO_ID["D00"], CLASS_TO_ID["D40"]]


def test_detection_collate_keeps_per_image_targets(all_in_val):
    ds = _ds(all_in_val)
    images, targets = detection_collate([ds[0], ds[1], ds[2]])
    assert len(images) == len(targets) == 3
    assert all(isinstance(im, torch.Tensor) for im in images)
    assert all(set(t) >= {"boxes", "labels"} for t in targets)


def test_split_selection_train_val_test(tmp_path):
    assignments = {
        "neg_0001.jpg": {"split": "dev", "fold": 0},
        "corrupt_0002.jpg": {"split": "dev", "fold": 1},
        "normal_0003.jpg": {"split": "dev", "fold": 0},
        "clip_0004.jpg": {"split": "test", "fold": None},
        "nosize_0005.jpg": {"split": "dev", "fold": 1},
    }
    sj = _make_split(tmp_path, assignments)

    def names(split, fold=None):
        ds = RddDataset(str(ANN), str(IMG), sj, split=split, fold=fold)
        return {pathlib.Path(s.image_path).name for s in ds.samples}

    assert names("val", 0) == {"neg_0001.jpg", "normal_0003.jpg"}
    assert names("train", 0) == {"corrupt_0002.jpg", "nosize_0005.jpg"}  # dev, fold!=0
    assert names("test") == {"clip_0004.jpg"}
    # val(0) and train(0) partition the dev set with no overlap
    assert names("val", 0).isdisjoint(names("train", 0))


def test_invalid_args_raise(all_in_val):
    with pytest.raises(ValueError):
        RddDataset(str(ANN), str(IMG), all_in_val, split="dev", fold=0)  # bad name
    with pytest.raises(ValueError):
        RddDataset(str(ANN), str(IMG), all_in_val, split="train")       # fold missing
    with pytest.raises(ValueError):
        RddDataset(str(ANN), str(IMG), all_in_val, split="test", fold=0)  # fold given
