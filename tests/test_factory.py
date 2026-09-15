"""Factory tests: build offline (pretrained=False) and prove both forward modes.

Uses random init and a 1-2 image mini-batch of small random tensors so it runs
in a couple of seconds on CPU with no weight download.
"""
from __future__ import annotations

import pytest
import torch

from models.factory import build_model

NUM_CLASSES = 5


@pytest.fixture(scope="module")
def model():
    # small min/max keeps the CPU forward fast; random init = no network.
    return build_model(name="retinanet", num_classes=NUM_CLASSES,
                       min_size=128, max_size=170, pretrained=False)


def _random_batch(n, num_boxes=2):
    images, targets = [], []
    for _ in range(n):
        images.append(torch.rand(3, 100, 120))
        # valid xyxy boxes inside the image; 1-indexed labels in [1, NUM_CLASSES)
        boxes = torch.tensor([[10.0, 10.0, 50.0, 60.0],
                              [20.0, 15.0, 90.0, 80.0]])[:num_boxes]
        labels = torch.randint(1, NUM_CLASSES, (num_boxes,), dtype=torch.int64)
        targets.append({"boxes": boxes, "labels": labels})
    return images, targets


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model(name="nope")


def test_classification_head_has_num_classes(model):
    head = model.head.classification_head
    assert head.num_classes == NUM_CLASSES
    # cls logits conv emits num_anchors * num_classes channels
    assert head.cls_logits.out_channels == head.num_anchors * NUM_CLASSES


def test_train_mode_returns_finite_loss_dict(model):
    model.train()
    images, targets = _random_batch(2)
    losses = model(images, targets)
    assert isinstance(losses, dict) and losses
    assert {"classification", "bbox_regression"} <= set(losses)
    for k, v in losses.items():
        assert torch.isfinite(v), f"{k} not finite: {v}"
    # loss actually backpropagates
    sum(losses.values()).backward()


def test_eval_mode_returns_prediction_dicts(model):
    model.eval()
    images, _ = _random_batch(2)
    with torch.no_grad():
        preds = model(images)
    assert isinstance(preds, list) and len(preds) == 2
    for p in preds:
        assert {"boxes", "labels", "scores"} <= set(p)
        n = p["boxes"].shape[0]
        assert p["labels"].shape[0] == n and p["scores"].shape[0] == n
        assert p["boxes"].shape[1] == 4
        if n:
            assert p["boxes"].dtype == torch.float32
            assert p["labels"].dtype == torch.int64
