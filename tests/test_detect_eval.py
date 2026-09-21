"""Known-answer tests for compute_map — CPU-only, no data, no model.

Feeds hand-built preds/targets with predictable IoU so mAP is known in advance.
"""
from __future__ import annotations

import torch

from evaluation.detect_eval import compute_map


def _b(rows):
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, 4)


def test_perfect_predictions_map50_is_one():
    gt = {"boxes": _b([[0, 0, 100, 100], [50, 50, 150, 150]]),
          "labels": torch.tensor([1, 2])}
    pred = {"boxes": gt["boxes"].clone(),
            "scores": torch.tensor([1.0, 1.0]),
            "labels": torch.tensor([1, 2])}
    res = compute_map([pred], [gt])
    assert res["map_50"] > 0.99
    assert res["map"] > 0.99  # perfect at every IoU threshold too


def test_empty_predictions_map_is_zero():
    gt = {"boxes": _b([[0, 0, 100, 100]]), "labels": torch.tensor([1])}
    pred = {"boxes": _b([]), "scores": torch.tensor([]),
            "labels": torch.tensor([], dtype=torch.int64)}
    res = compute_map([pred], [gt])
    # GT present, no detections -> AP 0 (torchmetrics may report 0 or -1)
    assert res["map"] <= 0.01


def test_shifted_box_ap50_greater_than_ap75():
    # GT [0,0,100,100] vs pred [25,0,125,100]: intersection 75x100=7500,
    # union 12500 -> IoU = 0.60. TP at .50, FP at .75.
    gt = {"boxes": _b([[0, 0, 100, 100]]), "labels": torch.tensor([1])}
    pred = {"boxes": _b([[25, 0, 125, 100]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1])}
    res = compute_map([pred], [gt])
    assert res["map_50"] > res["map_75"]
    assert res["map_50"] > 0.99   # counts at IoU 0.50
    assert res["map_75"] <= 0.01  # missed at IoU 0.75


def test_per_class_keys_use_dcodes():
    gt = {"boxes": _b([[0, 0, 100, 100], [10, 10, 60, 60]]),
          "labels": torch.tensor([1, 3])}  # D00, D20
    pred = {"boxes": gt["boxes"].clone(),
            "scores": torch.tensor([1.0, 1.0]),
            "labels": torch.tensor([1, 3])}
    res = compute_map([pred], [gt])
    assert set(res["per_class"]) <= {"D00", "D10", "D20", "D40"}
    assert "D00" in res["per_class"] and "D20" in res["per_class"]
