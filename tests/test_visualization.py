"""Unit tests for draw_detections — synthetic tensors, no model, CPU."""
from __future__ import annotations

import torch

from visualization.draw import draw_detections


def _img():
    return torch.rand(3, 64, 80)  # float CHW in [0,1]


def test_returns_uint8_same_shape():
    out = draw_detections(_img())
    assert out.dtype == torch.uint8
    assert out.shape == (3, 64, 80)


def test_draws_gt_and_preds_without_error():
    target = {"boxes": torch.tensor([[5.0, 5.0, 30.0, 40.0]]),
              "labels": torch.tensor([1])}
    pred = {"boxes": torch.tensor([[6.0, 6.0, 31.0, 41.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([2])}
    out = draw_detections(_img(), pred, target, score_thresh=0.3)
    assert out.dtype == torch.uint8 and out.shape == (3, 64, 80)


def test_score_threshold_filters_low_confidence():
    # a low-score pred below threshold must not change the image;
    # one above threshold must.
    img = (torch.zeros(3, 40, 40))  # flat image -> any drawing changes pixels
    pred_lo = {"boxes": torch.tensor([[2.0, 2.0, 20.0, 20.0]]),
               "scores": torch.tensor([0.1]), "labels": torch.tensor([1])}
    pred_hi = {"boxes": torch.tensor([[2.0, 2.0, 20.0, 20.0]]),
               "scores": torch.tensor([0.9]), "labels": torch.tensor([1])}
    base = draw_detections(img, score_thresh=0.3)
    lo = draw_detections(img, pred_lo, score_thresh=0.3)
    hi = draw_detections(img, pred_hi, score_thresh=0.3)
    assert torch.equal(base, lo)        # filtered out -> unchanged
    assert not torch.equal(base, hi)    # drawn -> changed


def test_empty_pred_and_target_is_safe():
    pred = {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.int64)}
    target = {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}
    out = draw_detections(_img(), pred, target)
    assert out.shape == (3, 64, 80)
