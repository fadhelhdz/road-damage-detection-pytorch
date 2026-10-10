"""Overlay detection boxes on an image for visual sanity-checking.

GT boxes in green, predictions (score >= threshold) in red with class+score.
Kept model-free so it unit-tests on synthetic tensors.
"""
from __future__ import annotations

import torch
from torchvision.utils import draw_bounding_boxes

from data.voc import ID_TO_CLASS


def _to_uint8(image: torch.Tensor) -> torch.Tensor:
    if image.dtype == torch.uint8:
        return image.cpu()
    return (image.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)


def _name(label: int) -> str:
    return ID_TO_CLASS.get(int(label), str(int(label)))


def draw_detections(image, pred=None, target=None, score_thresh=0.3) -> torch.Tensor:
    """Return a uint8 [3,H,W] image with GT (green) and kept preds (red) drawn.

    `pred` = {boxes, scores, labels}; `target` = {boxes, labels}. Predictions
    below `score_thresh` are dropped so the overlay isn't flooded with the
    low-confidence tail.
    """
    img = _to_uint8(image)

    if target is not None and target["boxes"].numel():
        gt_labels = [f"GT:{_name(l)}" for l in target["labels"].tolist()]
        img = draw_bounding_boxes(img, target["boxes"].cpu(), labels=gt_labels,
                                  colors="green", width=2)

    if pred is not None and pred["boxes"].numel():
        keep = pred["scores"].cpu() >= score_thresh
        boxes = pred["boxes"].cpu()[keep]
        if boxes.numel():
            scores = pred["scores"].cpu()[keep].tolist()
            labels = pred["labels"].cpu()[keep].tolist()
            tags = [f"{_name(l)}:{s:.2f}" for l, s in zip(labels, scores)]
            img = draw_bounding_boxes(img, boxes, labels=tags,
                                      colors="red", width=2)
    return img
