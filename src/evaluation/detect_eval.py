"""COCO-style detection evaluation (mAP + per-class AP).

Uses torchmetrics' MeanAveragePrecision (pycocotools backend). preds/targets are
lists of dicts in torchmetrics' format:
    pred   = {"boxes": FloatTensor[N,4] xyxy, "scores": FloatTensor[N], "labels": Int64[N]}
    target = {"boxes": FloatTensor[M,4] xyxy, "labels": Int64[M]}
Per-class AP is keyed by D-code via ID_TO_CLASS.
"""
from __future__ import annotations

import torch
from torchmetrics.detection import MeanAveragePrecision

from data.voc import ID_TO_CLASS


@torch.inference_mode()
def run_inference(model, loader, device):
    """Run the model over `loader` in eval mode; return (preds, targets).

    Both are lists of CPU dicts aligned per image, ready for compute_map.
    """
    model.eval()
    preds, targets = [], []
    for images, tgts in loader:
        images = [im.to(device, non_blocking=True) for im in images]
        outputs = model(images)
        for out in outputs:
            preds.append({
                "boxes": out["boxes"].detach().cpu(),
                "scores": out["scores"].detach().cpu(),
                "labels": out["labels"].detach().cpu(),
            })
        for t in tgts:
            targets.append({
                "boxes": t["boxes"].cpu(),
                "labels": t["labels"].cpu(),
            })
    return preds, targets


def compute_map(preds, targets, class_metrics=True) -> dict:
    """COCO mAP over preds/targets. Returns map / map_50 / map_75 (+ per_class).

    per_class maps each present class id -> AP@[.50:.95], labelled by D-code.
    """
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                  class_metrics=class_metrics)
    metric.update(preds, targets)
    res = metric.compute()

    out = {
        "map": float(res["map"]),        # mAP@[.50:.95]
        "map_50": float(res["map_50"]),
        "map_75": float(res["map_75"]),
    }
    if class_metrics:
        per_class: dict[str, float] = {}
        classes = res.get("classes")
        ap = res.get("map_per_class")
        if classes is not None and ap is not None and ap.ndim > 0:
            for cid, a in zip(classes.tolist(), ap.tolist()):
                per_class[ID_TO_CLASS.get(int(cid), str(cid))] = float(a)
        out["per_class"] = per_class
    return out
