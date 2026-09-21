#!/usr/bin/env python3
"""Evaluate a checkpoint on a fold's val (or test) split: mAP + per-class AP.

    python scripts/evaluate.py --config configs/baseline_retinanet.yaml \
        --checkpoint runs/fold0/last.pt --fold 0

Prints the metrics table and writes it to JSON (default: <ckpt_dir>/metrics.json).
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from data.dataset import RddDataset, detection_collate  # noqa: E402
from data.voc import VALID_CLASSES  # noqa: E402
from evaluation.detect_eval import compute_map, run_inference  # noqa: E402
from models.factory import build_model  # noqa: E402
from utils.config import load_config  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split-json", default=str(ROOT / "data/splits/czech_splits.json"))
    ap.add_argument("--ann-dir", default=None)
    ap.add_argument("--img-dir", default=None)
    ap.add_argument("--output", default=None, help="metrics JSON path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(args.device)

    root = pathlib.Path(cfg.data.dataset_root)
    ann_dir = args.ann_dir or str(root / "Czech/train/annotations/xmls")
    img_dir = args.img_dir or str(root / "Czech/train/images")

    fold = None if args.split == "test" else args.fold
    ds = RddDataset(ann_dir, img_dir, args.split_json, split=args.split, fold=fold)
    if args.limit is not None:
        ds = Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False,
                        num_workers=cfg.data.num_workers, collate_fn=detection_collate)

    min_size = cfg.data.image_size
    max_size = round(min_size * 683 / 512)
    model = build_model(name=cfg.model.name, num_classes=cfg.model.num_classes,
                        min_size=min_size, max_size=max_size, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    print(f"evaluating {args.checkpoint} on {args.split} "
          f"(fold={fold}, n={len(ds)}, device={device})")
    preds, targets = run_inference(model, loader, device)
    metrics = compute_map(preds, targets, class_metrics=True)

    print(f"\n  mAP@[.50:.95] : {metrics['map']:.4f}")
    print(f"  AP50          : {metrics['map_50']:.4f}")
    print(f"  AP75          : {metrics['map_75']:.4f}")
    print("  per-class AP@[.50:.95]:")
    for cls in VALID_CLASSES:
        val = metrics.get("per_class", {}).get(cls)
        print(f"    {cls}: {val:.4f}" if val is not None else f"    {cls}: n/a")

    out = pathlib.Path(args.output) if args.output else \
        pathlib.Path(args.checkpoint).with_name("metrics.json")
    payload = {
        "checkpoint": str(args.checkpoint),
        "split": args.split, "fold": fold,
        "epoch": ckpt.get("epoch"), "n_images": len(ds),
        "metrics": metrics,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
