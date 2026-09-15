#!/usr/bin/env python3
"""End-to-end plumbing proof on CPU: Dataset -> collate -> model, both modes.

Pulls a few real samples through RddDataset + detection_collate and pushes them
through the model in train mode (must return finite losses that backprop) and
eval mode (must return per-image prediction dicts). This is the "the wiring
works" check to run before spending a GPU epoch on it.

    python scripts/smoke_test.py                 # transfer-learning model (downloads COCO weights)
    python scripts/smoke_test.py --no-pretrained # random init, fully offline
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from data.dataset import RddDataset, detection_collate  # noqa: E402
from models.factory import build_model  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=str(ROOT / "data/rdd/Czech/train/annotations/xmls"))
    ap.add_argument("--img", default=str(ROOT / "data/rdd/Czech/train/images"))
    ap.add_argument("--split-json", default=str(ROOT / "data/splits/czech_splits.json"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=3, help="samples to pull")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    ds = RddDataset(args.ann, args.img, args.split_json,
                    split=args.split, fold=args.fold)
    print(f"dataset: split={args.split} fold={args.fold}  ->  {len(ds)} images")

    n = min(args.n, len(ds))
    images, targets = detection_collate([ds[i] for i in range(n)])
    images, targets = list(images), list(targets)
    print(f"pulled {n} samples; box counts per image: "
          f"{[int(t['boxes'].shape[0]) for t in targets]}")

    model = build_model(name="retinanet", num_classes=5,
                        pretrained=not args.no_pretrained)

    # --- train mode: finite losses + backward -------------------------------
    model.train()
    losses = model(images, targets)
    assert isinstance(losses, dict), f"train() should return a loss dict, got {type(losses)}"
    for k, v in losses.items():
        assert torch.isfinite(v), f"non-finite loss {k}={v}"
    total = sum(losses.values())
    total.backward()
    print("\ntrain() loss dict:")
    for k, v in losses.items():
        print(f"  {k:<22}: {float(v):.4f}")
    print(f"  {'sum':<22}: {float(total):.4f}  (backward ok)")

    # --- eval mode: per-image prediction dicts ------------------------------
    model.eval()
    with torch.no_grad():
        preds = model(images)
    assert isinstance(preds, list) and len(preds) == n
    for p in preds:
        assert {"boxes", "labels", "scores"} <= set(p)
        assert p["boxes"].shape[0] == p["labels"].shape[0] == p["scores"].shape[0]
    p0 = preds[0]
    print("\neval() prediction[0] shapes:")
    print(f"  boxes : {tuple(p0['boxes'].shape)}  ({p0['boxes'].dtype})")
    print(f"  labels: {tuple(p0['labels'].shape)}  ({p0['labels'].dtype})")
    print(f"  scores: {tuple(p0['scores'].shape)}  ({p0['scores'].dtype})")
    print("\nplumbing OK")


if __name__ == "__main__":
    main()
