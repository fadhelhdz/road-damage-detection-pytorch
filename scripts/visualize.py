#!/usr/bin/env python3
"""Save prediction overlays for a few val images — a visual sanity check.

    python scripts/visualize.py --config configs/baseline_retinanet.yaml \
        --checkpoint runs/fold0/last.pt --fold 0 --n 8 --score-thresh 0.3

Writes <output-dir>/<stem>.png with GT boxes (green) and predictions above
--score-thresh (red, class+score). Confirms the model detects sensible damage
rather than trusting the mAP number alone.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
from torchvision.io import write_png  # noqa: E402

from data.dataset import RddDataset  # noqa: E402
from models.factory import build_model  # noqa: E402
from utils.config import load_config  # noqa: E402
from visualization.draw import draw_detections  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=8, help="images to render")
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--split-json", default=str(ROOT / "data/splits/czech_splits.json"))
    ap.add_argument("--ann-dir", default=None)
    ap.add_argument("--img-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(args.device)
    root = pathlib.Path(cfg.data.dataset_root)
    ann_dir = args.ann_dir or str(root / "Czech/train/annotations/xmls")
    img_dir = args.img_dir or str(root / "Czech/train/images")

    fold = None if args.split == "test" else args.fold
    ds = RddDataset(ann_dir, img_dir, args.split_json, split=args.split, fold=fold)

    min_size = cfg.data.image_size
    max_size = round(min_size * 683 / 512)
    model = build_model(name=cfg.model.name, num_classes=cfg.model.num_classes,
                        min_size=min_size, max_size=max_size, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    out_dir = pathlib.Path(args.output_dir) if args.output_dir else \
        pathlib.Path(args.checkpoint).with_name("viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    # evenly spaced indices so we sample across the fold, not just the head
    n = min(args.n, len(ds))
    idxs = [round(i * (len(ds) - 1) / max(n - 1, 1)) for i in range(n)]

    print(f"rendering {n} images from {args.split} (fold={fold}, n={len(ds)})")
    with torch.inference_mode():
        for i in idxs:
            image, target = ds[i]
            pred = model([image.to(device)])[0]
            canvas = draw_detections(image, pred, target, args.score_thresh)
            stem = pathlib.Path(ds.samples[i].image_path).stem
            path = out_dir / f"{stem}.png"
            write_png(canvas, str(path))
            n_kept = int((pred["scores"].cpu() >= args.score_thresh).sum())
            print(f"  {stem}: GT={target['boxes'].shape[0]} "
                  f"pred>={args.score_thresh}={n_kept} -> {path.name}")

    print(f"\nsaved {n} overlays -> {out_dir}")


if __name__ == "__main__":
    main()
