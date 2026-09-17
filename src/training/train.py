#!/usr/bin/env python3
"""Train one CV fold: config -> data -> model -> loop -> checkpoints + logging.

    python src/training/train.py --config configs/baseline_retinanet.yaml --fold 0
    python src/training/train.py --config configs/smoke.yaml --limit 20   # CPU smoke

Selects train = dev folds != fold, val = fold (see RddDataset). Checkpoints
last.pt every epoch and lowest_val_loss.pt. Val uses the train-mode/no-grad
loss trick — val LOSS is a diagnostic only, NOT the model-selection criterion.
Real selection is val mAP (a later milestone); until then rely on last.pt plus
the per-epoch metrics.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

# run as `python src/training/train.py`: put src/ on the path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from data.dataset import RddDataset, detection_collate  # noqa: E402
from models.factory import build_model  # noqa: E402
from training.engine import evaluate_loss, train_one_epoch  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.seed import seed_everything  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


class RunLogger:
    """Thin scalar logger over TensorBoard / MLflow / stdout-only ('none')."""

    def __init__(self, backend, logdir, run_name, params):
        self.backend = backend
        self.tb = self.mlflow = None
        if backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(logdir))
        elif backend == "mlflow":
            import mlflow
            self.mlflow = mlflow
            mlflow.set_experiment(run_name)
            mlflow.start_run()
            mlflow.log_params(params)
        elif backend != "none":
            raise ValueError(f"unknown logger backend {backend!r}")

    def log_scalar(self, tag, value, step):
        if self.tb is not None:
            self.tb.add_scalar(tag, value, step)
        if self.mlflow is not None:
            self.mlflow.log_metric(tag.replace("/", "_"), value, step=step)

    def close(self):
        if self.tb is not None:
            self.tb.close()
        if self.mlflow is not None:
            self.mlflow.end_run()


def _maybe_limit(ds, limit):
    if limit is None or limit >= len(ds):
        return ds
    return Subset(ds, range(limit))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap train & val to N images (CPU smoke)")
    ap.add_argument("--output-dir", default=str(ROOT / "runs"))
    ap.add_argument("--split-json", default=str(ROOT / "data/splits/czech_splits.json"))
    ap.add_argument("--ann-dir", default=None, help="override; else <root>/Czech/train/annotations/xmls")
    ap.add_argument("--img-dir", default=None, help="override; else <root>/Czech/train/images")
    ap.add_argument("--logger", choices=["tensorboard", "mlflow", "none"], default="tensorboard")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed_everything(cfg.train.seed)
    device = torch.device(args.device)

    root = pathlib.Path(cfg.data.dataset_root)
    ann_dir = args.ann_dir or str(root / "Czech/train/annotations/xmls")
    img_dir = args.img_dir or str(root / "Czech/train/images")

    train_ds = _maybe_limit(
        RddDataset(ann_dir, img_dir, args.split_json, split="train", fold=args.fold),
        args.limit)
    val_ds = _maybe_limit(
        RddDataset(ann_dir, img_dir, args.split_json, split="val", fold=args.fold),
        args.limit)
    print(f"fold {args.fold}: train={len(train_ds)}  val={len(val_ds)}  device={device}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(
            f"empty dataset (train={len(train_ds)}, val={len(val_ds)}). No XMLs "
            f"matched this partition. Check that the data exists:\n"
            f"  ann-dir : {ann_dir}\n  img-dir : {img_dir}\n"
            f"  split   : {args.split_json}\n"
            f"On a box without the Czech data, pass --ann-dir/--img-dir/"
            f"--split-json pointing at a local dataset.")

    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers,
                              collate_fn=detection_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers,
                            collate_fn=detection_collate)

    min_size = cfg.data.image_size
    max_size = round(min_size * 683 / 512)
    model = build_model(name=cfg.model.name, num_classes=cfg.model.num_classes,
                        min_size=min_size, max_size=max_size,
                        pretrained=cfg.model.pretrained).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.train.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=cfg.train.lr, momentum=0.9,
                                    weight_decay=cfg.train.weight_decay)
    elif cfg.train.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=cfg.train.lr,
                                      weight_decay=cfg.train.weight_decay)
    else:  # adam
        optimizer = torch.optim.Adam(params, lr=cfg.train.lr,
                                     weight_decay=cfg.train.weight_decay)

    warmup_iters = min(cfg.train.warmup_iters, max(len(train_loader) - 1, 0))
    warmup_scheduler = (
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=max(warmup_iters, 1))
        if warmup_iters > 0 else None)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs)

    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    out_dir = pathlib.Path(args.output_dir) / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.logger, out_dir / "tb",
                       run_name=f"{cfg.model.name}_fold{args.fold}",
                       params=_flat_params(cfg, args))

    # val LOSS is NOT the model-selection criterion: a low val loss on a tiny or
    # negative-heavy val fold is meaningless noise (the smoke showed 0.0466 on a
    # single negative image). Real selection = val mAP, deferred to the eval
    # milestone. This file only records the lowest-val-loss epoch as a diagnostic
    # convenience; pick the final model with mAP, and lean on last.pt until then.
    lowest_val = float("inf")
    global_step = 0
    for epoch in range(cfg.train.epochs):
        train_loss, global_step = train_one_epoch(
            model, optimizer, train_loader, device, epoch=epoch,
            amp=use_amp, scaler=scaler, warmup_scheduler=warmup_scheduler,
            warmup_iters=warmup_iters, grad_clip=cfg.train.grad_clip,
            global_step=global_step, logger=logger)
        val_loss = evaluate_loss(model, val_loader, device, amp=use_amp)
        lr_now = optimizer.param_groups[0]["lr"]  # the LR this epoch trained at
        main_scheduler.step()

        logger.log_scalar("train/epoch_loss", train_loss, epoch)
        logger.log_scalar("val/epoch_loss", val_loss, epoch)
        print(f"epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"lr={lr_now:.2e}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "fold": args.fold,
            "config": dataclasses.asdict(cfg),
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_loss < lowest_val:
            lowest_val = val_loss
            torch.save(ckpt, out_dir / "lowest_val_loss.pt")
            print(f"  lowest val_loss so far={val_loss:.4f} -> "
                  f"{out_dir / 'lowest_val_loss.pt'} (diagnostic, not selection)")

    logger.close()
    print(f"done. checkpoints in {out_dir}")


def _flat_params(cfg, args):
    p = {}
    for section, sub in dataclasses.asdict(cfg).items():
        for k, v in sub.items():
            p[f"{section}.{k}"] = v
    p["fold"] = args.fold
    p["limit"] = args.limit
    return p


if __name__ == "__main__":
    main()
