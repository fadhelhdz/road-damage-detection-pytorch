"""Training/eval steps for detection models.

Kept free of config/CLI/logging so they're unit-testable in isolation:
train.py owns the wiring, these own the per-epoch mechanics.
"""
from __future__ import annotations

import math

import torch
from torch.nn.utils import clip_grad_norm_


def _to_device(images, targets, device):
    images = [im.to(device, non_blocking=True) for im in images]
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
               for t in targets]
    return images, targets


def train_one_epoch(model, optimizer, loader, device, *, epoch=0,
                    amp=False, scaler=None, warmup_scheduler=None,
                    warmup_iters=0, grad_clip=0.0, global_step=0, logger=None):
    """One training epoch. Returns (mean_loss, global_step).

    Linear LR warmup is stepped per-iteration for the first `warmup_iters`
    total iterations (i.e. across epoch 0); AMP autocast/scaler are used only
    when `amp` is on (CUDA); grads are clipped to `grad_clip` when > 0.
    """
    model.train()
    device_type = device.type
    running, n = 0.0, 0

    for images, targets in loader:
        images, targets = _to_device(images, targets, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, enabled=amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if not math.isfinite(loss.item()):
            raise RuntimeError(f"non-finite loss at step {global_step}: "
                               f"{ {k: float(v) for k, v in loss_dict.items()} }")

        if amp and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # per-iteration warmup only during the initial warmup window
        if warmup_scheduler is not None and global_step < warmup_iters:
            warmup_scheduler.step()

        running += loss.item()
        n += 1
        global_step += 1
        if logger is not None:
            logger.log_scalar("train/step_loss", loss.item(), global_step)
            logger.log_scalar("train/lr", optimizer.param_groups[0]["lr"],
                              global_step)

    return running / max(n, 1), global_step


@torch.no_grad()
def evaluate_loss(model, loader, device, *, amp=False):
    """Mean validation loss.

    A detector only returns a loss dict in *train* mode (eval mode returns
    decoded predictions). So we force train mode to get losses but wrap in
    no_grad to skip autograd — the "train-mode/no-grad" trick. BatchNorm/Dropout
    therefore use train-mode behavior, which is standard for a val-loss signal;
    real AP later will use eval mode.
    """
    was_training = model.training
    model.train()
    device_type = device.type
    running, n = 0.0, 0

    for images, targets in loader:
        images, targets = _to_device(images, targets, device)
        with torch.autocast(device_type=device_type, enabled=amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        running += loss.item()
        n += 1

    model.train(was_training)
    return running / max(n, 1)
