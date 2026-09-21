"""Detection-model factory.

`build_model` returns a torchvision detection model with its classification head
resized to our class count. Labels are 1-indexed (CLASS_TO_ID), so `num_classes`
counts background as index 0: 4 damage classes -> num_classes=5. This holds for
RetinaNet/FCOS (sigmoid focal, no explicit background class) *and* Faster R-CNN
(softmax, class 0 = background) — torchvision builds the one-hot / logit target
by indexing at the label value, so the highest label (4) requires column 4 to
exist, i.e. num_classes >= 5. Column 0 is simply a dead channel for the
sigmoid models. Keeping one uniform label scheme is why we standardize on 5.

Adding a model is a new `_build_*` branch below, not a rewrite.
"""
from __future__ import annotations

from functools import partial

import torch.nn as nn


def build_model(name: str = "retinanet", num_classes: int = 5,
                min_size: int = 512, max_size: int = 683,
                trainable_backbone_layers: int = 3, pretrained: bool = True):
    """Build a detection model with a `num_classes`-way classification head.

    pretrained=True loads COCO weights and swaps the head (transfer learning);
    pretrained=False builds a fresh, download-free model (random init) — used by
    the fast unit tests.
    """
    builders = {
        "retinanet": _build_retinanet,
        # "fcos": _build_fcos,               # future: same head-swap shape
        # "fasterrcnn": _build_fasterrcnn,
    }
    if name not in builders:
        raise ValueError(f"unknown model {name!r}; known: {sorted(builders)}")
    return builders[name](num_classes, min_size, max_size,
                          trainable_backbone_layers, pretrained)


def _build_retinanet(num_classes, min_size, max_size,
                     trainable_backbone_layers, pretrained):
    from torchvision.models.detection import (
        RetinaNet_ResNet50_FPN_Weights,
        retinanet_resnet50_fpn,
    )
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    if pretrained:
        # Transfer learning: load full COCO weights (90+1 classes).
        model = retinanet_resnet50_fpn(
            weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1,
            min_size=min_size, max_size=max_size,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    else:
        # Fresh model, no network access (random init).
        model = retinanet_resnet50_fpn(
            weights=None, weights_backbone=None,
            min_size=min_size, max_size=max_size,
            trainable_backbone_layers=trainable_backbone_layers,
        )

    # ALWAYS swap the classification head to our num_classes + GroupNorm(32),
    # regardless of `pretrained`. This keeps the architecture identical whether
    # or not we started from COCO weights, so a checkpoint trained with
    # pretrained=True loads cleanly into a pretrained=False eval build. (Passing
    # num_classes to the constructor instead would give a *different* head — no
    # GroupNorm — and break state_dict loading.)
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.backbone.out_channels  # 256
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels, num_anchors, num_classes,
        norm_layer=partial(nn.GroupNorm, 32),
    )
    return model
