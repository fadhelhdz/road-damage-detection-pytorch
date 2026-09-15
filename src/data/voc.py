from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

VALID_CLASSES = ("D00", "D10", "D20", "D40")

# Single source of truth for the torchvision detection label ids: 1-indexed,
# with 0 reserved for background. Shared by parser/Dataset/evaluator; wired in
# at M4.
CLASS_TO_ID = {c: i + 1 for i, c in enumerate(VALID_CLASSES)}

# Inverse, for decoding model predictions back to D-codes. 0 (background) has no
# entry — detectors never emit it as a prediction label.
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}

# Keys are lowercased/stripped. Accepts both the D-codes and the descriptive
# names used in the raw XML. The four descriptive names below are the only
# object labels present in the Czech train subset (verified: full parse of
# 2,829 XMLs). The D-code keys never fire for Czech but are kept as harmless
# carries for other RDD2022 regions.
NAME_TO_CODE = {
    "d00": "D00", "longitudinal crack": "D00",
    "d10": "D10", "transverse crack": "D10",
    "d20": "D20", "alligator crack": "D20",
    "d40": "D40", "pothole": "D40",
}


@dataclass
class Sample:
    image_path: str
    width: int
    height: int
    boxes: np.ndarray   # (N, 4) xyxy float32; (0, 4) if none
    labels: list[str]   # length N; only D00/D10/D20/D40


def parse_voc_xml(xml_path: str, images_dir: str,
                  stats: Counter | None = None) -> Sample:
    """Parse one Pascal-VOC XML into a Sample.

    - Returns boxes (0, 4) and empty labels for negative images (no <object>).
    - Drops any object whose <name> is not in NAME_TO_CODE (the 'other
      corruption' case); counts it as 'dropped_unknown_class'.
    - Reads width/height from <size>; falls back to the image header only when
      <size> is missing/zero. Clips each box to [0, w]/[0, h], then validates
      xmax>xmin and ymax>ymin. Invalid/degenerate boxes are counted, not raised.

    Pass a shared Counter as `stats` to accumulate corruption counts across a
    whole dataset; omit it and counting is a no-op.
    """
    bump = (lambda k: stats.update([k])) if stats is not None else (lambda k: None)

    root = ET.parse(xml_path).getroot()

    fn = root.findtext("filename") or Path(xml_path).with_suffix(".jpg").name
    image_path = str(Path(images_dir) / fn)
    width, height = _read_size(root, image_path, bump)

    boxes: list[list[float]] = []
    labels: list[str] = []
    for obj in root.findall("object"):
        code = NAME_TO_CODE.get((obj.findtext("name") or "").strip().lower())
        if code is None:
            bump("dropped_unknown_class")
            continue

        bb = obj.find("bndbox")
        try:
            xmin, ymin, xmax, ymax = (
                float(bb.findtext(t)) for t in ("xmin", "ymin", "xmax", "ymax")
            )
        except (AttributeError, TypeError, ValueError):
            bump("invalid_box")
            continue

        if width > 0:
            xmin = min(max(xmin, 0.0), width)
            xmax = min(max(xmax, 0.0), width)
        if height > 0:
            ymin = min(max(ymin, 0.0), height)
            ymax = min(max(ymax, 0.0), height)

        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(code)
        else:
            bump("invalid_box")

    arr = (np.asarray(boxes, dtype=np.float32) if boxes
           else np.zeros((0, 4), dtype=np.float32))
    return Sample(image_path, width, height, arr, labels)


def _read_size(root, image_path: str, bump) -> tuple[int, int]:
    size = root.find("size")
    w = h = 0
    if size is not None:
        try:
            w = int(float(size.findtext("width") or 0))
            h = int(float(size.findtext("height") or 0))
        except ValueError:
            w = h = 0
    if w > 0 and h > 0:
        return w, h

    bump("size_from_image")
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return im.width, im.height
    except Exception:
        bump("size_unknown")
        logger.warning("No usable <size> and cannot read image: %s", image_path)
        return 0, 0  # 0 => clipping skipped, raw boxes kept