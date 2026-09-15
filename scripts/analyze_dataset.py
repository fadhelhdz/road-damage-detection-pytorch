#!/usr/bin/env python3
"""Analyze the Czech train subset of RDD2022.

Walks the annotation XMLs (and their images), then prints a human-readable
report and saves a machine-readable one. Covers:

  * image / XML / negative-image counts (and img<->xml mismatches)
  * per-class object counts, reconciled against the dataset card's 1,745
  * dropped 'other corruption' (and any other unknown) labels
  * invalid + clipped box counts (clip-then-validate, mirroring src/data/voc.py)
  * box size distribution: areas, COCO S/M/L buckets, aspect ratios
  * a perceptual-hash near-duplicate probe -> duplicate CLUSTERS, which is the
    evidence for choosing a group-aware split (dupes must not straddle train/val)

Run from the project root:  python scripts/analyze_dataset.py
"""
from __future__ import annotations

import argparse
import json
import sys
import pathlib
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

# --- make `src` importable and anchor data paths to the project root ----------
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data.voc import parse_voc_xml, VALID_CLASSES, NAME_TO_CODE  # noqa: E402
from data.dedup import hash_images, cluster_from_hashes  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
CARD_TOTAL_OBJECTS = 1745  # claim in docs/dataset_card.md, to reconcile against


# --- geometry (xyxy, no +1 — matches the iou_xyxy convention) -----------------
def box_area_xyxy(b: np.ndarray) -> np.ndarray:
    """Area of xyxy boxes. Same convention as iou_xyxy: (x2-x1)*(y2-y1)."""
    w = np.clip(b[:, 2] - b[:, 0], 0, None)
    h = np.clip(b[:, 3] - b[:, 1], 0, None)
    return w * h


def _read_size(root) -> tuple[int, int]:
    size = root.find("size")
    if size is None:
        return 0, 0
    try:
        return (int(float(size.findtext("width") or 0)),
                int(float(size.findtext("height") or 0)))
    except ValueError:
        return 0, 0


# --- annotation pass ----------------------------------------------------------
def analyze_annotations(ann_dir: pathlib.Path, img_dir: str) -> dict:
    """Single raw pass for geometry detail; cross-checked against the parser."""
    xml_paths = sorted(ann_dir.glob("*.xml"))

    per_class = Counter()
    corruption = Counter()          # unknown labels, by raw name
    negatives = 0
    n_clipped = 0                   # >=1 coord out of bounds (still valid after clip)
    n_invalid = 0                   # degenerate before/after clip, or unparseable
    missing_images = []
    kept_boxes = []                 # clipped, valid boxes -> for size distribution
    kept_labels = []

    for xp in xml_paths:
        root = ET.parse(xp).getroot()
        w, h = _read_size(root)
        objs = root.findall("object")
        if not objs:
            negatives += 1

        # image existence (uses the same filename logic as the parser)
        fn = root.findtext("filename") or xp.with_suffix(".jpg").name
        if not (pathlib.Path(img_dir) / fn).exists():
            missing_images.append(fn)

        kept_here = 0
        for obj in objs:
            code = NAME_TO_CODE.get((obj.findtext("name") or "").strip().lower())
            if code is None:
                corruption[(obj.findtext("name") or "").strip()] += 1
                continue
            bb = obj.find("bndbox")
            try:
                xmin, ymin, xmax, ymax = (float(bb.findtext(t))
                                          for t in ("xmin", "ymin", "xmax", "ymax"))
            except (AttributeError, TypeError, ValueError):
                n_invalid += 1
                continue

            was_clipped = False
            if w > 0:
                cx0, cx1 = min(max(xmin, 0.0), w), min(max(xmax, 0.0), w)
                was_clipped |= (cx0 != xmin) or (cx1 != xmax)
                xmin, xmax = cx0, cx1
            if h > 0:
                cy0, cy1 = min(max(ymin, 0.0), h), min(max(ymax, 0.0), h)
                was_clipped |= (cy0 != ymin) or (cy1 != ymax)
                ymin, ymax = cy0, cy1

            if xmax > xmin and ymax > ymin:
                if was_clipped:
                    n_clipped += 1
                per_class[code] += 1
                kept_here += 1
                kept_boxes.append([xmin, ymin, xmax, ymax])
                kept_labels.append(code)
            else:
                n_invalid += 1

    boxes = (np.asarray(kept_boxes, dtype=np.float32)
             if kept_boxes else np.zeros((0, 4), np.float32))
    return {
        "n_xml": len(xml_paths),
        "negatives": negatives,
        "per_class": dict(per_class),
        "corruption": dict(corruption),
        "n_clipped": n_clipped,
        "n_invalid": n_invalid,
        "missing_images": missing_images,
        "boxes": boxes,
        "labels": kept_labels,
    }


def crosscheck_with_parser(ann_dir: pathlib.Path, img_dir: str) -> Counter:
    """Run the actual training parser; its counts must agree with the raw pass."""
    stats = Counter()
    per_class = Counter()
    for xp in sorted(ann_dir.glob("*.xml")):
        s = parse_voc_xml(str(xp), img_dir, stats)
        per_class.update(s.labels)
    stats["_per_class"] = dict(per_class)
    return stats


# --- size distribution --------------------------------------------------------
def size_distribution(boxes: np.ndarray, labels: list[str]) -> dict:
    if len(boxes) == 0:
        return {}
    areas = box_area_xyxy(boxes)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    aspect = np.where(h > 0, w / np.maximum(h, 1e-6), 0.0)  # w/h

    # COCO buckets (px^2): small <32^2, medium 32^2..96^2, large >96^2.
    s = int((areas < 32 ** 2).sum())
    m = int(((areas >= 32 ** 2) & (areas < 96 ** 2)).sum())
    lg = int((areas >= 96 ** 2).sum())

    def pct(a):
        return {p: round(float(np.percentile(a, p)), 1) for p in (5, 25, 50, 75, 95)}

    per_class_area = {}
    lab = np.array(labels)
    for c in VALID_CLASSES:
        mask = lab == c
        if mask.any():
            per_class_area[c] = {
                "median_area": round(float(np.median(areas[mask])), 1),
                "median_aspect_wh": round(float(np.median(aspect[mask])), 2),
            }

    return {
        "coco_buckets": {"small": s, "medium": m, "large": lg},
        "area_percentiles": pct(areas),
        "aspect_wh_percentiles": pct(aspect),
        "per_class": per_class_area,
    }


# --- near-duplicate probe -----------------------------------------------------
# Clustering (dhash + union-find) lives in src/data/dedup.py so this report and
# the group-aware split in src/data/split.py cluster identically.
def near_duplicate_probe(img_dir: str, filenames: list[str],
                         hash_size: int = 8, threshold: int = 5,
                         sample: int | None = None) -> dict:
    names = list(filenames)
    sampled = False
    if sample and sample < len(names):
        rng = np.random.default_rng(0)
        names = list(rng.choice(names, size=sample, replace=False))
        sampled = True

    hashes, valid = hash_images(img_dir, names, hash_size)
    if len(hashes) < 2:
        return {"error": "fewer than 2 images hashed", "n_hashed": len(hashes)}

    dup_clusters, n_pairs = cluster_from_hashes(hashes, valid, threshold)
    n = len(hashes)
    involved = sum(len(c) for c in dup_clusters)

    return {
        "n_hashed": n,
        "sampled": sampled,
        "hash_size": hash_size,
        "hamming_threshold": threshold,
        "near_dup_pairs": n_pairs,
        "n_clusters": len(dup_clusters),
        "images_in_clusters": involved,
        "pct_in_clusters": round(100 * involved / n, 2),
        "largest_cluster": len(dup_clusters[0]) if dup_clusters else 0,
        "clusters_preview": dup_clusters[:20],
    }


# --- driver -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=str(ROOT / "data/rdd/Czech/train/annotations/xmls"))
    ap.add_argument("--img", default=str(ROOT / "data/rdd/Czech/train/images"))
    ap.add_argument("--out", default=str(ROOT / "reports/dataset_analysis.json"))
    ap.add_argument("--npz", default=str(ROOT / "reports/box_geometry.npz"))
    ap.add_argument("--hash-size", type=int, default=8)
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--sample", type=int, default=None,
                    help="hash only N images (faster; note: undercounts clusters)")
    ap.add_argument("--no-dup", action="store_true", help="skip the image hashing pass")
    args = ap.parse_args()

    ann_dir = pathlib.Path(args.ann)
    img_dir = args.img
    if not ann_dir.exists():
        sys.exit(f"annotation dir not found: {ann_dir}")

    # image files on disk
    img_files = [p.name for p in pathlib.Path(img_dir).glob("*")
                 if p.suffix.lower() in IMAGE_EXTS] if pathlib.Path(img_dir).exists() else []

    a = analyze_annotations(ann_dir, img_dir)
    parser_stats = crosscheck_with_parser(ann_dir, img_dir)

    # reconcile per-class: raw pass vs parser vs card
    total_objs = sum(a["per_class"].values())
    parser_total = sum(parser_stats.get("_per_class", {}).values())
    agree = (a["per_class"] == parser_stats.get("_per_class", {}))

    size = size_distribution(a["boxes"], a["labels"])

    dup = {}
    if not args.no_dup and img_files:
        dup = near_duplicate_probe(img_dir, img_files, args.hash_size,
                                   args.threshold, args.sample)

    report = {
        "counts": {
            "images_on_disk": len(img_files),
            "xml_files": a["n_xml"],
            "negative_images": a["negatives"],
            "missing_images_for_xml": len(a["missing_images"]),
        },
        "per_class_objects": a["per_class"],
        "reconciliation": {
            "sum_per_class": total_objs,
            "parser_sum": parser_total,
            "card_total": CARD_TOTAL_OBJECTS,
            "matches_card": total_objs == CARD_TOTAL_OBJECTS,
            "raw_pass_agrees_with_parser": agree,
        },
        "corruption": {
            "dropped_unknown_objects": sum(a["corruption"].values()),
            "by_name": a["corruption"],
        },
        "geometry": {
            "clipped_boxes": a["n_clipped"],
            "invalid_boxes": a["n_invalid"],
            "size_distribution": size,
        },
        "near_duplicates": dup,
    }

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    if len(a["boxes"]):
        np.savez_compressed(args.npz, boxes=a["boxes"],
                            labels=np.array(a["labels"]))

    _print_report(report, a)
    print(f"\nsaved JSON  -> {args.out}")
    if len(a["boxes"]):
        print(f"saved boxes -> {args.npz}")


def _print_report(r: dict, a: dict) -> None:
    c = r["counts"]
    print("=" * 60)
    print("CZECH TRAIN — DATASET ANALYSIS")
    print("=" * 60)
    print(f"images on disk        : {c['images_on_disk']}")
    print(f"xml files             : {c['xml_files']}")
    print(f"negative images       : {c['negative_images']}"
          f"  ({100*c['negative_images']/max(c['xml_files'],1):.1f}% of xml)")
    print(f"xml w/ missing image  : {c['missing_images_for_xml']}")
    print("-" * 60)
    print("per-class objects:")
    for cls in VALID_CLASSES:
        print(f"  {cls}: {r['per_class_objects'].get(cls, 0)}")
    rec = r["reconciliation"]
    tick = "OK" if rec["matches_card"] else "MISMATCH"
    print(f"  total = {rec['sum_per_class']}  vs card {rec['card_total']}  [{tick}]")
    print(f"  raw-pass == parser: {rec['raw_pass_agrees_with_parser']}")
    print("-" * 60)
    print(f"dropped unknown labels: {r['corruption']['dropped_unknown_objects']}"
          f"  {r['corruption']['by_name'] or ''}")
    print(f"clipped boxes         : {r['geometry']['clipped_boxes']}")
    print(f"invalid boxes         : {r['geometry']['invalid_boxes']}")
    sd = r["geometry"]["size_distribution"]
    if sd:
        b = sd["coco_buckets"]
        print(f"size buckets (S/M/L)  : {b['small']} / {b['medium']} / {b['large']}")
        print(f"area px^2 (5/50/95)   : {sd['area_percentiles'][5]} / "
              f"{sd['area_percentiles'][50]} / {sd['area_percentiles'][95]}")
        print(f"aspect w/h (5/50/95)  : {sd['aspect_wh_percentiles'][5]} / "
              f"{sd['aspect_wh_percentiles'][50]} / {sd['aspect_wh_percentiles'][95]}")
    d = r["near_duplicates"]
    if d and "error" not in d:
        print("-" * 60)
        note = "  [SAMPLED — undercounts]" if d.get("sampled") else ""
        print(f"near-dup probe (thr={d['hamming_threshold']}){note}")
        print(f"  hashed              : {d['n_hashed']}")
        print(f"  near-dup pairs      : {d['near_dup_pairs']}")
        print(f"  duplicate clusters  : {d['n_clusters']}")
        print(f"  images in clusters  : {d['images_in_clusters']} "
              f"({d['pct_in_clusters']}%)  largest={d['largest_cluster']}")


if __name__ == "__main__":
    main()