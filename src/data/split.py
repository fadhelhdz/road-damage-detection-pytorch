"""Group-aware, stratified train/val/test split for the Czech RDD2022 subset.

Produces a committed JSON of ``{image_filename: {"split", "fold"}}`` that is the
single source of truth every training/eval run reads. No images are stored — the
file is tiny, versioned, and fully reproducible from (ann_dir, img_dir, seed).

Design:
  * Groups   — near-duplicate images (dhash + union-find, src/data/dedup.py)
    share a group id so a duplicate can never straddle two splits and leak.
  * Strata   — each image is labelled by its *rarest present* class so the
    scarce classes are spread evenly across every partition.
  * Frozen test — StratifiedGroupKFold(7), fold 0's held-out part. Never touched
    during development.
  * 5 CV folds — a second StratifiedGroupKFold(5) over everything that is not
    test; each dev image carries the fold index in which it is the validation
    set.

Run:  python scripts/make_splits.py   # writes data/splits/czech_splits.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from data.dedup import cluster_near_duplicates
from data.voc import VALID_CLASSES, parse_voc_xml

# repo root: src/data/split.py -> parents[2]
ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data/splits/czech_splits.json"

N_TEST_SPLITS = 7   # frozen test = 1/7 of the data (fold 0 of a 7-way split)
N_DEV_FOLDS = 5     # 5-fold CV over the remaining 6/7

# Rarest-first priority for the stratification label. D20 is scarcest in the
# Czech subset and D00 the most common, so a multi-class image is represented by
# its scarcest class — that is the count we most need balanced across folds.
_RARITY_ORDER = ("D20", "D40", "D10", "D00")


def build_stratification_label(sample) -> str:
    """Rarest present class (D20>D40>D10>D00), or 'negative' if no objects."""
    present = set(sample.labels)
    for cls in _RARITY_ORDER:
        if cls in present:
            return cls
    return "negative"


def assign_groups(filenames, clusters) -> dict[str, int]:
    """Each cluster -> shared group id; each unclustered image -> unique singleton id."""
    to_group: dict[str, int] = {}
    gid = 0
    for cluster in clusters:               # multi-image near-duplicate clusters
        for fn in cluster:
            to_group[fn] = gid
        gid += 1
    for fn in filenames:                   # everything else is its own group
        if fn not in to_group:
            to_group[fn] = gid
            gid += 1
    return to_group


def _load(ann_dir, img_dir):
    """Parse every XML once. Returns (filenames, samples) in sorted XML order."""
    ann_dir = pathlib.Path(ann_dir)
    filenames, samples = [], []
    for xp in sorted(ann_dir.glob("*.xml")):
        s = parse_voc_xml(str(xp), str(img_dir))
        filenames.append(pathlib.Path(s.image_path).name)
        samples.append(s)
    return filenames, samples


def make_splits(ann_dir, img_dir, dup_threshold=6, seed=42) -> dict:
    """Return {filename: {'split': 'test'|'dev', 'fold': int|None}}.

    Frozen test via StratifiedGroupKFold(7)[fold0]; 5 CV folds on the rest.
    Seed everything. Save to data/splits/czech_splits.json (committed, versioned).
    """
    # Deterministic without torch: StratifiedGroupKFold uses its own
    # random_state and dhash has no RNG. Seed numpy/random anyway to document
    # intent and guard any future RNG step; skip torch seeding (a no-op for the
    # split, and it warns on a torch-less box).
    import random
    random.seed(seed)
    np.random.seed(seed)

    filenames, samples = _load(ann_dir, img_dir)
    if not filenames:
        raise SystemExit(f"no XML annotations found under {ann_dir}")

    clusters = cluster_near_duplicates(img_dir, filenames, threshold=dup_threshold)
    groups_map = assign_groups(filenames, clusters)

    strat = np.array([build_stratification_label(s) for s in samples])
    groups = np.array([groups_map[fn] for fn in filenames])
    X = np.arange(len(filenames))

    # 1) frozen test = fold 0 of a 7-way stratified, group-aware split.
    skf_test = StratifiedGroupKFold(n_splits=N_TEST_SPLITS, shuffle=True,
                                    random_state=seed)
    dev_idx, test_idx = next(iter(skf_test.split(X, strat, groups)))

    assignment: dict[str, dict] = {}
    for i in test_idx:
        assignment[filenames[i]] = {"split": "test", "fold": None}

    # 2) 5-fold CV over the non-test remainder; fold = the fold where the image
    #    is the held-out validation set.
    skf_dev = StratifiedGroupKFold(n_splits=N_DEV_FOLDS, shuffle=True,
                                   random_state=seed)
    for fold, (_, val_local) in enumerate(
            skf_dev.split(X[dev_idx], strat[dev_idx], groups[dev_idx])):
        for j in val_local:
            i = int(dev_idx[j])
            assignment[filenames[i]] = {"split": "dev", "fold": fold}

    _write_json(assignment, samples, filenames, groups_map, clusters,
                dup_threshold, seed)
    return assignment


def _write_json(assignment, samples, filenames, groups_map, clusters,
                threshold, seed, out=None) -> None:
    out = pathlib.Path(out) if out is not None else DEFAULT_OUT
    per_class = Counter()
    negatives = 0
    for s in samples:
        if s.labels:
            per_class.update(s.labels)
        else:
            negatives += 1
    meta = {
        "threshold": threshold,
        "seed": seed,
        "n_dev_folds": N_DEV_FOLDS,
        "n_test_splits": N_TEST_SPLITS,
        "created_from": {
            "n_images": len(filenames),
            "n_groups": len(set(groups_map.values())),
            "n_dup_clusters": len(clusters),
            "negatives": negatives,
            "per_class_objects": {c: per_class.get(c, 0) for c in VALID_CLASSES},
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"_meta": meta, "assignments": assignment}, f, indent=2,
                  sort_keys=True)


# --- reporting ----------------------------------------------------------------
def per_split_class_table(assignment, samples, filenames) -> str:
    """Per-split x per-class object counts (+ images / negatives / groups)."""
    by_name = {fn: s for fn, s in zip(filenames, samples)}
    order = ["test"] + [f"dev{f}" for f in range(N_DEV_FOLDS)]
    rows: dict[str, Counter] = {k: Counter() for k in order}
    imgs: dict[str, int] = {k: 0 for k in order}

    for fn, a in assignment.items():
        key = "test" if a["split"] == "test" else f"dev{a['fold']}"
        s = by_name[fn]
        imgs[key] += 1
        if s.labels:
            rows[key].update(s.labels)
        else:
            rows[key]["negative"] += 1

    cols = list(VALID_CLASSES) + ["negative", "images"]
    head = f"{'split':<7}" + "".join(f"{c:>10}" for c in cols)
    lines = [head, "-" * len(head)]
    for k in order:
        cells = [rows[k].get(c, 0) for c in VALID_CLASSES]
        cells += [rows[k].get("negative", 0), imgs[k]]
        lines.append(f"{k:<7}" + "".join(f"{v:>10}" for v in cells))
    total = [sum(rows[k].get(c, 0) for k in order) for c in VALID_CLASSES]
    total += [sum(rows[k].get("negative", 0) for k in order),
              sum(imgs[k] for k in order)]
    lines.append("-" * len(head))
    lines.append(f"{'total':<7}" + "".join(f"{v:>10}" for v in total))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ann", default=str(ROOT / "data/rdd/Czech/train/annotations/xmls"))
    ap.add_argument("--img", default=str(ROOT / "data/rdd/Czech/train/images"))
    ap.add_argument("--threshold", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if not pathlib.Path(args.ann).exists():
        sys.exit(f"annotation dir not found: {args.ann}")

    assignment = make_splits(args.ann, args.img, args.threshold, args.seed)
    filenames, samples = _load(args.ann, args.img)
    print(per_split_class_table(assignment, samples, filenames))
    print(f"\nsaved split -> {DEFAULT_OUT}")


if __name__ == "__main__":
    main()
