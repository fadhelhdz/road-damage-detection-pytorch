"""Trust tests for the group-aware stratified split (src/data/split.py).

No real dataset required: a synthetic VOC set is generated on the fly (distinct
noise images + a few deliberate duplicate pairs) so the three invariants that
make the split trustworthy can be asserted directly:

  1. No group crosses splits  — every group id lives in exactly one of
     {test, dev-fold-0, ... dev-fold-4}; a near-duplicate never leaks.
  2. Coverage — every class appears in every dev fold and in test, so per-class
     AP is computable everywhere.
  3. Determinism — same seed -> byte-identical assignment.
"""
from __future__ import annotations

import pathlib
from collections import Counter

import numpy as np
import pytest

from data.dedup import cluster_near_duplicates
from data.split import (
    N_DEV_FOLDS,
    assign_groups,
    build_stratification_label,
    make_splits,
)
from data.voc import VALID_CLASSES, Sample

# 30 images per strat class keeps every partition (1 test + 5 dev folds) well
# above 1 per class, so coverage does not hinge on a lucky fold boundary.
PER_CLASS = 30
STRATA = list(VALID_CLASSES) + ["negative"]
SIZE = 48


@pytest.fixture(autouse=True)
def _redirect_output(tmp_path, monkeypatch):
    """Never let a test write the real committed split file."""
    monkeypatch.setattr("data.split.DEFAULT_OUT",
                        tmp_path / "czech_splits.json")


def _xml(filename: str, label: str | None) -> str:
    objs = ""
    if label is not None:
        objs = f"""
  <object><name>{label}</name>
    <bndbox><xmin>5</xmin><ymin>5</ymin><xmax>25</xmax><ymax>30</ymax></bndbox>
  </object>"""
    return (f"<annotation><filename>{filename}</filename>"
            f"<size><width>{SIZE}</width><height>{SIZE}</height>"
            f"<depth>3</depth></size>{objs}</annotation>")


@pytest.fixture(scope="module")
def synthetic_ds(tmp_path_factory):
    """Build a synthetic VOC dataset; return (ann_dir, img_dir, dup_pairs)."""
    from PIL import Image

    base = tmp_path_factory.mktemp("voc")
    ann = base / "ann"
    img = base / "img"
    ann.mkdir()
    img.mkdir()
    rng = np.random.default_rng(0)

    dup_pairs: list[tuple[str, str]] = []
    idx = 0
    for cls in STRATA:
        label = None if cls == "negative" else cls
        for k in range(PER_CLASS):
            stem = f"{cls}_{k:03d}"
            fn = f"{stem}.jpg"
            arr = rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
            Image.fromarray(arr).save(img / fn)
            (ann / f"{stem}.xml").write_text(_xml(fn, label))

            # every 12th image (non-negative) gets an exact-copy duplicate ->
            # a 2-image near-duplicate cluster with the same class.
            if label is not None and k % 12 == 0:
                dfn = f"{stem}_dup.jpg"
                Image.fromarray(arr).save(img / dfn)
                (ann / f"{stem}_dup.xml").write_text(_xml(dfn, label))
                dup_pairs.append((fn, dfn))
            idx += 1

    return str(ann), str(img), dup_pairs


def _load_samples(ann_dir, img_dir):
    from data.voc import parse_voc_xml
    out = {}
    for xp in sorted(pathlib.Path(ann_dir).glob("*.xml")):
        s = parse_voc_xml(str(xp), img_dir)
        out[pathlib.Path(s.image_path).name] = s
    return out


# --- unit-level checks on the helpers ----------------------------------------
def test_stratification_label_picks_rarest_present():
    assert build_stratification_label(Sample("x", 1, 1, None, [])) == "negative"
    # D20 is rarest -> wins even alongside a common class
    s = Sample("x", 1, 1, None, ["D00", "D20", "D40"])
    assert build_stratification_label(s) == "D20"
    assert build_stratification_label(Sample("x", 1, 1, None, ["D00", "D10"])) == "D10"


def test_assign_groups_shares_ids_within_cluster_and_singles_elsewhere():
    files = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    clusters = [["a.jpg", "b.jpg"]]
    g = assign_groups(files, clusters)
    assert g["a.jpg"] == g["b.jpg"]              # cluster shares an id
    assert len({g["c.jpg"], g["d.jpg"], g["a.jpg"]}) == 3  # singletons distinct
    assert len(set(g.values())) == 3             # 1 cluster + 2 singletons


# --- the three trust invariants ----------------------------------------------
def test_duplicates_form_the_expected_clusters(synthetic_ds):
    """Guard: the only near-dup clusters are the copies we planted (deterministic)."""
    _, img, dup_pairs = synthetic_ds
    filenames = sorted(p.name for p in pathlib.Path(img).glob("*.jpg"))
    clusters = cluster_near_duplicates(img, filenames, threshold=6)
    got = {tuple(sorted(c)) for c in clusters}
    want = {tuple(sorted(p)) for p in dup_pairs}
    assert got == want


def test_invariant_1_no_group_crosses_splits(synthetic_ds):
    ann, img, dup_pairs = synthetic_ds
    assignment = make_splits(ann, img, dup_threshold=6, seed=42)

    filenames = list(assignment)
    clusters = cluster_near_duplicates(img, filenames, threshold=6)
    groups = assign_groups(filenames, clusters)

    where = lambda fn: (assignment[fn]["split"], assignment[fn]["fold"])
    by_group: dict[int, set] = {}
    for fn, gid in groups.items():
        by_group.setdefault(gid, set()).add(where(fn))
    # each group lands in exactly one (split, fold) cell
    assert all(len(cells) == 1 for cells in by_group.values())
    # and the planted duplicate pairs concretely co-locate
    for a, b in dup_pairs:
        assert where(a) == where(b)


def test_invariant_2_every_class_in_every_fold_and_test(synthetic_ds):
    ann, img, _ = synthetic_ds
    assignment = make_splits(ann, img, dup_threshold=6, seed=42)
    samples = _load_samples(ann, img)

    partitions = ["test"] + [f"dev{f}" for f in range(N_DEV_FOLDS)]
    counts = {p: Counter() for p in partitions}
    for fn, a in assignment.items():
        key = "test" if a["split"] == "test" else f"dev{a['fold']}"
        counts[key].update(samples[fn].labels)

    for p in partitions:
        for cls in VALID_CLASSES:
            assert counts[p][cls] > 0, f"class {cls} missing from {p}"


def test_invariant_3_determinism(synthetic_ds):
    ann, img, _ = synthetic_ds
    a1 = make_splits(ann, img, dup_threshold=6, seed=42)
    a2 = make_splits(ann, img, dup_threshold=6, seed=42)
    assert a1 == a2
    # a different seed must actually change something (shuffle is wired in)
    a3 = make_splits(ann, img, dup_threshold=6, seed=7)
    assert a3 != a1


def test_split_partitions_are_disjoint_and_complete(synthetic_ds):
    ann, img, _ = synthetic_ds
    assignment = make_splits(ann, img, dup_threshold=6, seed=42)
    n_files = len(list(pathlib.Path(img).glob("*.jpg")))
    assert len(assignment) == n_files
    for a in assignment.values():
        if a["split"] == "test":
            assert a["fold"] is None
        else:
            assert a["split"] == "dev" and 0 <= a["fold"] < N_DEV_FOLDS
