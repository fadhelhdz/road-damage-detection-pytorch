"""Perceptual-hash near-duplicate clustering.

Single implementation of the dhash + union-find clustering used both by the
dataset analysis report (scripts/analyze_dataset.py) and by the group-aware
train/val split (src/data/split.py). Sharing this module is what makes the
split's groups *provably* the same near-duplicate clusters the analysis reports:
one implementation, one definition of "duplicate".

A cluster is a connected component of images whose dhash Hamming distance is
<= `threshold`. `cluster_near_duplicates` returns only the multi-image clusters
(size >= 2); every other image is a singleton the caller can treat as its own
group.
"""
from __future__ import annotations

import pathlib
from collections import defaultdict

import numpy as np


def dhash(path: str, hash_size: int = 8) -> int | None:
    """Difference hash of an image as a `hash_size*hash_size`-bit integer.

    Returns None when the file cannot be opened/decoded (missing, corrupt).
    """
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
            px = np.asarray(im, dtype=np.int16)
    except Exception:
        return None
    diff = px[:, 1:] > px[:, :-1]
    h = 0
    for bit in diff.flatten():
        h = (h << 1) | int(bit)
    return h


def _popcount64(a: np.ndarray) -> np.ndarray:
    return np.unpackbits(a.astype(np.uint64).view(np.uint8).reshape(-1, 8),
                         axis=1).sum(axis=1)


class _DSU:
    """Union-find over integer node ids."""

    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def hash_images(img_dir: str, filenames: list[str],
                hash_size: int = 8) -> tuple[list[int], list[str]]:
    """dhash every file; return (hashes, filenames) for the ones that decoded."""
    hashes, valid = [], []
    for fn in filenames:
        h = dhash(str(pathlib.Path(img_dir) / fn), hash_size)
        if h is not None:
            hashes.append(h)
            valid.append(fn)
    return hashes, valid


def cluster_from_hashes(hashes: list[int], names: list[str],
                        threshold: int = 6) -> tuple[list[list[str]], int]:
    """Union images whose hashes are within `threshold` Hamming distance.

    Returns (multi_image_clusters, n_pairs). Clusters are size >= 2, each
    sorted, and the list is sorted largest-first. `n_pairs` is the raw count of
    within-threshold pairs (a diagnostic for the analysis report).
    """
    n = len(hashes)
    if n < 2:
        return [], 0
    H = np.array(hashes, dtype=np.uint64)
    dsu = _DSU(n)
    n_pairs = 0
    for i in range(n - 1):
        d = _popcount64(H[i + 1:] ^ H[i])
        for j in np.nonzero(d <= threshold)[0]:
            n_pairs += 1
            dsu.union(i, i + 1 + int(j))

    groups = defaultdict(list)
    for idx, fn in enumerate(names):
        groups[dsu.find(idx)].append(fn)
    clusters = [sorted(v) for v in groups.values() if len(v) > 1]
    clusters.sort(key=len, reverse=True)
    return clusters, n_pairs


def cluster_near_duplicates(img_dir: str, filenames: list[str],
                            threshold: int = 6,
                            hash_size: int = 8) -> list[list[str]]:
    """Near-duplicate clusters (size >= 2) among `filenames` under `img_dir`.

    Images that fail to hash, and images with no near-duplicate, are simply
    absent from the result — the caller treats every unclustered image as its
    own singleton group.
    """
    hashes, valid = hash_images(img_dir, filenames, hash_size)
    clusters, _ = cluster_from_hashes(hashes, valid, threshold)
    return clusters
