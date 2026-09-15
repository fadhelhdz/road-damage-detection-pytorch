"""Parser tests that run on a CPU box without the real dataset.

Fixtures live in tests/fixtures/mini_voc/. The placeholder images are tiny and
are (re)generated on demand by the `placeholder_images` fixture below, so the
repo needs no committed binaries — the XMLs are the real, hand-written fixtures.
"""
from __future__ import annotations

import sys
import pathlib
from collections import Counter

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.voc import parse_voc_xml, Sample  # noqa: E402

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "mini_voc"
ANN = FIX / "annotations"
IMG_DIR = FIX / "images"

# filename -> (width, height) for the placeholder images.
_IMAGES = {
    "neg_0001.jpg": (24, 24),
    "corrupt_0002.jpg": (24, 24),
    "normal_0003.jpg": (24, 24),
    "clip_0004.jpg": (24, 24),
    "nosize_0005.jpg": (24, 24),  # header must be readable for the size fallback
}


@pytest.fixture(scope="session", autouse=True)
def placeholder_images():
    """Ensure a matching placeholder image exists for every fixture XML."""
    from PIL import Image
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for name, (w, h) in _IMAGES.items():
        p = IMG_DIR / name
        if not p.exists():
            Image.new("RGB", (w, h), (127, 127, 127)).save(p)
    yield


def _parse(stem: str, stats: Counter | None = None) -> Sample:
    return parse_voc_xml(str(ANN / f"{stem}.xml"), str(IMG_DIR), stats)


def test_negative_image_has_no_boxes():
    s = _parse("neg_0001")
    assert s.labels == []
    assert s.boxes.shape == (0, 4)
    assert s.boxes.dtype == np.float32
    assert s.width == 24 and s.height == 24
    assert s.image_path.endswith("images/neg_0001.jpg")


def test_other_corruption_label_is_dropped():
    stats = Counter()
    s = _parse("corrupt_0002", stats)
    # the 'other corruption' object is gone; the valid one survives
    assert s.labels == ["D10"]
    assert s.boxes.shape == (1, 4)
    assert stats["dropped_unknown_class"] == 1
    assert stats["invalid_box"] == 0


def test_normal_multiclass():
    s = _parse("normal_0003")
    assert s.labels == ["D00", "D40"]
    assert s.boxes.shape == (2, 4)
    np.testing.assert_allclose(s.boxes[0], [1, 1, 8, 20])
    np.testing.assert_allclose(s.boxes[1], [10, 10, 20, 20])


def test_clip_kept_and_degenerate_dropped():
    stats = Counter()
    s = _parse("clip_0004", stats)
    # alligator box clipped from xmax=30 -> 24 and kept; degenerate pothole dropped
    assert s.labels == ["D20"]
    np.testing.assert_allclose(s.boxes[0], [18, 2, 24, 20])
    assert stats["invalid_box"] == 1
    assert stats["dropped_unknown_class"] == 0


def test_size_fallback_reads_image_header():
    stats = Counter()
    s = _parse("nosize_0005", stats)
    # no <size> in XML -> dimensions come from the placeholder image (24x24)
    assert (s.width, s.height) == (24, 24)
    assert stats["size_from_image"] == 1
    assert s.labels == ["D10"]
    np.testing.assert_allclose(s.boxes[0], [2, 2, 12, 12])


def test_shared_stats_accumulate_across_files():
    stats = Counter()
    for stem in ("neg_0001", "corrupt_0002", "normal_0003", "clip_0004", "nosize_0005"):
        _parse(stem, stats)
    # exactly one unknown drop (corrupt) and one invalid box (clip) across the set
    assert stats["dropped_unknown_class"] == 1
    assert stats["invalid_box"] == 1
    assert stats["size_from_image"] == 1


def test_boxes_are_float32_and_2d_even_when_empty():
    for stem in ("neg_0001", "normal_0003"):
        s = _parse(stem)
        assert s.boxes.ndim == 2 and s.boxes.shape[1] == 4
        assert s.boxes.dtype == np.float32