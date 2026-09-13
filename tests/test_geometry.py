import numpy as np

from evaluation.geometry import iou_xyxy

def test_identical_boxes():
    boxes_a = [[0, 0, 10, 10]]
    boxes_b = [[0, 0, 10, 10]]

    result = iou_xyxy(boxes_a, boxes_b)

    assert np.isclose(result[0, 0], 1.0)

def test_disjoint_boxes():
    boxes_a = [[0, 0, 10, 10]]
    boxes_b = [[20, 20, 30, 30]]

    result = iou_xyxy(boxes_a, boxes_b)

    assert np.isclose(result[0, 0], 0.0)

def test_half_overlap():
    boxes_a = [[0, 0, 10, 10]]
    boxes_b = [[5, 0, 15, 10]]

    result = iou_xyxy(boxes_a, boxes_b)

    # intersection = 5 * 10 = 50
    # unio = 100 + 100 - 50 = 150
    # IoU = 50 / 150 = 1/3
    assert np.isclose(result[0, 0], 1/3)

def test_empty_input():
    boxes_a = np.empty((0, 4))
    boxes_b = np.array([
        [0, 0, 10, 10],
        [20, 20, 30, 30]
    ])

    result = iou_xyxy(boxes_a, boxes_b)

    assert result.shape == (0, 2)
    assert result.size == 0

def test_zero_area_box():
    boxes_a = [[0, 0, 0, 10]]
    boxes_b = [[0, 0, 10, 10]]

    result = iou_xyxy(boxes_a, boxes_b)

    assert result[0, 0] == 0.0
    assert not np.isnan(result).any()

def test_matrix_shape_and_values():
    boxes_a = np.array([
        [0, 0, 10, 10],
        [0, 0, 20, 20]
    ])
    boxes_b = np.array([
        [0, 0, 10, 10],
        [5, 5, 15, 15],
        [20, 20, 30, 30]
    ])

    result = iou_xyxy(boxes_a, boxes_b)

    expected = np.array([
        [1.0, 25 / 175, 0.0],
        [0.25, 100 / 400, 0.0]
    ])

    assert result.shape == (2, 3)
    assert np.allclose(result, expected)

def test_containment():
    boxes_a = [[2, 2, 4, 4]]
    boxes_b = [[0, 0, 10, 10]]

    result = iou_xyxy(boxes_a, boxes_b)

    # Small box is fully contained inside the large box
    # intersection = area_small = 4
    # union = area_large = 100
    # IoU = 4 / 100 = 0.04
    assert np.isclose(result[0, 0], 0.04)