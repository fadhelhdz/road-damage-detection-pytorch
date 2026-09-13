import numpy as np

def iou_xyxy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes.

    Args:
        boxes_a: (N, 4) array, each row (x1, y1, x2, y2), x2>=x1, y2>=y1.
        boxes_b: (M, 4) array, same format.
    Returns:
        (N, M) array; element [i, j] = IoU(boxes_a[i], boxes_b[j]) in [0, 1].
    """
    
    boxes_a = np.asarray(boxes_a, dtype=float)
    boxes_b = np.asarray(boxes_b, dtype=float)

    N = boxes_a.shape[0]
    M = boxes_b.shape[0]

    # Handle empty inputs
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=float)

    # Broadcast boxes to (N, M, 4)
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    # Intersection dimensions; clamp to zero for non-overlapping boxes
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    # Box areas; clamp to zero for degenerate/invalid boxes
    area1 = (
        np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0])
        * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    )
    area2 = (
        np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0])
        * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    )

    union_area = area1[:, None] + area2[None, :] - inter_area

    # Only divide where union > 0; degenerate boxes get IoU = 0
    iou = np.divide(
        inter_area,
        union_area,
        out=np.zeros_like(inter_area),
        where=union_area > 0
    )

    return iou

