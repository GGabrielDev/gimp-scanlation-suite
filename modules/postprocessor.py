import numpy as np
import sys

def nms(boxes, confidences, threshold):
    """
    Standard Non-Maximum Suppression (NMS) for filtering overlapping bboxes.
    boxes: numpy array of shape (M, 4) with [xmin, ymin, xmax, ymax]
    confidences: numpy array of shape (M,)
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = confidences.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep

def postprocess_boxes(preds, confidence_threshold, image_size, orig_w, orig_h, resized_w, resized_h, x_offset=0, y_offset=0):
    """
    Decodes predictions, handles normalized vs pixel coordinates,
    computes bounding boxes [xmin, ymin, xmax, ymax], and maps them back
    to the original coordinates space.
    """
    if preds.shape[1] < 5:
        raise ValueError(f"Detector predictions have too few columns: {preds.shape}")

    # For ogkalu YOLO model, there is no objectness column. Columns 4 and 5 are class probabilities.
    # We take the maximum probability across class columns.
    confidences = np.max(preds[:, 4:], axis=1)
    mask = confidences >= confidence_threshold
    valid_preds = preds[mask]
    valid_confs = confidences[mask]

    sys.stderr.write(
        f"[Koharu Postprocessor] tile=({x_offset},{y_offset},{orig_w},{orig_h}) "
        f"total_preds={len(preds)} valid_preds={len(valid_preds)} threshold={confidence_threshold:.2f}\n"
    )

    if len(valid_preds) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # Check if predictions are normalized (all coordinate columns values <= 1.01)
    # The first 4 columns are: x_center, y_center, w, h
    is_normalized = np.max(valid_preds[:, :4]) <= 1.01
    if is_normalized:
        valid_preds = valid_preds.copy()
        valid_preds[:, :4] *= image_size

    x_center = valid_preds[:, 0]
    y_center = valid_preds[:, 1]
    w = valid_preds[:, 2]
    h = valid_preds[:, 3]

    w_ratio = orig_w / resized_w
    h_ratio = orig_h / resized_h

    bbox_dilation = 1.0
    x1 = (x_center - w / 2.0) * w_ratio - bbox_dilation + x_offset
    x2 = (x_center + w / 2.0) * w_ratio + bbox_dilation + x_offset
    y1 = (y_center - h / 2.0) * h_ratio - bbox_dilation + y_offset
    y2 = (y_center + h / 2.0) * h_ratio + bbox_dilation + y_offset

    boxes = np.column_stack([x1, y1, x2, y2]).astype(np.float32)
    return boxes, valid_confs.astype(np.float32)
