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

def postprocess_boxes(preds, confidence_threshold, image_size, orig_w, orig_h, resized_w, resized_h, left_pad, top_pad, class_filter="Text Only", x_offset=0, y_offset=0):
    """
    Decodes predictions, handles normalized vs pixel coordinates,
    computes bounding boxes [xmin, ymin, xmax, ymax], and maps them back
    to the original coordinates space.
    """
    if preds.shape[1] < 5:
        raise ValueError(f"Detector predictions have too few columns: {preds.shape}")

    # For ogkalu YOLO model / legacy YOLO:
    if preds.shape[1] == 6:
        if class_filter == "Text Only":
            confidences = preds[:, 4]  # class 0 (text)
            mask = confidences >= confidence_threshold
        elif class_filter == "Bubbles Only":
            confidences = preds[:, 5]  # class 1 (bubble)
            mask = confidences >= confidence_threshold
        else:
            confidences = np.max(preds[:, 4:], axis=1)
            mask = confidences >= confidence_threshold
    else:
        if class_filter == "Text Only":
            if preds.shape[1] >= 7:
                confidences = np.max(preds[:, 5:7], axis=1)
                mask = confidences >= confidence_threshold
            else:
                confidences = np.max(preds[:, 4:], axis=1)
                mask = confidences >= confidence_threshold
        elif class_filter == "Bubbles Only":
            confidences = preds[:, 4]  # class 0 (bubble)
            mask = confidences >= confidence_threshold
        else:
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
    is_normalized = np.max(valid_preds[:, :4]) <= 1.01
    if is_normalized:
        valid_preds = valid_preds.copy()
        valid_preds[:, :4] *= image_size

    x_center = valid_preds[:, 0]
    y_center = valid_preds[:, 1]
    w = valid_preds[:, 2]
    h = valid_preds[:, 3]

    # Subtract padding offsets to get coordinates relative to the resized image area
    x_center_unpadded = x_center - left_pad
    y_center_unpadded = y_center - top_pad

    w_ratio = orig_w / resized_w
    h_ratio = orig_h / resized_h

    bbox_dilation = 1.0
    x1 = (x_center_unpadded - w / 2.0) * w_ratio - bbox_dilation + x_offset
    x2 = (x_center_unpadded + w / 2.0) * w_ratio + bbox_dilation + x_offset
    y1 = (y_center_unpadded - h / 2.0) * h_ratio - bbox_dilation + y_offset
    y2 = (y_center_unpadded + h / 2.0) * h_ratio + bbox_dilation + y_offset

    boxes = np.column_stack([x1, y1, x2, y2]).astype(np.float32)
    return boxes, valid_confs.astype(np.float32)

def postprocess_rtdetr_outputs(boxes_val, scores_val, labels_val, confidence_threshold, orig_w, orig_h, resized_w, resized_h, left_pad, top_pad, class_filter="Text Only", x_offset=0, y_offset=0):
    """
    Decodes RT-DETR outputs where boxes are scaled to image_size.
    Maps them to original dimensions and adds offsets.
    """
    boxes = boxes_val[0]
    scores = scores_val[0]
    labels = labels_val[0] if labels_val is not None else None

    # If scores is 2D (e.g. [300, num_classes])
    if scores.ndim == 2:
        scores = np.max(scores, axis=-1)

    mask = scores >= confidence_threshold

    # Apply class filter if labels are available
    if labels is not None:
        if class_filter == "Text Only":
            # In ogkalu: 1 is text_bubble, 2 is text_free
            class_mask = (labels == 1) | (labels == 2)
            mask = mask & class_mask
        elif class_filter == "Bubbles Only":
            # In ogkalu: 0 is bubble
            class_mask = (labels == 0)
            mask = mask & class_mask

    valid_boxes = boxes[mask]
    valid_confs = scores[mask]

    if labels is not None:
        valid_labels = labels[mask]
        sys.stderr.write(f"[Koharu Postprocessor] RT-DETR filtered classes: {valid_labels.tolist()}\n")

    sys.stderr.write(
        f"[Koharu Postprocessor] RT-DETR tile=({x_offset},{y_offset}) "
        f"total_preds={len(boxes)} valid_preds={len(valid_boxes)} threshold={confidence_threshold:.2f}\n"
    )

    if len(valid_boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    w_ratio = orig_w / resized_w
    h_ratio = orig_h / resized_h

    # Subtract padding offsets to get coordinates relative to the resized image area, then scale back
    x1 = (valid_boxes[:, 0] - left_pad) * w_ratio + x_offset
    y1 = (valid_boxes[:, 1] - top_pad) * h_ratio + y_offset
    x2 = (valid_boxes[:, 2] - left_pad) * w_ratio + x_offset
    y2 = (valid_boxes[:, 3] - top_pad) * h_ratio + y_offset

    final_boxes = np.column_stack([x1, y1, x2, y2]).astype(np.float32)
    return final_boxes, valid_confs.astype(np.float32)

