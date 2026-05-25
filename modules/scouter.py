import os
import sys
import glob
import numpy as np

# Bootstrapping local venv site-packages so we can import onnxruntime
plugin_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
venv_paths = glob.glob(os.path.join(plugin_dir, "venv", "lib", "python*", "site-packages"))
if venv_paths:
    sys.path.insert(0, venv_paths[0])

import onnxruntime as ort
import gi
gi.require_version('Gegl', '0.4')
from gi.repository import Gegl

MODEL_IMAGE_SIZE = 1024


def resize_image(img, new_w, new_h):
    """
    Bilinear interpolation to resize a numpy image of shape (H, W, C).
    Done in pure NumPy to avoid dependency on OpenCV or PIL.
    """
    h, w, c = img.shape
    x = np.linspace(0, w - 1, new_w)
    y = np.linspace(0, h - 1, new_h)

    x_indices = np.floor(x).astype(np.int32)
    y_indices = np.floor(y).astype(np.int32)

    x_indices_next = np.minimum(x_indices + 1, w - 1)
    y_indices_next = np.minimum(y_indices + 1, h - 1)

    x_weight = (x - x_indices).reshape(1, new_w, 1)
    y_weight = (y - y_indices).reshape(new_h, 1, 1)

    p00 = img[y_indices][:, x_indices]
    p10 = img[y_indices][:, x_indices_next]
    p01 = img[y_indices_next][:, x_indices]
    p11 = img[y_indices_next][:, x_indices_next]

    interpolated = (
        p00 * (1 - x_weight) * (1 - y_weight) +
        p10 * x_weight * (1 - y_weight) +
        p01 * (1 - x_weight) * y_weight +
        p11 * x_weight * y_weight
    )
    return interpolated.astype(np.uint8)


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


def _normalize_yolo_predictions(outputs):
    """
    Normalize common YOLO ONNX output layouts to a 2D array of shape (N, C),
    where columns are:
      [x_center, y_center, w, h, objectness, class_0, class_1, ...]
    """
    for i, out in enumerate(outputs):
        sys.stderr.write(f"[Koharu Scouter] output[{i}] shape={getattr(out, 'shape', None)}\n")

    for out in outputs:
        shape = getattr(out, "shape", None)
        if shape is None:
            continue

        if len(shape) == 3 and shape[0] == 1:
            preds = out[0]
            if preds.ndim != 2:
                continue

            # Already (N, C)
            if preds.shape[1] >= 6:
                return preds

            # Transpose from (C, N) to (N, C)
            if preds.shape[0] >= 6:
                return preds.T

    predictions = outputs[0]
    shape = getattr(predictions, "shape", None)
    raise ValueError(f"Unexpected detector output shape: {shape}")


def _run_model_on_image(session, input_name, img_np, confidence_threshold, nms_threshold, x_offset=0, y_offset=0):
    """
    Run detector on an RGB numpy image and return boxes/confidences mapped back
    to the original image coordinate space with optional offsets.
    """
    orig_h, orig_w, _ = img_np.shape

    image_size = MODEL_IMAGE_SIZE
    if orig_w >= orig_h:
        resized_w = image_size
        resized_h = int(image_size * orig_h / orig_w)
    else:
        resized_w = int(image_size * orig_w / orig_h)
        resized_h = image_size

    resized_w = max(1, resized_w)
    resized_h = max(1, resized_h)

    resized_img = resize_image(img_np, resized_w, resized_h)

    padded_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    padded_img[0:resized_h, 0:resized_w, :] = resized_img

    input_data = padded_img.astype(np.float32) / 255.0
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)

    outputs = session.run(None, {input_name: input_data})
    preds = _normalize_yolo_predictions(outputs)

    if preds.shape[1] < 6:
        raise ValueError(f"Detector predictions have too few columns: {preds.shape}")

    obj_conf = preds[:, 4]
    if preds.shape[1] == 6:
        class_conf = preds[:, 5]
    else:
        class_conf = np.max(preds[:, 5:], axis=1)

    confidences = obj_conf * class_conf
    mask = confidences >= confidence_threshold
    valid_preds = preds[mask]
    valid_confs = confidences[mask]

    sys.stderr.write(
        f"[Koharu Scouter] tile=({x_offset},{y_offset},{orig_w},{orig_h}) total_preds={len(preds)} valid_preds={len(valid_preds)} threshold={confidence_threshold:.2f}\n"
    )

    if len(valid_preds) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

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


def _compute_sliding_window_starts(full_size, tile_size, step_size):
    """
    Compute deterministic sliding-window start coordinates that always include
    the trailing edge window.
    """
    if full_size <= tile_size:
        return [0]

    last_start = full_size - tile_size
    starts = list(range(0, last_start + 1, step_size))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def detect_text_bubbles(layer, confidence_threshold=0.22, nms_threshold=0.55):
    """
    Extracts GeglBuffer pixel data, runs ONNX text-bubble detector,
    and returns a list of bounding box coordinates [[xmin, ymin, xmax, ymax], ...].
    Uses full-image plus overlapping tiled inference to improve recall on large/stylized text.
    """
    buffer = layer.get_buffer()
    rect = buffer.get_extent()
    full_w = rect.width
    full_h = rect.height

    raw_data = buffer.get(rect, 1.0, "RGB u8", Gegl.AbyssPolicy.NONE)
    img_np = np.frombuffer(raw_data, dtype=np.uint8)
    img_np = img_np.reshape((full_h, full_w, 3))

    model_path = os.path.expanduser("~/Projects/gimp-scanlation-suite/models/comic-text-detector.onnx")
    if not os.path.exists(model_path):
        sys.stderr.write(f"[Koharu Scouter] Model file not found at: {model_path}\n")
        return []

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    all_boxes = []
    all_confs = []

    # Pass 1: full image using RGB to preserve title/text styling cues.
    boxes, confs = _run_model_on_image(
        session,
        input_name,
        img_np,
        confidence_threshold,
        nms_threshold,
        x_offset=0,
        y_offset=0,
    )
    if len(boxes) > 0:
        all_boxes.append(boxes)
        all_confs.append(confs)

    # Pass 2: overlapping model-sized tiled inference to preserve detail on
    # high-resolution pages while still covering the full image.
    tile_w = min(full_w, MODEL_IMAGE_SIZE)
    tile_h = min(full_h, MODEL_IMAGE_SIZE)
    step_x = max(1, tile_w // 2)
    step_y = max(1, tile_h // 2)

    x_starts = _compute_sliding_window_starts(full_w, tile_w, step_x)
    y_starts = _compute_sliding_window_starts(full_h, tile_h, step_y)

    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(full_w, x0 + tile_w)
            y1 = min(full_h, y0 + tile_h)
            tile = img_np[y0:y1, x0:x1, :]

            # Skip tiles that are effectively the full image to avoid duplicate work.
            if x0 == 0 and y0 == 0 and tile.shape[1] == full_w and tile.shape[0] == full_h:
                continue

            tile_boxes, tile_confs = _run_model_on_image(
                session,
                input_name,
                tile,
                confidence_threshold,
                nms_threshold,
                x_offset=x0,
                y_offset=y0,
            )
            if len(tile_boxes) > 0:
                all_boxes.append(tile_boxes)
                all_confs.append(tile_confs)

    if not all_boxes:
        return []

    boxes = np.vstack(all_boxes)
    confidences = np.concatenate(all_confs)

    boxes[:, 0] = np.clip(boxes[:, 0], 0, full_w)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, full_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, full_h)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, full_h)

    valid_geom = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid_geom]
    confidences = confidences[valid_geom]

    if len(boxes) == 0:
        return []

    keep = nms(boxes, confidences, nms_threshold)
    sys.stderr.write(
        f"[Koharu Scouter] merged_boxes={len(boxes)} kept_after_nms={len(keep)} nms_threshold={nms_threshold:.2f}\n"
    )
    final_boxes = boxes[keep]

    return final_boxes.tolist()
