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

from modules import preprocessor
from modules import postprocessor


def _run_model_on_image(session, input_name, img_np, confidence_threshold, nms_threshold, class_filter="Text Only", x_offset=0, y_offset=0):
    """
    Run detector on a numpy image tile.
    """
    orig_h, orig_w, _ = img_np.shape

    # Dynamic model input size detection
    try:
        input_shape = session.get_inputs()[0].shape
        if len(input_shape) >= 4 and isinstance(input_shape[2], int):
            image_size = input_shape[2]
        else:
            image_size = 640
    except Exception:
        image_size = 640

    # Preprocessing
    input_data, resized_w, resized_h, left_pad, top_pad = preprocessor.prepare_input_tensor(img_np, image_size)

    inputs = session.get_inputs()
    input_names = [i.name for i in inputs]

    if "orig_target_sizes" in input_names:
        # Construct orig_target_sizes input
        orig_target_sizes_input = next(i for i in inputs if i.name == "orig_target_sizes")
        type_str = orig_target_sizes_input.type
        if "int64" in type_str:
            orig_sizes = np.array([[image_size, image_size]], dtype=np.int64)
        elif "int32" in type_str:
            orig_sizes = np.array([[image_size, image_size]], dtype=np.int32)
        else:
            orig_sizes = np.array([[image_size, image_size]], dtype=np.float32)

        # Inference
        feed_dict = {
            input_name: input_data,
            "orig_target_sizes": orig_sizes
        }
        outputs = session.run(None, feed_dict)

        # Map outputs to a dictionary by name
        output_names = [o.name for o in session.get_outputs()]
        output_dict = {name: val for name, val in zip(output_names, outputs)}

        # Find boxes, scores, and labels outputs
        boxes_val = None
        scores_val = None
        labels_val = None
        for name, val in output_dict.items():
            if "box" in name.lower():
                boxes_val = val
            elif "score" in name.lower():
                scores_val = val
            elif "label" in name.lower():
                labels_val = val

        # Fallback based on shape if names don't match
        if boxes_val is None or scores_val is None or labels_val is None:
            for val in outputs:
                if val.ndim == 3 and val.shape[2] == 4:
                    boxes_val = val
                elif val.ndim == 2:
                    if val.dtype == np.int64 or val.dtype == np.int32:
                        labels_val = val
                    else:
                        scores_val = val

        if boxes_val is None or scores_val is None:
            raise ValueError("Could not identify boxes or scores outputs from the model.")

        return postprocessor.postprocess_rtdetr_outputs(
            boxes_val, scores_val, labels_val, confidence_threshold, orig_w, orig_h, resized_w, resized_h, left_pad, top_pad, class_filter, x_offset, y_offset
        )
    else:
        # Inference with single image input
        outputs = session.run(None, {input_name: input_data})

        # Implement defensive tensor slicing and dynamic logging of outputs[0] shape
        output0 = outputs[0]
        shape = getattr(output0, "shape", None)
        sys.stderr.write(f"[Koharu Scouter] outputs[0] shape={shape}\n")

        if shape is not None and len(shape) == 3 and shape[0] == 1:
            if shape[2] == 6:
                # (1, anchors, 6)
                preds = output0[0]
            elif shape[1] == 6:
                # (1, 6, anchors) -> Transpose to (anchors, 6)
                preds = output0[0].T
            else:
                # Fallback handling based on dimensions
                if shape[2] == 6 or (shape[2] != 6 and shape[1] == 6):
                    preds = output0[0].T if shape[1] == 6 else output0[0]
                else:
                    preds = output0[0].T if shape[1] < shape[2] else output0[0]
        else:
            # Fallback
            preds = output0
            if preds.ndim == 3 and preds.shape[0] == 1:
                preds = preds[0]
            if preds.ndim == 2:
                if preds.shape[1] != 6 and preds.shape[0] == 6:
                    preds = preds.T

        # Postprocessing
        return postprocessor.postprocess_boxes(
            preds, confidence_threshold, image_size, orig_w, orig_h,
            resized_w, resized_h, left_pad, top_pad, class_filter, x_offset, y_offset
        )


def _compute_sliding_window_starts(full_size, tile_size, step_size):
    """
    Compute sliding window start positions.
    """
    if full_size <= tile_size:
        return [0]
    last_start = full_size - tile_size
    starts = list(range(0, last_start + 1, step_size))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def detect_text_bubbles(layer, model_path, confidence_threshold=0.22, nms_threshold=0.55, class_filter="Text Only"):
    """
    Extracts GeglBuffer pixel data, converts to high-contrast grayscale,
    runs ONNX text-bubble detector on full image and sliding window tiles,
    applies NMS, and returns final bounding boxes.
    """
    buffer = layer.get_buffer()
    rect = buffer.get_extent()
    full_w = rect.width
    full_h = rect.height

    raw_data = buffer.get(rect, 1.0, "RGB u8", Gegl.AbyssPolicy.NONE)
    img_np = np.frombuffer(raw_data, dtype=np.uint8)
    img_np = img_np.reshape((full_h, full_w, 3))

    # Preprocessing: Convert RGB input to grayscale and restack to 3 channels
    img_np = preprocessor.to_grayscale(img_np)

    if not model_path or not os.path.exists(model_path):
        sys.stderr.write(f"[Koharu Scouter] Model path '{model_path}' not found.\n")
        return []

    # Enforce CPU Execution Provider
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Dynamic model image size detection for sliding window sizing
    try:
        input_shape = session.get_inputs()[0].shape
        if len(input_shape) >= 4 and isinstance(input_shape[2], int):
            model_image_size = input_shape[2]
        else:
            model_image_size = 640
    except Exception:
        model_image_size = 640

    all_boxes = []
    all_confs = []

    # Pass 1: Full image detection
    boxes, confs = _run_model_on_image(
        session, input_name, img_np, confidence_threshold, nms_threshold,
        class_filter=class_filter, x_offset=0, y_offset=0
    )
    if len(boxes) > 0:
        all_boxes.append(boxes)
        all_confs.append(confs)

    # Pass 2: Overlapping sliding-window tiled inference
    tile_w = min(full_w, model_image_size)
    tile_h = min(full_h, model_image_size)
    step_x = max(1, tile_w // 2)
    step_y = max(1, tile_h // 2)

    x_starts = _compute_sliding_window_starts(full_w, tile_w, step_x)
    y_starts = _compute_sliding_window_starts(full_h, tile_h, step_y)

    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(full_w, x0 + tile_w)
            y1 = min(full_h, y0 + tile_h)
            tile = img_np[y0:y1, x0:x1, :]

            # Skip if tile matches the full image
            if x0 == 0 and y0 == 0 and tile.shape[1] == full_w and tile.shape[0] == full_h:
                continue

            tile_boxes, tile_confs = _run_model_on_image(
                session, input_name, tile, confidence_threshold, nms_threshold,
                class_filter=class_filter, x_offset=x0, y_offset=y0
            )
            if len(tile_boxes) > 0:
                all_boxes.append(tile_boxes)
                all_confs.append(tile_confs)

    if not all_boxes:
        return []

    boxes = np.vstack(all_boxes)
    confidences = np.concatenate(all_confs)

    # Clip coordinates to GIMP layer bounds
    boxes[:, 0] = np.clip(boxes[:, 0], 0, full_w)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, full_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, full_h)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, full_h)

    # Filter invalid bounding boxes (where min >= max)
    valid_geom = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid_geom]
    confidences = confidences[valid_geom]

    if len(boxes) == 0:
        return []

    # Final NMS merging
    keep = postprocessor.nms(boxes, confidences, nms_threshold)
    sys.stderr.write(
        f"[Koharu Scouter] merged_boxes={len(boxes)} kept_after_nms={len(keep)} "
        f"nms_threshold={nms_threshold:.2f}\n"
    )
    final_boxes = boxes[keep]

    return final_boxes.tolist()
