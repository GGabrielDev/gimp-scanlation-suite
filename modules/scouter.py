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
    
    areas = (x2 - x1) * (y2 - y1)
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


def detect_text_bubbles(layer, confidence_threshold=0.4, nms_threshold=0.35):
    """
    Extracts GeglBuffer pixel data, runs ONNX text-bubble detector,
    and returns a list of bounding box coordinates [[xmin, ymin, xmax, ymax], ...].
    """
    # 1. GeglBuffer extraction
    buffer = layer.get_buffer()
    rect = buffer.get_extent()
    orig_w = rect.width
    orig_h = rect.height
    
    # Extract RGB u8 pixel data from GeglBuffer.
    # Babl automatically converts transparent, grayscale, or RGBA layer pixels to raw RGB bytes.
    raw_data = buffer.get(rect, 1.0, "RGB u8", Gegl.AbyssPolicy.NONE)
    img_np = np.frombuffer(raw_data, dtype=np.uint8)
    img_np = img_np.reshape((orig_h, orig_w, 3))
    
    # Convert RGB array to high-contrast grayscale array using standard luminosity formula,
    # then stack back to 3 channels to match model expectation.
    gray_img = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    img_np = np.stack((gray_img,) * 3, axis=-1)
    
    # 2. Preprocessing
    image_size = 1024
    if orig_w >= orig_h:
        resized_w = image_size
        resized_h = int(image_size * orig_h / orig_w)
    else:
        resized_w = int(image_size * orig_w / orig_h)
        resized_h = image_size
        
    resized_w = max(1, resized_w)
    resized_h = max(1, resized_h)
    
    # Resize keeping aspect ratio
    resized_img = resize_image(img_np, resized_w, resized_h)
    
    # Pad bottom-right with zero (black background) to 1024x1024
    padded_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    padded_img[0:resized_h, 0:resized_w, :] = resized_img
    
    # Normalize to [0, 1] and permute to BCHW (1, 3, 1024, 1024)
    input_data = padded_img.astype(np.float32) / 255.0
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)
    
    # 3. Model path and session initialization
    # Path is absolute under the active workspace models/ directory
    model_path = os.path.expanduser("~/Projects/gimp-scanlation-suite/models/comic-text-detector.onnx")
    
    if not os.path.exists(model_path):
        # Log error to sys.stderr so as not to pollute GIMP wire protocol
        sys.stderr.write(f"[Koharu Scouter] Model file not found at: {model_path}\n")
        return []
        
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    # Run the ONNX model
    outputs = session.run(None, {input_name: input_data})
    
    # Locate yolo predictions tensor (shape 1, num_anchors, 6)
    predictions = None
    for out in outputs:
        if len(out.shape) == 3 and out.shape[0] == 1 and out.shape[2] == 6:
            predictions = out
            break
    if predictions is None:
        predictions = outputs[0]
        
    preds = predictions[0] # Squeeze batch dimension: (num_anchors, 6)
    
    # 4. Filter predictions by confidence threshold
    # Coordinates mapping: x_center, y_center, w, h, objectness, class_score
    obj_conf = preds[:, 4]
    class_conf = preds[:, 5]
    confidences = obj_conf * class_conf
    
    mask = confidences >= confidence_threshold
    valid_preds = preds[mask]
    valid_confs = confidences[mask]
    
    if len(valid_preds) == 0:
        return []
        
    x_center = valid_preds[:, 0]
    y_center = valid_preds[:, 1]
    w = valid_preds[:, 2]
    h = valid_preds[:, 3]
    
    # Scale coordinates back to original size (excluding padding margins)
    w_ratio = orig_w / resized_w
    h_ratio = orig_h / resized_h
    
    # Bbox dilation of 1px matching the Rust pipeline
    bbox_dilation = 1.0
    x1 = (x_center - w / 2.0) * w_ratio - bbox_dilation
    x2 = (x_center + w / 2.0) * w_ratio + bbox_dilation
    y1 = (y_center - h / 2.0) * h_ratio - bbox_dilation
    y2 = (y_center + h / 2.0) * h_ratio + bbox_dilation
    
    # Clamp coordinates to original image boundaries
    x1 = np.clip(x1, 0, orig_w)
    x2 = np.clip(x2, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    y2 = np.clip(y2, 0, orig_h)
    
    boxes = np.column_stack([x1, y1, x2, y2])
    
    # 5. Non-Maximum Suppression
    keep = nms(boxes, valid_confs, nms_threshold)
    final_boxes = boxes[keep]
    
    # Convert numpy float coordinates to python float lists
    return final_boxes.tolist()
