import os
import sys
import gi
import numpy as np

gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('GObject', '2.0')
from gi.repository import GObject
gi.require_version('GLib', '2.0')
from gi.repository import GLib
gi.require_version('Gegl', '0.4')
from gi.repository import Gegl

try:
    from modules import scouter
except Exception as e:
    sys.stderr.write(f"[Scanlation Detector] Failed to import scouter in runner: {e}\n")
    scouter = None

try:
    from modules import model_manager
except Exception as e:
    sys.stderr.write(f"[Scanlation Detector] Failed to import model_manager in runner: {e}\n")
    model_manager = None

def run_detect_processing(procedure, image, drawables, config):
    """
    Runs local ONNX inference or Tight Pathing contouring.
    """
    detection_mode = config.get_property("detection-mode") or "Initial Detection"
    if detection_mode == "Tight Pathing":
        return run_tight_pathing_processing(procedure, image, drawables, config)

    # Parameters extraction
    detector_model = config.get_property("detector-model")
    confidence = config.get_property("confidence")
    class_filter = config.get_property("class-filter")

    Gimp.message(f"[Scanlation Detector] Running '{detector_model}' with threshold={confidence:.2f}...")

    # 1. Verification of active layer
    if not drawables:
        Gimp.message("Error: No active drawable/layer selected.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())
        
    active_layer = drawables[0]

    # 2. Call model manager to ensure model exists
    if model_manager is None:
        Gimp.message("Error: Model manager module could not be imported. Check venv dependencies.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    # Map detector-model string to Hugging Face repo and filename
    if "pp-doclayoutv3" in detector_model.lower():
        model_id = "alex-dinh/PP-DocLayoutV3-ONNX"
        filename = "PP-DocLayoutV3.onnx"
    else:
        model_id = "ogkalu/comic-text-and-bubble-detector"
        filename = "detector.onnx"

    # Pump events to prevent UI freeze
    while GLib.MainContext.default().iteration(False):
        pass

    # Check if the model needs to be downloaded
    plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    models_dir = os.path.join(plugin_dir, "models")
    local_path = os.path.join(models_dir, filename)
    if not os.path.exists(local_path):
        Gimp.message(f"[Scanlation Detector] Downloading model '{model_id}/{filename}'... This may take a moment.")
        # Pump events again
        while GLib.MainContext.default().iteration(False):
            pass

    try:
        model_path = model_manager.ensure_model_exists(model_id, filename)
    except Exception as e:
        sys.stderr.write(f"[Scanlation Detector] Model acquisition failed: {e}\n")
        Gimp.message(f"Model download failed. Check GIMP error logs.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    # Pump events again
    while GLib.MainContext.default().iteration(False):
        pass

    # 3. Call scouter to detect bounding boxes
    if scouter is None:
        Gimp.message("Error: Scouter module could not be imported. Check venv dependencies.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    try:
        boxes = scouter.detect_text_bubbles(active_layer, model_path, confidence_threshold=confidence, class_filter=class_filter)
    except Exception as e:
        # Route all exceptions/debug output to stderr to protect GIMP's wire protocol
        sys.stderr.write(f"[Scanlation Detector] Inference error: {e}\n")
        Gimp.message(f"Inference failed. Check GIMP error logs.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    if not boxes:
        Gimp.message("No text bubbles detected.")
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    # 4. Create native GIMP path containing the bounding boxes
    try:
        # Dynamic check for Gimp.Path vs Gimp.Vectors (GIMP 3.0 API abstraction)
        if hasattr(Gimp, "Path"):
            path_class = Gimp.Path
        else:
            path_class = Gimp.Vectors
            
        if hasattr(Gimp, "PathStrokeType"):
            stroke_type = Gimp.PathStrokeType.BEZIER
        elif hasattr(Gimp, "VectorsStrokeType"):
            stroke_type = Gimp.VectorsStrokeType.BEZIER
        else:
            stroke_type = 0  # Fallback integer value representing Bezier stroke

        vectors = path_class.new(image, f"Detected Bubbles ({len(boxes)})")

        for box in boxes:
            xmin, ymin, xmax, ymax = box
            # Define 4-point closed rectangular Bezier stroke.
            # In GIMP paths, each anchor requires triplets: (handle_in_x/y, anchor_x/y, handle_out_x/y)
            points = [
                xmin, ymin, xmin, ymin, xmin, ymin,  # Top-left corner
                xmax, ymin, xmax, ymin, xmax, ymin,  # Top-right corner
                xmax, ymax, xmax, ymax, xmax, ymax,  # Bottom-right corner
                xmin, ymax, xmin, ymax, xmin, ymax   # Bottom-left corner
            ]
            vectors.stroke_new_from_points(stroke_type, points, True)

        # Insert path into the image stack
        if hasattr(image, "insert_path"):
            image.insert_path(vectors, None, -1)
        elif hasattr(image, "add_path"):
            image.add_path(vectors, None, -1)
        elif hasattr(image, "add_vectors"):
            image.add_vectors(vectors, -1)
        else:
            sys.stderr.write("[Scanlation Detector] Failed to add path: API unsupported.\n")
            
        Gimp.message(f"Successfully generated paths for {len(boxes)} text bubbles.")
    except Exception as e:
        sys.stderr.write(f"[Scanlation Detector] Path creation failed: {e}\n")
        Gimp.message("Failed to generate path layers.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

def run_tight_pathing_processing(procedure, image, drawables, config):
    if not drawables:
        Gimp.message("Error: No active drawable/layer selected.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())
    
    active_layer = drawables[0]
    tight_path_name = config.get_property("tight-path-name")
    tight_path_mode = config.get_property("tight-path-mode") or "Speech Bubble"
    dilation_radius = config.get_property("tight-path-dilation") or 8
    
    # 1. Locate the target path
    target_path = None
    if tight_path_name:
        for p in image.get_paths():
            if p.get_name() == tight_path_name:
                target_path = p
                break
    if not target_path:
        selected = image.get_selected_paths()
        if selected:
            target_path = selected[0]
        else:
            paths = image.get_paths()
            if paths:
                target_path = paths[0]
                
    if not target_path:
        Gimp.message("Error: No path selected or found in the image. Please run Initial Detection first.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())
        
    Gimp.message(f"[Scanlation Detector] Running Tight Pathing ({tight_path_mode}) on path '{target_path.get_name()}'...")
    
    # 2. Get active layer offsets and pixels
    try:
        buffer = active_layer.get_buffer()
        rect = buffer.get_extent()
        full_w = rect.width
        full_h = rect.height
        success, offset_x, offset_y = active_layer.get_offsets()
        if not success:
            offset_x, offset_y = 0, 0
            
        raw_data = buffer.get(rect, 1.0, "RGB u8", Gegl.AbyssPolicy.NONE)
        img_np = np.frombuffer(raw_data, dtype=np.uint8).reshape((full_h, full_w, 3))
    except Exception as e:
        sys.stderr.write(f"[Scanlation Detector] Failed to read active layer pixels: {e}\n")
        Gimp.message("Failed to read active layer pixels. Check error log.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())
        
    # Convert image to grayscale for analysis
    img_gray = (0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]).astype(np.uint8)
    
    # 3. Process each stroke in the path
    strokes = target_path.get_strokes()
    if not strokes:
        Gimp.message("Error: The selected path has no strokes.")
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())
        
    # Determine GIMP Path Classes
    if hasattr(Gimp, "Path"):
        path_class = Gimp.Path
    else:
        path_class = Gimp.Vectors
        
    if hasattr(Gimp, "PathStrokeType"):
        stroke_type = Gimp.PathStrokeType.BEZIER
    elif hasattr(Gimp, "VectorsStrokeType"):
        stroke_type = Gimp.VectorsStrokeType.BEZIER
    else:
        stroke_type = 0
        
    tight_vectors = path_class.new(image, f"[Tight] {target_path.get_name()}")
    
    strokes_added = 0
    for stroke_id in strokes:
        res = target_path.stroke_get_points(stroke_id)
        coords = None
        # Retrieve points coordinates exactly as OCR does
        if isinstance(res, tuple) or isinstance(res, list):
            for item in res:
                if isinstance(item, list) or isinstance(item, tuple):
                    if len(item) > 0 and isinstance(item[0], (int, float)):
                        coords = list(item)
                        break
            if coords is None:
                for item in res:
                    if hasattr(item, "controlpoints"):
                        coords = list(item.controlpoints)
                        break
                    elif hasattr(item, "points"):
                        coords = list(item.points)
                        break
        else:
            if hasattr(res, "controlpoints"):
                coords = list(res.controlpoints)
            elif hasattr(res, "points"):
                coords = list(res.points)
                
        if not coords:
            continue
            
        x_coords = coords[0::2]
        y_coords = coords[1::2]
        if not x_coords or not y_coords:
            continue
            
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        
        # 4. Crop the region and do the tight pathing
        # Map canvas-relative coordinates to layer-relative index positions
        x0 = int(np.clip(xmin - offset_x, 0, full_w))
        x1 = int(np.clip(xmax - offset_x, 0, full_w))
        y0 = int(np.clip(ymin - offset_y, 0, full_h))
        y1 = int(np.clip(ymax - offset_y, 0, full_h))
        
        crop_w = x1 - x0
        crop_h = y1 - y0
        if crop_w <= 3 or crop_h <= 3:
            continue
            
        crop_gray = img_gray[y0:y1, x0:x1]
        
        # Run custom segmenter
        if tight_path_mode == "Speech Bubble":
            contours = process_speech_bubble(crop_gray)
        else:
            contours = process_floating_text(crop_gray, dilation_radius)
            
        for contour in contours:
            # Simplify each contour
            simplified = rdp_simplify(contour, 1.2)
            if len(simplified) < 3:
                continue
                
            # Convert back to canvas coordinates and construct Bezier format
            gimp_points = []
            for cy, cx in simplified:
                canvas_x = cx + x0 + offset_x
                canvas_y = cy + y0 + offset_y
                gimp_points.extend([canvas_x, canvas_y, canvas_x, canvas_y, canvas_x, canvas_y])
                
            tight_vectors.stroke_new_from_points(stroke_type, gimp_points, True)
            strokes_added += 1
            
    # 5. Insert new path into image stack
    if strokes_added > 0:
        if hasattr(image, "insert_path"):
            image.insert_path(tight_vectors, None, -1)
        elif hasattr(image, "add_path"):
            image.add_path(tight_vectors, None, -1)
        elif hasattr(image, "add_vectors"):
            image.add_vectors(tight_vectors, -1)
        Gimp.message(f"Successfully generated tight path with {strokes_added} strokes.")
    else:
        Gimp.message("No tight path strokes could be generated.")
        
    Gimp.displays_flush()
    return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

def process_speech_bubble(crop_gray):
    h, w = crop_gray.shape
    # 1. Search for brightest pixel near center for seed selection
    cy, cx = h // 2, w // 2
    search_radius = min(15, h // 4, w // 4)
    best_y, best_x = cy, cx
    max_val = -1
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            y, x = cy + dy, cx + dx
            if 0 <= y < h and 0 <= x < w:
                if crop_gray[y, x] > max_val:
                    max_val = crop_gray[y, x]
                    best_y, best_x = y, x
                    
    # Flood-fill light pixels from seed
    mask = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    queue = [(best_y, best_x)]
    visited[best_y, best_x] = True
    mask[best_y, best_x] = True
    
    head = 0
    while head < len(queue):
        curr_y, curr_x = queue[head]
        head += 1
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = curr_y + dy, curr_x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                if crop_gray[ny, nx] > 140:
                    visited[ny, nx] = True
                    mask[ny, nx] = True
                    queue.append((ny, nx))
                    
    # Fill internal holes (solidify text inside bubble)
    filled_mask = fill_bubble_holes(mask)
    
    # Trace contour
    contour = trace_contour_moore(filled_mask)
    return [contour] if contour else []

def fill_bubble_holes(mask):
    h, w = mask.shape
    reachable = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    
    queue = []
    # Borders seeds
    for x in range(w):
        for y in [0, h - 1]:
            if not mask[y, x] and not visited[y, x]:
                visited[y, x] = True
                reachable[y, x] = True
                queue.append((y, x))
    for y in range(h):
        for x in [0, w - 1]:
            if not mask[y, x] and not visited[y, x]:
                visited[y, x] = True
                reachable[y, x] = True
                queue.append((y, x))
                
    head = 0
    while head < len(queue):
        cy, cx = queue[head]
        head += 1
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                if not mask[ny, nx]:
                    visited[ny, nx] = True
                    reachable[ny, nx] = True
                    queue.append((ny, nx))
                    
    return mask | (~mask & ~reachable)

def process_floating_text(crop_gray, dilation_radius):
    # Threshold text pixels (dark characters)
    text_mask = crop_gray < 120
    
    # Dilate using iterative binary shifting to group letters
    dilated = dilate_mask(text_mask, dilation_radius)
    
    # Trace all separate components
    contours = trace_all_contours(dilated)
    return contours

def dilate_mask(mask, radius):
    dilated = np.copy(mask)
    h, w = mask.shape
    for _ in range(radius):
        prev = np.copy(dilated)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            y0_src = max(0, -dy)
            y1_src = min(h, h - dy)
            x0_src = max(0, -dx)
            x1_src = min(w, w - dx)
            
            y0_dst = max(0, dy)
            y1_dst = min(h, h + dy)
            x0_dst = max(0, dx)
            x1_dst = min(w, w + dx)
            
            dilated[y0_dst:y1_dst, x0_dst:x1_dst] |= prev[y0_src:y1_src, x0_src:x1_src]
    return dilated

def trace_all_contours(mask):
    temp_mask = np.copy(mask)
    h, w = temp_mask.shape
    contours = []
    
    for y in range(h):
        for x in range(w):
            if temp_mask[y, x]:
                # Extract connected component
                comp_mask = flood_fill_component(temp_mask, y, x)
                contour = trace_contour_moore(comp_mask)
                if len(contour) > 4:
                    contours.append(contour)
                temp_mask[comp_mask] = False
    return contours

def flood_fill_component(mask, sy, sx):
    h, w = mask.shape
    comp = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    
    queue = [(sy, sx)]
    visited[sy, sx] = True
    comp[sy, sx] = True
    
    head = 0
    while head < len(queue):
        cy, cx = queue[head]
        head += 1
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                if mask[ny, nx]:
                    visited[ny, nx] = True
                    comp[ny, nx] = True
                    queue.append((ny, nx))
    return comp

def trace_contour_moore(mask):
    H, W = mask.shape
    padded = np.zeros((H + 2, W + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    
    start_y, start_x = None, None
    for y in range(1, H + 1):
        for x in range(1, W + 1):
            if padded[y, x]:
                start_y, start_x = y, x
                break
        if start_y is not None:
            break
            
    if start_y is None:
        return []
        
    directions = [
        (-1, 0),   # N
        (-1, 1),   # NE
        (0, 1),    # E
        (1, 1),    # SE
        (1, 0),    # S
        (1, -1),   # SW
        (0, -1),   # W
        (-1, -1)   # NW
    ]
    
    contour = []
    curr_y, curr_x = start_y, start_x
    back_dir_idx = 6 # Start scanning from West
    
    max_steps = (H + 2) * (W + 2) * 2
    step = 0
    while step < max_steps:
        contour.append((curr_y - 1, curr_x - 1))
        
        found = False
        start_search = (back_dir_idx + 1) % 8
        for i in range(8):
            idx = (start_search + i) % 8
            dy, dx = directions[idx]
            ny, nx = curr_y + dy, curr_x + dx
            if padded[ny, nx]:
                back_dir_idx = (idx + 4) % 8
                curr_y, curr_x = ny, nx
                found = True
                break
                
        if not found:
            break
            
        if curr_y == start_y and curr_x == start_x:
            break
            
        step += 1
        
    return contour

def distance_point_to_line(p, p1, p2):
    y, x = p[0], p[1]
    y1, x1 = p1[0], p1[1]
    y2, x2 = p2[0], p2[1]
    
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return np.sqrt((x - x1)**2 + (y - y1)**2)
        
    l2 = dx**2 + dy**2
    t = ((x - x1) * dx + (y - y1) * dy) / l2
    t = max(0.0, min(1.0, t))
    
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    return np.sqrt((x - proj_x)**2 + (y - proj_y)**2)

def rdp_simplify(points, epsilon):
    if len(points) < 3:
        return points
        
    dmax = 0.0
    index = 0
    end = len(points) - 1
    
    for i in range(1, end):
        d = distance_point_to_line(points[i], points[0], points[end])
        if d > dmax:
            index = i
            dmax = d
            
    if dmax > epsilon:
        results1 = rdp_simplify(points[:index+1], epsilon)
        results2 = rdp_simplify(points[index:], epsilon)
        return results1[:-1] + results2
    else:
        return [points[0], points[end]]
