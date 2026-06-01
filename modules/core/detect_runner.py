import os
import sys
import gi

gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('GObject', '2.0')
from gi.repository import GObject
gi.require_version('GLib', '2.0')
from gi.repository import GLib

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
    Runs local ONNX inference on the selected drawable and registers
    the detected bounding boxes as native GIMP Paths (Gimp.Vectors).
    """
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
