import os
import sys
import numpy as np
import threading
import time
import io
import base64
from PIL import Image, ImageDraw
import gi

gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('GLib', '2.0')
from gi.repository import GLib
gi.require_version('Gegl', '0.4')
from gi.repository import Gegl

try:
    from modules import model_manager
except Exception as e:
    sys.stderr.write(f"[Scanlation Inpaint] Failed to import model_manager in runner: {e}\n")
    model_manager = None

def run_inpaint_processing(procedure, image, drawables, config):
    """
    Core inpainting processing runner.
    """
    # Parameters extraction
    inpaint_model = config.get_property("inpaint-model") or "lama-manga"
    dilation = config.get_property("dilation")
    inference_mode = config.get_property("inference-mode") or "Local"
    api_url = config.get_property("api-url") or "http://localhost:7890"

    sys.stderr.write(f"[Scanlation Inpaint] Running in {inference_mode} mode using '{inpaint_model}' (dilation={dilation}px)...\n")

    # 1. Verification of active layer
    if not drawables:
        Gimp.message("Error: No active drawable/layer selected.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())
        
    active_layer = drawables[0]

    # Resolve the best source layer for inpainting (skip text/group/system layers)
    def find_base_artwork_layer(img, act_layer):
        suitable_layers = []
        
        def traverse_layers(layers):
            for layer in layers:
                if hasattr(layer, "get_children"):
                    try:
                        children = Gimp.Item.get_children(layer)
                        if children:
                            traverse_layers(children)
                            continue
                    except Exception:
                        pass
                
                if hasattr(Gimp, "TextLayer") and isinstance(layer, Gimp.TextLayer):
                    continue
                if hasattr(layer, "get_text") and layer.get_text() is not None:
                    continue
                    
                name = layer.get_name()
                if name.startswith("[Inpaint]") or name in ["OCR Transcriptions", "Translated Text", "Detected Bubbles", "Curved Text"]:
                    continue
                    
                try:
                    parent = layer.get_parent()
                    if parent:
                        pname = parent.get_name()
                        if any(k in pname for k in ["OCR", "Translate", "Bubble", "Inpaint", "Curved"]):
                            continue
                except Exception:
                    pass
                    
                suitable_layers.append(layer)

        traverse_layers(img.get_layers())
        if suitable_layers:
            return suitable_layers[-1]
        return act_layer

    active_layer = find_base_artwork_layer(image, active_layer)

    # Check local import requirements
    if inference_mode == "Local":
        if model_manager is None:
            Gimp.message("Error: Model Manager could not be imported.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())
        try:
            import onnxruntime as ort
        except ImportError:
            Gimp.message("Error: onnxruntime is not installed in the virtual environment. Please run dispatcher or install it.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())
    else:
        try:
            from modules import remote_client
        except ImportError as e:
            Gimp.message(f"Error: Remote client module could not be imported: {e}")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    # Pump events
    while GLib.MainContext.default().iteration(False):
        pass

    # 2. Locate the path layer (Detected Bubbles or fallback)
    target_path = None
    paths = image.get_paths()
    for p in paths:
        if p.get_name().startswith("Detected Bubbles"):
            target_path = p
            break
            
    if not target_path:
        selected = image.get_selected_paths()
        if selected:
            target_path = selected[0]
            
    if not target_path and paths:
        target_path = paths[0]
        
    if not target_path:
        Gimp.message("Error: No paths/vectors found in the image. Please run detection first.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    sys.stderr.write(f"[Scanlation Inpaint] Reading bounding boxes from path: '{target_path.get_name()}'...\n")

    # 3. Retrieve strokes and parse coordinates
    bounding_boxes = []
    stroke_anchors = []
    try:
        strokes = target_path.get_strokes()
        for stroke_id in strokes:
            res = target_path.stroke_get_points(stroke_id)
            coords = None
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
                sys.stderr.write(f"[Scanlation Inpaint] Skipping stroke {stroke_id}: no coordinates retrieved.\n")
                continue

            x_coords = coords[0::2]
            y_coords = coords[1::2]
            if not x_coords or not y_coords:
                continue
                
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)
            
            bounding_boxes.append((xmin, ymin, xmax, ymax))

            # Interpolate Bezier curve between nodes to trace curves/rounded parts precisely
            closed = True
            try:
                closed = target_path.stroke_is_closed(stroke_id)
            except Exception:
                pass

            anchors = []
            if len(coords) >= 6:
                num_nodes = len(coords) // 6
                for j in range(num_nodes):
                    next_j = (j + 1) % num_nodes
                    idx_j = j * 6
                    
                    p0 = (coords[idx_j + 2], coords[idx_j + 3]) # anchor
                    p1 = (coords[idx_j + 4], coords[idx_j + 5]) # handle out
                    
                    idx_next = next_j * 6
                    p2 = (coords[idx_next], coords[idx_next + 1]) # handle in
                    p3 = (coords[idx_next + 2], coords[idx_next + 3]) # anchor
                    
                    if next_j == 0 and not closed:
                        break
                        
                    # Evaluate 15 points per Bezier segment for high-resolution curves
                    num_bezier_points = 15
                    for step in range(num_bezier_points):
                        t = step / float(num_bezier_points)
                        x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
                        y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
                        anchors.append((x, y))
                
                if not closed and num_nodes > 0:
                    idx_last = (num_nodes - 1) * 6
                    anchors.append((coords[idx_last + 2], coords[idx_last + 3]))
            else:
                anchors = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2) if i + 1 < len(coords)]
            
            stroke_anchors.append(anchors)
    except Exception as e:
        sys.stderr.write(f"[Scanlation Inpaint] Failed to parse paths/strokes: {e}\n")
        Gimp.message("Failed to extract coordinates from paths.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    if not bounding_boxes:
        Gimp.message("No valid bounding boxes found in the selected path.")
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    # Check local model weights presence if in Local Mode
    plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    if inference_mode == "Local":
        if inpaint_model == "lama-manga":
            repo = "mayocream/lama-manga-onnx"
            filename = "lama-manga.onnx"
            local_filename = "lama-manga.onnx"
        elif inpaint_model == "aot-inpainting":
            repo = "ogkalu/aot-inpainting"
            filename = "aot.onnx"
            local_filename = "aot-inpainting.onnx"
        elif inpaint_model in ["sd-inpainting", "anime-inpaint", "sdxl-inpainting"]:
            Gimp.message(f"Error: {inpaint_model} is only supported in Remote inference mode (running on the server with GPU). Please switch Inference Mode to Remote.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())
        else:
            Gimp.message(f"Error: Unknown local model option '{inpaint_model}'")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        models_dir = os.path.join(plugin_dir, "models")
        model_file_path = os.path.join(models_dir, local_filename)
        if not os.path.exists(model_file_path):
            Gimp.message(f"[Scanlation Inpaint] Local model '{inpaint_model}' weights not found. Downloading (approx. 100MB-200MB). This may take a moment...")
            while GLib.MainContext.default().iteration(False):
                pass

    # 4. Extract pixel buffer and construct mask
    try:
        buffer = active_layer.get_buffer()
        rect = buffer.get_extent()
        full_w = rect.width
        full_h = rect.height

        # Retrieve layer offsets
        success, offset_x, offset_y = active_layer.get_offsets()
        if not success:
            offset_x, offset_y = 0, 0

        # Pump events
        while GLib.MainContext.default().iteration(False):
            pass

        sys.stderr.write(f"[Scanlation Inpaint] Fetching active layer pixel buffer ({full_w}x{full_h})...\n")
        raw_data = buffer.get(rect, 1.0, "RGB u8", Gegl.AbyssPolicy.NONE)
        img_np = np.frombuffer(raw_data, dtype=np.uint8).reshape((full_h, full_w, 3))
    except Exception as e:
        sys.stderr.write(f"[Scanlation Inpaint] Failed to read layer pixels: {e}\n")
        Gimp.message("Failed to read active layer pixels.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    # Construct full-canvas mask
    mask_np = np.zeros((full_h, full_w), dtype=np.uint8)
    mask_pil = Image.fromarray(mask_np)
    draw = ImageDraw.Draw(mask_pil)

    for idx, box in enumerate(bounding_boxes):
        xmin, ymin, xmax, ymax = box
        anchors = stroke_anchors[idx] if idx < len(stroke_anchors) else []
        
        if len(anchors) >= 3:
            # Shift anchor coordinates by offset_x, offset_y
            shifted_anchors = [(int(ax - offset_x), int(ay - offset_y)) for ax, ay in anchors]
            draw.polygon(shifted_anchors, fill=255)
        else:
            x0 = int(np.clip(xmin - offset_x, 0, full_w))
            x1 = int(np.clip(xmax - offset_x, 0, full_w))
            y0 = int(np.clip(ymin - offset_y, 0, full_h))
            y1 = int(np.clip(ymax - offset_y, 0, full_h))
            if x1 > x0 and y1 > y0:
                draw.rectangle([x0, y0, x1, y1], fill=255)
                
    mask_np = np.array(mask_pil)

    # Dilation
    if dilation > 0:
        try:
            from PIL import ImageFilter
            mask_pil = Image.fromarray(mask_np)
            mask_pil = mask_pil.filter(ImageFilter.MaxFilter(size=2 * dilation + 1))
            mask_np = np.array(mask_pil)
        except Exception as dil_err:
            sys.stderr.write(f"[Scanlation Inpaint] Mask dilation failed: {dil_err}\n")

    # 5. Run inference with background thread and event loop pumping
    result_container = []
    error_container = []

    def progress_cb(percentage, message):
        def update_ui(pct, msg):
            if msg:
                Gimp.progress_set_text(msg)
            Gimp.progress_update(pct)
            return False
        GLib.idle_add(update_ui, percentage, message)

    if inference_mode == "Local":
        def worker():
            try:
                progress_cb(0.1, f"Loading model '{inpaint_model}'...")
                model_path = model_manager.ensure_model_exists(repo, filename, local_filename=local_filename)
                session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                
                progress_cb(0.4, "Running crop-based local inpainting...")
                
                # Work on a copy of the original image pixels
                out_img_np = img_np.copy()
                
                for idx, box in enumerate(bounding_boxes):
                    xmin, ymin, xmax, ymax = box
                    w = xmax - xmin
                    h = ymax - ymin
                    if w <= 0 or h <= 0:
                        continue
                        
                    # Centered square with a margin of 1.6x max dimension
                    side = int(max(w, h) * 1.6)
                    if side < 256:
                        side = 256
                    
                    cx = (xmin + xmax) // 2
                    cy = (ymin + ymax) // 2
                    
                    x0 = int(cx - side // 2)
                    x1 = x0 + side
                    y0 = int(cy - side // 2)
                    y1 = y0 + side
                    
                    x0_clipped = max(0, x0)
                    x1_clipped = min(full_w, x1)
                    y0_clipped = max(0, y0)
                    y1_clipped = min(full_h, y1)
                    
                    crop_w = x1_clipped - x0_clipped
                    crop_h = y1_clipped - y0_clipped
                    if crop_w <= 0 or crop_h <= 0:
                        continue
                        
                    crop_img = img_np[y0_clipped:y1_clipped, x0_clipped:x1_clipped]
                    crop_mask = mask_np[y0_clipped:y1_clipped, x0_clipped:x1_clipped]
                    
                    # Resize to 512x512
                    crop_img_pil = Image.fromarray(crop_img).resize((512, 512), Image.Resampling.BILINEAR)
                    crop_mask_pil = Image.fromarray(crop_mask).resize((512, 512), Image.Resampling.NEAREST)
                    
                    crop_img_512 = np.array(crop_img_pil)
                    crop_mask_512 = np.array(crop_mask_pil)
                    
                    img_feed = crop_img_512.astype(np.float32) / 255.0
                    img_feed = np.transpose(img_feed, (2, 0, 1))
                    img_feed = np.expand_dims(img_feed, axis=0)
                    
                    mask_feed = crop_mask_512.astype(np.float32) / 255.0
                    mask_feed = np.expand_dims(mask_feed, axis=0)
                    mask_feed = np.expand_dims(mask_feed, axis=0)
                    
                    # Zero out the masked region in the input image for correct in-distribution inference
                    img_feed = img_feed * (1.0 - mask_feed)
                    
                    input_names = [i.name for i in session.get_inputs()]
                    feeds = {}
                    for name in input_names:
                        if "image" in name.lower() or "input" in name.lower():
                            feeds[name] = img_feed
                        elif "mask" in name.lower():
                            feeds[name] = mask_feed
                    if len(feeds) < 2:
                        feeds = {input_names[0]: img_feed, input_names[1]: mask_feed}
                        
                    outputs = session.run(None, feeds)
                    out_crop = outputs[0]
                    
                    out_crop = np.squeeze(out_crop, axis=0)
                    out_crop = np.transpose(out_crop, (1, 2, 0))
                    out_crop = np.clip(out_crop * 255.0, 0.0, 255.0).astype(np.uint8)
                    
                    # Resize back
                    out_crop_pil = Image.fromarray(out_crop).resize((crop_w, crop_h), Image.Resampling.BILINEAR)
                    out_crop_original = np.array(out_crop_pil)
                    
                    # Blend using the original crop mask
                    mask_area = (crop_mask > 0)[:, :, np.newaxis]
                    out_img_np[y0_clipped:y1_clipped, x0_clipped:x1_clipped] = np.where(
                        mask_area,
                        out_crop_original,
                        out_img_np[y0_clipped:y1_clipped, x0_clipped:x1_clipped]
                    )
                    
                    progress_cb(0.4 + 0.5 * (idx + 1) / len(bounding_boxes), f"Processed region {idx+1}/{len(bounding_boxes)}...")
                
                result_container.append(out_img_np)
                progress_cb(1.0, "Done.")
            except Exception as ex:
                error_container.append(ex)
    else:
        def worker():
            try:
                progress_cb(0.1, "Encoding image and mask to PNG...")
                # Encode active layer image
                pil_img = Image.fromarray(img_np)
                buf_img = io.BytesIO()
                pil_img.save(buf_img, format="PNG")
                img_b64 = base64.b64encode(buf_img.getvalue()).decode("utf-8")
                
                # Encode mask
                pil_mask = Image.fromarray(mask_np)
                buf_mask = io.BytesIO()
                pil_mask.save(buf_mask, format="PNG")
                mask_b64 = base64.b64encode(buf_mask.getvalue()).decode("utf-8")
                
                progress_cb(0.3, "Offloading to remote dispatcher server...")
                
                options = {
                    "bounding_boxes": bounding_boxes,
                    "prompt": config.get_property("prompt") or "",
                    "negative_prompt": config.get_property("negative-prompt") or "",
                    "steps": config.get_property("steps") or 25,
                    "guidance_scale": config.get_property("guidance-scale") or 7.5,
                    "auto_prompt": config.get_property("auto-prompt") or False,
                    "consensus_arbiter": config.get_property("consensus-arbiter") or "DeepSeek"
                }
                res_b64_list = remote_client.dispatch_batch(
                    "inpaint",
                    inpaint_model,
                    [img_b64, mask_b64],
                    api_url,
                    options=options,
                    progress_callback=progress_cb
                )
                
                if not res_b64_list:
                    raise RuntimeError("No result received from remote dispatcher.")
                
                res_b64 = res_b64_list[0]
                img_data = base64.b64decode(res_b64)
                inpainted_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
                final_img_np = np.array(inpainted_pil)
                
                result_container.append(final_img_np)
                progress_cb(1.0, "Done.")
            except Exception as ex:
                error_container.append(ex)

    sys.stderr.write(f"[Scanlation Inpaint] Dispatching inpainting thread in background...\n")
    Gimp.progress_init(f"Inpainting dialogue regions ({inpaint_model})...")
    
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()

    # Pump events until worker finishes
    while t.is_alive():
        while GLib.MainContext.default().iteration(False):
            pass
        time.sleep(0.05)

    Gimp.progress_end()

    # Check results
    if not result_container:
        err_msg = error_container[0] if error_container else "Unknown error occurred during inpainting"
        sys.stderr.write(f"[Scanlation Inpaint] Inference error: {err_msg}\n")
        Gimp.message(f"Inpainting failed: {err_msg}")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    final_img = result_container[0]

    # 6. Save result as non-destructive layer above active_layer
    try:
        # Duplicate the active layer (preserves transparency/alpha settings/size)
        copy_layer = active_layer.copy()
        copy_layer.set_name(f"[Inpaint] {active_layer.get_name()}")
    except Exception as copy_err:
        sys.stderr.write(f"[Scanlation Inpaint] Failed to duplicate layer: {copy_err}\n")
        Gimp.message("Failed to create inpainting layer copy.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    # Insert layer exactly above active layer FIRST.
    # In GIMP 3, the layer must be attached to the image hierarchy before writing to its buffer.
    try:
        parent = active_layer.get_parent()
        siblings = parent.get_children() if parent else image.get_layers()
        try:
            idx = siblings.index(active_layer)
            image.insert_layer(copy_layer, parent, idx)
        except ValueError:
            image.insert_layer(copy_layer, parent, 0)
    except Exception as insert_err:
        sys.stderr.write(f"[Scanlation Inpaint] Failed to insert layer: {insert_err}\n")
        try:
            image.insert_layer(copy_layer, None, 0)
        except Exception:
            Gimp.message("Failed to insert the inpainted layer into image.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    # Now that the layer is attached to the image, write the output pixels into its buffer
    try:
        copy_buffer = copy_layer.get_buffer()
        copy_rect = copy_buffer.get_extent()
        copy_buffer.set(copy_rect, "RGB u8", final_img.tobytes())
    except Exception as write_err:
        sys.stderr.write(f"[Scanlation Inpaint] Failed to write inpainted buffer: {write_err}\n")
        Gimp.message("Failed to write inpainted pixels to the copied layer.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    # Activate the new layer
    try:
        image.set_selected_drawables([copy_layer])
    except Exception as sel_err:
        sys.stderr.write(f"[Scanlation Inpaint] Failed to set active layer: {sel_err}\n")

    # Flush display
    try:
        Gimp.displays_flush()
    except Exception:
        pass

    Gimp.message(f"Inpainting complete! Created non-destructive layer '{copy_layer.get_name()}' above '{active_layer.get_name()}'.")
    return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())
