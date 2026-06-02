import os
import sys
import numpy as np
import gi
import re

gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
from gi.repository import GLib
from gi.repository import Gegl
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

try:
    from modules import ocr_engine
except ImportError as e:
    sys.stderr.write(f"[Scanlation Suite] Failed to import ocr_engine: {e}\n")
    ocr_engine = None

try:
    from modules import remote_client
except ImportError as e:
    remote_client = None

def clean_and_normalize_text(text, half_to_full=True):
    """
    Applies custom normalization rules to clean the recognized OCR text.
    """
    if not text:
        return ""
    
    text = text.strip()
    
    if half_to_full:
        chars = []
        for c in text:
            code = ord(c)
            if code == 0x0020:
                chars.append(chr(0x3000))
            elif 0x0021 <= code <= 0x007E:
                chars.append(chr(code + 0xfee0))
            else:
                chars.append(c)
        text = "".join(chars)
        
    text = re.sub(r'\.{2,}', '...', text)
    text = re.sub(r'…+', '...', text)
    text = re.sub(r'・{2,}', '...', text)
    text = text.replace('…', '...')
    
    return text

import json

def load_context_cache():
    cache_path = os.path.expanduser("~/.gimp_scanlation_ocr_context_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            sys.stderr.write(f"[Scanlation OCR] Failed to load context cache: {e}\n")
    return {}

def save_context_cache(cache):
    cache_path = os.path.expanduser("~/.gimp_scanlation_ocr_context_cache.json")
    try:
        if len(cache) > 100:
            keys_to_remove = list(cache.keys())[:len(cache) - 100]
            for k in keys_to_remove:
                del cache[k]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        sys.stderr.write(f"[Scanlation OCR] Failed to save context cache: {e}\n")

def find_cached_hint(cache, image_key, xmin, ymin, xmax, ymax):
    if image_key not in cache:
        return ""
    img_cache = cache[image_key]
    
    exact_key = f"{int(xmin)},{int(ymin)},{int(xmax)},{int(ymax)}"
    if exact_key in img_cache:
        return img_cache[exact_key]
        
    for coord_str, hint in img_cache.items():
        try:
            parts = [int(p) for p in coord_str.split(",")]
            if len(parts) == 4:
                cx_min, cy_min, cx_max, cy_max = parts
                if (abs(cx_min - xmin) <= 5 and 
                    abs(cy_min - ymin) <= 5 and 
                    abs(cx_max - xmax) <= 5 and 
                    abs(cy_max - ymax) <= 5):
                    return hint
        except Exception:
            continue
            
    return ""

def run_ocr_processing(procedure, image, active_layer, bounding_boxes, config, run_mode):
    """
    Executes Japanese Manga OCR processing, including GIMP crops, Remote VLM/Local OCR,
    per-block options configuration, and GIMP TextLayer creation.
    """
    ocr_engine_param = config.get_property("ocr-engine") or "PaddleOCR"
    half_to_full = config.get_property("half-to-full")
    inference_mode = config.get_property("inference-mode") or "Local"
    api_url = config.get_property("api-url") or "http://localhost:7890"
    target_lang = config.get_property("target-language") or "Japanese"
    source_lang = config.get_property("source-language") or "Japanese"
    material_type = config.get_property("material-type") or "manga"
    ensemble_consensus = config.get_property("ensemble-consensus")
    consensus_expert_b = config.get_property("consensus-expert-b") or "PaddleOCR_Manga"
    consensus_arbiter = config.get_property("consensus-arbiter") or "DeepSeek"
    enable_thinking = config.get_property("enable-thinking")
    configure_per_path = config.get_property("configure-per-path")

    plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    if inference_mode == "Local":
        if ocr_engine is None:
            Gimp.message("Error: OCR engine module could not be imported. Check venv dependencies.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())
    else:
        if remote_client is None:
            Gimp.message("Error: Remote client module could not be imported.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    # Check local model weights presence if in Local Mode
    if inference_mode == "Local":
        models_dir = os.path.join(plugin_dir, "models")
        gguf_path = os.path.join(models_dir, "PaddleOCR-VL-1.5-Q4_K_M.gguf")
        projector_path = os.path.join(models_dir, "PaddleOCR-VL-1.5-mmproj.gguf")
        
        if not os.path.exists(gguf_path) or not os.path.exists(projector_path):
            Gimp.message("[Scanlation OCR] OCR model weights not found locally. Downloading PaddleOCR-VL-1.5 GGUF and vision projector (approx. 180MB total). This may take a moment...")
            while GLib.MainContext.default().iteration(False):
                pass

    # Resolve the best source layer for cropping pixels (skip text/group/system layers)
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

    source_layer = find_base_artwork_layer(image, active_layer)

    # Extract pixel crops
    try:
        buffer = source_layer.get_buffer()
        rect = buffer.get_extent()
        full_w = rect.width
        full_h = rect.height

        success, offset_x, offset_y = source_layer.get_offsets()
        if not success:
            offset_x, offset_y = 0, 0

        # Pump events
        while GLib.MainContext.default().iteration(False):
            pass

        sys.stderr.write(f"[Scanlation OCR] Fetching active layer pixel buffer ({full_w}x{full_h})...\n")
        raw_data = buffer.get(rect, 1.0, "RGB u8", Gegl.AbyssPolicy.NONE)
        img_np = np.frombuffer(raw_data, dtype=np.uint8).reshape((full_h, full_w, 3))
    except Exception as e:
        sys.stderr.write(f"[Scanlation OCR] Failed to read layer pixels: {e}\n")
        Gimp.message("Failed to read active layer pixels.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    # Pump events
    while GLib.MainContext.default().iteration(False):
        pass

    sys.stderr.write(f"[Scanlation OCR] run_ocr_processing called with {len(bounding_boxes)} bounding boxes.\n")

    # Crop all regions
    crops = []
    valid_boxes = []
    for i, box in enumerate(bounding_boxes):
        xmin, ymin, xmax, ymax = box
        
        x0 = int(np.clip(xmin - offset_x, 0, full_w))
        x1 = int(np.clip(xmax - offset_x, 0, full_w))
        y0 = int(np.clip(ymin - offset_y, 0, full_h))
        y1 = int(np.clip(ymax - offset_y, 0, full_h))

        if x1 <= x0 or y1 <= y0:
            sys.stderr.write(f"[Scanlation OCR] Box {i} ({box}) skipped: empty intersection (x0={x0}, x1={x1}, y0={y0}, y1={y1}, full_w={full_w}, full_h={full_h}, offset_x={offset_x}, offset_y={offset_y})\n")
            continue

        crops.append(img_np[y0:y1, x0:x1, :])
        valid_boxes.append(box)

    if not crops:
        Gimp.message("No valid cropped text regions to process.")
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    ocr_results = []

    # Load cached context hints
    cache = load_context_cache()
    image_key = None
    try:
        gfile = image.get_file()
        if gfile:
            image_key = gfile.get_uri()
    except Exception:
        pass
    if not image_key:
        try:
            image_key = image.get_name()
        except Exception:
            image_key = "unknown_image"

    # Initialize per-path options
    per_path_options = []
    for i in range(len(crops)):
        box = valid_boxes[i]
        xmin, ymin, xmax, ymax = box
        cached_hint = find_cached_hint(cache, image_key, xmin, ymin, xmax, ymax)
        per_path_options.append({
            "enable_thinking": enable_thinking,
            "context_hint": cached_hint
        })

    if inference_mode == "Remote" and configure_per_path and run_mode == Gimp.RunMode.INTERACTIVE:
        from gi.repository import GdkPixbuf
        import io
        import base64
        import threading
        from PIL import Image

        def numpy_to_pixbuf(np_arr, max_width=120, max_height=80):
            pil_img = Image.fromarray(np_arr)
            pil_img.thumbnail((max_width, max_height))
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            
            loader = GdkPixbuf.PixbufLoader.new_with_type("png")
            loader.write(buffered.getvalue())
            loader.close()
            return loader.get_pixbuf()

        def run_single_row_analysis(btn, crop_img, hint_entry):
            btn.set_sensitive(False)
            btn.set_label("⏳")
            
            def run_analysis():
                try:
                    pil_img = Image.fromarray(crop_img)
                    buffered = io.BytesIO()
                    pil_img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    options = {
                        "analyze_style": True,
                        "source_language": source_lang,
                        "material_type": material_type
                    }
                    
                    results = remote_client.dispatch_batch(
                        task_type="ocr",
                        model_id=consensus_expert_b,
                        batch_payload=[{"image_data": img_str}],
                        api_url=api_url,
                        options=options
                    )
                    
                    if results and results[0]:
                        description = results[0].strip()
                        GLib.idle_add(lambda: hint_entry.set_text(description))
                    else:
                        GLib.idle_add(lambda: hint_entry.set_text("No analysis returned."))
                except Exception as ex:
                    sys.stderr.write(f"[Scanlation OCR] Style analysis failed: {ex}\n")
                    GLib.idle_add(lambda: hint_entry.set_text("Analysis failed."))
                finally:
                    GLib.idle_add(lambda: btn.set_sensitive(True))
                    GLib.idle_add(lambda: btn.set_label("✨"))
            
            t = threading.Thread(target=run_analysis)
            t.daemon = True
            t.start()

        def on_analyze_clicked(button, crop_img, hint_entry):
            run_single_row_analysis(button, crop_img, hint_entry)

        def on_analyze_all_clicked(btn_all, individual_buttons):
            btn_all.set_sensitive(False)
            btn_all.set_label("Analyzing All...")
            
            rows_to_analyze = []
            for btn, crop_img, hint_entry in individual_buttons:
                btn.set_sensitive(False)
                rows_to_analyze.append((btn, crop_img, hint_entry))
                
            def run_sequential_analysis():
                try:
                    for btn, crop_img, hint_entry in rows_to_analyze:
                        GLib.idle_add(lambda b=btn: b.set_label("⏳"))
                        
                        try:
                            pil_img = Image.fromarray(crop_img)
                            buffered = io.BytesIO()
                            pil_img.save(buffered, format="PNG")
                            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                            
                            options = {
                                "analyze_style": True,
                                "source_language": source_lang,
                                "material_type": material_type
                            }
                            
                            results = remote_client.dispatch_batch(
                                task_type="ocr",
                                model_id=consensus_expert_b,
                                batch_payload=[{"image_data": img_str}],
                                api_url=api_url,
                                options=options
                            )
                            
                            if results and results[0]:
                                description = results[0].strip()
                                GLib.idle_add(lambda e=hint_entry, d=description: e.set_text(d))
                            else:
                                GLib.idle_add(lambda e=hint_entry: e.set_text("No analysis returned."))
                        except Exception as ex:
                            sys.stderr.write(f"[Scanlation OCR] Style analysis failed: {ex}\n")
                            GLib.idle_add(lambda e=hint_entry: e.set_text("Analysis failed."))
                        finally:
                            GLib.idle_add(lambda b=btn: b.set_sensitive(True))
                            GLib.idle_add(lambda b=btn: b.set_label("✨"))
                finally:
                    GLib.idle_add(lambda: btn_all.set_sensitive(True))
                    GLib.idle_add(lambda: btn_all.set_label("✨ Analyze All"))
            
            t = threading.Thread(target=run_sequential_analysis)
            t.daemon = True
            t.start()

        desc_dialog = Gtk.Dialog(title="Configure Options Per Text Block", parent=None, flags=0)
        desc_dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OK, Gtk.ResponseType.OK)
        desc_dialog.set_default_size(650, 480)

        content_area = desc_dialog.get_content_area()
        
        individual_buttons = []

        title_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        title_hbox.set_margin_top(12)
        title_hbox.set_margin_bottom(6)
        title_hbox.set_margin_start(12)
        title_hbox.set_margin_end(12)
        
        lbl = Gtk.Label()
        lbl.set_markup("<span size='large' weight='bold' foreground='#3584e4'>Per-Block Configuration</span>")
        lbl.set_xalign(0.0)
        title_hbox.pack_start(lbl, True, True, 0)
        
        btn_analyze_all = Gtk.Button(label="✨ Analyze All")
        btn_analyze_all.connect("clicked", on_analyze_all_clicked, individual_buttons)
        title_hbox.pack_end(btn_analyze_all, False, False, 0)
        
        content_area.pack_start(title_hbox, False, False, 0)
        
        sub_lbl = Gtk.Label()
        sub_lbl.set_text("Review cropped images, override reasoning, or add custom context/hints per block.")
        sub_lbl.set_margin_bottom(12)
        sub_lbl.set_xalign(0.0)
        sub_lbl.set_margin_start(12)
        content_area.pack_start(sub_lbl, False, False, 0)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_margin_start(12)
        scrolled.set_margin_end(12)
        scrolled.set_margin_bottom(12)
        
        list_grid = Gtk.Grid()
        list_grid.set_column_spacing(18)
        list_grid.set_row_spacing(18)
        list_grid.set_margin_top(6)
        list_grid.set_margin_bottom(6)
        list_grid.set_margin_start(6)
        list_grid.set_margin_end(6)
        
        h_img = Gtk.Label()
        h_img.set_markup("<b>Preview</b>")
        h_img.set_xalign(0.0)
        list_grid.attach(h_img, 0, 0, 1, 1)
        
        h_reason = Gtk.Label()
        h_reason.set_markup("<b>Enable Reasoning</b>")
        h_reason.set_xalign(0.0)
        list_grid.attach(h_reason, 1, 0, 1, 1)
        
        h_hint = Gtk.Label()
        h_hint.set_markup("<b>Additional Context Hint</b>")
        h_hint.set_xalign(0.0)
        list_grid.attach(h_hint, 2, 0, 1, 1)

        rows_widgets = []
        
        is_ds = False
        active_model = consensus_arbiter if (ocr_engine_param == "Ensemble" or ensemble_consensus) else ocr_engine_param
        if active_model and "deepseek" in active_model.lower():
            is_ds = True

        for idx, crop in enumerate(crops):
            row_idx = idx + 1
            
            try:
                pixbuf = numpy_to_pixbuf(crop)
                img_widget = Gtk.Image.new_from_pixbuf(pixbuf)
            except Exception:
                img_widget = Gtk.Label(label="[No Preview]")
            
            img_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
            img_box.pack_start(img_widget, False, False, 0)
            img_box.set_size_request(120, 80)
            list_grid.attach(img_box, 0, row_idx, 1, 1)
            
            chk_row = Gtk.CheckButton()
            chk_row.set_active(enable_thinking)
            chk_row.set_sensitive(is_ds)
            
            chk_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
            chk_box.pack_start(chk_row, False, False, 0)
            list_grid.attach(chk_box, 1, row_idx, 1, 1)
            
            entry_hint = Gtk.Entry()
            entry_hint.set_placeholder_text("e.g. whispering, sound effect, screaming")
            entry_hint.set_width_chars(30)
            entry_hint.set_text(per_path_options[idx]["context_hint"])
            
            btn_analyze = Gtk.Button(label="✨")
            btn_analyze.connect("clicked", on_analyze_clicked, crop, entry_hint)
            individual_buttons.append((btn_analyze, crop, entry_hint))
            
            hint_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            hint_box.pack_start(entry_hint, True, True, 0)
            hint_box.pack_start(btn_analyze, False, False, 0)
            list_grid.attach(hint_box, 2, row_idx, 1, 1)
            
            rows_widgets.append((chk_row, entry_hint))

        scrolled.add(list_grid)
        content_area.pack_start(scrolled, True, True, 0)
        
        desc_dialog.show_all()
        response = desc_dialog.run()
        if response == Gtk.ResponseType.OK:
            img_cache = cache.setdefault(image_key, {})
            for idx, (chk_row, entry_hint) in enumerate(rows_widgets):
                hint_text = entry_hint.get_text().strip()
                per_path_options[idx]["enable_thinking"] = chk_row.get_active()
                per_path_options[idx]["context_hint"] = hint_text
                
                # Save to cache
                box = valid_boxes[idx]
                xmin, ymin, xmax, ymax = box
                box_key = f"{int(xmin)},{int(ymin)},{int(xmax)},{int(ymax)}"
                img_cache[box_key] = hint_text
            
            save_context_cache(cache)
            desc_dialog.destroy()
        else:
            desc_dialog.destroy()
            return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

    # Run inference depending on mode
    if inference_mode == "Local":
        # Run local inference sequentially
        for i, crop in enumerate(crops):
            box = valid_boxes[i]
            sys.stderr.write(f"[Scanlation OCR] Performing local OCR on region {i+1}/{len(crops)}...\n")
            while GLib.MainContext.default().iteration(False):
                pass
                
            try:
                res_list = ocr_engine.extract_text_from_crops([crop])
                raw_text = res_list[0] if res_list else ""
                normalized_text = clean_and_normalize_text(raw_text, half_to_full=half_to_full)
                ocr_results.append(normalized_text)
                sys.stderr.write(f"[Scanlation OCR] Region {i} bounding box {box} -> '{normalized_text}'\n")
            except Exception as ocr_err:
                sys.stderr.write(f"[Scanlation OCR] Local inference error on region {i}: {ocr_err}\n")
                ocr_results.append("")
            
            while GLib.MainContext.default().iteration(False):
                pass
    else:
        # Run remote dispatch in a background thread to prevent UI freezing
        import io
        import base64
        from PIL import Image
        import time
        import threading
        
        # Serialize crops to base64 PNGs
        sys.stderr.write("[Scanlation OCR] Serializing crops to base64 PNGs...\n")
        batch_payload = []
        for idx, crop in enumerate(crops):
            pil_img = Image.fromarray(crop)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            item_options = per_path_options[idx] if idx < len(per_path_options) else {}
            batch_payload.append({
                "image_data": img_str,
                "enable_thinking": item_options.get("enable_thinking", enable_thinking),
                "context_hint": item_options.get("context_hint", "")
            })

        result_container = []
        error_container = []

        def progress_cb(percentage, message):
            def update_ui(pct, msg):
                if msg:
                    Gimp.progress_set_text(msg)
                Gimp.progress_update(pct)
                return False
            GLib.idle_add(update_ui, percentage, message)

        def worker():
            try:
                options = {
                    "target_language": target_lang,
                    "source_language": source_lang,
                    "material_type": material_type,
                    "half_to_full": half_to_full,
                    "consensus_expert_b": consensus_expert_b,
                    "consensus_arbiter": consensus_arbiter,
                    "enable_thinking": enable_thinking
                }
                task_type = "ensemble_ocr" if (ocr_engine_param == "Ensemble" or ensemble_consensus) else "ocr"
                res = remote_client.dispatch_batch(
                    task_type,
                    ocr_engine_param,
                    batch_payload,
                    api_url,
                    options=options,
                    progress_callback=progress_cb
                )
                result_container.append(res)
            except Exception as ex:
                error_container.append(ex)

        sys.stderr.write(f"[Scanlation OCR] Sending {len(crops)} crops to remote dispatcher at {api_url}...\n")
        Gimp.progress_init("Initializing consensus OCR...")
        
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()

        # Pump GTK event loop while waiting for remote completion
        while t.is_alive():
            while GLib.MainContext.default().iteration(False):
                pass
            time.sleep(0.05)

        Gimp.progress_end()

        if error_container:
            sys.stderr.write(f"[Scanlation OCR] Remote dispatch failed: {error_container[0]}\n")
            Gimp.message(f"Remote OCR failed: {error_container[0]}")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        if result_container:
            raw_results = result_container[0]
            for i, raw_text in enumerate(raw_results):
                box = valid_boxes[i]
                normalized_text = clean_and_normalize_text(raw_text, half_to_full=half_to_full)
                ocr_results.append(normalized_text)
                sys.stderr.write(f"[Scanlation OCR] Region {i} bounding box {box} -> '{normalized_text}'\n")

    # Create GIMP text layers for the recognized text blocks
    try:
        group_name = "OCR Transcriptions"
        group_layer = None
        for layer in image.get_layers():
            if layer.get_name() == group_name:
                group_layer = layer
                break

        if not group_layer:
            # Create new Layer Group
            if hasattr(Gimp, "GroupLayer"):
                group_layer = Gimp.GroupLayer.new(image)
                group_layer.set_name(group_name)
                image.insert_layer(group_layer, None, -1)
            else:
                group_layer = None
    except Exception as group_err:
        sys.stderr.write(f"[Scanlation OCR] Failed to resolve/create layer group: {group_err}\n")
        group_layer = None

    # Resolve a valid Gimp.Font object
    font = None
    try:
        if hasattr(Gimp, "context_get_font"):
            ctx_font = Gimp.context_get_font()
            if ctx_font and hasattr(Gimp.Font, "get_by_name"):
                font_name = ctx_font.get_name()
                font = Gimp.Font.get_by_name(f"{font_name} Bold")
                if not font:
                    font = ctx_font
        if not font and hasattr(Gimp, "Font") and hasattr(Gimp.Font, "get_by_name"):
            font = Gimp.Font.get_by_name("Sans-serif Bold")
            if not font:
                font = Gimp.Font.get_by_name("Sans Bold")
            if not font:
                font = Gimp.Font.get_by_name("Sans-serif")
    except Exception as font_err:
        sys.stderr.write(f"[Scanlation OCR] Failed to resolve font: {font_err}\n")

    # Insert text layers for each recognized box
    existing_children = []
    if group_layer:
        try:
            existing_children = Gimp.Item.get_children(group_layer)
            if existing_children is None:
                existing_children = []
            else:
                existing_children = list(existing_children)
        except Exception as e:
            sys.stderr.write(f"[Scanlation OCR] Failed to get children of group layer: {e}\n")

    for i, text in enumerate(ocr_results):
        if not text.strip():
            continue
        box = valid_boxes[i]
        xmin, ymin, xmax, ymax = box
        
        try:
            # Check for and remove duplicate child layer at similar offsets
            for child in list(existing_children):
                success, tx, ty = child.get_offsets()
                if success:
                    if abs(tx - xmin) <= 5 and abs(ty - ymin) <= 5:
                        try:
                            sys.stderr.write(f"[Scanlation OCR] Overwriting old text layer: '{child.get_name()}' at ({tx}, {ty})\n")
                            image.remove_layer(child)
                            existing_children.remove(child)
                        except Exception as rm_err:
                            sys.stderr.write(f"[Scanlation OCR] Failed to remove duplicate child layer: {rm_err}\n")

            if font:
                text_layer = Gimp.TextLayer.new(image, text, font, 32, Gimp.Unit.pixel())
            else:
                text_layer = None
                sys.stderr.write(f"[Scanlation OCR] Cannot create text layer: no font found.\n")

            if text_layer:
                text_layer.set_offsets(int(xmin), int(ymin))
                image.insert_layer(text_layer, group_layer, -1)
        except Exception as layer_err:
            sys.stderr.write(f"[Scanlation OCR] Failed to create text layer for region {i}: {layer_err}\n")

    # Display summary message
    non_empty = [t for t in ocr_results if t.strip()]
    Gimp.message(f"OCR Complete! Processed {len(valid_boxes)} regions, recognized {len(non_empty)} text blocks.\nText layers added to the 'OCR Transcriptions' group.")
    
    return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())
