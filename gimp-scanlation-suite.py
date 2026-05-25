#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
GIMP 3 Scanlation Suite Scaffolding
Part of the Koharu Manga Translation Project

This GIMP 3 Python plugin registers four procedures under the 'Filters > Scanlation' menu:
1. Detect Text & Bubbles: Bounding box and mask detection (YOLO/Segmentation).
2. OCR Selected Blocks: Japanese Text recognition (ViT + BERT / MangaOCR).
3. Inpaint / Erase Text: Cleans dialogue bounds (LaMa/AOT-Inpaint).
4. Translate & Render: Connects to local LLMs (via Koharu API) and typesets translations.

Each procedure pops up a GimpUi / GTK3 configuration dialog.
"""

import sys
import os
import glob

# Dynamically inject local venv site-packages into sys.path
plugin_dir = os.path.dirname(os.path.realpath(__file__))
venv_paths = glob.glob(os.path.join(plugin_dir, "venv", "lib", "python*", "site-packages"))
if venv_paths:
    sys.path.insert(0, venv_paths[0])

import gi
gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('GimpUi', '3.0')
from gi.repository import GimpUi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import GObject
from gi.repository import GLib

try:
    from modules import scouter
except ImportError as e:
    sys.stderr.write(f"[Koharu Suite] Failed to import scouter: {e}\n")
    scouter = None

try:
    from modules import model_manager
except ImportError as e:
    sys.stderr.write(f"[Koharu Suite] Failed to import model_manager: {e}\n")
    model_manager = None

try:
    from modules import ocr_engine
except ImportError as e:
    sys.stderr.write(f"[Koharu Suite] Failed to import ocr_engine: {e}\n")
    ocr_engine = None

import numpy as np


def clean_and_normalize_text(text, half_to_full=True):
    """
    Applies custom normalization rules to clean the recognized OCR text.
    """
    if not text:
        return ""
    
    # 1. Strip whitespace
    text = text.strip()
    
    # 2. Convert half-width ASCII to full-width CJK if enabled
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
        
    # 3. Collapse repetitive dots/punctuation
    import re
    text = re.sub(r'\.{2,}', '...', text)
    text = re.sub(r'…+', '...', text)
    text = re.sub(r'・{2,}', '...', text)
    
    # 4. Replace unicode ellipsis with standard dot notation
    text = text.replace('…', '...')
    
    return text


# We import additional GI modules for future use (e.g. Gegl for buffer manipulations)
gi.require_version('Gegl', '0.4')
from gi.repository import Gegl



class GimpScanlationSuite(Gimp.PlugIn):

    # 1. Register procedures names
    def do_query_procedures(self):
        return [
            "gimp-scanlation-detect",
            "gimp-scanlation-ocr",
            "gimp-scanlation-inpaint",
            "gimp-scanlation-translate"
        ]

    # 2. Create procedure instances and configure arguments
    def do_create_procedure(self, name):
        if name == "gimp-scanlation-detect":
            procedure = Gimp.ImageProcedure.new(
                self, name, Gimp.PDBProcType.PLUGIN, self.run_detect, None
            )
            procedure.set_image_types("RGB*, GRAY*")
            procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
            procedure.set_menu_label("1. Detect Text and Bubbles...")
            procedure.add_menu_path("<Image>/Filters/Scanlation")
            
            procedure.set_documentation(
                "Detect speech bubbles and text regions on the canvas.",
                "Runs Yolov8/Segmentation detection to find bubble and text bounds.",
                name
            )
            procedure.set_attribution("Koharu Contributors", "GPL-3.0", "2026")

            # Arguments
            procedure.add_string_argument(
                "detector-model",
                "_Detector Model",
                "Model to run (e.g. ogkalu/comic-text-and-bubble-detector, PP-DocLayoutV3)",
                "ogkalu/comic-text-and-bubble-detector",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_double_argument(
                "confidence",
                "Confidence _Threshold",
                "Minimum model detection confidence",
                0.0, 1.0, 0.30,
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "class-filter",
                "Class _Filter",
                "Classes to detect: All, Text Only, Bubbles Only",
                "Text Only",
                GObject.ParamFlags.READWRITE
            )
            return procedure

        elif name == "gimp-scanlation-ocr":
            procedure = Gimp.ImageProcedure.new(
                self, name, Gimp.PDBProcType.PLUGIN, self.run_ocr, None
            )
            procedure.set_image_types("RGB*, GRAY*")
            procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
            procedure.set_menu_label("2. OCR Selected Blocks...")
            procedure.add_menu_path("<Image>/Filters/Scanlation")
            
            procedure.set_documentation(
                "Extract Japanese characters from speech bubbles.",
                "Performs VisionEncoderDecoder (ViT + Bert) OCR on text regions.",
                name
            )
            procedure.set_attribution("Koharu Contributors", "GPL-3.0", "2026")

            # Arguments
            procedure.add_string_argument(
                "ocr-engine",
                "_OCR Engine",
                "OCR engine/model to use",
                "PaddleOCR",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_boolean_argument(
                "half-to-full",
                "_Convert Half-width to Full-width",
                "Post-process ASCII characters to full-width CJK alternatives",
                True,
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "inference-mode",
                "_Inference Mode",
                "Where to run inference (Local or Remote)",
                "Local",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "api-url",
                "_API URL",
                "Dispatcher API server URL (e.g. http://localhost:7890)",
                "http://localhost:7890",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "target-language",
                "_Target Language",
                "Target language context for the VLM prompt",
                "Japanese",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "source-language",
                "_Source Language",
                "Language of the source text",
                "Japanese",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "material-type",
                "_Material Type",
                "Type of media (manga, doujinshi, doujinshi_nsfw, comic, light_novel)",
                "manga",
                GObject.ParamFlags.READWRITE
            )
            return procedure

        elif name == "gimp-scanlation-inpaint":
            procedure = Gimp.ImageProcedure.new(
                self, name, Gimp.PDBProcType.PLUGIN, self.run_inpaint, None
            )
            procedure.set_image_types("RGB*, GRAY*")
            procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
            procedure.set_menu_label("3. Inpaint / Erase Text...")
            procedure.add_menu_path("<Image>/Filters/Scanlation")
            
            procedure.set_documentation(
                "Erase original text and fill background using inpainting.",
                "Applies LaMa or AOT-Inpainting model to clear text masks.",
                name
            )
            procedure.set_attribution("Koharu Contributors", "GPL-3.0", "2026")

            # Arguments
            procedure.add_string_argument(
                "inpaint-model",
                "_Inpaint Model",
                "Inpainting model to run (e.g. lama-manga, aot-inpainting)",
                "lama-manga",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_int_argument(
                "dilation",
                "Mask _Dilation (px)",
                "Number of pixels to expand the text mask before inpainting",
                0, 50, 4,
                GObject.ParamFlags.READWRITE
            )
            return procedure

        elif name == "gimp-scanlation-translate":
            procedure = Gimp.ImageProcedure.new(
                self, name, Gimp.PDBProcType.PLUGIN, self.run_translate, None
            )
            procedure.set_image_types("RGB*, GRAY*")
            procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
            procedure.set_menu_label("4. Translate & Render...")
            procedure.add_menu_path("<Image>/Filters/Scanlation")
            
            procedure.set_documentation(
                "Translate dialogues and typeset them onto the canvas.",
                "Queries LLM translation server and renders vertical CJK or horizontal text.",
                name
            )
            procedure.set_attribution("Koharu Contributors", "GPL-3.0", "2026")

            # Arguments
            procedure.add_string_argument(
                "source-lang",
                "_Source Language",
                "Language to translate from",
                "ja",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "target-lang",
                "_Target Language",
                "Language to translate to",
                "en",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "api-url",
                "Koharu _API / LLM URL",
                "Endpoint for local LLM or Koharu translator server",
                "http://localhost:4000",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "font-family",
                "_Font Family",
                "Font family to use for rendering translated text",
                "Sans-serif",
                GObject.ParamFlags.READWRITE
            )
            return procedure

        return None

    # 3. Execution Callbacks
    def run_detect(self, procedure, run_mode, image, drawables, config, run_data):
        """
        Executes Bubble/Text Region Detection.
        
        Runs local ONNX inference on the selected drawable and registers
        the detected bounding boxes as native GIMP Paths (Gimp.Vectors).
        """
        GimpUi.init("gimp-scanlation-detect")

        if run_mode == Gimp.RunMode.INTERACTIVE:
            dialog = GimpUi.ProcedureDialog.new(procedure, config, "Detect Text & Bubbles")
            
            # Premium styled header using GTK3 custom layout
            vbox = dialog.get_content_area()
            
            header_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            header_box.set_margin_top(12)
            header_box.set_margin_bottom(12)
            header_box.set_margin_start(12)
            header_box.set_margin_end(12)
            
            title_label = Gtk.Label()
            title_label.set_markup("<span size='large' weight='bold' foreground='#3584e4'>Koharu Text &amp; Bubble Detector</span>")
            title_label.set_xalign(0.0)
            header_box.pack_start(title_label, False, False, 0)
            
            desc_label = Gtk.Label()
            desc_label.set_text("Identifies regions of Japanese manga text and their corresponding speech bubble shapes. The detected bounds are registered as native GIMP Paths (Vectors) for verification.")
            desc_label.set_line_wrap(True)
            desc_label.set_xalign(0.0)
            header_box.pack_start(desc_label, False, False, 0)
            
            vbox.pack_start(header_box, False, False, 0)
            vbox.show_all()
            
            # Automatically populate properties
            dialog.fill(None)
            
            if not dialog.run():
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

        # Parameters extraction
        detector_model = config.get_property("detector-model")
        confidence = config.get_property("confidence")
        class_filter = config.get_property("class-filter")

        Gimp.message(f"[Koharu Detector] Running '{detector_model}' with threshold={confidence:.2f}...")

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
        models_dir = os.path.join(plugin_dir, "models")
        local_path = os.path.join(models_dir, filename)
        if not os.path.exists(local_path):
            Gimp.message(f"[Koharu Detector] Downloading model '{model_id}/{filename}'... This may take a moment.")
            # Pump events again
            while GLib.MainContext.default().iteration(False):
                pass

        try:
            model_path = model_manager.ensure_model_exists(model_id, filename)
        except Exception as e:
            sys.stderr.write(f"[Koharu Detector] Model acquisition failed: {e}\n")
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
            sys.stderr.write(f"[Koharu Detector] Inference error: {e}\n")
            Gimp.message(f"Inference failed. Check GIMP error logs.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        if not boxes:
            Gimp.message("No text bubbles detected.")
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

        # 3. Create native GIMP path containing the bounding boxes
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
                sys.stderr.write("[Koharu Detector] Failed to add path: API unsupported.\n")
                
            Gimp.message(f"Successfully generated paths for {len(boxes)} text bubbles.")
        except Exception as e:
            sys.stderr.write(f"[Koharu Detector] Path creation failed: {e}\n")
            Gimp.message("Failed to generate path layers.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    def run_ocr(self, procedure, run_mode, image, drawables, config, run_data):
        """
        Executes Japanese Manga OCR (Optical Character Recognition).
        """
        GimpUi.init("gimp-scanlation-ocr")

        if run_mode == Gimp.RunMode.INTERACTIVE:
            dialog = GimpUi.ProcedureDialog.new(procedure, config, "OCR Selected Blocks")
            
            vbox = dialog.get_content_area()
            
            # Premium Header Box
            header_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            header_box.set_margin_top(12)
            header_box.set_margin_bottom(12)
            header_box.set_margin_start(12)
            header_box.set_margin_end(12)
            
            title_label = Gtk.Label()
            title_label.set_markup("<span size='large' weight='bold' foreground='#3584e4'>Koharu Manga OCR Engine</span>")
            title_label.set_xalign(0.0)
            header_box.pack_start(title_label, False, False, 0)
            
            desc_label = Gtk.Label()
            desc_label.set_text("Performs optical character recognition (OCR) on text regions using local or remote VLM inference.")
            desc_label.set_line_wrap(True)
            desc_label.set_xalign(0.0)
            header_box.pack_start(desc_label, False, False, 0)
            vbox.pack_start(header_box, False, False, 0)

            # Custom Grid for dropdown selectors
            grid = Gtk.Grid()
            grid.set_column_spacing(12)
            grid.set_row_spacing(12)
            grid.set_margin_start(12)
            grid.set_margin_end(12)
            grid.set_margin_bottom(12)
            
            # Inference Mode Select
            inf_label = Gtk.Label()
            inf_label.set_markup("<b>Inference Mode:</b>")
            inf_label.set_xalign(0.0)
            grid.attach(inf_label, 0, 0, 1, 1)
            
            combo_inf = Gtk.ComboBoxText()
            combo_inf.append_text("Local")
            combo_inf.append_text("Remote")
            grid.attach(combo_inf, 1, 0, 1, 1)
            
            # OCR Model/Engine Select
            model_label = Gtk.Label()
            model_label.set_markup("<b>OCR Model / Engine:</b>")
            model_label.set_xalign(0.0)
            grid.attach(model_label, 0, 1, 1, 1)
            
            combo_model = Gtk.ComboBoxText()
            grid.attach(combo_model, 1, 1, 1, 1)

            # Source Language Select
            src_lang_label = Gtk.Label()
            src_lang_label.set_markup("<b>Source Language:</b>")
            src_lang_label.set_xalign(0.0)
            grid.attach(src_lang_label, 0, 2, 1, 1)

            combo_src_lang = Gtk.ComboBoxText()
            src_langs = ["Japanese", "English", "Chinese", "Korean"]
            for lang in src_langs:
                combo_src_lang.append_text(lang)
            grid.attach(combo_src_lang, 1, 2, 1, 1)

            # Material Type Select
            material_label = Gtk.Label()
            material_label.set_markup("<b>Material Type:</b>")
            material_label.set_xalign(0.0)
            grid.attach(material_label, 0, 3, 1, 1)

            combo_material = Gtk.ComboBoxText()
            materials = ["manga", "doujinshi", "doujinshi_nsfw", "comic", "light_novel"]
            for mat in materials:
                combo_material.append_text(mat)
            grid.attach(combo_material, 1, 3, 1, 1)
            
            vbox.pack_start(grid, False, False, 0)
            
            # Set initial values based on config
            inf_val = config.get_property("inference-mode") or "Local"
            if inf_val == "Remote":
                combo_inf.set_active(1)
            else:
                combo_inf.set_active(0)

            src_val = config.get_property("source-language") or "Japanese"
            if src_val in src_langs:
                combo_src_lang.set_active(src_langs.index(src_val))
            else:
                combo_src_lang.set_active(0)

            mat_val = config.get_property("material-type") or "manga"
            if mat_val in materials:
                combo_material.set_active(materials.index(mat_val))
            else:
                combo_material.set_active(0)
                
            # Populating dropdown safely in the idle loop
            def populate_dropdown(models):
                combo_model.remove_all()
                for m in models:
                    combo_model.append_text(m)
                
                stored_model = config.get_property("ocr-engine")
                if stored_model in models:
                    combo_model.set_active(models.index(stored_model))
                else:
                    combo_model.set_active(0)
                return False

            import threading
            
            def load_remote_models_bg():
                from modules import remote_client
                api_url = config.get_property("api-url") or "http://localhost:7890"
                models = remote_client.get_available_models("ocr", api_url)
                GLib.idle_add(populate_dropdown, models)

            def update_model_dropdown():
                current_mode = config.get_property("inference-mode")
                if current_mode == "Remote":
                    # Non-blocking remote query in a background thread
                    t = threading.Thread(target=load_remote_models_bg)
                    t.daemon = True
                    t.start()
                else:
                    populate_dropdown(["PaddleOCR"])

            def on_inf_changed(widget):
                val = widget.get_active_text()
                config.set_property("inference-mode", val)
                update_model_dropdown()
                
            combo_inf.connect("changed", on_inf_changed)

            def on_model_changed(widget):
                val = widget.get_active_text()
                if val:
                    config.set_property("ocr-engine", val)
                    
            combo_model.connect("changed", on_model_changed)

            def on_src_lang_changed(widget):
                val = widget.get_active_text()
                if val:
                    config.set_property("source-language", val)

            combo_src_lang.connect("changed", on_src_lang_changed)

            def on_material_changed(widget):
                val = widget.get_active_text()
                if val:
                    config.set_property("material-type", val)

            combo_material.connect("changed", on_material_changed)

            update_model_dropdown()
            vbox.show_all()
            
            # Render remaining free-text and checkbox arguments
            dialog.fill(["api-url", "target-language", "half-to-full"])
            
            if not dialog.run():
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

        # Parameters extraction
        ocr_engine_param = config.get_property("ocr-engine") or "PaddleOCR"
        half_to_full = config.get_property("half-to-full")
        inference_mode = config.get_property("inference-mode") or "Local"
        api_url = config.get_property("api-url") or "http://localhost:7890"
        target_lang = config.get_property("target-language") or "Japanese"
        source_lang = config.get_property("source-language") or "Japanese"
        material_type = config.get_property("material-type") or "manga"

        if inference_mode == "Local" and ocr_engine_param == "Ensemble":
            Gimp.message("Error: Ensemble OCR mode is only supported in Remote mode. Please start the dispatcher server and select Remote mode.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        Gimp.message(f"[Koharu OCR] Running in {inference_mode} mode using '{ocr_engine_param}'...")

        # 1. Verification of active layer and engine imports
        if not drawables:
            Gimp.message("Error: No active drawable/layer selected.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())
            
        active_layer = drawables[0]

        if inference_mode == "Local":
            if ocr_engine is None:
                Gimp.message("Error: OCR engine module could not be imported. Check venv dependencies.")
                return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())
        else:
            # Remote mode requires remote_client and requests
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

        Gimp.message(f"[Koharu OCR] Reading bounding boxes from path: '{target_path.get_name()}'...")

        # 3. Retrieve strokes and parse coordinates
        bounding_boxes = []
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
                    sys.stderr.write(f"[Koharu OCR] Skipping stroke {stroke_id}: no coordinates retrieved.\n")
                    continue

                x_coords = coords[0::2]
                y_coords = coords[1::2]
                if not x_coords or not y_coords:
                    continue
                    
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)
                
                bounding_boxes.append((xmin, ymin, xmax, ymax))
        except Exception as e:
            sys.stderr.write(f"[Koharu OCR] Failed to parse paths/strokes: {e}\n")
            Gimp.message("Failed to extract coordinates from paths.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        if not bounding_boxes:
            Gimp.message("No valid text bounding boxes found in the selected path.")
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

        # Check local model weights presence if in Local Mode
        if inference_mode == "Local":
            models_dir = os.path.join(plugin_dir, "models")
            gguf_path = os.path.join(models_dir, "PaddleOCR-VL-1.5-Q4_K_M.gguf")
            projector_path = os.path.join(models_dir, "PaddleOCR-VL-1.5-mmproj.gguf")
            
            if not os.path.exists(gguf_path) or not os.path.exists(projector_path):
                Gimp.message("[Koharu OCR] OCR model weights not found locally. Downloading PaddleOCR-VL-1.5 GGUF and vision projector (approx. 180MB total). This may take a moment...")
                while GLib.MainContext.default().iteration(False):
                    pass

        # 4. Extract pixel crops
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

            Gimp.message(f"[Koharu OCR] Fetching active layer pixel buffer ({full_w}x{full_h})...")
            raw_data = buffer.get(rect, 1.0, "RGB u8", Gegl.AbyssPolicy.NONE)
            img_np = np.frombuffer(raw_data, dtype=np.uint8).reshape((full_h, full_w, 3))
        except Exception as e:
            sys.stderr.write(f"[Koharu OCR] Failed to read layer pixels: {e}\n")
            Gimp.message("Failed to read active layer pixels.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        # Pump events
        while GLib.MainContext.default().iteration(False):
            pass

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
                sys.stderr.write(f"[Koharu OCR] Box {i} has empty intersection with layer: {box}\n")
                continue

            crops.append(img_np[y0:y1, x0:x1, :])
            valid_boxes.append(box)

        if not crops:
            Gimp.message("No valid cropped text regions to process.")
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

        ocr_results = []

        # 5. Run inference depending on mode
        if inference_mode == "Local":
            # Run local inference sequentially
            for i, crop in enumerate(crops):
                box = valid_boxes[i]
                Gimp.message(f"[Koharu OCR] Performing local OCR on region {i+1}/{len(crops)}...")
                while GLib.MainContext.default().iteration(False):
                    pass
                    
                try:
                    res_list = ocr_engine.extract_text_from_crops([crop])
                    raw_text = res_list[0] if res_list else ""
                    normalized_text = clean_and_normalize_text(raw_text, half_to_full=half_to_full)
                    ocr_results.append(normalized_text)
                    sys.stderr.write(f"[Koharu OCR] Region {i} bounding box {box} -> '{normalized_text}'\n")
                except Exception as ocr_err:
                    sys.stderr.write(f"[Koharu OCR] Local inference error on region {i}: {ocr_err}\n")
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
            
            # Serialize crops to base64 JPEGs
            sys.stderr.write("[Koharu OCR] Serializing crops to base64 JPEGs...\n")
            batch_payload = []
            for crop in crops:
                pil_img = Image.fromarray(crop)
                buffered = io.BytesIO()
                pil_img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                batch_payload.append(img_str)

            result_container = []
            error_container = []

            def worker():
                try:
                    options = {
                        "target_language": target_lang,
                        "source_language": source_lang,
                        "material_type": material_type,
                        "half_to_full": half_to_full
                    }
                    task_type = "ensemble_ocr" if ocr_engine_param == "Ensemble" else "ocr"
                    res = remote_client.dispatch_batch(task_type, ocr_engine_param, batch_payload, api_url, options=options)
                    result_container.append(res)
                except Exception as ex:
                    error_container.append(ex)

            Gimp.message(f"[Koharu OCR] Sending {len(crops)} crops to remote dispatcher at {api_url}...")
            
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()

            # Pump GTK event loop while waiting for remote completion
            while t.is_alive():
                while GLib.MainContext.default().iteration(False):
                    pass
                time.sleep(0.05)

            if error_container:
                sys.stderr.write(f"[Koharu OCR] Remote dispatch failed: {error_container[0]}\n")
                Gimp.message(f"Remote OCR failed: {error_container[0]}")
                return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

            if result_container:
                raw_results = result_container[0]
                for i, raw_text in enumerate(raw_results):
                    box = valid_boxes[i]
                    normalized_text = clean_and_normalize_text(raw_text, half_to_full=half_to_full)
                    ocr_results.append(normalized_text)
                    sys.stderr.write(f"[Koharu OCR] Region {i} bounding box {box} -> '{normalized_text}'\n")

        # Create GIMP text layers for the recognized text blocks
        try:
            # Look for an existing "OCR Transcriptions" layer group and delete it to prevent overlays
            group_name = "OCR Transcriptions"
            for layer in image.get_layers():
                if layer.get_name() == group_name:
                    image.remove_layer(layer)
                    break

            # Create new Layer Group
            if hasattr(Gimp, "GroupLayer"):
                group_layer = Gimp.GroupLayer.new(image)
                group_layer.set_name(group_name)
                image.insert_layer(group_layer, None, -1)
            else:
                group_layer = None
        except Exception as group_err:
            sys.stderr.write(f"[Koharu OCR] Failed to create layer group: {group_err}\n")
            group_layer = None

        # Resolve a valid Gimp.Font object (preferring bold)
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
            sys.stderr.write(f"[Koharu OCR] Failed to resolve font: {font_err}\n")

        # Insert text layers for each recognized box
        for i, text in enumerate(ocr_results):
            if not text.strip():
                continue
            box = valid_boxes[i]
            xmin, ymin, xmax, ymax = box
            
            try:
                if font:
                    # Create GIMP text layer (image, text, Gimp.Font object, size, unit)
                    text_layer = Gimp.TextLayer.new(image, text, font, 32, Gimp.Unit.pixel())
                else:
                    text_layer = None
                    sys.stderr.write(f"[Koharu OCR] Cannot create text layer: no font found.\n")

                if text_layer:
                    # Set position/offsets
                    text_layer.set_offsets(int(xmin), int(ymin))
                    # Add to the group layer (parent) if it was successfully created, otherwise root stack
                    image.insert_layer(text_layer, group_layer, -1)
            except Exception as layer_err:
                sys.stderr.write(f"[Koharu OCR] Failed to create text layer for region {i}: {layer_err}\n")

        # Display summary message
        non_empty = [t for t in ocr_results if t.strip()]
        Gimp.message(f"OCR Complete! Processed {len(valid_boxes)} regions, recognized {len(non_empty)} text blocks.\nText layers added to the 'OCR Transcriptions' group.")
        
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    def run_inpaint(self, procedure, run_mode, image, drawables, config, run_data):
        """
        Executes Inpainting to erase text and fill background.
        
        Rust ONNX Pipeline Equivalents:
        1. Preprocessing:
           - Obtains source image and mask (based on detected bounding boxes).
           - Optional scaling: scales down image/mask if speed is prioritized.
           - Pad height and width to modulo 8 (or 256 depending on model) using symmetric padding
             to prevent border artifacts during convolutional downsampling/upsampling.
        2. ONNX Model Inference (LaMa - Large Mask Inpainting):
           - Executes `lama_manga.onnx` session.
           - Input feeds: `image` (float32, scaled [0, 1]) and `mask` (binary float32).
           - Output is the reconstructed / inpainted image canvas.
        3. Postprocessing:
           - Multiplies output by 255.0, clips, and transposes back to standard HWC layout.
           - Blends the inpainted patches back into GIMP's active layer boundaries.
        """
        GimpUi.init("gimp-scanlation-inpaint")

        if run_mode == Gimp.RunMode.INTERACTIVE:
            dialog = GimpUi.ProcedureDialog.new(procedure, config, "Inpaint / Erase Text")
            
            vbox = dialog.get_content_area()
            
            header_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            header_box.set_margin_top(12)
            header_box.set_margin_bottom(12)
            header_box.set_margin_start(12)
            header_box.set_margin_end(12)
            
            title_label = Gtk.Label()
            title_label.set_markup("<span size='large' weight='bold' foreground='#3584e4'>Koharu Inpainting Tool</span>")
            title_label.set_xalign(0.0)
            header_box.pack_start(title_label, False, False, 0)
            
            desc_label = Gtk.Label()
            desc_label.set_text("Cleans text regions from active layers by applying a Fast Fourier Transform-based LaMa/AOT-Inpainting model to fill the text masks natively.")
            desc_label.set_line_wrap(True)
            desc_label.set_xalign(0.0)
            header_box.pack_start(desc_label, False, False, 0)
            
            vbox.pack_start(header_box, False, False, 0)
            vbox.show_all()
            
            dialog.fill(None)
            
            if not dialog.run():
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

        inpaint_model = config.get_property("inpaint-model")
        dilation = config.get_property("dilation")

        Gimp.message(f"[Koharu Inpaint] Erasing text using '{inpaint_model}' (dilation={dilation}px)...")

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    def run_translate(self, procedure, run_mode, image, drawables, config, run_data):
        """
        Executes Translation & Typesetting.
        
        Rust/LLM Integration:
        1. Context Loading & Layout Mapping:
           - Groups OCR text fragments in reading order (top-right to bottom-left for standard manga).
           - Prepares prompt package including the Japanese source dialogues.
        2. LLM Translation Inference:
           - Sends payload to Koharu LLM Manager (via HTTP `/v1/chat/completions` or MCP client).
           - Reuses local GGUF models (e.g. Qwen/Gemma) loaded into memory via `llama.cpp` runtime.
        3. Typesetting/Rendering:
           - Layout maps translation blocks into the boundary constraints of detected text regions.
           - Employs OpenType script-aware text layout (shaping glyph metrics, applying line-breaks,
             and choosing horizontal or vertical text paths).
           - Modifies/Rasterizes the resulting translation onto new GIMP text layers.
        """
        GimpUi.init("gimp-scanlation-translate")

        if run_mode == Gimp.RunMode.INTERACTIVE:
            dialog = GimpUi.ProcedureDialog.new(procedure, config, "Translate & Render")
            
            vbox = dialog.get_content_area()
            
            header_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            header_box.set_margin_top(12)
            header_box.set_margin_bottom(12)
            header_box.set_margin_start(12)
            header_box.set_margin_end(12)
            
            title_label = Gtk.Label()
            title_label.set_markup("<span size='large' weight='bold' foreground='#3584e4'>Koharu Typesetting &amp; Translation</span>")
            title_label.set_xalign(0.0)
            header_box.pack_start(title_label, False, False, 0)
            
            desc_label = Gtk.Label()
            desc_label.set_text("Translates extracted dialogue nodes using local LLMs/MT engines and overlays formatted typography layers back onto the GIMP image structure.")
            desc_label.set_line_wrap(True)
            desc_label.set_xalign(0.0)
            header_box.pack_start(desc_label, False, False, 0)
            
            vbox.pack_start(header_box, False, False, 0)
            vbox.show_all()
            
            dialog.fill(None)
            
            if not dialog.run():
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

        src_lang = config.get_property("source-lang")
        tgt_lang = config.get_property("target-lang")
        api_url = config.get_property("api-url")
        font_family = config.get_property("font-family")

        Gimp.message(f"[Koharu Translator] Translating {src_lang}->{tgt_lang} via {api_url} (Font: {font_family})...")

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


if __name__ == "__main__":
    Gimp.main(GimpScanlationSuite.__gtype__, sys.argv)
