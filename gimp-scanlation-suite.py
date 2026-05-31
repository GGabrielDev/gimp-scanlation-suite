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
gi.require_version('Babl', '0.1')
from gi.repository import Babl

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
            "gimp-scanlation-translate",
            "gimp-scanlation-typeset"
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
                "ensemble-consensus",
                "_Ensemble Consensus Mode",
                "Use the ensemble mixture of experts pipeline with the selected model as the arbiter",
                False,
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
            procedure.add_string_argument(
                "consensus-expert-b",
                "_Extractor B",
                "Vision-capable OCR model to act as Extractor B in consensus pipeline",
                "PaddleOCR_Manga",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "consensus-arbiter",
                "_Arbiter",
                "Model to act as Arbiter in consensus pipeline",
                "DeepSeek",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_boolean_argument(
                "enable-thinking",
                "_Enable Thinking/Reasoning",
                "Enable reasoning/thinking mode for API models that support it",
                False,
                GObject.ParamFlags.READWRITE
            )
            procedure.add_boolean_argument(
                "configure-per-path",
                "_Configure Options Per Path",
                "Review each text block, toggle reasoning per-path, and add context hints before processing",
                False,
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
            procedure.add_string_argument(
                "inference-mode",
                "_Inference Mode",
                "Execute inpainting locally or offload to remote server",
                "Local",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "api-url",
                "_API URL",
                "Remote dispatcher daemon URL",
                "http://localhost:7890",
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
                "Japanese",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "target-lang",
                "_Target Language",
                "Language to translate to",
                "English",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "api-url",
                "Koharu _API / LLM URL",
                "Endpoint for local LLM or Koharu translator server",
                "http://localhost:7890",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "inference-mode",
                "Inference _Mode",
                "Execute translation locally or offload to remote server",
                "Remote",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "translation-model",
                "Translation _Model",
                "Model to run (e.g. DeepSeek, JP_Arbiter_8B, etc.)",
                "DeepSeek",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_boolean_argument(
                "enable-thinking",
                "Enable _Thinking",
                "Enable reasoning toggle for DeepSeek models",
                False,
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "global-context",
                "_Global Context",
                "Additional global instructions/prompt context for translation",
                "",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "reading-order",
                "Reading _Order Heuristic",
                "Sort dialogues: Japanese (RTL), Western (LTR), Top-to-Bottom, Creation Order",
                "Japanese (RTL)",
                GObject.ParamFlags.READWRITE
            )
            return procedure

        elif name == "gimp-scanlation-typeset":
            procedure = Gimp.ImageProcedure.new(
                self, name, Gimp.PDBProcType.PLUGIN, self.run_typeset, None
            )
            procedure.set_image_types("RGB*, GRAY*")
            procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
            procedure.set_menu_label("5. Typeset / Render Dialogue...")
            procedure.add_menu_path("<Image>/Filters/Scanlation")
            
            procedure.set_documentation(
                "Apply font styling, text-wrapping, and auto-fitting to translated layers.",
                "Formats text layers in the 'Translated Text' group to fit speech bubbles.",
                name
            )
            procedure.set_attribution("Koharu Contributors", "GPL-3.0", "2026")
            
            # Arguments for Typesetting
            procedure.add_string_argument(
                "font-family",
                "_Font Family",
                "Font family to use for rendering",
                "CCYadaYadaYada",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_int_argument(
                "base-font-size",
                "_Base Font Size",
                "Initial font size before auto-scaling",
                6, 72, 18,
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "alignment",
                "_Text Alignment",
                "Justification: Center, Left, Right",
                "Center",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_boolean_argument(
                "auto-fit",
                "_Auto-fit to Bubble",
                "Automatically wrap and scale font size to fit bubble path",
                True,
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
            src_langs = ["Japanese", "English"]
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

            # Extractor B Select
            expert_b_label = Gtk.Label()
            expert_b_label.set_markup("<b>Extractor B:</b>")
            expert_b_label.set_xalign(0.0)
            grid.attach(expert_b_label, 0, 4, 1, 1)

            combo_expert_b = Gtk.ComboBoxText()
            grid.attach(combo_expert_b, 1, 4, 1, 1)

            # Arbiter Select
            arbiter_label = Gtk.Label()
            arbiter_label.set_markup("<b>Arbiter:</b>")
            arbiter_label.set_xalign(0.0)
            grid.attach(arbiter_label, 0, 5, 1, 1)

            combo_arbiter = Gtk.ComboBoxText()
            grid.attach(combo_arbiter, 1, 5, 1, 1)

            # Enable Thinking/Reasoning Checkbox
            chk_thinking = Gtk.CheckButton(label="Enable Thinking/Reasoning")
            grid.attach(chk_thinking, 0, 6, 2, 1)

            # Note label
            note_label = Gtk.Label()
            note_label.set_markup("<span size='small' foreground='#888888'>* Note: manga_ocr is used as Extractor A (first pass) by default.</span>")
            note_label.set_xalign(0.0)
            grid.attach(note_label, 0, 7, 2, 1)
            
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

            # Set thinking checkbox state
            thinking_val = config.get_property("enable-thinking")
            chk_thinking.set_active(thinking_val)
            
            # Fetch the GIMP procedure dialog's auto-generated ensemble-consensus checkbutton
            chk_ensemble = dialog.get_widget("ensemble-consensus", GObject.TYPE_NONE)
            
            # Thinking sensitivity logic
            def update_thinking_sensitivity():
                is_ensemble = chk_ensemble.get_active()
                if is_ensemble:
                    active_model = config.get_property("consensus-arbiter") or ""
                else:
                    active_model = config.get_property("ocr-engine") or ""
                is_ds = "deepseek" in active_model.lower()
                chk_thinking.set_sensitive(is_ds)
                
            # Toggle sensitivity of single-model vs. consensus selectors based on checkbox
            def update_consensus_sensitivity(widget=None):
                is_ensemble = chk_ensemble.get_active()
                combo_model.set_sensitive(not is_ensemble)
                combo_expert_b.set_sensitive(is_ensemble)
                combo_arbiter.set_sensitive(is_ensemble)
                update_thinking_sensitivity()

            chk_ensemble.connect("toggled", update_consensus_sensitivity)
                
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

                combo_expert_b.remove_all()
                for m in models:
                    combo_expert_b.append_text(m)
                stored_expert_b = config.get_property("consensus-expert-b")
                if stored_expert_b in models:
                    combo_expert_b.set_active(models.index(stored_expert_b))
                else:
                    if "PaddleOCR_Manga" in models:
                        combo_expert_b.set_active(models.index("PaddleOCR_Manga"))
                    elif "PaddleOCR" in models:
                        combo_expert_b.set_active(models.index("PaddleOCR"))
                    else:
                        combo_expert_b.set_active(0)

                combo_arbiter.remove_all()
                for m in models:
                    combo_arbiter.append_text(m)
                stored_arbiter = config.get_property("consensus-arbiter")
                if stored_arbiter in models:
                    combo_arbiter.set_active(models.index(stored_arbiter))
                else:
                    if "DeepSeek-V4-Flash" in models:
                        combo_arbiter.set_active(models.index("DeepSeek-V4-Flash"))
                    elif "DeepSeek" in models:
                        combo_arbiter.set_active(models.index("DeepSeek"))
                    elif "JP_Arbiter_8B" in models:
                        combo_arbiter.set_active(models.index("JP_Arbiter_8B"))
                    else:
                        combo_arbiter.set_active(0)
                
                update_thinking_sensitivity()
                update_consensus_sensitivity()
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
                update_thinking_sensitivity()
                    
            combo_model.connect("changed", on_model_changed)

            def on_expert_b_changed(widget):
                val = widget.get_active_text()
                if val:
                    config.set_property("consensus-expert-b", val)

            combo_expert_b.connect("changed", on_expert_b_changed)

            def on_arbiter_changed(widget):
                val = widget.get_active_text()
                if val:
                    config.set_property("consensus-arbiter", val)
                update_thinking_sensitivity()

            combo_arbiter.connect("changed", on_arbiter_changed)

            def on_thinking_toggled(widget):
                config.set_property("enable-thinking", widget.get_active())

            chk_thinking.connect("toggled", on_thinking_toggled)

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
            dialog.fill(["api-url", "target-language", "ensemble-consensus", "configure-per-path", "half-to-full"])
            
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
        ensemble_consensus = config.get_property("ensemble-consensus")
        consensus_expert_b = config.get_property("consensus-expert-b") or "PaddleOCR_Manga"
        consensus_arbiter = config.get_property("consensus-arbiter") or "DeepSeek"
        enable_thinking = config.get_property("enable-thinking")
        configure_per_path = config.get_property("configure-per-path")

        if inference_mode == "Local" and (ocr_engine_param == "Ensemble" or ensemble_consensus):
            Gimp.message("Error: Ensemble OCR mode is only supported in Remote mode. Please start the dispatcher server and select Remote mode.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        sys.stderr.write(f"[Koharu OCR] Running in {inference_mode} mode using '{ocr_engine_param}' (Ensemble consensus={ensemble_consensus})...\n")

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

        sys.stderr.write(f"[Koharu OCR] Reading bounding boxes from path: '{target_path.get_name()}'...\n")

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

            sys.stderr.write(f"[Koharu OCR] Fetching active layer pixel buffer ({full_w}x{full_h})...\n")
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

        # Initialize per-path options
        per_path_options = []
        for _ in range(len(crops)):
            per_path_options.append({
                "enable_thinking": enable_thinking,
                "context_hint": ""
            })

        if inference_mode == "Remote" and configure_per_path and run_mode == Gimp.RunMode.INTERACTIVE:
            from gi.repository import GdkPixbuf
            import io
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

            desc_dialog = Gtk.Dialog(title="Configure Options Per Text Block", parent=None, flags=0)
            desc_dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OK, Gtk.ResponseType.OK)
            desc_dialog.set_default_size(650, 480)

            content_area = desc_dialog.get_content_area()
            
            lbl = Gtk.Label()
            lbl.set_markup("<span size='large' weight='bold' foreground='#3584e4'>Per-Block Configuration</span>")
            lbl.set_margin_top(12)
            lbl.set_margin_bottom(6)
            lbl.set_xalign(0.0)
            lbl.set_margin_start(12)
            content_area.pack_start(lbl, False, False, 0)
            
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
                except Exception as ex_pb:
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
                
                hint_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
                hint_box.pack_start(entry_hint, True, True, 0)
                list_grid.attach(hint_box, 2, row_idx, 1, 1)
                
                rows_widgets.append((chk_row, entry_hint))

            scrolled.add(list_grid)
            content_area.pack_start(scrolled, True, True, 0)
            
            desc_dialog.show_all()
            response = desc_dialog.run()
            if response == Gtk.ResponseType.OK:
                for idx, (chk_row, entry_hint) in enumerate(rows_widgets):
                    per_path_options[idx]["enable_thinking"] = chk_row.get_active()
                    per_path_options[idx]["context_hint"] = entry_hint.get_text().strip()
                desc_dialog.destroy()
            else:
                desc_dialog.destroy()
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

        # 5. Run inference depending on mode
        if inference_mode == "Local":
            # Run local inference sequentially
            for i, crop in enumerate(crops):
                box = valid_boxes[i]
                sys.stderr.write(f"[Koharu OCR] Performing local OCR on region {i+1}/{len(crops)}...\n")
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
            
            # Serialize crops to base64 PNGs
            sys.stderr.write("[Koharu OCR] Serializing crops to base64 PNGs...\n")
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

            sys.stderr.write(f"[Koharu OCR] Sending {len(crops)} crops to remote dispatcher at {api_url}...\n")
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
        
        Saves output non-destructively to a new layer named `[Inpaint] <Original Layer Name>`
        placed directly above the active layer for quick comparison and toggle visibility.
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

            # Inpaint Model Select
            model_label = Gtk.Label()
            model_label.set_markup("<b>Inpaint Model:</b>")
            model_label.set_xalign(0.0)
            grid.attach(model_label, 0, 1, 1, 1)

            combo_model = Gtk.ComboBoxText()
            combo_model.append_text("lama-manga")
            combo_model.append_text("aot-inpainting")
            grid.attach(combo_model, 1, 1, 1, 1)

            vbox.pack_start(grid, False, False, 0)

            # Set initial values based on config
            inf_val = config.get_property("inference-mode") or "Local"
            if inf_val == "Remote":
                combo_inf.set_active(1)
            else:
                combo_inf.set_active(0)

            model_val = config.get_property("inpaint-model") or "lama-manga"
            if model_val == "aot-inpainting":
                combo_model.set_active(1)
            else:
                combo_model.set_active(0)

            def on_inf_changed(widget):
                val = widget.get_active_text()
                config.set_property("inference-mode", val)
            combo_inf.connect("changed", on_inf_changed)

            def on_model_changed(widget):
                val = widget.get_active_text()
                config.set_property("inpaint-model", val)
            combo_model.connect("changed", on_model_changed)

            vbox.show_all()
            
            # Render remaining free-text and slider arguments
            dialog.fill(["dilation", "api-url"])
            
            if not dialog.run():
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

        # Parameters extraction
        inpaint_model = config.get_property("inpaint-model") or "lama-manga"
        dilation = config.get_property("dilation")
        inference_mode = config.get_property("inference-mode") or "Local"
        api_url = config.get_property("api-url") or "http://localhost:7890"

        sys.stderr.write(f"[Koharu Inpaint] Running in {inference_mode} mode using '{inpaint_model}' (dilation={dilation}px)...\n")

        # 1. Verification of active layer
        if not drawables:
            Gimp.message("Error: No active drawable/layer selected.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())
            
        active_layer = drawables[0]

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

        sys.stderr.write(f"[Koharu Inpaint] Reading bounding boxes from path: '{target_path.get_name()}'...\n")

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
                    sys.stderr.write(f"[Koharu Inpaint] Skipping stroke {stroke_id}: no coordinates retrieved.\n")
                    continue

                x_coords = coords[0::2]
                y_coords = coords[1::2]
                if not x_coords or not y_coords:
                    continue
                    
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)
                
                bounding_boxes.append((xmin, ymin, xmax, ymax))
        except Exception as e:
            sys.stderr.write(f"[Koharu Inpaint] Failed to parse paths/strokes: {e}\n")
            Gimp.message("Failed to extract coordinates from paths.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        if not bounding_boxes:
            Gimp.message("No valid bounding boxes found in the selected path.")
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

        # Check local model weights presence if in Local Mode
        if inference_mode == "Local":
            if inpaint_model == "lama-manga":
                repo = "mayocream/lama-manga-onnx"
                filename = "lama-manga.onnx"
                local_filename = "lama-manga.onnx"
            elif inpaint_model == "aot-inpainting":
                repo = "ogkalu/aot-inpainting"
                filename = "aot.onnx"
                local_filename = "aot-inpainting.onnx"
            else:
                Gimp.message(f"Error: Unknown local model option '{inpaint_model}'")
                return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

            models_dir = os.path.join(plugin_dir, "models")
            model_file_path = os.path.join(models_dir, local_filename)
            if not os.path.exists(model_file_path):
                Gimp.message(f"[Koharu Inpaint] Local model '{inpaint_model}' weights not found. Downloading (approx. 100MB-200MB). This may take a moment...")
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

            sys.stderr.write(f"[Koharu Inpaint] Fetching active layer pixel buffer ({full_w}x{full_h})...\n")
            raw_data = buffer.get(rect, 1.0, "RGB u8", Gegl.AbyssPolicy.NONE)
            img_np = np.frombuffer(raw_data, dtype=np.uint8).reshape((full_h, full_w, 3))
        except Exception as e:
            sys.stderr.write(f"[Koharu Inpaint] Failed to read layer pixels: {e}\n")
            Gimp.message("Failed to read active layer pixels.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        # Construct full-canvas mask
        mask_np = np.zeros((full_h, full_w), dtype=np.uint8)
        for box in bounding_boxes:
            xmin, ymin, xmax, ymax = box
            x0 = int(np.clip(xmin - offset_x, 0, full_w))
            x1 = int(np.clip(xmax - offset_x, 0, full_w))
            y0 = int(np.clip(ymin - offset_y, 0, full_h))
            y1 = int(np.clip(ymax - offset_y, 0, full_h))
            
            if x1 > x0 and y1 > y0:
                mask_np[y0:y1, x0:x1] = 255

        # Dilation
        if dilation > 0:
            try:
                from PIL import Image, ImageFilter
                mask_pil = Image.fromarray(mask_np)
                mask_pil = mask_pil.filter(ImageFilter.MaxFilter(size=2 * dilation + 1))
                mask_np = np.array(mask_pil)
            except Exception as dil_err:
                sys.stderr.write(f"[Koharu Inpaint] Mask dilation failed: {dil_err}\n")

        # 5. Run inference with background thread and event loop pumping
        import threading
        import time
        import io
        import base64
        from PIL import Image

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
                        "bounding_boxes": bounding_boxes
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

        sys.stderr.write(f"[Koharu Inpaint] Dispatching inpainting thread in background...\n")
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
            sys.stderr.write(f"[Koharu Inpaint] Inference error: {err_msg}\n")
            Gimp.message(f"Inpainting failed: {err_msg}")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        final_img = result_container[0]

        # 6. Save result as non-destructive layer above active_layer
        try:
            # Duplicate the active layer (preserves transparency/alpha settings/size)
            copy_layer = active_layer.copy()
            copy_layer.set_name(f"[Inpaint] {active_layer.get_name()}")
        except Exception as copy_err:
            sys.stderr.write(f"[Koharu Inpaint] Failed to duplicate layer: {copy_err}\n")
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
            sys.stderr.write(f"[Koharu Inpaint] Failed to insert layer: {insert_err}\n")
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
            sys.stderr.write(f"[Koharu Inpaint] Failed to write inpainted buffer: {write_err}\n")
            Gimp.message("Failed to write inpainted pixels to the copied layer.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        # Activate the new layer
        try:
            image.set_selected_drawables([copy_layer])
        except Exception as sel_err:
            sys.stderr.write(f"[Koharu Inpaint] Failed to set active layer: {sel_err}\n")

        # Flush display
        try:
            Gimp.displays_flush()
        except Exception:
            pass

        Gimp.message(f"Inpainting complete! Created non-destructive layer '{copy_layer.get_name()}' above '{active_layer.get_name()}'.")
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    def run_translate(self, procedure, run_mode, image, drawables, config, run_data):
        """
        Translates dialogue blocks contextually using local or remote LLMs,
        handling reading order sorting, character tracking, and auto-fit rendering.
        """
        GimpUi.init("gimp-scanlation-translate")

        # 1. Verification of active layer
        if not drawables:
            Gimp.message("Error: No active drawable/layer selected.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())
            
        active_layer = drawables[0]

        # Retrieve active layer offsets and dimensions
        try:
            # Find the original background/manga layer for previews
            preview_layer = active_layer
            for layer in reversed(image.get_layers()):
                # Skip group layers
                if hasattr(layer, "get_children") and Gimp.Item.get_children(layer) is not None:
                    continue
                # Skip text layers
                if hasattr(Gimp, "TextLayer") and isinstance(layer, Gimp.TextLayer):
                    continue
                name = layer.get_name()
                if name.startswith("[Inpaint]") or name in ["OCR Transcriptions", "Translated Text", "Detected Bubbles"]:
                    continue
                preview_layer = layer
                break

            sys.stderr.write(f"[Koharu Translator] Using layer '{preview_layer.get_name()}' for preview cropping.\n")

            buffer = preview_layer.get_buffer()
            rect = buffer.get_extent()
            full_w = rect.width
            full_h = rect.height
            success, offset_x, offset_y = preview_layer.get_offsets()
            if not success:
                offset_x, offset_y = 0, 0
                
            raw_data = buffer.get(rect, 1.0, "RGB u8", Gegl.AbyssPolicy.NONE)
            img_np = np.frombuffer(raw_data, dtype=np.uint8).reshape((full_h, full_w, 3))
        except Exception as e:
            sys.stderr.write(f"[Koharu Translator] Failed to read preview layer pixels: {e}\n")
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
            except Exception as active_err:
                sys.stderr.write(f"[Koharu Translator] Failed to read active layer fallback pixels: {active_err}\n")
                Gimp.message("Failed to read active layer pixels.")
                return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

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

        sys.stderr.write(f"[Koharu Translator] Reading bounding boxes from path: '{target_path.get_name()}'...\n")

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
                    continue

                x_coords = coords[0::2]
                y_coords = coords[1::2]
                if not x_coords or not y_coords:
                    continue
                    
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)
                
                bounding_boxes.append((xmin, ymin, xmax, ymax))
        except Exception as e:
            sys.stderr.write(f"[Koharu Translator] Failed to parse paths/strokes: {e}\n")
            Gimp.message("Failed to extract coordinates from paths.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        if not bounding_boxes:
            Gimp.message("No valid bounding boxes found in the selected path.")
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

        # 4. Locate the "OCR Transcriptions" group layer and extract text layers
        ocr_group = None
        for layer in image.get_layers():
            if layer.get_name() == "OCR Transcriptions":
                ocr_group = layer
                break

        ocr_texts = []
        if ocr_group:
            try:
                children = Gimp.Item.get_children(ocr_group)
                for child in children:
                    if hasattr(child, "get_text"):
                        success, tx, ty = child.get_offsets()
                        text_val = child.get_text() or ""
                        if text_val.strip():
                            ocr_texts.append((tx, ty, text_val))
            except Exception as ocr_read_err:
                sys.stderr.write(f"[Koharu Translator] Failed to read OCR layers: {ocr_read_err}\n")

        # Map OCR texts back to bounding boxes spatially (overlap with 20px tolerance)
        box_ocr_texts = []
        for box in bounding_boxes:
            xmin, ymin, xmax, ymax = box
            matched = []
            for tx, ty, text in ocr_texts:
                if (xmin - 20 <= tx <= xmax + 20) and (ymin - 20 <= ty <= ymax + 20):
                    matched.append((ty, tx, text))
            if matched:
                matched.sort() # Top-to-bottom spatial order within bubble
                combined_text = "\n".join([m[2] for m in matched])
                box_ocr_texts.append(combined_text)
            else:
                box_ocr_texts.append("")

        # Initialize bubble states in original stroke/path detection order
        bubble_states = []
        for idx, box in enumerate(bounding_boxes):
            bubble_states.append({
                "original_index": idx,
                "box": box,
                "text": box_ocr_texts[idx],
                "speaker": "Unassigned / Narrative",
                "context": "",
                "skip": False
            })

        def sort_bubble_states(heuristic):
            if heuristic == "Japanese (RTL)":
                def get_sort_key(state):
                    xmin, ymin, xmax, ymax = state["box"]
                    cx = (xmin + xmax) / 2.0
                    cy = (ymin + ymax) / 2.0
                    col = int((full_w - cx) / max(1.0, full_w / 3.0))
                    return (col, cy)
                bubble_states.sort(key=get_sort_key)
            elif heuristic == "Western (LTR)":
                def get_sort_key(state):
                    xmin, ymin, xmax, ymax = state["box"]
                    cx = (xmin + xmax) / 2.0
                    cy = (ymin + ymax) / 2.0
                    col = int(cx / max(1.0, full_w / 3.0))
                    return (col, cy)
                bubble_states.sort(key=get_sort_key)
            elif heuristic == "Top-to-Bottom":
                def get_sort_key(state):
                    xmin, ymin, xmax, ymax = state["box"]
                    cx = (xmin + xmax) / 2.0
                    cy = (ymin + ymax) / 2.0
                    return (cy, cx)
                bubble_states.sort(key=get_sort_key)
            elif heuristic == "Creation Order":
                bubble_states.sort(key=lambda s: s["original_index"])

        initial_reading_order = config.get_property("reading-order") or "Japanese (RTL)"
        sort_bubble_states(initial_reading_order)


        # 5. Build Gtk Dialog interface (tabbed Gtk.Notebook)
        if run_mode == Gimp.RunMode.INTERACTIVE:
            dialog = Gtk.Dialog(title="Koharu Translation & Typesetting", parent=None, flags=0)
            dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OK, Gtk.ResponseType.OK)
            dialog.set_default_size(750, 580)
            
            content_area = dialog.get_content_area()
            
            # Premium Header Box
            header_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            header_box.set_margin_top(12)
            header_box.set_margin_bottom(6)
            header_box.set_margin_start(12)
            header_box.set_margin_end(12)
            
            title_label = Gtk.Label()
            title_label.set_markup("<span size='large' weight='bold' foreground='#3584e4'>Koharu Typesetting &amp; Translation</span>")
            title_label.set_xalign(0.0)
            header_box.pack_start(title_label, False, False, 0)
            
            desc_label = Gtk.Label()
            desc_label.set_text("Review and configure translation parameters, map characters, and verify text boxes before typesetting.")
            desc_label.set_xalign(0.0)
            header_box.pack_start(desc_label, False, False, 0)
            content_area.pack_start(header_box, False, False, 0)
            
            notebook = Gtk.Notebook()
            content_area.pack_start(notebook, True, True, 0)
            
            # --- TAB 1: Translation Settings ---
            grid_settings = Gtk.Grid()
            grid_settings.set_column_spacing(12)
            grid_settings.set_row_spacing(12)
            grid_settings.set_margin_top(12)
            grid_settings.set_margin_bottom(12)
            grid_settings.set_margin_start(12)
            grid_settings.set_margin_end(12)
            
            # Source Language
            lbl_src = Gtk.Label(label="Source Language:")
            lbl_src.set_xalign(0.0)
            grid_settings.attach(lbl_src, 0, 0, 1, 1)
            combo_src = Gtk.ComboBoxText()
            for lang in ["Japanese", "English"]:
                combo_src.append_text(lang)
            combo_src.set_active(0)
            grid_settings.attach(combo_src, 1, 0, 1, 1)
            
            # Target Language
            lbl_tgt = Gtk.Label(label="Target Language:")
            lbl_tgt.set_xalign(0.0)
            grid_settings.attach(lbl_tgt, 0, 1, 1, 1)
            combo_tgt = Gtk.ComboBoxText()
            for lang in ["English", "Spanish", "French", "German", "Japanese"]:
                combo_tgt.append_text(lang)
            combo_tgt.set_active(0)
            grid_settings.attach(combo_tgt, 1, 1, 1, 1)
            
            # Inference Mode
            lbl_inf = Gtk.Label(label="Inference Mode:")
            lbl_inf.set_xalign(0.0)
            grid_settings.attach(lbl_inf, 0, 2, 1, 1)
            combo_inf = Gtk.ComboBoxText()
            combo_inf.append_text("Remote")
            combo_inf.append_text("Local")
            combo_inf.set_active(0)
            grid_settings.attach(combo_inf, 1, 2, 1, 1)
            
            # API URL
            lbl_api = Gtk.Label(label="API / LLM URL:")
            lbl_api.set_xalign(0.0)
            grid_settings.attach(lbl_api, 0, 3, 1, 1)
            entry_api = Gtk.Entry()
            entry_api.set_text(config.get_property("api-url") or "http://localhost:7890")
            grid_settings.attach(entry_api, 1, 3, 1, 1)
            
            # Model
            lbl_model = Gtk.Label(label="Translation Model:")
            lbl_model.set_xalign(0.0)
            grid_settings.attach(lbl_model, 0, 4, 1, 1)
            combo_model = Gtk.ComboBoxText()
            grid_settings.attach(combo_model, 1, 4, 1, 1)
            
            # Reasoning Toggle
            chk_thinking = Gtk.CheckButton(label="Enable Thinking / Reasoning (DeepSeek)")
            chk_thinking.set_active(config.get_property("enable-thinking"))
            grid_settings.attach(chk_thinking, 0, 5, 2, 1)
            
            # Reading Order Heuristic
            lbl_ro = Gtk.Label(label="Reading Order Heuristic:")
            lbl_ro.set_xalign(0.0)
            grid_settings.attach(lbl_ro, 0, 6, 1, 1)
            
            combo_ro = Gtk.ComboBoxText()
            ro_options = ["Japanese (RTL)", "Western (LTR)", "Top-to-Bottom", "Creation Order"]
            for ro in ro_options:
                combo_ro.append_text(ro)
            stored_ro = config.get_property("reading-order") or "Japanese (RTL)"
            if stored_ro in ro_options:
                combo_ro.set_active(ro_options.index(stored_ro))
            else:
                combo_ro.set_active(0)
            grid_settings.attach(combo_ro, 1, 6, 1, 1)
            
            # Global Context Label
            lbl_global = Gtk.Label(label="Global Scene Context / Style Prompts:")
            lbl_global.set_xalign(0.0)
            grid_settings.attach(lbl_global, 0, 7, 2, 1)
            
            # Global Context TextView
            scroll_global = Gtk.ScrolledWindow()
            scroll_global.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
            scroll_global.set_size_request(-1, 80)
            txt_global = Gtk.TextView()
            txt_global.set_wrap_mode(Gtk.WrapMode.WORD)
            buf_global = txt_global.get_buffer()
            buf_global.set_text(config.get_property("global-context") or "")
            scroll_global.add(txt_global)
            grid_settings.attach(scroll_global, 0, 8, 2, 1)
            
            notebook.append_page(grid_settings, Gtk.Label(label="Settings"))
            
            # --- TAB 3: Characters ---
            grid_chars = Gtk.Grid()
            grid_chars.set_column_spacing(12)
            grid_chars.set_row_spacing(12)
            grid_chars.set_margin_top(12)
            grid_chars.set_margin_bottom(12)
            grid_chars.set_margin_start(12)
            grid_chars.set_margin_end(12)
            
            lbl_chars_desc = Gtk.Label()
            lbl_chars_desc.set_markup("<b>Define Characters in this scene (dynamically populated in dropdowns):</b>")
            lbl_chars_desc.set_xalign(0.0)
            grid_chars.attach(lbl_chars_desc, 0, 0, 2, 1)
            
            char_entries = []
            for c_idx in range(5):
                lbl_c = Gtk.Label(label=f"Character {c_idx+1}:")
                lbl_c.set_xalign(0.0)
                grid_chars.attach(lbl_c, 0, c_idx + 1, 1, 1)
                
                ent_c = Gtk.Entry()
                ent_c.set_text(f"Character {c_idx+1}")
                grid_chars.attach(ent_c, 1, c_idx + 1, 1, 1)
                char_entries.append(ent_c)
                
            notebook.append_page(grid_chars, Gtk.Label(label="Characters"))
            
            # --- TAB 2: Dialogue Blocks ---
            scroll_blocks = Gtk.ScrolledWindow()
            scroll_blocks.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
            
            box_blocks = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
            box_blocks.set_margin_top(12)
            box_blocks.set_margin_bottom(12)
            box_blocks.set_margin_start(12)
            box_blocks.set_margin_end(12)
            
            row_widgets = []
            
            from gi.repository import GdkPixbuf
            
            def get_crop_pixbuf(np_crop, max_w=100, max_h=70):
                try:
                    from PIL import Image
                    import io
                    pil_img = Image.fromarray(np_crop)
                    pil_img.thumbnail((max_w, max_h))
                    buffered = io.BytesIO()
                    pil_img.save(buffered, format="PNG")
                    loader = GdkPixbuf.PixbufLoader.new_with_type("png")
                    loader.write(buffered.getvalue())
                    loader.close()
                    return loader.get_pixbuf()
                except Exception:
                    return None
            
            crops = []
            for box in bounding_boxes:
                xmin, ymin, xmax, ymax = box
                x0 = int(np.clip(xmin - offset_x, 0, full_w))
                x1 = int(np.clip(xmax - offset_x, 0, full_w))
                y0 = int(np.clip(ymin - offset_y, 0, full_h))
                y1 = int(np.clip(ymax - offset_y, 0, full_h))
                if x1 > x0 and y1 > y0:
                    crops.append(img_np[y0:y1, x0:x1, :])
                else:
                    crops.append(None)

            def save_current_edits():
                for row_idx, (combo_spk, txt_src, entry_hint, chk_exclude, _) in enumerate(row_widgets):
                    state = bubble_states[row_idx]
                    state["speaker"] = combo_spk.get_active_text() or "Unassigned / Narrative"
                    buf = txt_src.get_buffer()
                    state["text"] = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), True)
                    state["context"] = entry_hint.get_text()
                    state["skip"] = chk_exclude.get_active()

            def rebuild_dialogue_queue():
                for child in box_blocks.get_children():
                    child.destroy()
                del row_widgets[:]
                
                for idx, state in enumerate(bubble_states):
                    box = state["box"]
                    frame = Gtk.Frame()
                    frame.set_label(f"Bubble #{idx+1} (Reading Order)")
                    
                    grid_row = Gtk.Grid()
                    grid_row.set_column_spacing(10)
                    grid_row.set_row_spacing(6)
                    grid_row.set_margin_top(8)
                    grid_row.set_margin_bottom(8)
                    grid_row.set_margin_start(8)
                    grid_row.set_margin_end(8)
                    
                    # Preview Image
                    np_crop = crops[state["original_index"]]
                    if np_crop is not None:
                        pb = get_crop_pixbuf(np_crop)
                        if pb:
                            img_widget = Gtk.Image.new_from_pixbuf(pb)
                        else:
                            img_widget = Gtk.Label(label="[No Preview]")
                    else:
                        img_widget = Gtk.Label(label="[No Preview]")
                    
                    img_widget.set_size_request(100, 70)
                    grid_row.attach(img_widget, 0, 0, 1, 2)
                    
                    # Exclude checkbox
                    chk_exclude = Gtk.CheckButton(label="Skip translation")
                    chk_exclude.set_active(state["skip"])
                    grid_row.attach(chk_exclude, 1, 0, 1, 1)
                    
                    # Context Hint Entry
                    entry_hint = Gtk.Entry()
                    entry_hint.set_placeholder_text("e.g. whispering, angry")
                    entry_hint.set_width_chars(20)
                    entry_hint.set_text(state["context"])
                    
                    # Speaker Dropdown
                    combo_spk = Gtk.ComboBoxText()
                    
                    def refresh_speakers(spk_combo=combo_spk):
                        active_text = spk_combo.get_active_text()
                        spk_combo.remove_all()
                        spk_combo.append_text("Unassigned / Narrative")
                        spk_combo.append_text("SFX / Onomatopoeia")
                        for ent in char_entries:
                            name = ent.get_text().strip()
                            if name:
                                spk_combo.append_text(name)
                        
                        found = False
                        for i in range(spk_combo.get_model().iter_n_children(None)):
                            spk_combo.set_active(i)
                            if spk_combo.get_active_text() == active_text:
                                found = True
                                break
                        if not found:
                            spk_combo.set_active(0)
                    
                    refresh_speakers()
                    
                    target_spk = state["speaker"]
                    model = combo_spk.get_model()
                    found = False
                    for i in range(model.iter_n_children(None)):
                        combo_spk.set_active(i)
                        if combo_spk.get_active_text() == target_spk:
                            found = True
                            break
                    if not found:
                        combo_spk.set_active(0)
                        
                    grid_row.attach(combo_spk, 2, 0, 1, 1)
                    grid_row.attach(entry_hint, 3, 0, 1, 1)
                    
                    # Source Text (TextView inside ScrolledWindow for multi-line editing)
                    scroll_src = Gtk.ScrolledWindow()
                    scroll_src.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
                    scroll_src.set_size_request(300, 45)
                    txt_src = Gtk.TextView()
                    txt_src.set_wrap_mode(Gtk.WrapMode.WORD)
                    buf_src = txt_src.get_buffer()
                    buf_src.set_text(state["text"])
                    scroll_src.add(txt_src)
                    
                    grid_row.attach(scroll_src, 1, 1, 3, 1)
                    
                    frame.add(grid_row)
                    box_blocks.pack_start(frame, False, False, 0)
                    
                    row_widgets.append((combo_spk, txt_src, entry_hint, chk_exclude, refresh_speakers))

            rebuild_dialogue_queue()

            def on_reading_order_changed(widget):
                save_current_edits()
                new_ro = combo_ro.get_active_text()
                sort_bubble_states(new_ro)
                rebuild_dialogue_queue()
                box_blocks.show_all()
            
            combo_ro.connect("changed", on_reading_order_changed)

            def on_char_name_changed(widget):
                for row in row_widgets:
                    row[4]()
            
            for ent in char_entries:
                ent.connect("changed", on_char_name_changed)
                
            scroll_blocks.add(box_blocks)
            notebook.append_page(scroll_blocks, Gtk.Label(label="Dialogue Queue"))
            
            # Dynamic Model Populating
            def populate_dropdown(models):
                combo_model.remove_all()
                filtered = [m for m in models if any(w in m.lower() for w in ["deepseek", "arbiter", "llama", "qwen", "gemma", "mistral", "chat", "jp"])]
                if not filtered:
                    filtered = ["DeepSeek", "JP_Arbiter_8B"]
                for m in filtered:
                    combo_model.append_text(m)
                
                stored_model = config.get_property("translation-model") or "DeepSeek"
                if stored_model in filtered:
                    combo_model.set_active(filtered.index(stored_model))
                else:
                    combo_model.set_active(0)
                
                update_thinking_sensitivity()
            
            def update_thinking_sensitivity():
                active_model = combo_model.get_active_text() or ""
                is_ds = "deepseek" in active_model.lower()
                chk_thinking.set_sensitive(is_ds)
            
            combo_model.connect("changed", lambda widget: update_thinking_sensitivity())
            
            import threading
            
            def load_remote_models_bg():
                from modules import remote_client
                api_url = entry_api.get_text().strip() or "http://localhost:7890"
                models = remote_client.get_available_models("translate", api_url)
                GLib.idle_add(populate_dropdown, models)

            def update_model_dropdown():
                current_mode = combo_inf.get_active_text()
                if current_mode == "Remote":
                    t = threading.Thread(target=load_remote_models_bg)
                    t.daemon = True
                    t.start()
                else:
                    populate_dropdown(["JP_Arbiter_8B"])
            
            combo_inf.connect("changed", lambda widget: update_model_dropdown())
            update_model_dropdown()
            
            dialog.show_all()
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                # Save settings to config
                src_lang = combo_src.get_active_text()
                tgt_lang = combo_tgt.get_active_text()
                inf_mode = combo_inf.get_active_text()
                api_url = entry_api.get_text().strip()
                trans_model = combo_model.get_active_text()
                enable_thinking = chk_thinking.get_active()
                reading_order = combo_ro.get_active_text()
                
                buf_global_ctx = txt_global.get_buffer()
                global_ctx = buf_global_ctx.get_text(buf_global_ctx.get_start_iter(), buf_global_ctx.get_end_iter(), True).strip()
                
                config.set_property("source-lang", src_lang)
                config.set_property("target-lang", tgt_lang)
                config.set_property("api-url", api_url)
                config.set_property("inference-mode", inf_mode)
                config.set_property("translation-model", trans_model)
                config.set_property("enable-thinking", enable_thinking)
                config.set_property("global-context", global_ctx)
                config.set_property("reading-order", reading_order)
                
                # Save current edits from widgets to bubble_states
                save_current_edits()
                
                # Extract dialog queue rows
                payload = []
                included_box_indices = []
                for idx, state in enumerate(bubble_states):
                    if state["skip"]:
                        continue
                    
                    src_text = state["text"].strip()
                    if not src_text:
                        continue
                        
                    speaker = state["speaker"]
                    if speaker == "Unassigned / Narrative":
                        speaker = ""
                    elif speaker == "SFX / Onomatopoeia":
                        speaker = "SFX"
                        
                    context = state["context"].strip()
                    
                    payload.append({
                        "index": idx + 1,
                        "text": src_text,
                        "speaker": speaker,
                        "context": context
                    })
                    included_box_indices.append(state["original_index"])
                
                dialog.destroy()
            else:
                dialog.destroy()
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())
        else:
            # Non-interactive fallback values
            src_lang = config.get_property("source-lang") or "Japanese"
            tgt_lang = config.get_property("target-lang") or "English"
            api_url = config.get_property("api-url") or "http://localhost:7890"
            inf_mode = config.get_property("inference-mode") or "Remote"
            trans_model = config.get_property("translation-model") or "DeepSeek"
            enable_thinking = config.get_property("enable-thinking")
            global_ctx = config.get_property("global-context") or ""
            reading_order = config.get_property("reading-order") or "Japanese (RTL)"
            
            # Sort states
            sort_bubble_states(reading_order)
            
            payload = []
            included_box_indices = []
            for state in bubble_states:
                src_text = state["text"].strip()
                if src_text:
                    payload.append({
                        "index": state["original_index"] + 1,
                        "text": src_text,
                        "speaker": "",
                        "context": ""
                    })
                    included_box_indices.append(state["original_index"])

        if not payload:
            Gimp.message("No dialogue blocks selected for translation.")
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

        # 6. Execute Translation Request
        translated_results = []
        try:
            Gimp.progress_init(f"Translating dialogue via {trans_model}...")
            
            options = {
                "source_language": src_lang,
                "target_language": tgt_lang,
                "global_context": global_ctx,
                "enable_thinking": enable_thinking
            }
            
            from modules import remote_client
            res_list = remote_client.dispatch_batch(
                "translate",
                trans_model,
                payload,
                api_url,
                options=options,
                progress_callback=lambda pct, msg: Gimp.progress_update(pct)
            )
            
            Gimp.progress_end()
            
            if not res_list:
                raise RuntimeError("No translations returned from the server.")
                
            translated_results = res_list
        except Exception as trans_err:
            sys.stderr.write(f"[Koharu Translator] Translation failed: {trans_err}\n")
            Gimp.message(f"Translation failed: {trans_err}")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        # Create new "Translated Text" group layer
        try:
            group_name = "Translated Text"
            for layer in image.get_layers():
                if layer.get_name() == group_name:
                    image.remove_layer(layer)
                    break

            if hasattr(Gimp, "GroupLayer"):
                group_layer = Gimp.GroupLayer.new(image)
                group_layer.set_name(group_name)
                image.insert_layer(group_layer, None, -1)
            else:
                group_layer = None
        except Exception as group_err:
            sys.stderr.write(f"[Koharu Translator] Failed to create layer group: {group_err}\n")
            group_layer = None

        default_font = None
        try:
            if hasattr(Gimp, "Font") and hasattr(Gimp.Font, "get_by_name"):
                default_font = Gimp.Font.get_by_name("Sans-serif")
        except Exception:
            pass

        # Render translations onto new text layers in GIMP (raw, unstyled text layers centered in bubbles)
        for i, translation in enumerate(translated_results):
            if not translation.strip():
                continue
            box_idx = included_box_indices[i]
            box = bounding_boxes[box_idx]
            xmin, ymin, xmax, ymax = box
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            
            try:
                text_layer = Gimp.TextLayer.new(image, translation, default_font, 14, Gimp.Unit.pixel())
                if text_layer:
                    text_layer.set_justification(Gimp.TextJustification.CENTER)
                    image.insert_layer(text_layer, group_layer, -1)
                    
                    rect_t = text_layer.get_buffer().get_extent()
                    tw = rect_t.width
                    th = rect_t.height
                    
                    tx = cx - tw // 2
                    ty = cy - th // 2
                    
                    text_layer.set_offsets(int(tx), int(ty))
            except Exception as render_err:
                sys.stderr.write(f"[Koharu Translator] Failed to render bubble {box_idx+1}: {render_err}\n")

        # Display completed GIMP message
        Gimp.message(f"Translation complete! Saved {len(translated_results)} translated bubbles in 'Translated Text' layer group. Please run '5. Typeset / Render Dialogue...' to format them.")
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    def run_typeset(self, procedure, run_mode, image, drawables, config, run_data):
        """
        Formats and typesets translated dialogue layers with per-bubble editing,
        searchable font picker, manga presets, and live canvas preview.
        """
        import textwrap

        GimpUi.init("gimp-scanlation-typeset")

        # ── 1. Locate the 'Translated Text' layer group at root ──
        translated_group = None
        for layer in image.get_layers():
            if layer.get_name() == "Translated Text":
                translated_group = layer
                break

        if not translated_group:
            Gimp.message("Error: 'Translated Text' layer group not found.\nPlease run translation first.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        # Extract text layers from translated_group
        translated_texts = []
        try:
            children = Gimp.Item.get_children(translated_group)
            for child in children:
                if hasattr(child, "get_text"):
                    success, tx, ty = child.get_offsets()
                    text_val = child.get_text() or ""
                    if text_val.strip():
                        translated_texts.append((child, tx, ty, text_val))
        except Exception as read_err:
            sys.stderr.write(f"[Koharu Typesetter] Failed to read layers: {read_err}\n")
            Gimp.message("Failed to read translated text layers.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        if not translated_texts:
            Gimp.message("No text layers found in 'Translated Text' group to typeset.")
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

        # ── 2. Locate the path layer (Detected Bubbles or fallback) ──
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
            Gimp.message("Error: No paths/vectors found. Please run detection first.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        # ── 3. Parse bounding boxes from path strokes ──
        bounding_boxes = []
        try:
            strokes = target_path.get_strokes()
            for stroke_id in strokes:
                res = target_path.stroke_get_points(stroke_id)
                coords = None
                if isinstance(res, (tuple, list)):
                    for item in res:
                        if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], (int, float)):
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
                bounding_boxes.append((min(x_coords), min(y_coords), max(x_coords), max(y_coords)))
        except Exception as e:
            sys.stderr.write(f"[Koharu Typesetter] Failed to parse paths: {e}\n")
            Gimp.message("Failed to extract bounding boxes from paths.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        if not bounding_boxes:
            Gimp.message("No valid bounding boxes found in the path.")
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

        # ── 4. Map translated layers to bounding boxes ──
        # Also include unmatched text layers (no bounding box) so user can still typeset them
        mapped_bubbles = []
        used_children = set()
        for box in bounding_boxes:
            xmin, ymin, xmax, ymax = box
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            matched_child = None
            matched_text = ""
            best_dist = 999999.0
            for child, tx, ty, text in translated_texts:
                if id(child) in used_children:
                    continue
                dist = np.sqrt((tx - cx)**2 + (ty - cy)**2)
                if dist < best_dist:
                    best_dist = dist
                    matched_child = child
                    matched_text = text
            if matched_child:
                used_children.add(id(matched_child))
                mapped_bubbles.append({
                    "box": box,
                    "layer": matched_child,
                    "text": matched_text,
                    "font_family": config.get_property("font-family") or "CCYadaYadaYada",
                    "font_size": config.get_property("base-font-size") or 18,
                    "alignment": config.get_property("alignment") or "Center",
                    "auto_fit": True,
                    "letter_spacing": 0.0,
                    "line_spacing": 0.0,
                    "color_hex": "#000000",
                    "direction": "Horizontal",
                    "preset": "Dialogue",
                })

        # Include any remaining text layers that didn't match a bounding box
        for child, tx, ty, text in translated_texts:
            if id(child) not in used_children:
                # Use layer bounds as a synthetic bounding box
                lw = child.get_width()
                lh = child.get_height()
                mapped_bubbles.append({
                    "box": (tx, ty, tx + lw, ty + lh),
                    "layer": child,
                    "text": text,
                    "font_family": config.get_property("font-family") or "CCYadaYadaYada",
                    "font_size": config.get_property("base-font-size") or 18,
                    "alignment": config.get_property("alignment") or "Center",
                    "auto_fit": True,
                    "letter_spacing": 0.0,
                    "line_spacing": 0.0,
                    "color_hex": "#000000",
                    "direction": "Horizontal",
                    "preset": "Dialogue",
                })
                sys.stderr.write(f"[Koharu Typesetter] Text layer '{text[:30]}...' had no matching bounding box, using layer bounds\n")

        # ── 5. Gather available font names from GIMP ──
        all_font_names = []
        try:
            pdb = Gimp.get_pdb()
            if pdb:
                # Approach 1: Gimp.ValueArray with string arg
                try:
                    args = Gimp.ValueArray.new(1)
                    args.insert(0, GObject.Value(GObject.TYPE_STRING, ""))
                    result = pdb.run_procedure_argv("gimp-fonts-get-list", args)
                    if result and result.length() > 1:
                        str_val = result.index(1)
                        if hasattr(str_val, "data"):
                            all_font_names = list(str_val.data)
                        elif isinstance(str_val, (list, tuple)):
                            all_font_names = list(str_val)
                except Exception as e1:
                    sys.stderr.write(f"[Koharu Typesetter] Font enum approach 1 failed: {e1}\n")

                # Approach 2: run_procedure with list arg
                if not all_font_names:
                    try:
                        result = pdb.run_procedure("gimp-fonts-get-list", [GObject.Value(GObject.TYPE_STRING, "")])
                        if result and result.length() > 1:
                            str_val = result.index(1)
                            if hasattr(str_val, "data"):
                                all_font_names = list(str_val.data)
                            elif isinstance(str_val, (list, tuple)):
                                all_font_names = list(str_val)
                    except Exception as e2:
                        sys.stderr.write(f"[Koharu Typesetter] Font enum approach 2 failed: {e2}\n")

            # Approach 3: fc-list system command as final fallback
            if not all_font_names:
                try:
                    import subprocess
                    fc_out = subprocess.check_output(["fc-list", "--format", "%{family}\n"], text=True, timeout=5)
                    seen = set()
                    for line in fc_out.strip().split("\n"):
                        # fc-list can return comma-separated families
                        for fam in line.split(","):
                            fam = fam.strip()
                            if fam and fam not in seen:
                                seen.add(fam)
                                all_font_names.append(fam)
                except Exception as e3:
                    sys.stderr.write(f"[Koharu Typesetter] fc-list fallback failed: {e3}\n")
        except Exception as font_list_err:
            sys.stderr.write(f"[Koharu Typesetter] Could not enumerate fonts: {font_list_err}\n")

        if not all_font_names:
            all_font_names = ["Sans-serif", "Serif", "Monospace"]
        all_font_names = sorted(set(all_font_names), key=lambda s: s.lower())
        sys.stderr.write(f"[Koharu Typesetter] Loaded {len(all_font_names)} fonts\n")

        # ── Non-interactive mode: apply defaults to all ──
        if run_mode != Gimp.RunMode.INTERACTIVE:
            font_family = config.get_property("font-family") or "CCYadaYadaYada"
            base_font_size = config.get_property("base-font-size") or 18
            alignment = config.get_property("alignment") or "Center"
            auto_fit = config.get_property("auto-fit") if config.get_property("auto-fit") is not None else True

            font = self._resolve_font(font_family)
            if not font:
                Gimp.message(f"Error: Font '{font_family}' could not be resolved.")
                return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

            justification = {"Center": Gimp.TextJustification.CENTER,
                             "Left": Gimp.TextJustification.LEFT,
                             "Right": Gimp.TextJustification.RIGHT}.get(alignment, Gimp.TextJustification.CENTER)

            image.undo_group_start()
            for bubble in mapped_bubbles:
                self._apply_typeset_to_bubble(image, translated_group, bubble, font, base_font_size,
                                              justification, auto_fit, 0.0, 0.0, "#000000", "Horizontal")
            image.undo_group_end()
            Gimp.displays_flush()
            Gimp.message(f"Typesetting complete! Formatted {len(mapped_bubbles)} bubbles.")
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

        # ══════════════════════════════════════════════════════════════
        # ── 6. INTERACTIVE: Build the two-panel typesetter window ──
        # ══════════════════════════════════════════════════════════════

        # Manga Presets definition
        PRESETS = {
            "Dialogue": {
                "font_family": "CCYadaYadaYada", "font_size": 18,
                "alignment": "Center", "letter_spacing": 0.0, "line_spacing": 0.0,
                "color_hex": "#000000", "direction": "Horizontal", "auto_fit": True,
            },
            "Shout / Title": {
                "font_family": "CCHushHush", "font_size": 24,
                "alignment": "Center", "letter_spacing": 0.0, "line_spacing": 0.0,
                "color_hex": "#000000", "direction": "Horizontal", "auto_fit": True,
            },
            "Narration": {
                "font_family": "Georgia", "font_size": 16,
                "alignment": "Left", "letter_spacing": 0.0, "line_spacing": 2.0,
                "color_hex": "#000000", "direction": "Horizontal", "auto_fit": True,
            },
            "Whisper / Thought": {
                "font_family": "CCYadaYadaYada", "font_size": 14,
                "alignment": "Center", "letter_spacing": 0.0, "line_spacing": 0.0,
                "color_hex": "#555555", "direction": "Horizontal", "auto_fit": True,
            },
            "SFX Vertical": {
                "font_family": "CCHushHush", "font_size": 28,
                "alignment": "Center", "letter_spacing": 4.0, "line_spacing": -4.0,
                "color_hex": "#000000", "direction": "Vertical Stack", "auto_fit": False,
            },
            "SFX Horizontal": {
                "font_family": "CCHushHush", "font_size": 28,
                "alignment": "Center", "letter_spacing": 2.0, "line_spacing": 0.0,
                "color_hex": "#000000", "direction": "Horizontal", "auto_fit": True,
            },
        }

        # State tracking
        current_selection = [0]  # index into mapped_bubbles
        cancelled = [False]

        # Start undo group for live preview
        image.undo_group_start()

        # ── Build Window ──
        win = Gtk.Window(title="Koharu Typesetter")
        win.set_default_size(820, 620)
        win.set_keep_above(True)

        main_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        win.add(main_hbox)

        # ╔═════════════════════════════════════════╗
        # ║  LEFT PANEL: Bubble List                ║
        # ╚═════════════════════════════════════════╝
        left_frame = Gtk.Frame(label="  Dialogue Bubbles  ")
        left_frame.set_size_request(260, -1)
        left_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        left_vbox.set_margin_top(8)
        left_vbox.set_margin_bottom(8)
        left_vbox.set_margin_start(8)
        left_vbox.set_margin_end(8)

        # Scrollable list
        scroll_list = Gtk.ScrolledWindow()
        scroll_list.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll_list.set_vexpand(True)

        listbox = Gtk.ListBox()
        listbox.set_selection_mode(Gtk.SelectionMode.SINGLE)

        for idx, bubble in enumerate(mapped_bubbles):
            row = Gtk.ListBoxRow()
            row_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            row_box.set_margin_top(6)
            row_box.set_margin_bottom(6)
            row_box.set_margin_start(8)
            row_box.set_margin_end(8)

            lbl_idx = Gtk.Label()
            preview_text = bubble["text"][:40] + ("..." if len(bubble["text"]) > 40 else "")
            lbl_idx.set_markup(f"<b>Bubble {idx + 1}</b>  <span foreground='#888888' size='small'>[{bubble['preset']}]</span>")
            lbl_idx.set_xalign(0.0)
            row_box.pack_start(lbl_idx, False, False, 0)

            lbl_preview = Gtk.Label(label=preview_text)
            lbl_preview.set_xalign(0.0)
            lbl_preview.set_line_wrap(True)
            lbl_preview.set_max_width_chars(30)
            row_box.pack_start(lbl_preview, False, False, 0)

            row.add(row_box)
            row._bubble_index = idx
            row._label_widget = lbl_idx
            listbox.add(row)

        scroll_list.add(listbox)
        left_vbox.pack_start(scroll_list, True, True, 0)

        # Apply All to Selected Preset button
        btn_preset_all = Gtk.Button(label="Apply Preset to All Bubbles")
        left_vbox.pack_start(btn_preset_all, False, False, 4)

        left_frame.add(left_vbox)
        main_hbox.pack_start(left_frame, False, False, 0)

        # Vertical separator
        main_hbox.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 0)

        # ╔═════════════════════════════════════════╗
        # ║  RIGHT PANEL: Property Editor           ║
        # ╚═════════════════════════════════════════╝
        right_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        right_vbox.set_hexpand(True)

        # Header
        header = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        header.set_margin_top(10)
        header.set_margin_start(12)
        header.set_margin_end(12)
        header.set_margin_bottom(6)
        lbl_title = Gtk.Label()
        lbl_title.set_markup("<span size='large' weight='bold' foreground='#3584e4'>Koharu Typesetter</span>")
        lbl_title.set_xalign(0.0)
        header.pack_start(lbl_title, False, False, 0)
        lbl_sub = Gtk.Label(label="Edit properties per-bubble or apply manga presets in bulk.")
        lbl_sub.set_xalign(0.0)
        header.pack_start(lbl_sub, False, False, 0)
        right_vbox.pack_start(header, False, False, 0)
        right_vbox.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), False, False, 4)

        # Scrollable properties area
        scroll_props = Gtk.ScrolledWindow()
        scroll_props.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll_props.set_vexpand(True)

        props_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        props_box.set_margin_top(8)
        props_box.set_margin_bottom(8)
        props_box.set_margin_start(12)
        props_box.set_margin_end(12)

        # ── Section: Manga Preset ──
        lbl_preset_section = Gtk.Label()
        lbl_preset_section.set_markup("<b>Manga Preset</b>")
        lbl_preset_section.set_xalign(0.0)
        props_box.pack_start(lbl_preset_section, False, False, 0)

        preset_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        combo_preset = Gtk.ComboBoxText()
        for pname in PRESETS.keys():
            combo_preset.append_text(pname)
        combo_preset.set_active(0)
        preset_hbox.pack_start(combo_preset, True, True, 0)

        btn_apply_preset = Gtk.Button(label="Load Preset")
        preset_hbox.pack_start(btn_apply_preset, False, False, 0)
        props_box.pack_start(preset_hbox, False, False, 0)

        props_box.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), False, False, 4)

        # ── Section: Font ──
        lbl_font_section = Gtk.Label()
        lbl_font_section.set_markup("<b>Font Family</b>  <span size='small' foreground='#888888'>(type to search)</span>")
        lbl_font_section.set_xalign(0.0)
        props_box.pack_start(lbl_font_section, False, False, 0)

        entry_font_search = Gtk.Entry()
        entry_font_search.set_placeholder_text("Search fonts...")
        entry_font_search.set_text(mapped_bubbles[0]["font_family"])
        props_box.pack_start(entry_font_search, False, False, 0)

        # Font list (filtered)
        font_scroll = Gtk.ScrolledWindow()
        font_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        font_scroll.set_size_request(-1, 120)
        font_scroll.set_min_content_height(80)

        font_listbox = Gtk.ListBox()
        font_listbox.set_selection_mode(Gtk.SelectionMode.SINGLE)
        font_scroll.add(font_listbox)
        props_box.pack_start(font_scroll, False, False, 0)

        in_font_update = [False]

        def populate_font_list(filter_text="", selected_font=None):
            for child in font_listbox.get_children():
                font_listbox.remove(child)
            ft = filter_text.lower()
            count = 0
            for fname in all_font_names:
                if ft and ft not in fname.lower():
                    continue
                row = Gtk.ListBoxRow()
                lbl = Gtk.Label(label=fname)
                lbl.set_xalign(0.0)
                lbl.set_margin_start(6)
                lbl.set_margin_top(2)
                lbl.set_margin_bottom(2)
                row.add(lbl)
                row._font_name = fname
                font_listbox.add(row)
                if selected_font and fname == selected_font:
                    font_listbox.select_row(row)
                count += 1
                if count >= 100:
                    break
            font_listbox.show_all()

        populate_font_list(mapped_bubbles[0]["font_family"], mapped_bubbles[0]["font_family"])

        def on_font_search_changed(entry):
            if in_font_update[0]:
                return
            populate_font_list(entry.get_text())
        entry_font_search.connect("changed", on_font_search_changed)

        def on_font_selected(lb, row):
            if row and hasattr(row, "_font_name"):
                in_font_update[0] = True
                entry_font_search.set_text(row._font_name)
                in_font_update[0] = False
                idx = current_selection[0]
                mapped_bubbles[idx]["font_family"] = row._font_name
        font_listbox.connect("row-selected", on_font_selected)

        props_box.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), False, False, 4)

        # ── Section: Text Properties ──
        lbl_text_section = Gtk.Label()
        lbl_text_section.set_markup("<b>Text Properties</b>")
        lbl_text_section.set_xalign(0.0)
        props_box.pack_start(lbl_text_section, False, False, 0)

        grid_props = Gtk.Grid()
        grid_props.set_column_spacing(12)
        grid_props.set_row_spacing(8)

        # Font Size
        grid_props.attach(Gtk.Label(label="Font Size (px):"), 0, 0, 1, 1)
        spin_font_size = Gtk.SpinButton.new_with_range(6.0, 72.0, 1.0)
        spin_font_size.set_value(mapped_bubbles[0]["font_size"])
        grid_props.attach(spin_font_size, 1, 0, 1, 1)

        # Alignment
        grid_props.attach(Gtk.Label(label="Alignment:"), 0, 1, 1, 1)
        combo_alignment = Gtk.ComboBoxText()
        for a in ["Center", "Left", "Right"]:
            combo_alignment.append_text(a)
        combo_alignment.set_active(0)
        grid_props.attach(combo_alignment, 1, 1, 1, 1)

        # Letter Spacing
        grid_props.attach(Gtk.Label(label="Letter Spacing:"), 0, 2, 1, 1)
        spin_letter = Gtk.SpinButton.new_with_range(-10.0, 20.0, 0.5)
        spin_letter.set_value(0.0)
        grid_props.attach(spin_letter, 1, 2, 1, 1)

        # Line Spacing
        grid_props.attach(Gtk.Label(label="Line Spacing:"), 0, 3, 1, 1)
        spin_line = Gtk.SpinButton.new_with_range(-10.0, 20.0, 0.5)
        spin_line.set_value(0.0)
        grid_props.attach(spin_line, 1, 3, 1, 1)

        # Text Color + Eyedropper
        grid_props.attach(Gtk.Label(label="Text Color:"), 0, 4, 1, 1)
        color_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        color_btn = Gtk.ColorButton()
        try:
            from gi.repository import Gdk
            rgba_default = Gdk.RGBA()
            rgba_default.parse("#000000")
            color_btn.set_rgba(rgba_default)
        except Exception:
            pass
        color_hbox.pack_start(color_btn, True, True, 0)

        btn_eyedropper = Gtk.Button(label="🎨 Pick from Canvas")
        btn_eyedropper.set_tooltip_text(
            "Use GIMP's Color Picker tool (eyedropper) on the canvas first,\n"
            "then click this button to grab the foreground color.")
        def on_eyedropper_clicked(btn):
            try:
                fg_color = None
                # Try direct API first
                if hasattr(Gimp, 'context_get_foreground'):
                    fg_color = Gimp.context_get_foreground()
                # Fallback: PDB call
                if not fg_color:
                    pdb = Gimp.get_pdb()
                    if pdb:
                        result = pdb.run_procedure("gimp-context-get-foreground", [])
                        if result and result.length() > 1:
                            val = result.index(1)
                            if hasattr(val, "get_value"):
                                fg_color = val.get_value()
                            else:
                                fg_color = val

                if fg_color:
                    r = g = b = 0.0
                    # Try multiple Gegl.Color extraction methods
                    extracted = False
                    if hasattr(fg_color, 'get_rgba'):
                        try:
                            ret = fg_color.get_rgba()
                            if hasattr(ret, 'red'):
                                r, g, b = ret.red, ret.green, ret.blue
                                extracted = True
                            elif isinstance(ret, (tuple, list)) and len(ret) >= 3:
                                r, g, b = ret[0], ret[1], ret[2]
                                extracted = True
                        except Exception as e_rgba:
                            sys.stderr.write(f"[Koharu Typesetter] get_rgba failed: {e_rgba}\n")
                    if not extracted and hasattr(fg_color, 'get_components'):
                        try:
                            comps = fg_color.get_components()
                            if len(comps) >= 3:
                                r, g, b = comps[0], comps[1], comps[2]
                                extracted = True
                        except Exception as e_comps:
                            sys.stderr.write(f"[Koharu Typesetter] get_components failed: {e_comps}\n")
                    if not extracted and hasattr(fg_color, 'get_bytes'):
                        try:
                            fmt = Babl.format("R'G'B' u8")
                            data = fg_color.get_bytes(fmt)
                            if data and len(data) >= 3:
                                r, g, b = data[0]/255.0, data[1]/255.0, data[2]/255.0
                                extracted = True
                        except Exception as e_bytes:
                            sys.stderr.write(f"[Koharu Typesetter] get_bytes failed: {e_bytes}\n")

                    from gi.repository import Gdk
                    rgba = Gdk.RGBA()
                    # Clamp values - they should be 0.0-1.0 floats
                    rgba.red = max(0.0, min(1.0, float(r)))
                    rgba.green = max(0.0, min(1.0, float(g)))
                    rgba.blue = max(0.0, min(1.0, float(b)))
                    rgba.alpha = 1.0
                    color_btn.set_rgba(rgba)
                    sys.stderr.write(f"[Koharu Typesetter] Picked FG color: R={rgba.red:.2f} G={rgba.green:.2f} B={rgba.blue:.2f}\n")
                else:
                    sys.stderr.write("[Koharu Typesetter] Could not get foreground color\n")
            except Exception as pick_err:
                sys.stderr.write(f"[Koharu Typesetter] Eyedropper pick failed: {pick_err}\n")
        btn_eyedropper.connect("clicked", on_eyedropper_clicked)
        color_hbox.pack_start(btn_eyedropper, False, False, 0)
        grid_props.attach(color_hbox, 1, 4, 1, 1)

        # Auto-fit
        chk_autofit = Gtk.CheckButton(label="Auto-fit text to bubble")
        chk_autofit.set_active(True)
        grid_props.attach(chk_autofit, 0, 5, 2, 1)

        # Direction
        grid_props.attach(Gtk.Label(label="Direction:"), 0, 6, 1, 1)
        combo_direction = Gtk.ComboBoxText()
        combo_direction.append_text("Horizontal")
        combo_direction.append_text("Vertical Stack")
        combo_direction.set_active(0)
        grid_props.attach(combo_direction, 1, 6, 1, 1)

        # Left-align all grid labels
        for child in grid_props.get_children():
            if isinstance(child, Gtk.Label):
                child.set_xalign(0.0)

        props_box.pack_start(grid_props, False, False, 0)
        scroll_props.add(props_box)
        right_vbox.pack_start(scroll_props, True, True, 0)

        # ── Bottom Button Bar ──
        right_vbox.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), False, False, 0)
        btn_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        btn_bar.set_margin_top(8)
        btn_bar.set_margin_bottom(8)
        btn_bar.set_margin_start(12)
        btn_bar.set_margin_end(12)

        btn_preview = Gtk.Button(label="⟳ Preview")
        btn_preview.get_style_context().add_class("suggested-action")
        btn_bar.pack_start(btn_preview, True, True, 0)

        btn_apply_selected = Gtk.Button(label="✓ Apply Selected")
        btn_bar.pack_start(btn_apply_selected, True, True, 0)

        btn_apply_all = Gtk.Button(label="✓✓ Apply All & Close")
        btn_apply_all.get_style_context().add_class("suggested-action")
        btn_bar.pack_start(btn_apply_all, True, True, 0)

        btn_cancel = Gtk.Button(label="✕ Cancel")
        btn_cancel.get_style_context().add_class("destructive-action")
        btn_bar.pack_start(btn_cancel, False, False, 0)

        right_vbox.pack_start(btn_bar, False, False, 0)
        main_hbox.pack_start(right_vbox, True, True, 0)

        # ══════════════════════════════════════════════════════════════
        # ── 7. Wire up all signal handlers ──
        # ══════════════════════════════════════════════════════════════

        def save_current_to_state():
            """Save the current widget values into the selected bubble's state dict."""
            idx = current_selection[0]
            b = mapped_bubbles[idx]
            b["font_family"] = entry_font_search.get_text().strip()
            b["font_size"] = int(spin_font_size.get_value())
            b["alignment"] = combo_alignment.get_active_text() or "Center"
            b["letter_spacing"] = spin_letter.get_value()
            b["line_spacing"] = spin_line.get_value()
            b["auto_fit"] = chk_autofit.get_active()
            b["direction"] = combo_direction.get_active_text() or "Horizontal"
            b["preset"] = combo_preset.get_active_text() or "Dialogue"
            try:
                from gi.repository import Gdk
                rgba = color_btn.get_rgba()
                r = int(rgba.red * 255)
                g = int(rgba.green * 255)
                bb = int(rgba.blue * 255)
                b["color_hex"] = f"#{r:02x}{g:02x}{bb:02x}"
            except Exception:
                b["color_hex"] = "#000000"

        def load_state_to_widgets(idx):
            """Load a bubble's state dict into the property widgets."""
            b = mapped_bubbles[idx]
            in_font_update[0] = True
            entry_font_search.set_text(b["font_family"])
            in_font_update[0] = False
            populate_font_list(b["font_family"], b["font_family"])

            spin_font_size.set_value(b["font_size"])
            align_map = {"Center": 0, "Left": 1, "Right": 2}
            combo_alignment.set_active(align_map.get(b["alignment"], 0))
            spin_letter.set_value(b["letter_spacing"])
            spin_line.set_value(b["line_spacing"])
            chk_autofit.set_active(b["auto_fit"])
            dir_map = {"Horizontal": 0, "Vertical Stack": 1}
            combo_direction.set_active(dir_map.get(b["direction"], 0))

            preset_names = list(PRESETS.keys())
            if b["preset"] in preset_names:
                combo_preset.set_active(preset_names.index(b["preset"]))
            else:
                combo_preset.set_active(0)

            try:
                from gi.repository import Gdk
                rgba = Gdk.RGBA()
                rgba.parse(b["color_hex"])
                color_btn.set_rgba(rgba)
            except Exception:
                pass

        def on_bubble_selected(lb, row):
            if row and hasattr(row, "_bubble_index"):
                save_current_to_state()
                current_selection[0] = row._bubble_index
                load_state_to_widgets(row._bubble_index)
        listbox.connect("row-selected", on_bubble_selected)

        def on_load_preset_clicked(btn):
            preset_name = combo_preset.get_active_text()
            if preset_name and preset_name in PRESETS:
                p = PRESETS[preset_name]
                idx = current_selection[0]
                b = mapped_bubbles[idx]
                b.update({
                    "font_family": p["font_family"],
                    "font_size": p["font_size"],
                    "alignment": p["alignment"],
                    "letter_spacing": p["letter_spacing"],
                    "line_spacing": p["line_spacing"],
                    "color_hex": p["color_hex"],
                    "direction": p["direction"],
                    "auto_fit": p["auto_fit"],
                    "preset": preset_name,
                })
                load_state_to_widgets(idx)
                # Update list label
                row = listbox.get_row_at_index(idx)
                if row and hasattr(row, "_label_widget"):
                    preview_text = b["text"][:40] + ("..." if len(b["text"]) > 40 else "")
                    row._label_widget.set_markup(f"<b>Bubble {idx + 1}</b>  <span foreground='#888888' size='small'>[{preset_name}]</span>")
        btn_apply_preset.connect("clicked", on_load_preset_clicked)

        def on_preset_all_clicked(btn):
            preset_name = combo_preset.get_active_text()
            if preset_name and preset_name in PRESETS:
                p = PRESETS[preset_name]
                for i, b in enumerate(mapped_bubbles):
                    b.update({
                        "font_family": p["font_family"],
                        "font_size": p["font_size"],
                        "alignment": p["alignment"],
                        "letter_spacing": p["letter_spacing"],
                        "line_spacing": p["line_spacing"],
                        "color_hex": p["color_hex"],
                        "direction": p["direction"],
                        "auto_fit": p["auto_fit"],
                        "preset": preset_name,
                    })
                    row = listbox.get_row_at_index(i)
                    if row and hasattr(row, "_label_widget"):
                        row._label_widget.set_markup(f"<b>Bubble {i + 1}</b>  <span foreground='#888888' size='small'>[{preset_name}]</span>")
                load_state_to_widgets(current_selection[0])
        btn_preset_all.connect("clicked", on_preset_all_clicked)

        def on_preview_clicked(btn):
            save_current_to_state()
            idx = current_selection[0]
            b = mapped_bubbles[idx]
            font = self._resolve_font(b["font_family"])
            if not font:
                Gimp.message(f"Font '{b['font_family']}' not found.")
                return

            justification = {"Center": Gimp.TextJustification.CENTER,
                             "Left": Gimp.TextJustification.LEFT,
                             "Right": Gimp.TextJustification.RIGHT}.get(b["alignment"], Gimp.TextJustification.CENTER)

            new_layer = self._apply_typeset_to_bubble(
                image, translated_group, b, font, b["font_size"],
                justification, b["auto_fit"], b["letter_spacing"],
                b["line_spacing"], b["color_hex"], b["direction"])
            if new_layer:
                b["layer"] = new_layer
            Gimp.displays_flush()
        btn_preview.connect("clicked", on_preview_clicked)

        def on_apply_selected_clicked(btn):
            save_current_to_state()
            idx = current_selection[0]
            b = mapped_bubbles[idx]
            font = self._resolve_font(b["font_family"])
            if not font:
                Gimp.message(f"Font '{b['font_family']}' not found.")
                return

            justification = {"Center": Gimp.TextJustification.CENTER,
                             "Left": Gimp.TextJustification.LEFT,
                             "Right": Gimp.TextJustification.RIGHT}.get(b["alignment"], Gimp.TextJustification.CENTER)

            new_layer = self._apply_typeset_to_bubble(
                image, translated_group, b, font, b["font_size"],
                justification, b["auto_fit"], b["letter_spacing"],
                b["line_spacing"], b["color_hex"], b["direction"])
            if new_layer:
                b["layer"] = new_layer
            Gimp.displays_flush()

            # Auto-advance to the next bubble
            next_idx = idx + 1
            if next_idx < len(mapped_bubbles):
                next_row = listbox.get_row_at_index(next_idx)
                if next_row:
                    listbox.select_row(next_row)
        btn_apply_selected.connect("clicked", on_apply_selected_clicked)

        def on_apply_all_clicked(btn):
            save_current_to_state()
            for b in mapped_bubbles:
                font = self._resolve_font(b["font_family"])
                if not font:
                    continue
                justification = {"Center": Gimp.TextJustification.CENTER,
                                 "Left": Gimp.TextJustification.LEFT,
                                 "Right": Gimp.TextJustification.RIGHT}.get(b["alignment"], Gimp.TextJustification.CENTER)
                new_layer = self._apply_typeset_to_bubble(
                    image, translated_group, b, font, b["font_size"],
                    justification, b["auto_fit"], b["letter_spacing"],
                    b["line_spacing"], b["color_hex"], b["direction"])
                if new_layer:
                    b["layer"] = new_layer
            image.undo_group_end()
            Gimp.displays_flush()
            win.destroy()
            Gtk.main_quit()
        btn_apply_all.connect("clicked", on_apply_all_clicked)

        def on_cancel_clicked(btn):
            cancelled[0] = True
            image.undo_group_end()
            # Undo the entire group
            try:
                Gimp.get_pdb().run_procedure("gimp-image-undo", [GObject.Value(Gimp.Image.__gtype__, image)])
            except Exception:
                pass
            Gimp.displays_flush()
            win.destroy()
            Gtk.main_quit()
        btn_cancel.connect("clicked", on_cancel_clicked)

        def on_window_delete(w, event):
            on_cancel_clicked(None)
            return True
        win.connect("delete-event", on_window_delete)

        # Select first bubble by default
        first_row = listbox.get_row_at_index(0)
        if first_row:
            listbox.select_row(first_row)
        load_state_to_widgets(0)

        win.show_all()
        Gtk.main()

        if cancelled[0]:
            return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

        Gimp.message(f"Typesetting complete! Formatted {len(mapped_bubbles)} dialogue bubbles.")
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    # ── Helper: Resolve a font name to a Gimp.Font object ──
    def _resolve_font(self, font_family):
        """Try to resolve font_family to a Gimp.Font, with fallbacks."""
        font = None
        fallbacks = [
            font_family,
            "CCYadaYadaYada",
            "CCHushHush",
            "Liberation Serif",
            "Georgia",
            "Serif",
            "Sans-serif",
            "Sans",
        ]
        try:
            if hasattr(Gimp, "Font") and hasattr(Gimp.Font, "get_by_name"):
                for name in fallbacks:
                    if not name:
                        continue
                    font = Gimp.Font.get_by_name(name)
                    if font:
                        if name != font_family:
                            sys.stderr.write(f"[Koharu Typesetter] Font '{font_family}' not found, fell back to '{name}'\n")
                        break
        except Exception as e:
            sys.stderr.write(f"[Koharu Typesetter] Font resolve error: {e}\n")
        return font

    # ── Helper: Apply typeset styling to a single bubble ──
    def _apply_typeset_to_bubble(self, image, group, bubble, font, font_size,
                                  justification, auto_fit, letter_spacing,
                                  line_spacing, color_hex, direction):
        """Remove old layer, create new styled TextLayer, insert into group, center in bubble box."""
        import textwrap

        box = bubble["box"]
        text_content = bubble["text"]
        old_layer = bubble["layer"]

        xmin, ymin, xmax, ymax = box
        w = max(10, int(xmax - xmin))
        h = max(10, int(ymax - ymin))
        cx = int((xmin + xmax) / 2)
        cy = int((ymin + ymax) / 2)

        # Handle vertical stacking: one character per line
        if direction == "Vertical Stack":
            display_text = "\n".join(list(text_content.replace(" ", "")))
            final_size = font_size
        elif auto_fit:
            display_text, final_size = self._fit_and_wrap(text_content, font_size, w, h)
        else:
            avg_cw = max(1.0, font_size * 0.52)
            cpl = max(5, int(w / avg_cw))
            display_text = "\n".join(textwrap.wrap(text_content, width=cpl))
            final_size = font_size

        try:
            # Remove existing layer
            if old_layer and old_layer.is_valid():
                image.remove_layer(old_layer)

            text_layer = Gimp.TextLayer.new(image, display_text, font, final_size, Gimp.Unit.pixel())
            if not text_layer:
                return None

            text_layer.set_justification(justification)

            # Letter spacing
            if letter_spacing != 0.0:
                try:
                    text_layer.set_letter_spacing(letter_spacing)
                except Exception:
                    pass

            # Line spacing
            if line_spacing != 0.0:
                try:
                    text_layer.set_line_spacing(line_spacing)
                except Exception:
                    pass

            # Text color
            try:
                text_color = Gegl.Color.new(color_hex)
                text_layer.set_color(text_color)
            except Exception:
                pass

            # Insert into group
            image.insert_layer(text_layer, group, -1)

            # Center in bubble
            rect_t = text_layer.get_buffer().get_extent()
            tw = rect_t.width
            th = rect_t.height
            tx = cx - tw // 2
            ty = cy - th // 2
            text_layer.set_offsets(int(tx), int(ty))

            return text_layer
        except Exception as err:
            sys.stderr.write(f"[Koharu Typesetter] Render error: {err}\n")
            return None

    # ── Helper: Fit and wrap text to bubble dimensions ──
    def _fit_and_wrap(self, text, font_size_init, max_width, max_height):
        """Scale font down and wrap text until it fits the bubble area."""
        import textwrap
        font_size = font_size_init
        while font_size >= 10:
            avg_cw = max(1.0, font_size * 0.52)
            cpl = max(5, int(max_width / avg_cw))
            lines = textwrap.wrap(text, width=cpl)
            total_h = len(lines) * (font_size * 1.3)
            if total_h <= max_height or font_size == 10:
                return "\n".join(lines), font_size
            font_size -= 2
        return text, font_size




if __name__ == "__main__":
    Gimp.main(GimpScanlationSuite.__gtype__, sys.argv)
