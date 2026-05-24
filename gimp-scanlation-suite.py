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
                "Detector Model",
                "Model to run (e.g. anime-text-yolo, comic-text-bubble-detector)",
                "anime-text-yolo",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_double_argument(
                "confidence",
                "Confidence Threshold",
                "Minimum model detection confidence",
                0.0, 1.0, 0.45,
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
                "OCR Engine",
                "OCR engine to use (e.g. manga-ocr, mit48px-ocr, paddleocr)",
                "manga-ocr",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_boolean_argument(
                "half-to-full",
                "Convert Half-width to Full-width",
                "Post-process ASCII characters to full-width CJK alternatives",
                True,
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
                "Inpaint Model",
                "Inpainting model to run (e.g. lama-manga, aot-inpainting)",
                "lama-manga",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_int_argument(
                "dilation",
                "Mask Dilation (px)",
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
                "Source Language",
                "Language to translate from",
                "ja",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "target-lang",
                "Target Language",
                "Language to translate to",
                "en",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "api-url",
                "Koharu API / LLM URL",
                "Endpoint for local LLM or Koharu translator server",
                "http://localhost:4000",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "font-family",
                "Font Family",
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

        Gimp.message(f"[Koharu Detector] Running '{detector_model}' with threshold={confidence:.2f}...")

        # 1. Verification of active layer
        if not drawables:
            Gimp.message("Error: No active drawable/layer selected.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())
            
        active_layer = drawables[0]

        # 2. Call scouter to detect bounding boxes
        if scouter is None:
            Gimp.message("Error: Scouter module could not be imported. Check venv dependencies.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        try:
            boxes = scouter.detect_text_bubbles(active_layer, confidence_threshold=confidence)
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
        
        Rust ONNX Pipeline Equivalents:
        1. Preprocessing:
           - In `manga_ocr_onnx_inference.py`, each cropped text block is converted to grayscale,
             then padded/resized to 224x224 using bilinear/area interpolation.
           - Rescaled (division by 255.0) and normalized: (pixel - 0.5) / 0.5.
           - Transposed from HWC (224, 224, 3) to CHW (3, 224, 224) and batched.
        2. ONNX Model Inference (ViT-BERT EncoderDecoder):
           - Encoder: Feed `pixel_values` to get `encoder_hidden_states`.
           - Decoder: Loop generated token IDs up to maximum length.
             In each iteration, run decoder session using `encoder_hidden_states` and `input_ids` (token_ids list).
             Decoder output logits are processed to select the highest probability token (argmax/greedy).
             Loop terminates when `eos_token` (ID=3) is encountered.
        3. Postprocessing:
           - Decodes token IDs to strings using tokenizer vocab dictionary.
           - Strips whitespace.
           - Normalises text with `jaconv` and custom rules: maps halfwidth ASCII to fullwidth CJK (`halfwidth_to_fullwidth`),
             collapses repetitive dots/punctuation (`collapse_dots`), and replaces unicode ellipsis with standard dot notation.
        """
        GimpUi.init("gimp-scanlation-ocr")

        if run_mode == Gimp.RunMode.INTERACTIVE:
            dialog = GimpUi.ProcedureDialog.new(procedure, config, "OCR Selected Blocks")
            
            vbox = dialog.get_content_area()
            
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
            desc_label.set_text("Performs optical character recognition (OCR) on the grayscaled bounds of text regions using a VisionEncoderDecoder Transformer model.")
            desc_label.set_line_wrap(True)
            desc_label.set_xalign(0.0)
            header_box.pack_start(desc_label, False, False, 0)
            
            vbox.pack_start(header_box, False, False, 0)
            vbox.show_all()
            
            dialog.fill(None)
            
            if not dialog.run():
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

        ocr_engine = config.get_property("ocr-engine")
        half_to_full = config.get_property("half-to-full")

        Gimp.message(f"[Koharu OCR] Running '{ocr_engine}' (half-to-full postprocess={half_to_full})...")

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
