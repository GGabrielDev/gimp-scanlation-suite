#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
GIMP 3 Scanlation Suite Scaffolding

This GIMP 3 Python plugin registers four procedures under the 'Filters > Scanlation' menu:
1. Detect Text & Bubbles: Bounding box and mask detection (YOLO/Segmentation).
2. OCR Selected Blocks: Japanese Text recognition (ViT + BERT / MangaOCR).
3. Inpaint / Erase Text: Cleans dialogue dialogue bounds (LaMa/AOT-Inpaint).
4. Translate & Render: Connects to local LLMs (via Remote API) and typesets translations.

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
gi.require_version('Gegl', '0.4')
from gi.repository import Gegl
import numpy as np

try:
    from modules import scouter
except ImportError as e:
    sys.stderr.write(f"[Scanlation Suite] Failed to import scouter: {e}\n")
    scouter = None

try:
    from modules import model_manager
except ImportError as e:
    sys.stderr.write(f"[Scanlation Suite] Failed to import model_manager: {e}\n")
    model_manager = None

try:
    from modules import ocr_engine
except ImportError as e:
    sys.stderr.write(f"[Scanlation Suite] Failed to import ocr_engine: {e}\n")
    ocr_engine = None


class GimpScanlationSuite(Gimp.PlugIn):

    # 1. Register procedures names
    def do_query_procedures(self):
        return [
            "gimp-scanlation-detect",
            "gimp-scanlation-ocr",
            "gimp-scanlation-inpaint",
            "gimp-scanlation-translate",
            "gimp-scanlation-curve-typeset"
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
            procedure.set_attribution("Scanlation Suite Contributors", "GPL-3.0", "2026")

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
            procedure.set_attribution("Scanlation Suite Contributors", "GPL-3.0", "2026")

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
            procedure.set_attribution("Scanlation Suite Contributors", "GPL-3.0", "2026")

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
            procedure.add_string_argument(
                "prompt",
                "Diffusion _Prompt",
                "Prompt for SD Inpainting (blank for auto/general)",
                "",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "negative-prompt",
                "Diffusion _Negative Prompt",
                "Things to avoid in SD Inpainting",
                "color, colorful, low quality, blurry, bad anatomy",
                GObject.ParamFlags.READWRITE
            )
            procedure.add_int_argument(
                "steps",
                "Diffusion _Steps",
                "Number of denoising steps for SD",
                1, 100, 25,
                GObject.ParamFlags.READWRITE
            )
            procedure.add_double_argument(
                "guidance-scale",
                "Diffusion _Guidance Scale",
                "CFG Scale for SD",
                1.0, 20.0, 7.5,
                GObject.ParamFlags.READWRITE
            )
            procedure.add_boolean_argument(
                "auto-prompt",
                "Auto _Prompt via VLM",
                "Query the VLM to automatically generate the inpainting prompt",
                False,
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "consensus-arbiter",
                "VLM Arbiter",
                "VLM model used for auto-prompting",
                "DeepSeek",
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
            procedure.set_attribution("Scanlation Suite Contributors", "GPL-3.0", "2026")

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
                "Remote _API / LLM URL",
                "Endpoint for local LLM or remote translator server",
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

        elif name == "gimp-scanlation-curve-typeset":
            procedure = Gimp.ImageProcedure.new(
                self, name, Gimp.PDBProcType.PLUGIN, self.run_curve_typeset, None
            )
            procedure.set_image_types("RGB*, GRAY*")
            procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
            procedure.set_menu_label("5. Curve and Stroke Text...")
            procedure.add_menu_path("<Image>/Filters/Scanlation")
            
            procedure.set_documentation(
                "Create a non-destructive outline and fill for text along a path.",
                "Fills path with active FG/text color, and outlines it with active BG color on separate layers.",
                name
            )
            procedure.set_attribution("Scanlation Suite Contributors", "GPL-3.0", "2026")

            # Arguments
            procedure.add_int_argument(
                "stroke-width",
                "Stroke _Width (px)",
                "Width of the outline/stroke",
                1, 50, 4,
                GObject.ParamFlags.READWRITE
            )
            procedure.add_boolean_argument(
                "grow-selection",
                "_Grow Selection Method",
                "Grow the selection to fill background outline (Method A) instead of direct path stroking (Method B)",
                True,
                GObject.ParamFlags.READWRITE
            )
            procedure.add_boolean_argument(
                "use-text-color",
                "Use _Text Layer Color",
                "Attempt to extract the fill color from the text layer's properties instead of active Foreground color",
                True,
                GObject.ParamFlags.READWRITE
            )
            procedure.add_string_argument(
                "path-name",
                "_Path Name",
                "The target path name for curving the text",
                "",
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
            from modules.ui.detect_dialog import show_detect_dialog
            ok = show_detect_dialog(procedure, config)
            if not ok:
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

        from modules.core.detect_runner import run_detect_processing
        return run_detect_processing(procedure, image, drawables, config)

    def run_ocr(self, procedure, run_mode, image, drawables, config, run_data):
        """
        Executes Japanese Manga OCR (Optical Character Recognition).
        """
        GimpUi.init("gimp-scanlation-ocr")

        # Verification of active layer
        if not drawables:
            Gimp.message("Error: No active drawable/layer selected.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())
            
        active_layer = drawables[0]

        # Locate target path layer
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

        sys.stderr.write(f"[Scanlation OCR] Reading bounding boxes from path: '{target_path.get_name()}'...\n")

        # Retrieve strokes and parse coordinates
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
                    sys.stderr.write(f"[Scanlation OCR] Skipping stroke {stroke_id}: no coordinates retrieved.\n")
                    continue

                x_coords = coords[0::2]
                y_coords = coords[1::2]
                if not x_coords or not y_coords:
                    continue
                    
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)
                
                bounding_boxes.append((xmin, ymin, xmax, ymax))
        except Exception as e:
            sys.stderr.write(f"[Scanlation OCR] Failed to parse paths/strokes: {e}\n")
            Gimp.message("Failed to extract coordinates from paths.")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

        if not bounding_boxes:
            Gimp.message("No valid text bounding boxes found in the selected path.")
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

        if run_mode == Gimp.RunMode.INTERACTIVE:
            from modules.ui.ocr_dialog import show_ocr_dialog
            selected_indices = show_ocr_dialog(procedure, config, image, bounding_boxes)
            if selected_indices is None:
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())
            if not selected_indices:
                Gimp.message("No regions selected for OCR.")
                return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())
        else:
            selected_indices = list(range(len(bounding_boxes)))

        # Filter bounding boxes to only selected checklist indices
        bounding_boxes = [bounding_boxes[i] for i in selected_indices]

        # Execute OCR processing loop
        from modules.core.ocr_runner import run_ocr_processing
        return run_ocr_processing(procedure, image, active_layer, bounding_boxes, config, run_mode)

    def run_inpaint(self, procedure, run_mode, image, drawables, config, run_data):
        """
        Executes Inpainting to erase text and fill background.
        
        Saves output non-destructively to a new layer named `[Inpaint] <Original Layer Name>`
        placed directly above the active layer for quick comparison and toggle visibility.
        """
        GimpUi.init("gimp-scanlation-inpaint")

        if run_mode == Gimp.RunMode.INTERACTIVE:
            from modules.ui.inpaint_dialog import show_inpaint_dialog
            ok = show_inpaint_dialog(procedure, config)
            if not ok:
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

        from modules.core.inpaint_runner import run_inpaint_processing
        return run_inpaint_processing(procedure, image, drawables, config)

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

        # Retrieve active layer offsets and dimensions for preview crops
        try:
            preview_layer = active_layer
            for layer in reversed(image.get_layers()):
                if hasattr(layer, "get_children") and Gimp.Item.get_children(layer) is not None:
                    continue
                if hasattr(Gimp, "TextLayer") and isinstance(layer, Gimp.TextLayer):
                    continue
                name = layer.get_name()
                if name.startswith("[Inpaint]") or name in ["OCR Transcriptions", "Translated Text", "Detected Bubbles"]:
                    continue
                preview_layer = layer
                break

            sys.stderr.write(f"[Scanlation Translator] Using layer '{preview_layer.get_name()}' for preview cropping.\n")

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
            sys.stderr.write(f"[Scanlation Translator] Failed to read preview layer pixels: {e}\n")
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
                sys.stderr.write(f"[Scanlation Translator] Failed to read active layer fallback pixels: {active_err}\n")
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

        sys.stderr.write(f"[Scanlation Translator] Reading bounding boxes from path: '{target_path.get_name()}'...\n")

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
            sys.stderr.write(f"[Scanlation Translator] Failed to parse paths/strokes: {e}\n")
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
                sys.stderr.write(f"[Scanlation Translator] Failed to read OCR layers: {ocr_read_err}\n")

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

        if run_mode == Gimp.RunMode.INTERACTIVE:
            from modules.ui.translate_dialog import show_translate_dialog, sort_bubble_states
            initial_reading_order = config.get_property("reading-order") or "Japanese (RTL)"
            sort_bubble_states(bubble_states, initial_reading_order, full_w)
            
            res_val = show_translate_dialog(procedure, config, image, bounding_boxes, bubble_states, crops, full_w, full_h)
            if res_val is None:
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())
            
            payload, included_box_indices = res_val
        else:
            from modules.ui.translate_dialog import sort_bubble_states
            reading_order = config.get_property("reading-order") or "Japanese (RTL)"
            sort_bubble_states(bubble_states, reading_order, full_w)
            
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

        from modules.core.translate_runner import run_translate_processing
        return run_translate_processing(procedure, image, bounding_boxes, payload, included_box_indices, config, run_mode)

    def run_curve_typeset(self, procedure, run_mode, image, drawables, config, run_data):
        """
        Executes Curved Text Typesetting.
        """
        GimpUi.init("gimp-scanlation-curve-typeset")

        if run_mode == Gimp.RunMode.INTERACTIVE:
            from modules.ui.curve_typeset_dialog import show_curve_typeset_dialog
            ok = show_curve_typeset_dialog(procedure, config, image)
            if not ok:
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

        from modules.core.curve_typeset_runner import run_curve_typeset_processing
        return run_curve_typeset_processing(procedure, image, drawables, config)

if __name__ == "__main__":
    Gimp.main(GimpScanlationSuite.__gtype__, sys.argv)
