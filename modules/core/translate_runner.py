import sys
import gi

gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
from gi.repository import GLib

try:
    from modules import remote_client
except ImportError as e:
    remote_client = None

def run_translate_processing(procedure, image, bounding_boxes, payload, included_box_indices, config, run_mode):
    """
    Performs API translation queries, creates the 'Translated Text' layer group,
    and renders text layers inside GIMP.
    """
    src_lang = config.get_property("source-lang") or "Japanese"
    tgt_lang = config.get_property("target-lang") or "English"
    api_url = config.get_property("api-url") or "http://localhost:7890"
    trans_model = config.get_property("translation-model") or "DeepSeek"
    enable_thinking = config.get_property("enable-thinking")
    global_ctx = config.get_property("global-context") or ""

    if remote_client is None:
        Gimp.message("Error: Remote client module could not be imported.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    # Execute Translation Request
    translated_results = []
    try:
        Gimp.progress_init(f"Translating dialogue via {trans_model}...")
        
        options = {
            "source_language": src_lang,
            "target_language": tgt_lang,
            "global_context": global_ctx,
            "enable_thinking": enable_thinking
        }
        
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
        sys.stderr.write(f"[Scanlation Translator] Translation failed: {trans_err}\n")
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
        sys.stderr.write(f"[Scanlation Translator] Failed to create layer group: {group_err}\n")
        group_layer = None

    default_font = None
    try:
        if hasattr(Gimp, "Font") and hasattr(Gimp.Font, "get_by_name"):
            default_font = Gimp.Font.get_by_name("Sans-serif")
    except Exception:
        pass

    # Render translations onto new text layers in GIMP
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
            sys.stderr.write(f"[Scanlation Translator] Failed to render bubble {box_idx+1}: {render_err}\n")

    # Display completed GIMP message
    Gimp.message(f"Translation complete! Saved {len(translated_results)} translated layers in the 'Translated Text' layer group. You can now style and position them using standard GIMP text tools.")
    return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())
