import sys
import threading
import numpy as np
import gi

gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import GObject
from gi.repository import GLib

def sort_bubble_states(bubble_states, heuristic, full_w):
    """
    Sorts bubble_states in-place based on the reading order heuristic.
    """
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

def show_translate_dialog(procedure, config, image, bounding_boxes, bubble_states, crops, full_w, full_h):
    """
    Builds and displays the Gtk Translation and Character Mapping dialog.
    Returns:
        tuple (payload, included_box_indices) if accepted, or None if canceled.
    """
    dialog = Gtk.Dialog(title="Translation & Typesetting", parent=None, flags=0)
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
    title_label.set_markup("<span size='large' weight='bold' foreground='#3584e4'>Typesetting &amp; Translation</span>")
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

    def save_current_edits():
        for row_idx, (combo_spk, txt_src, entry_hint, chk_exclude, _) in enumerate(row_widgets):
            state = bubble_states[row_idx]
            state["speaker"] = combo_spk.get_active_text() or "Unassigned / Narrative"
            buf = txt_src.get_buffer()
            state["text"] = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), True)
            state["context"] = entry_hint.get_text()
            state["skip"] = chk_exclude.get_active()

    def move_bubble_state(index, offset):
        save_current_edits()
        target_idx = index + offset
        if 0 <= target_idx < len(bubble_states):
            bubble_states[index], bubble_states[target_idx] = bubble_states[target_idx], bubble_states[index]
            crops[index], crops[target_idx] = crops[target_idx], crops[index]
            rebuild_dialogue_queue()
            box_blocks.show_all()

    def rebuild_dialogue_queue():
        for child in box_blocks.get_children():
            child.destroy()
        del row_widgets[:]
        
        for idx, state in enumerate(bubble_states):
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
            
            # Source Text
            scroll_src = Gtk.ScrolledWindow()
            scroll_src.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
            scroll_src.set_size_request(300, 45)
            txt_src = Gtk.TextView()
            txt_src.set_wrap_mode(Gtk.WrapMode.WORD)
            buf_src = txt_src.get_buffer()
            buf_src.set_text(state["text"])
            scroll_src.add(txt_src)
            
            grid_row.attach(scroll_src, 1, 1, 3, 1)
            
            # Up/Down reordering buttons box
            btn_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            btn_box.set_valign(Gtk.Align.CENTER)
            
            btn_up = Gtk.Button(label="▲")
            btn_up.set_tooltip_text("Move Up in Reading Order")
            btn_up.set_sensitive(idx > 0)
            
            btn_down = Gtk.Button(label="▼")
            btn_down.set_tooltip_text("Move Down in Reading Order")
            btn_down.set_sensitive(idx < len(bubble_states) - 1)
            
            btn_box.pack_start(btn_up, False, False, 0)
            btn_box.pack_start(btn_down, False, False, 0)
            
            grid_row.attach(btn_box, 4, 0, 1, 2)
            
            def make_move_up_cb(i):
                return lambda button: move_bubble_state(i, -1)
            def make_move_down_cb(i):
                return lambda button: move_bubble_state(i, 1)
                
            btn_up.connect("clicked", make_move_up_cb(idx))
            btn_down.connect("clicked", make_move_down_cb(idx))
            
            frame.add(grid_row)
            box_blocks.pack_start(frame, False, False, 0)
            
            row_widgets.append((combo_spk, txt_src, entry_hint, chk_exclude, refresh_speakers))

    # AI Pre-analysis control bar
    queue_top_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
    queue_top_box.set_margin_bottom(6)
    
    btn_analyze = Gtk.Button(label="✨ Pre-Analyze Scene with AI")
    btn_analyze.set_tooltip_text("Query LLM to identify speakers, assign dialogue types, and suggest context/emotional hints")
    queue_top_box.pack_start(btn_analyze, False, False, 0)
    
    lbl_status = Gtk.Label(label="")
    lbl_status.set_xalign(0.0)
    queue_top_box.pack_start(lbl_status, True, True, 0)
    
    def set_status(msg):
        lbl_status.set_text(msg)
        
    def enable_analyze_btn():
        btn_analyze.set_sensitive(True)
        
    def apply_scene_analysis(data):
        characters = data.get("characters", [])
        for c_idx, name in enumerate(characters[:5]):
            char_entries[c_idx].set_text(name)
        on_char_name_changed(None)
        
        analysis_list = data.get("analysis", [])
        for item in analysis_list:
            idx = item.get("index") - 1
            if 0 <= idx < len(bubble_states):
                speaker = item.get("speaker")
                if speaker == "SFX":
                    state_speaker = "SFX / Onomatopoeia"
                elif not speaker or speaker == "Narrative" or speaker == "Unassigned / Narrative":
                    state_speaker = "Unassigned / Narrative"
                else:
                    state_speaker = speaker
                    
                bubble_states[idx]["speaker"] = state_speaker
                bubble_states[idx]["context"] = item.get("context", "")
                
        rebuild_dialogue_queue()
        box_blocks.show_all()
        lbl_status.set_text("Scene analysis complete! Speakers and hints updated.")
        
    def run_scene_analysis_bg():
        save_current_edits()
        payload = []
        for idx, state in enumerate(bubble_states):
            payload.append({
                "index": idx + 1,
                "text": state["text"]
            })
            
        from modules import remote_client
        api_url = entry_api.get_text().strip() or "http://localhost:7890"
        trans_model = combo_model.get_active_text() or "DeepSeek"
        enable_thinking = chk_thinking.get_active()
        options = {
            "analyze_scene": True,
            "enable-thinking": enable_thinking
        }
        
        try:
            results = remote_client.dispatch_batch("translate", trans_model, payload, api_url, options=options)
            if results:
                analysis_json = results[0]
                import json
                data = json.loads(analysis_json)
                GLib.idle_add(apply_scene_analysis, data)
            else:
                GLib.idle_add(set_status, "Server returned no results.")
        except Exception as err:
            GLib.idle_add(set_status, f"Analysis failed: {err}")
        finally:
            GLib.idle_add(enable_analyze_btn)
            
    def on_analyze_clicked(button):
        btn_analyze.set_sensitive(False)
        lbl_status.set_text("Analyzing scene with AI... Please wait.")
        t = threading.Thread(target=run_scene_analysis_bg)
        t.daemon = True
        t.start()
        
    btn_analyze.connect("clicked", on_analyze_clicked)

    rebuild_dialogue_queue()

    def on_reading_order_changed(widget):
        save_current_edits()
        new_ro = combo_ro.get_active_text()
        sort_bubble_states(bubble_states, new_ro, full_w)
        rebuild_dialogue_queue()
        box_blocks.show_all()
    
    combo_ro.connect("changed", on_reading_order_changed)

    def on_char_name_changed(widget):
        for row in row_widgets:
            row[4]()
    
    for ent in char_entries:
        ent.connect("changed", on_char_name_changed)
        
    scroll_blocks.add(box_blocks)
    
    queue_page_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    queue_page_vbox.set_border_width(12)
    queue_page_vbox.pack_start(queue_top_box, False, False, 0)
    queue_page_vbox.pack_start(scroll_blocks, True, True, 0)
    
    notebook.append_page(queue_page_vbox, Gtk.Label(label="Dialogue Queue"))
    
    def update_tooltip_and_desc(combo):
        model_id = combo.get_active_id()
        if model_id:
            from modules import remote_client
            metadata = remote_client.get_model_metadata(model_id)
            if metadata:
                desc = metadata.get("description", "")
                combo.set_tooltip_text(desc)
            else:
                combo.set_tooltip_text("")
        else:
            combo.set_tooltip_text("")

    # Dynamic Model Populating
    def populate_dropdown(models):
        combo_model.remove_all()
        for m in models:
            combo_model.append(m["model_id"], m["display_name"])
        
        model_ids = [m["model_id"] for m in models]
        stored_model = config.get_property("translation-model") or "DeepSeek"
        if stored_model in model_ids:
            combo_model.set_active_id(stored_model)
        else:
            if "DeepSeek" in model_ids:
                combo_model.set_active_id("DeepSeek")
            elif "JP_Arbiter_8B" in model_ids:
                combo_model.set_active_id("JP_Arbiter_8B")
            elif model_ids:
                combo_model.set_active(0)
        
        update_tooltip_and_desc(combo_model)
        update_thinking_sensitivity()
    
    def update_thinking_sensitivity():
        active_model = combo_model.get_active_id() or ""
        is_ds = "deepseek" in active_model.lower()
        chk_thinking.set_sensitive(is_ds)
    
    def on_model_changed(widget):
        update_tooltip_and_desc(widget)
        update_thinking_sensitivity()

    combo_model.connect("changed", on_model_changed)
    
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
            from modules import remote_client
            models = remote_client.get_available_models("translate", "")
            populate_dropdown(models)
    
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
        trans_model = combo_model.get_active_id()
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
        return payload, included_box_indices
    else:
        dialog.destroy()
        return None
