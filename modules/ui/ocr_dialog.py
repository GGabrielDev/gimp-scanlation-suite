import sys
import os
import threading
import gi

gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('GimpUi', '3.0')
from gi.repository import GimpUi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import GObject
from gi.repository import GLib

def show_ocr_dialog(procedure, config, image, bounding_boxes):
    """
    Builds and displays the interactive Manga OCR dialog.
    Returns:
        list of selected indices, or None if the dialog was canceled.
    """
    dialog = GimpUi.ProcedureDialog.new(procedure, config, "OCR Selected Blocks")
    dialog.set_default_size(600, 700)
    vbox = dialog.get_content_area()
    
    # Main scrolling window to prevent layout overflow on smaller screens
    scrolled_window = Gtk.ScrolledWindow()
    scrolled_window.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    scrolled_window.set_propagate_natural_width(True)
    scrolled_window.set_propagate_natural_height(True)
    
    # Internal vertical container
    scroll_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    scrolled_window.add(scroll_vbox)
    vbox.pack_start(scrolled_window, True, True, 0)
    
    # Premium Header Box
    header_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    header_box.set_margin_top(12)
    header_box.set_margin_bottom(12)
    header_box.set_margin_start(12)
    header_box.set_margin_end(12)
    
    title_label = Gtk.Label()
    title_label.set_markup("<span size='large' weight='bold' foreground='#3584e4'>Manga OCR Engine</span>")
    title_label.set_xalign(0.0)
    header_box.pack_start(title_label, False, False, 0)
    
    desc_label = Gtk.Label()
    desc_label.set_text("Performs optical character recognition (OCR) on text regions using local or remote VLM inference.")
    desc_label.set_line_wrap(True)
    desc_label.set_xalign(0.0)
    header_box.pack_start(desc_label, False, False, 0)
    scroll_vbox.pack_start(header_box, False, False, 0)

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
    
    scroll_vbox.pack_start(grid, False, False, 0)

    # Regions Checklist Frame
    regions_frame = Gtk.Frame(label="  Regions to Process (OCR Checklist)  ")
    regions_frame.set_margin_start(12)
    regions_frame.set_margin_end(12)
    regions_frame.set_margin_bottom(12)
    
    regions_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    regions_vbox.set_margin_top(8)
    regions_vbox.set_margin_bottom(8)
    regions_vbox.set_margin_start(12)
    regions_vbox.set_margin_end(12)
    
    # Select all / Deselect all buttons
    btn_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    btn_select_all = Gtk.Button(label="Select All")
    btn_deselect_all = Gtk.Button(label="Deselect All")
    btn_hbox.pack_start(btn_select_all, False, False, 0)
    btn_hbox.pack_start(btn_deselect_all, False, False, 0)
    regions_vbox.pack_start(btn_hbox, False, False, 2)
    
    # Scrollable list of regions
    scroll_regions = Gtk.ScrolledWindow()
    scroll_regions.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
    scroll_regions.set_size_request(-1, 150)
    scroll_regions.set_min_content_height(100)
    
    listbox_regions = Gtk.ListBox()
    listbox_regions.set_selection_mode(Gtk.SelectionMode.NONE)
    
    checkboxes = []
    for idx, box in enumerate(bounding_boxes):
        xmin, ymin, xmax, ymax = box
        w = int(xmax - xmin)
        h = int(ymax - ymin)
        
        row_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        row_hbox.set_margin_top(4)
        row_hbox.set_margin_bottom(4)
        row_hbox.set_margin_start(6)
        row_hbox.set_margin_end(6)
        
        chk = Gtk.CheckButton(label=f"Region {idx + 1}: {w}x{h} at ({int(xmin)}, {int(ymin)})")
        chk.set_active(True)
        row_hbox.pack_start(chk, True, True, 0)
        checkboxes.append(chk)
        
        btn_focus = Gtk.Button(label="🔍 Select")
        btn_focus.set_tooltip_text("Highlight this region on the GIMP canvas")
        
        def on_focus_clicked(btn, b=box):
            xmin_b, ymin_b, xmax_b, ymax_b = b
            try:
                Gimp.Image.select_rectangle(
                    image,
                    Gimp.ChannelOps.REPLACE,
                    float(xmin_b), float(ymin_b),
                    float(xmax_b - xmin_b), float(ymax_b - ymin_b)
                )
                Gimp.displays_flush()
            except Exception as select_err:
                sys.stderr.write(f"[Scanlation OCR] Failed to select region: {select_err}\n")
        
        btn_focus.connect("clicked", on_focus_clicked)
        row_hbox.pack_start(btn_focus, False, False, 0)
        
        row = Gtk.ListBoxRow()
        row.add(row_hbox)
        listbox_regions.add(row)
        
    scroll_regions.add(listbox_regions)
    regions_vbox.pack_start(scroll_regions, True, True, 0)
    
    # Select/Deselect All signal handlers
    def on_select_all_clicked(btn):
        for chk in checkboxes:
            chk.set_active(True)
    
    def on_deselect_all_clicked(btn):
        for chk in checkboxes:
            chk.set_active(False)
            
    btn_select_all.connect("clicked", on_select_all_clicked)
    btn_deselect_all.connect("clicked", on_deselect_all_clicked)
    
    regions_frame.add(regions_vbox)
    scroll_vbox.pack_start(regions_frame, False, False, 0)
    
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

    # Populating dropdown safely in the idle loop
    def populate_dropdown(ocr_models, arb_models):
        combo_model.remove_all()
        for m in ocr_models:
            combo_model.append(m["model_id"], m["display_name"])
        
        ocr_ids = [m["model_id"] for m in ocr_models]
        stored_model = config.get_property("ocr-engine")
        if stored_model in ocr_ids:
            combo_model.set_active_id(stored_model)
        else:
            if "PaddleOCR_Manga" in ocr_ids:
                combo_model.set_active_id("PaddleOCR_Manga")
            elif "PaddleOCR" in ocr_ids:
                combo_model.set_active_id("PaddleOCR")
            elif ocr_ids:
                combo_model.set_active(0)

        combo_expert_b.remove_all()
        for m in ocr_models:
            combo_expert_b.append(m["model_id"], m["display_name"])
        stored_expert_b = config.get_property("consensus-expert-b")
        if stored_expert_b in ocr_ids:
            combo_expert_b.set_active_id(stored_expert_b)
        else:
            if "PaddleOCR_Manga" in ocr_ids:
                combo_expert_b.set_active_id("PaddleOCR_Manga")
            elif "PaddleOCR" in ocr_ids:
                combo_expert_b.set_active_id("PaddleOCR")
            elif ocr_ids:
                combo_expert_b.set_active(0)

        combo_arbiter.remove_all()
        for m in arb_models:
            combo_arbiter.append(m["model_id"], m["display_name"])
        stored_arbiter = config.get_property("consensus-arbiter")
        arb_ids = [m["model_id"] for m in arb_models]
        if stored_arbiter in arb_ids:
            combo_arbiter.set_active_id(stored_arbiter)
        else:
            if "DeepSeek-V4-Flash" in arb_ids:
                combo_arbiter.set_active_id("DeepSeek-V4-Flash")
            elif "DeepSeek" in arb_ids:
                combo_arbiter.set_active_id("DeepSeek")
            elif "JP_Arbiter_8B" in arb_ids:
                combo_arbiter.set_active_id("JP_Arbiter_8B")
            elif arb_ids:
                combo_arbiter.set_active(0)
        
        # Update tooltips initially
        update_tooltip_and_desc(combo_model)
        update_tooltip_and_desc(combo_expert_b)
        update_tooltip_and_desc(combo_arbiter)
        
        update_thinking_sensitivity()
        update_consensus_sensitivity()
        return False
    
    def load_remote_models_bg():
        from modules import remote_client
        api_url = config.get_property("api-url") or "http://localhost:7890"
        ocr_models = remote_client.get_available_models("ocr_expert", api_url)
        arb_models = remote_client.get_available_models("ocr_arbiter", api_url)
        GLib.idle_add(populate_dropdown, ocr_models, arb_models)

    def update_model_dropdown():
        current_mode = config.get_property("inference-mode")
        if current_mode == "Remote":
            t = threading.Thread(target=load_remote_models_bg)
            t.daemon = True
            t.start()
        else:
            # Local mode fallbacks using metadata registry
            from modules import remote_client
            ocr_models = remote_client.get_available_models("ocr_expert", "")
            arb_models = remote_client.get_available_models("ocr_arbiter", "")
            
            # Ensure "PaddleOCR" local engine option is available
            local_ocr = {"model_id": "PaddleOCR", "display_name": "PaddleOCR (Local)", "description": "Local GGUF model optimized for Japanese text extraction"}
            if not any(m["model_id"] == "PaddleOCR" for m in ocr_models):
                ocr_models = [local_ocr] + ocr_models
                
            populate_dropdown(ocr_models, arb_models)

    def on_inf_changed(widget):
        val = widget.get_active_text()
        config.set_property("inference-mode", val)
        update_model_dropdown()
        
    combo_inf.connect("changed", on_inf_changed)

    def on_model_changed(widget):
        val = widget.get_active_id()
        if val:
            config.set_property("ocr-engine", val)
        update_tooltip_and_desc(widget)
        update_thinking_sensitivity()
            
    combo_model.connect("changed", on_model_changed)

    def on_expert_b_changed(widget):
        val = widget.get_active_id()
        if val:
            config.set_property("consensus-expert-b", val)
        update_tooltip_and_desc(widget)

    combo_expert_b.connect("changed", on_expert_b_changed)

    def on_arbiter_changed(widget):
        val = widget.get_active_id()
        if val:
            config.set_property("consensus-arbiter", val)
        update_tooltip_and_desc(widget)
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
    
    # Fetch additional GIMP procedure dialog auto-generated settings
    widget_api_url = dialog.get_widget("api-url", GObject.TYPE_NONE)
    widget_target_lang = dialog.get_widget("target-language", GObject.TYPE_NONE)
    widget_ensemble = dialog.get_widget("ensemble-consensus", GObject.TYPE_NONE)
    widget_configure_per_path = dialog.get_widget("configure-per-path", GObject.TYPE_NONE)
    widget_half_to_full = dialog.get_widget("half-to-full", GObject.TYPE_NONE)

    # Wrap these additional settings in a styled Frame
    settings_frame = Gtk.Frame(label="  System & Post-Processing Settings  ")
    settings_frame.set_margin_start(12)
    settings_frame.set_margin_end(12)
    settings_frame.set_margin_bottom(12)
    
    settings_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    settings_vbox.set_margin_top(8)
    settings_vbox.set_margin_bottom(8)
    settings_vbox.set_margin_start(12)
    settings_vbox.set_margin_end(12)
    
    settings_vbox.pack_start(widget_api_url, False, False, 0)
    settings_vbox.pack_start(widget_target_lang, False, False, 0)
    settings_vbox.pack_start(widget_ensemble, False, False, 0)
    settings_vbox.pack_start(widget_configure_per_path, False, False, 0)
    settings_vbox.pack_start(widget_half_to_full, False, False, 0)
    
    settings_frame.add(settings_vbox)
    scroll_vbox.pack_start(settings_frame, False, False, 0)
    
    vbox.show_all()
    
    if not dialog.run():
        return None

    selected_indices = [i for i, chk in enumerate(checkboxes) if chk.get_active()]
    sys.stderr.write(f"[Scanlation OCR] Dialog closed. Checkbox states: {[chk.get_active() for chk in checkboxes]}\n")
    sys.stderr.write(f"[Scanlation OCR] Selected indices returned: {selected_indices}\n")
    return selected_indices
