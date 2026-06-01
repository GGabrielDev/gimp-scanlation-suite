import gi

gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('GimpUi', '3.0')
from gi.repository import GimpUi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import GObject
from gi.repository import GLib

def show_inpaint_dialog(procedure, config):
    """
    Builds and displays the interactive Inpainting Tool dialog.
    Returns True if user presses OK, False if they cancel.
    """
    dialog = GimpUi.ProcedureDialog.new(procedure, config, "Inpaint / Erase Text")
    dialog.set_default_size(600, 680)
    
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
    
    # Header
    header_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    header_box.set_margin_top(12)
    header_box.set_margin_bottom(12)
    header_box.set_margin_start(12)
    header_box.set_margin_end(12)
    
    title_label = Gtk.Label()
    title_label.set_markup("<span size='large' weight='bold' foreground='#3584e4'>Inpainting Tool</span>")
    title_label.set_xalign(0.0)
    header_box.pack_start(title_label, False, False, 0)
    
    desc_label = Gtk.Label()
    desc_label.set_text("Cleans text regions from active layers by applying a Fast Fourier Transform-based LaMa/AOT-Inpainting model or Stable Diffusion to fill the text masks natively.")
    desc_label.set_line_wrap(True)
    desc_label.set_xalign(0.0)
    header_box.pack_start(desc_label, False, False, 0)
    
    scroll_vbox.pack_start(header_box, False, False, 0)

    # Combo box for Inference Mode
    combo_inf = Gtk.ComboBoxText()
    combo_inf.append_text("Local")
    combo_inf.append_text("Remote")
    
    # Combo box for Inpaint Model (Including new SD models)
    combo_model = Gtk.ComboBoxText()
    combo_model.append_text("lama-manga")
    combo_model.append_text("aot-inpainting")
    combo_model.append_text("sd-inpainting")
    combo_model.append_text("anime-inpaint")
    combo_model.append_text("sdxl-inpainting")

    # VLM Arbiter custom dropdown (Choose only between VLM models)
    combo_arbiter = Gtk.ComboBoxText()
    vlm_models = ["olmOCR2_Q4", "olmOCR2_Q6", "olmOCR2_Q8", "PaddleOCR_Manga"]
    for m in vlm_models:
        combo_arbiter.append_text(m)

    # Frame 1: General Settings
    gen_frame = Gtk.Frame(label="  General Settings  ")
    gen_frame.set_margin_start(12)
    gen_frame.set_margin_end(12)
    gen_frame.set_margin_bottom(12)
    
    gen_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    gen_vbox.set_margin_top(8)
    gen_vbox.set_margin_bottom(8)
    gen_vbox.set_margin_start(12)
    gen_vbox.set_margin_end(12)
    
    grid_gen = Gtk.Grid()
    grid_gen.set_column_spacing(12)
    grid_gen.set_row_spacing(8)
    
    # Inference Mode row
    lbl_inf = Gtk.Label(label="Inference Mode:")
    lbl_inf.set_xalign(0.0)
    lbl_inf.set_size_request(130, -1)
    grid_gen.attach(lbl_inf, 0, 0, 1, 1)
    grid_gen.attach(combo_inf, 1, 0, 1, 1)
    
    # Inpaint Model row
    lbl_model = Gtk.Label(label="Inpaint Model:")
    lbl_model.set_xalign(0.0)
    lbl_model.set_size_request(130, -1)
    grid_gen.attach(lbl_model, 0, 1, 1, 1)
    grid_gen.attach(combo_model, 1, 1, 1, 1)
    
    gen_vbox.pack_start(grid_gen, False, False, 0)
    
    # Dilation
    widget_dilation = dialog.get_widget("dilation", GObject.TYPE_NONE)
    gen_vbox.pack_start(widget_dilation, False, False, 0)
    
    gen_frame.add(gen_vbox)
    scroll_vbox.pack_start(gen_frame, False, False, 0)

    # Frame 2: VLM Auto-Prompting Settings
    vlm_frame = Gtk.Frame(label="  VLM Auto-Prompting Settings  ")
    vlm_frame.set_margin_start(12)
    vlm_frame.set_margin_end(12)
    vlm_frame.set_margin_bottom(12)
    
    vlm_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    vlm_vbox.set_margin_top(8)
    vlm_vbox.set_margin_bottom(8)
    vlm_vbox.set_margin_start(12)
    vlm_vbox.set_margin_end(12)
    
    # Auto Prompt via VLM CheckButton
    widget_auto_prompt = dialog.get_widget("auto-prompt", GObject.TYPE_NONE)
    vlm_vbox.pack_start(widget_auto_prompt, False, False, 0)
    
    # VLM Arbiter Dropdown row
    hbox_arbiter = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
    lbl_arb = Gtk.Label(label="VLM Arbiter:")
    lbl_arb.set_xalign(0.0)
    lbl_arb.set_size_request(130, -1)
    hbox_arbiter.pack_start(lbl_arb, False, False, 0)
    hbox_arbiter.pack_start(combo_arbiter, True, True, 0)
    vlm_vbox.pack_start(hbox_arbiter, False, False, 0)
    
    # API URL Entry (comes right after VLM Arbiter!)
    widget_api_url = dialog.get_widget("api-url", GObject.TYPE_NONE)
    vlm_vbox.pack_start(widget_api_url, False, False, 0)
    
    vlm_frame.add(vlm_vbox)
    scroll_vbox.pack_start(vlm_frame, False, False, 0)

    # Frame 3: Diffusion Model Settings
    diff_frame = Gtk.Frame(label="  Diffusion Model Settings  ")
    diff_frame.set_margin_start(12)
    diff_frame.set_margin_end(12)
    diff_frame.set_margin_bottom(12)
    
    diff_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    diff_vbox.set_margin_top(8)
    diff_vbox.set_margin_bottom(8)
    diff_vbox.set_margin_start(12)
    diff_vbox.set_margin_end(12)
    
    widget_prompt = dialog.get_widget("prompt", GObject.TYPE_NONE)
    widget_neg_prompt = dialog.get_widget("negative-prompt", GObject.TYPE_NONE)
    widget_steps = dialog.get_widget("steps", GObject.TYPE_NONE)
    widget_guidance = dialog.get_widget("guidance-scale", GObject.TYPE_NONE)
    
    diff_vbox.pack_start(widget_prompt, False, False, 0)
    diff_vbox.pack_start(widget_neg_prompt, False, False, 0)
    diff_vbox.pack_start(widget_steps, False, False, 0)
    diff_vbox.pack_start(widget_guidance, False, False, 0)
    
    diff_frame.add(diff_vbox)
    scroll_vbox.pack_start(diff_frame, False, False, 0)

    # Set initial values based on config
    inf_val = config.get_property("inference-mode") or "Local"
    if inf_val == "Remote":
        combo_inf.set_active(1)
    else:
        combo_inf.set_active(0)

    model_val = config.get_property("inpaint-model") or "lama-manga"
    models_list = ["lama-manga", "aot-inpainting", "sd-inpainting", "anime-inpaint", "sdxl-inpainting"]
    if model_val in models_list:
        combo_model.set_active(models_list.index(model_val))
    else:
        combo_model.set_active(0)

    stored_arbiter = config.get_property("consensus-arbiter") or "olmOCR2_Q4"
    if stored_arbiter in vlm_models:
        combo_arbiter.set_active(vlm_models.index(stored_arbiter))
    else:
        combo_arbiter.set_active(0)

    # Active states update callback
    def update_widget_states():
        # 1. Inference mode controls API URL visibility/sensitivity
        is_remote = (combo_inf.get_active_text() == "Remote")
        widget_api_url.set_sensitive(is_remote)
        
        # 2. Model type controls diffusion fields
        model_name = combo_model.get_active_text()
        is_diffusion = model_name in ["sd-inpainting", "anime-inpaint", "sdxl-inpainting"]
        
        widget_auto_prompt.set_sensitive(is_diffusion)
        
        if is_diffusion:
            is_auto_prompt = widget_auto_prompt.get_active()
            combo_arbiter.set_sensitive(is_auto_prompt)
            widget_prompt.set_sensitive(not is_auto_prompt)
            
            widget_neg_prompt.set_sensitive(True)
            widget_steps.set_sensitive(True)
            widget_guidance.set_sensitive(True)
        else:
            combo_arbiter.set_sensitive(False)
            widget_prompt.set_sensitive(False)
            widget_neg_prompt.set_sensitive(False)
            widget_steps.set_sensitive(False)
            widget_guidance.set_sensitive(False)

    # Connect signals
    def on_inf_changed(widget):
        val = widget.get_active_text()
        config.set_property("inference-mode", val)
        update_widget_states()
    combo_inf.connect("changed", on_inf_changed)

    def on_model_changed(widget):
        val = widget.get_active_text()
        config.set_property("inpaint-model", val)
        update_widget_states()
    combo_model.connect("changed", on_model_changed)

    def on_arbiter_changed(widget):
        val = widget.get_active_text()
        if val:
            config.set_property("consensus-arbiter", val)
    combo_arbiter.connect("changed", on_arbiter_changed)

    def on_auto_prompt_toggled(widget):
        update_widget_states()
    widget_auto_prompt.connect("toggled", on_auto_prompt_toggled)

    # Initialize states
    update_widget_states()

    vbox.show_all()
    
    return bool(dialog.run())
