import gi

gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('GimpUi', '3.0')
from gi.repository import GimpUi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import GLib

def show_inpaint_dialog(procedure, config):
    """
    Builds and displays the interactive Koharu Inpainting Tool dialog.
    Returns True if user presses OK, False if they cancel.
    """
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
    combo_model.append_text("sd-inpainting")
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
    elif model_val == "sd-inpainting":
        combo_model.set_active(2)
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
    dialog.fill([
        "dilation", 
        "api-url", 
        "prompt", 
        "negative-prompt", 
        "steps", 
        "guidance-scale", 
        "auto-prompt", 
        "consensus-arbiter"
    ])
    
    return bool(dialog.run())
