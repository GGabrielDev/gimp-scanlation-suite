import gi

gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('GimpUi', '3.0')
from gi.repository import GimpUi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import GLib

def show_curve_typeset_dialog(procedure, config, image):
    """
    Builds and displays the interactive Curved Text Typesetter dialog.
    Returns True if user presses OK, False if they cancel.
    """
    dialog = GimpUi.ProcedureDialog.new(procedure, config, "Typeset Curved Text")
    vbox = dialog.get_content_area()

    # Premium Header Box
    header_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    header_box.set_margin_top(12)
    header_box.set_margin_bottom(12)
    header_box.set_margin_start(12)
    header_box.set_margin_end(12)

    title_label = Gtk.Label()
    title_label.set_markup("<span size='large' weight='bold' foreground='#3584e4'>Curved Text Typesetter</span>")
    title_label.set_xalign(0.0)
    header_box.pack_start(title_label, False, False, 0)

    desc_label = Gtk.Label()
    desc_label.set_text("Fills and outlines text along a path in GIMP.\n\n"
                         "Instructions:\n"
                         "1. Draw a curve using the Paths Tool (B).\n"
                         "2. Select your text layer, right-click, and select 'Text along Path'.\n"
                         "3. Run this tool to automatically create the Curved Text Fill and Outline layers.")
    desc_label.set_line_wrap(True)
    desc_label.set_xalign(0.0)
    header_box.pack_start(desc_label, False, False, 0)
    vbox.pack_start(header_box, False, False, 0)

    # Custom Grid for path selection and other options
    grid = Gtk.Grid()
    grid.set_column_spacing(12)
    grid.set_row_spacing(12)
    grid.set_margin_start(12)
    grid.set_margin_end(12)
    grid.set_margin_bottom(12)

    # Path Selector
    path_label = Gtk.Label()
    path_label.set_markup("<b>Curved Text Path:</b>")
    path_label.set_xalign(0.0)
    grid.attach(path_label, 0, 0, 1, 1)

    combo_paths = Gtk.ComboBoxText()
    paths = image.get_paths()
    path_names = [p.get_name() for p in paths]
    
    for name in path_names:
        combo_paths.append_text(name)

    # Pre-select most likely path:
    # 1. Any path containing the active layer's name
    # 2. Or the last created path containing "along"
    active_drawable = image.get_selected_drawables()[0] if image.get_selected_drawables() else None
    active_name = active_drawable.get_name() if active_drawable else ""
    
    default_idx = 0
    for idx, name in enumerate(path_names):
        if active_name and active_name in name:
            default_idx = idx
            break
        elif "along" in name.lower() or "text" in name.lower():
            default_idx = idx

    if path_names:
        combo_paths.set_active(default_idx)
        config.set_property("path-name", path_names[default_idx])
    else:
        combo_paths.append_text("No paths found in image")
        combo_paths.set_active(0)

    grid.attach(combo_paths, 1, 0, 1, 1)
    vbox.pack_start(grid, False, False, 0)

    def on_path_changed(widget):
        val = widget.get_active_text()
        if val and val != "No paths found in image":
            config.set_property("path-name", val)
    combo_paths.connect("changed", on_path_changed)

    vbox.show_all()

    # Automatically render remaining arguments (stroke-width, grow-selection, use-text-color)
    dialog.fill(["stroke-width", "grow-selection", "use-text-color"])

    if not path_names:
        Gimp.message("Error: No paths/vectors found in the image. Please right-click a text layer and run 'Text along Path' first.")
        return False

    return bool(dialog.run())
