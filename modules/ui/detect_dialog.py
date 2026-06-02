import gi

gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('GimpUi', '3.0')
from gi.repository import GimpUi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import GObject
from gi.repository import GLib

def show_detect_dialog(procedure, config, image):
    """
    Builds and displays the interactive Text & Bubble Detector dialog with tabs.
    Returns True if the user pressed OK, False if they canceled/closed.
    """
    dialog = GimpUi.ProcedureDialog.new(procedure, config, "Detect Text & Bubbles")
    dialog.set_default_size(450, 480)
    
    vbox = dialog.get_content_area()
    
    # Premium styled header using GTK3 custom layout
    header_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    header_box.set_margin_top(12)
    header_box.set_margin_bottom(12)
    header_box.set_margin_start(12)
    header_box.set_margin_end(12)
    
    title_label = Gtk.Label()
    title_label.set_markup("<span size='large' weight='bold' foreground='#3584e4'>Text &amp; Bubble Detector</span>")
    title_label.set_xalign(0.0)
    header_box.pack_start(title_label, False, False, 0)
    
    desc_label = Gtk.Label()
    desc_label.set_text("Identifies regions of Japanese manga text and speech bubbles on the canvas. Runs YOLO models for initial layout detection, or tightens existing paths around boundaries.")
    desc_label.set_line_wrap(True)
    desc_label.set_xalign(0.0)
    header_box.pack_start(desc_label, False, False, 0)
    
    vbox.pack_start(header_box, False, False, 0)
    
    # Create Gtk.Notebook for tabbed interface
    notebook = Gtk.Notebook()
    notebook.set_margin_start(12)
    notebook.set_margin_end(12)
    notebook.set_margin_bottom(12)
    vbox.pack_start(notebook, True, True, 0)
    
    # ---------------- PAGE 1: Initial Detection ----------------
    page1_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
    page1_box.set_border_width(12)
    
    widget_model = dialog.get_widget("detector-model", GObject.TYPE_NONE)
    widget_conf = dialog.get_widget("confidence", GObject.TYPE_NONE)
    widget_filter = dialog.get_widget("class-filter", GObject.TYPE_NONE)
    
    page1_box.pack_start(widget_model, False, False, 0)
    page1_box.pack_start(widget_conf, False, False, 0)
    page1_box.pack_start(widget_filter, False, False, 0)
    
    label_tab1 = Gtk.Label(label="Initial Detection")
    notebook.append_page(page1_box, label_tab1)
    
    # ---------------- PAGE 2: Tight Pathing ----------------
    page2_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    page2_box.set_border_width(12)
    
    grid = Gtk.Grid()
    grid.set_column_spacing(12)
    grid.set_row_spacing(10)
    page2_box.pack_start(grid, False, False, 0)
    
    # Path selector dropdown
    lbl_paths = Gtk.Label(label="Target Path:")
    lbl_paths.set_xalign(0.0)
    lbl_paths.set_size_request(130, -1)
    grid.attach(lbl_paths, 0, 0, 1, 1)
    
    combo_paths = Gtk.ComboBoxText()
    paths = image.get_paths()
    path_names = [p.get_name() for p in paths]
    for name in path_names:
        combo_paths.append_text(name)
        
    selected_paths = image.get_selected_paths() if hasattr(image, "get_selected_paths") else []
    selected_name = selected_paths[0].get_name() if selected_paths else ""
    
    default_idx = 0
    if selected_name in path_names:
        default_idx = path_names.index(selected_name)
    else:
        for idx, name in enumerate(path_names):
            if "detected" in name.lower() or "bubble" in name.lower():
                default_idx = idx
                break
                
    if path_names:
        combo_paths.set_active(default_idx)
        config.set_property("tight-path-name", path_names[default_idx])
    else:
        combo_paths.append_text("No paths found in image")
        combo_paths.set_active(0)
        config.set_property("tight-path-name", "")
        
    grid.attach(combo_paths, 1, 0, 1, 1)
    
    # Tight path mode selector
    lbl_mode = Gtk.Label(label="Tighten Mode:")
    lbl_mode.set_xalign(0.0)
    grid.attach(lbl_mode, 0, 1, 1, 1)
    
    combo_mode = Gtk.ComboBoxText()
    combo_mode.append_text("Auto (Smart Detection)")
    combo_mode.append_text("Speech Bubble")
    combo_mode.append_text("Floating Text / SFX")
    grid.attach(combo_mode, 1, 1, 1, 1)
    
    # Dilation property slider
    widget_dilation = dialog.get_widget("tight-path-dilation", GObject.TYPE_NONE)
    page2_box.pack_start(widget_dilation, False, False, 0)
    
    label_tab2 = Gtk.Label(label="Tight Pathing")
    notebook.append_page(page2_box, label_tab2)
    
    # Set initial widget values & sensitivities
    mode_val = config.get_property("tight-path-mode") or "Auto"
    if mode_val == "Speech Bubble":
        combo_mode.set_active(1)
        widget_dilation.set_sensitive(False)
    elif mode_val == "Floating Text / SFX":
        combo_mode.set_active(2)
        widget_dilation.set_sensitive(True)
    else:
        combo_mode.set_active(0)
        widget_dilation.set_sensitive(True)
        
    # Toggle dilation sensitivity based on mode
    def update_tight_widgets():
        mode = combo_mode.get_active_text()
        widget_dilation.set_sensitive(mode in ["Auto (Smart Detection)", "Floating Text / SFX"])
        
    # Connect UI signal handlers to config properties
    def on_path_changed(widget):
        val = widget.get_active_text()
        if val and val != "No paths found in image":
            config.set_property("tight-path-name", val)
    combo_paths.connect("changed", on_path_changed)
    
    def on_mode_changed(widget):
        val = widget.get_active_text()
        if val:
            mapped_val = "Auto" if "Auto" in val else val
            config.set_property("tight-path-mode", mapped_val)
            update_tight_widgets()
    combo_mode.connect("changed", on_mode_changed)
    
    # Sync notebook tab switch with detection-mode property
    def on_page_switched(widget, page, page_num):
        if page_num == 0:
            config.set_property("detection-mode", "Initial Detection")
        else:
            config.set_property("detection-mode", "Tight Pathing")
            
    notebook.connect("switch-page", on_page_switched)
    
    # Set default notebook tab page from previous session
    det_mode = config.get_property("detection-mode") or "Initial Detection"
    if det_mode == "Tight Pathing" and path_names:
        notebook.set_current_page(1)
        config.set_property("detection-mode", "Tight Pathing")
    else:
        notebook.set_current_page(0)
        config.set_property("detection-mode", "Initial Detection")
        
    vbox.show_all()
    
    # Prevent duplicate automatic properties rendering
    dialog.fill([])
    
    return bool(dialog.run())
