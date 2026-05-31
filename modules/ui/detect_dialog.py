import gi

gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('GimpUi', '3.0')
from gi.repository import GimpUi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import GLib

def show_detect_dialog(procedure, config):
    """
    Builds and displays the interactive Koharu Text & Bubble Detector dialog.
    Returns True if the user pressed OK, False if they canceled/closed.
    """
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
    
    # Automatically populate properties (detector-model, confidence, class-filter)
    dialog.fill(None)
    
    return bool(dialog.run())
