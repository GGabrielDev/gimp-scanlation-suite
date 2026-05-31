import gi

gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('GLib', '2.0')
from gi.repository import GLib

def run_curve_typeset_processing(procedure, image, drawables, config):
    """
    Runner for the curved text typesetter.
    """
    if not drawables:
        Gimp.message("Error: No active drawable/layer selected.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())
        
    text_layer = drawables[0]
    path_name = config.get_property("path-name")
    stroke_width = config.get_property("stroke-width") or 4
    grow_selection = config.get_property("grow-selection")
    use_text_color = config.get_property("use-text-color")

    if not path_name:
        Gimp.message("Error: No target path selected in configuration.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    # 1. Locate the path object
    target_path = None
    for p in image.get_paths():
        if p.get_name() == path_name:
            target_path = p
            break

    if not target_path:
        Gimp.message(f"Error: Path '{path_name}' not found in the image.")
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error())

    # 2. Determine text and outline colors
    Gimp.context_push()
    
    text_color = Gimp.context_get_foreground()  # fallback default
    if use_text_color:
        try:
            # Query text color from GIMP PDB procedure
            success, color = Gimp.get_pdb().run_procedure('gimp-text-layer-get-color', [text_layer])
            if success:
                # The returned color value is at index 1 of the GValueArray
                text_color = color.index(1) if hasattr(color, 'index') else color
        except Exception:
            pass

    outline_color = Gimp.context_get_background()

    # 3. Create Curved Text Layer Group
    group_name = f"[Curved Text] {text_layer.get_name()}"
    
    # Remove existing group if name collisions occur to allow quick retries
    for layer in image.get_layers():
        if layer.get_name() == group_name:
            image.remove_layer(layer)
            break

    group_layer = Gimp.GroupLayer.new(image)
    group_layer.set_name(group_name)

    # Insert group layer directly above the original text layer
    parent = text_layer.get_parent()
    siblings = parent.get_children() if parent else image.get_layers()
    try:
        idx = siblings.index(text_layer)
        image.insert_layer(group_layer, parent, idx)
    except ValueError:
        image.insert_layer(group_layer, parent, 0)

    # 4. Create Curved Text Fill Layer
    img_w = image.get_width()
    img_h = image.get_height()
    
    fill_layer = Gimp.Layer.new(
        image,
        "Curved Text Fill",
        img_w,
        img_h,
        Gimp.ImageType.RGBA_IMAGE,
        100.0,
        Gimp.LayerMode.NORMAL
    )
    image.insert_layer(fill_layer, group_layer, 0)

    # Select the path as selection
    image.select_item(Gimp.ChannelOps.REPLACE, target_path)

    # Fill selection with text color
    Gimp.context_set_foreground(text_color)
    fill_layer.edit_fill(Gimp.FillType.FOREGROUND)

    # 5. Create Curved Text Outline Layer
    outline_layer = Gimp.Layer.new(
        image,
        "Curved Text Outline",
        img_w,
        img_h,
        Gimp.ImageType.RGBA_IMAGE,
        100.0,
        Gimp.LayerMode.NORMAL
    )
    image.insert_layer(outline_layer, group_layer, 1)

    if grow_selection:
        # Method A: Grow Selection
        Gimp.Selection.grow(image, stroke_width)
        Gimp.context_set_background(outline_color)
        outline_layer.edit_fill(Gimp.FillType.BACKGROUND)
    else:
        # Method B: Direct Path Stroking (Strokes use active Foreground color)
        image.select_item(Gimp.ChannelOps.REPLACE, target_path)
        Gimp.context_set_line_width(stroke_width)
        Gimp.context_set_foreground(outline_color)
        outline_layer.edit_stroke_item(target_path)

    # 6. Clean up selection and hide original text layer
    Gimp.Selection.none(image)
    
    try:
        text_layer.set_visible(False)
    except Exception:
        pass

    Gimp.context_pop()
    Gimp.displays_flush()

    Gimp.message(f"Curved text successfully typeset into group: '{group_name}'")
    return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())
