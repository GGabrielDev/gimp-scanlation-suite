# Manga Typesetting & Graphic Design Guide

This guide covers professional typesetting standards for manga scanlation and provides a step-by-step walkthrough on how to perform advanced graphic design tasks—such as curving text along a path—using GIMP 3.0.

---

## 1. Professional Manga Lettering Standards

Manga lettering is an art form. To make your scanlation look clean and professional, follow these industry-standard rules:

### A. Font Selection Hierarchy
* **Main Dialogue:** Use a dedicated comic dialogue font. The default **CC Yada Yada Yada** (or similar comic sans-serifs) is designed for readability.
* **Shouting / Exclamations:** Use bold, blocky fonts like **CC Hush Hush** (or similar display/comic title fonts).
* **Narrations & Whispers:** 
  * Narration boxes typically use a clean Serif font (e.g., **Georgia**, **Liberation Serif**) to distinguish narrator voice from dialogue.
  * Whispers/internal thoughts often use small, light sans-serif or italicized fonts.
* **SFX & Onomatopoeia:** Bold, highly stylized serif or block display fonts that match the original hand-drawn Japanese characters.

### B. Speech Bubble Layout Principles
* **Oval Wrapping (Diamond Profile):** Because speech bubbles are generally circular or oval, your text should wrap into a diamond or oval shape (short lines at the top and bottom, longer lines in the middle).
  * *Bad Layout:*
    ```
    I am going to the
    supermarket right
    now.
    ```
  * *Good Layout:*
    ```
    I am going
    to the supermarket
    right now.
    ```
* **Text Centering:** Dialogue text should always be horizontally and vertically centered within the speech bubble.
* **Auto-fit and Safety Margins:** Leave a comfort margin (approx. 10–15% of bubble width) between the text edges and the bubble stroke to ensure the text doesn't feel cramped.

### C. Spacing Adjustments
* **Line Spacing (Leading):** Adjust line spacing depending on layout:
  * For standard dialogue, keep it at default or slightly tight.
  * For stacked vertical text or SFX, pack characters tightly by applying **negative line spacing** (e.g., `-4px` or `-8px`) to make them feel cohesive.
* **Letter Spacing (Tracking):** Increase letter spacing slightly for wide sound effects or titles to add dramatic weight.

---

## 2. Horizontal vs. Vertical Stack Settings

Manga is traditionally written vertically in Japanese, but translated horizontally to English. However, some elements require vertical stacks:

* **When to use Vertical Stacks:** Short exclamations (e.g. "EH?", "WHAT?!", "AH!"), sound effects, or text in narrow, tall vertical speech bubbles.
* **How to format Vertical Stacks:** 
  * Character-by-character stacking:
    ```
    W
    H
    A
    T
    ```
  * In the **Koharu Typesetter** plugin, selecting the **Vertical Stack** alignment automatically formats your text character-by-character and applies tighter negative line spacing (`-4px` to `-8px`) to make sure the English letters stack cleanly without wide gaps.

---

## 3. How to Curve Text in GIMP 3.0 (Text along Path)

Curving text (e.g., around a character's head or conforming to a curved SFX blast) is best done manually using GIMP's vector paths tool, as it gives you complete control over bezier curves.

Follow these step-by-step instructions to curve text in GIMP:

### Step 1: Create the Guide Path
1. Select the **Paths Tool** (shortcut `B`) from the toolbox.
2. Click on the canvas to place start and end control points along the curve where you want the text to go.
3. Click and drag the control handles to bend the path into the desired arc.

### Step 2: Create the Text Layer
1. Select the **Text Tool** (shortcut `T`).
2. Click on the canvas, type your dialogue, and choose your font/size.
3. Keep the text layer selected in the Layers dialog.

### Step 3: Map Text to the Path
1. In the **Layers** panel, right-click on your text layer.
2. Select **"Text along Path"** from the context menu.
3. *What happens:* GIMP will automatically project the letter shapes as vector curves wrapped along the active path. You will see a red outline of the curved letters on your canvas.
4. You can now hide or delete the original flat text layer.

### Step 4: Render the Curved Text
Since GIMP creates a vector outline, you need to stroke or fill it to make it visible:
1. Create a **new transparent layer** (e.g., name it `Curved Text Render`) and select it.
2. Open the **Paths** tab (usually docked next to Layers, or open via `Windows -> Dockable Dialogs -> Paths`).
3. Right-click the newly generated text path (it will be named something like `text_content`) and choose:
   * **Path to Selection:** Creates a selection marquee matching the letters. You can now fill this selection with black, white, or a gradient using the Bucket Fill (`Shift + B`) or Gradient (`G`) tool.
   * **Stroke Path:** Opens a dialog to outline the path with a solid line (great for giving your text a thick comic border/outline).

### Step 5: Adjust and Move
* Use the **Move Tool** (`M`) set to **Path mode** (in Tool Options, toggle "Move: Path" instead of "Layer") if you need to reposition the vector outline before rendering.
