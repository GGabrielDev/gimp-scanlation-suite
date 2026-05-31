# Manga Typesetting & Graphic Design Guide (GIMP 3.0)

This guide covers professional typesetting standards for manga scanlation and provides a step-by-step walkthrough on how to use GIMP 3.0's native tools to achieve industry-standard, high-quality manga lettering.

---

## 1. Professional Manga Lettering Standards

Manga lettering is a critical part of scanlation. To make your work look clean and professional, follow these typesetting conventions:

### A. Font Selection Hierarchy
* **Main Dialogue:** Use dedicated comic dialogue fonts (e.g., **CCYadaYadaYada**, **Wild Words**, **Comic Sans** variant). These are designed with wide horizontal proportions to fit speech bubbles comfortably.
* **Shouting / Exclamations:** Use bold, hand-drawn-style title fonts (e.g., **CCHushHush**, **Komics**, **Impact**).
* **Narrations & Boxes:** 
  * Narration boxes typically use a clean Serif font (e.g., **Georgia**, **Liberation Serif**, **Times New Roman**) to distinguish the narrator's voice from dialogue.
  * Internal thoughts/whispers often use italicized, thin, or small sans-serif fonts.
* **SFX & Onomatopoeia:** Bold, highly stylized decorative fonts that mimic the energy of the original Japanese hand-drawn sound effects.

### B. Speech Bubble Layout (Diamond Profile)
Because speech bubbles are circular or oval, your text should wrap into a **diamond or oval shape** (shorter lines at the top and bottom, longer lines in the middle).
* *Bad Layout:*
  ```
  I am going to the
  supermarket right
  now.
  ```
* *Good Layout (Centered & Balanced):*
  ```
  I am going
  to the supermarket
  right now.
  ```

---

## 2. Formatting Text Layers in GIMP

To input and style text manually in GIMP, use the following techniques:

### A. Text Box Modes: Dynamic vs. Fixed
1. Select the **Text Tool** (shortcut `T`).
2. Click on the canvas to create a text layer.
3. In GIMP's **Tool Options** panel (usually on the left):
   * **Dynamic:** The text box expands automatically as you type. Use this for short sound effects or labels.
   * **Fixed:** Click and drag on the canvas to draw a box. The text will wrap automatically when it reaches the borders. Use this for standard dialogue bubbles. You can resize the box using the square handles on the corners.

### B. Spacing Adjustments
You can customize line and letter spacing to fit narrow bubbles:
* **Line Spacing (Leading):** In the Text Tool Options panel, find the vertical spacing slider (labeled with two vertical lines and letters). Decreasing this (negative values) packs text rows closer together.
  * *Tip:* For dialogue, tight line spacing looks cleaner. For vertical stack letters, set negative spacing (e.g., `-6px`) so the text block doesn't stretch too long.
* **Letter Spacing (Tracking):** Find the horizontal spacing slider (labeled with horizontal letters/arrows). Increasing this spreads out letters, which is useful for dramatic exclamations or decorative sound effects.

### C. Color Sampling & Selection
1. Select the **Color Picker tool** (shortcut `O`).
2. Click on the original page artwork to sample any color (such as original sound effect colors or specific tones). GIMP sets this as the active foreground color.
3. Switch to the **Text Tool** (`T`), double-click the text on the canvas to select it, and click the color swatch in the Text Tool Options (or the on-screen formatting popup) to apply the sampled foreground color.

---

## 3. Centering Text in Speech Bubbles

To center text perfectly inside speech bubbles:

1. Select your text layer.
2. In the **Tool Options** panel of the **Text Tool**, click the **Centered** alignment icon.
3. Select GIMP's **Alignment Tool** (shortcut `Q`).
4. Click on your background image or the speech bubble shape.
5. In the Alignment Tool Options panel:
   * Select **Align: Active Layer**.
   * Under *Relative to*, choose **First item** (which is the bubble you clicked).
   * Click the **Align Center of Target** button (horizontal line) and the **Align Middle of Target** button (vertical line). GIMP will center the text layer perfectly.

---

## 4. Manual Vertical Text Stacking
Standard GIMP does not have a native "vertical CJK" engine for horizontal-to-vertical English text. To write vertical stacked text manually:
1. Select the **Text Tool** (`T`) and click the canvas.
2. Type your text, hitting **Enter** after every single character:
   ```
   W
   H
   A
   T
   ```
3. Set the text alignment to **Centered**.
4. To reduce the vertical gap between the letters, go to **Tool Options** and adjust the **Line Spacing** to a negative value (e.g., `-6px` or `-10px` depending on the font size) until the letters stack naturally.

---

## 5. Adding Outlines / Borders to Text (Stroking)
Manga text often overlaps artwork or black panel borders. Adding a white border/outline (halo) makes it legible:

1. Right-click your text layer in the **Layers** panel.
2. Select **Alpha to Selection** (this wraps a selection marquee around your text).
3. In GIMP's main menu, go to **Select > Grow...**
4. Enter `2` to `4` pixels (depending on the image resolution) and click OK.
5. Create a **New Transparent Layer** and name it `Text Outline`.
6. Drag this layer **below** your text layer in the Layers panel.
7. Set your foreground color to White.
8. Use the **Bucket Fill Tool** (shortcut `Shift + B`) to fill the expanded selection on the `Text Outline` layer.

---

## 6. How to Curve Text in GIMP (Text along Path)
To wrap text along curved speech bubbles or dynamic sound waves:

### Step 1: Draw the Curve
1. Select the **Paths Tool** (shortcut `B`).
2. Click on the canvas to place start and end control points where you want your text to curve.
3. Click and drag the handles to bend the path into the desired arc.

### Step 2: Create the Text Layer
1. Select the **Text Tool** (`T`).
2. Click the canvas and type your text, styling it with the correct font and size.
3. Keep the text layer selected in the Layers panel.

### Step 3: Project the Text
1. Right-click the text layer in the **Layers** panel and select **Text along Path**.
2. *What happens:* GIMP creates a vector path outline of the text projected along your curve.
3. You can now hide or delete the original flat text layer.

### Step 4: Fill the Curved Path
1. Go to the **Paths** tab (docked next to the Layers tab, or open it via `Windows > Dockable Dialogs > Paths`).
2. Right-click the newly generated text path (named `[Text Layer Name]`) and select **Path to Selection**.
3. Create a **New Layer** named `Curved Text Fill`.
4. Set your text color as the active foreground color, select the **Bucket Fill Tool** (`Shift + B`), and click inside the selection (or press `Ctrl + ,`) to fill the letters.

### Step 5: Add Outline / Stroke to the Curved Text
There are two ways to outline curved text depending on the look you want:

#### Method A: Stroking via Selection (Thick Comic Outlines - Recommended)
This matches standard dialogue text outlining:
1. With the selection from **Step 4** still active (or right-click the text path in the **Paths** tab and select **Path to Selection**), go to the menu: **Select > Grow...**
2. Enter the outline thickness (e.g., `3` to `5` pixels) and click OK.
3. Create a **New Layer** named `Curved Text Outline` and position it **below** the `Curved Text Fill` layer.
4. Set your foreground color to White (or your desired outline color).
5. Select the **Bucket Fill Tool** (`Shift + B`) and fill the selection on the `Curved Text Outline` layer.

#### Method B: Direct Path Stroking (Thin Border Line / Hollow Style)
This draws a crisp border line directly centered on the path outline of the characters:
1. Create a **New Layer** named `Curved Text Outline` and position it **below** (or above) the `Curved Text Fill` layer.
2. Set your foreground color to the outline color.
3. In the **Paths** tab, right-click the text path and select **Stroke Path...** (or click the Stroke icon at the bottom of the panel).
4. In the dialog:
   * Select **Stroke line** -> **Solid color**.
   * Set the **Line width** (e.g., `2px` or `4px`).
   * Click **Stroke**.

