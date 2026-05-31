# GIMP Scanlation Suite (Koharu Project)

An agentic, AI-powered scanlation toolkit for GIMP 3.0. It automates speech bubble detection, text recognition (OCR), background inpainting/erasure, and contextual typesetting translations.

## Setup & Installation

### 1. Font Installation
The typesetting engine is configured to default to **`CC Yada Yada Yada`** for dialogue text and **`CC Hush Hush`** for titles/sound-effects, with standard Serif fonts as fallbacks.

If you have the `.ttf` files locally, install them so GIMP 3 can load them:

#### **Linux (Ubuntu/Debian/Arch/Fedora)**
1. Copy your `.ttf` files to your user's font directory:
   ```bash
   mkdir -p ~/.local/share/fonts
   cp *.ttf ~/.local/share/fonts/
   ```
2. Rebuild the system font cache:
   ```bash
   fc-cache -f -v
   ```
3. Restart GIMP.

#### **Windows**
1. Right-click the `.ttf` files and click **Install** (or **Install for all users**).
2. Restart GIMP.

---

### 2. Remote Dispatcher Server (VLM / LLM / Inpaint)
Heavy ML tasks (OCR, Inpainting, and LLM translation) are offloaded to a FastAPI remote dispatcher.

To run the remote dispatcher server, follow the setup instructions in `server/README.md`.

---

## Scanlation Pipeline

1. **Detect Text & Bubbles**: Runs segmentation/YOLO to identify speech bounds.
2. **OCR Selected Blocks**: Performs optical character recognition on crops. Matches consensus OCR results.
3. **Inpaint / Erase Text**: Cleans dialogue bounding boxes using `lama-manga` or `aot-inpainting` models.
4. **Translate & Render**:
   * Auto-sorts dialogue in standard Japanese reading order (top-right to bottom-left).
   * Spatially connects vector path bounds to the OCR'd text.
   * Prompts LLM contextually (passing entire page flow, speaker names, and context hints).
   * Formats translations dynamically (word-wrap and font downscaling to fit bubble bounds).
