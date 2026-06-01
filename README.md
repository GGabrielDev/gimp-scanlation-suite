# GIMP Scanlation Suite

An agentic, AI-powered scanlation toolkit for GIMP 3.0. It automates speech bubble detection, text recognition (OCR), background inpainting/erasure, and contextual translations.

## Setup & Installation

Heavy ML tasks (OCR, Inpainting, and LLM translation) are offloaded to a FastAPI remote dispatcher.

To run the remote dispatcher server, follow the setup instructions in `server/README.md`.

---

## Scanlation Pipeline

1. **Detect Text & Bubbles**: Runs segmentation/YOLO to identify speech bounds.
2. **OCR Selected Blocks**: Performs optical character recognition on crops. Matches consensus OCR results.
3. **Inpaint / Erase Text**: Cleans dialogue bounding boxes using `lama-manga` or `aot-inpainting` models.
4. **Translate & Render**:
   * Auto-sorts dialogue in standard Japanese reading order (top-right to bottom-left).
   * Prompts LLM contextually (passing entire page flow, speaker names, and context hints).
   * Saves translations as editable text layers inside the "Translated Text" group in GIMP.
   * See [docs/typesetting_guide.md](docs/typesetting_guide.md) for detailed guidelines on manually styling, spacing, outlining, and curving the text inside GIMP.
