# GIMP 3 Scanlation Suite - Architecture & Constraints

## 1. GIMP API & Environment (CRITICAL)
- **API Version:** This project targets GIMP 3.0+. You MUST use PyGObject (GTK3) via GObject Introspection (`from gi.repository import Gimp, GimpUi, Gegl, Gtk`).
- **Forbidden API:** NEVER use GIMP 2.10 `gimpfu` or Python 2 syntax.
- **Wire Protocol Safety:** NEVER use standard `print()` statements in the global scope. This corrupts GIMP's wire protocol (causing `unexpected EOF`). All debug output must be routed to `sys.stderr`.
- **Venv Bootstrapping:** System Python cannot see local virtual environments. The main script must dynamically inject the local `venv/lib/python3.*/site-packages` into `sys.path` before importing third-party modules like `onnxruntime` or `requests`.

## 2. Compute Splitting (Hardware Architecture)
- **Vision (Local):** Image segmentation and OCR models execute locally on the CPU using `onnxruntime`. 
- **LLM (Remote):** Heavy language translation is strictly offloaded to the remote AMD BC250 server. The plugin must use HTTP POST requests targeting an OpenAI-compatible endpoint (e.g., `http://<SERVER_IP>:8080/v1`). Do not attempt to run LLM inference locally.

## 3. Modularity & UX
- **The Suite:** The plugin registers as a suite of distinct tools under the menu `<Image>/Filters/Scanlation/`.
- **Non-Destructive Workflows:** Use GIMP Paths (Vectors) for bounding boxes to allow manual user adjustment before triggering the inpainting or translation steps. Output translated text as native, editable GIMP Text Layers (`Gimp.TextLayer`).

## 4. File Writing Protocol & Boundaries
- When multiple directories are mounted via `--add-dir`, treat all external repositories as **read-only reference material**.
- **Absolute Paths Only:** Always explicitly write output files to the absolute path of the active project: `~/Projects/gimp-scanlation-suite/`. Never drop files into reference directories.

## 5. Development Environment & Symlink Awareness (CRITICAL)
- **The Setup:** The project resides in `~/Projects/gimp-scanlation-suite/`. A symlink bridges this directory to `~/.config/GIMP/3.0/plug-ins/gimp-scanlation-suite`.
- **Traceback Interpretation:** If a Python error traceback points to the `~/.config/GIMP/` path, DO NOT attempt to copy, move, or recreate the file there. The symlink is intentional. Treat the error as originating from the `~/Projects/` source file and apply fixes only to the source file.
- **Python Virtual Environment:** The plugin relies on a virtual environment (`venv`) that MUST be created with system site packages enabled (`--system-site-packages`) so that GIMP's internal `gi` (PyGObject) library can be resolved alongside local AI dependencies.
