# Koharu Universal Remote Dispatch Server

This directory contains the remote dispatcher server for offloading heavy Vision-Language Model (VLM) OCR processing from GIMP.

## Architecture & Directory Structure

To deploy this on a remote server, ensure the following directory layout is present:
```
gimp-scanlation-suite/
├── modules/
│   ├── __init__.py
│   └── model_manager.py      # Handles Hugging Face model downloads on the server
└── server/
    ├── main.py               # FastAPI application entry point
    ├── core/                 # Server configuration and prompt dictionary
    ├── routes/               # API route submodules
    ├── services/             # VLM, OCR, and Inpainting ML services
    ├── requirements.txt      # Server python package list
    └── README.md             # Setup guide (this file)
```

---

## Installation Guide

### 1. Set Up Virtual Environment
Create a python virtual environment on your remote server:
```bash
python3 -m venv venv
```

Activate it depending on your current shell:

- **For bash/zsh**:
  ```bash
  source venv/bin/activate
  ```
- **For fish**:
  ```fish
  source venv/bin/activate.fish
  ```
- **For csh/tcsh**:
  ```csh
  source venv/bin/activate.csh
  ```

### 2. Install llama-cpp-python with GPU Acceleration
To compile `llama-cpp-python` with hardware acceleration on your GPU, set the appropriate compile flags before installing.

#### **For NVIDIA GPUs (CUDA)**:
Ensure CUDA Toolkit is installed and run:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python==0.3.23
```

#### **For AMD/Intel/NVIDIA GPUs (Vulkan)**:
Ensure Vulkan SDK is installed and run:
```bash
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python==0.3.23
```

> [!IMPORTANT]
> **Vulkan Build Requirements**:
> Compiling `llama-cpp-python` with Vulkan requires the Vulkan shader compiler (`glslc`), Vulkan development headers, and SPIR-V headers. If you get a CMake build error (such as missing `spirv/unified1/spirv.hpp` or `vulkan/vulkan.h`), install the required system libraries:
> - **Ubuntu/Debian**:
>   ```bash
>   sudo apt update
>   sudo apt install -y glslc libvulkan-dev spirv-headers
>   # Note: On older releases, glslc is provided by: sudo apt install -y shaderc
>   ```
> - **Arch Linux**: `sudo pacman -S shaderc vulkan-headers spirv-headers`
> - **Fedora/RHEL**: `sudo dnf install shaderc vulkan-headers spirv-headers`
> - **macOS (Vulkan via MoltenVK)**: `brew install shaderc spirv-headers`
> - **Other / Manual**: Install the official [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) which includes `glslc` and all headers.

#### **For Apple Silicon (Metal)**:
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python==0.3.23
```

---

### 3. Install Server Dependencies
Install the remaining packages listed in `requirements.txt`:
```bash
pip install -r server/requirements.txt
```

> [!TIP]
> **AMD GPU PyTorch Acceleration (ROCm)**:
> The standard `requirements.txt` installation pulls the default PyTorch package, which will execute on the **CPU** by default on AMD systems.
> If you want PyTorch (and SDXL Inpainting) to run with hardware acceleration on an AMD GPU:
> 
> * **Option A: Debian/Ubuntu (Pip-based)**:
>   1. Install the ROCm-enabled PyTorch build in your virtual environment:
>      ```bash
>      pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/rocm6.0
>      ```
>      *(Note: Replace `rocm6.0` with your target ROCm version if different, e.g., `rocm6.1`)*
> 
> * **Option B: Arch Linux / CachyOS (System-based)**:
>   If you are running Arch Linux or CachyOS, it is highly recommended to use the optimized system packages:
>   1. Install the ROCm PyTorch package via `pacman`:
>      ```bash
>      sudo pacman -S python-pytorch-opt-rocm
>      ```
>      *(Use `python-pytorch-rocm` if you do not want AVX2 optimizations).*
>   2. Recreate your virtual environment to include system site-packages (or change `include-system-site-packages = true` in `venv/pyvenv.cfg`):
>      ```bash
>      python3 -m venv venv --system-site-packages
>      source venv/bin/activate
>      pip install -r server/requirements.txt
>      ```
>      *(Note: Make sure not to run `pip install torch` inside the venv, as it will overwrite the system's ROCm-enabled version with a CPU-only one).*
> 
> * **For Unsupported AMD GPUs (e.g. BC-250 / gfx1013 "Cyan Skillfish")**:
>   AMD ROCm does not natively support the gfx1013 architecture. You must override the gfx version before running PyTorch to trick it into using compatible kernels (RDNA2 `gfx1030` is recommended):
>   ```bash
>   export HSA_OVERRIDE_GFX_VERSION=10.3.0
>   python3 -m uvicorn server.main:app --host 0.0.0.0 --port 7890
>   ```
>   *(Note: We have also added auto-configuration in `server/main.py` which attempts to set `HSA_OVERRIDE_GFX_VERSION=10.3.0` automatically if not already set, but exporting it explicitly in your shell is always safer.)*


---

## Running the Server

Start the FastAPI daemon using `uvicorn`:
```bash
python3 -m uvicorn server.main:app --host 0.0.0.0 --port 7890
```
- `--host 0.0.0.0` allows the server to accept connections from other computers.
- `--port 7890` matches GIMP's default port.

---

## Connecting from GIMP

1. Open GIMP on your local computer.
2. Run **Filters > Scanlation > 2. OCR Selected Blocks...**
3. Set **Inference Mode** to `Remote`.
4. Set **API URL** to `http://<your-server-ip>:7890` (replacing `<your-server-ip>` with the actual IP address or domain name of your remote server).
5. The **OCR Model / Engine** dropdown will dynamically query your server for supported models and display them!
