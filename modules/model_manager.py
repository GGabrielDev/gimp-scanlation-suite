import os
import sys
import glob
import shutil

# Dynamically inject local venv site-packages into sys.path
plugin_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
venv_paths = glob.glob(os.path.join(plugin_dir, "venv", "lib", "python*", "site-packages"))
if venv_paths:
    sys.path.insert(0, venv_paths[0])

try:
    from huggingface_hub import hf_hub_download
except ImportError as e:
    sys.stderr.write(f"[Model Manager] Failed to import huggingface_hub: {e}\n")
    hf_hub_download = None

def ensure_model_exists(model_id, filename):
    """
    Downloads the specified ONNX model from Hugging Face hub if it doesn't already exist locally.
    Returns the absolute path to the local model file.
    """
    models_dir = os.path.expanduser("~/Projects/gimp-scanlation-suite/models")
    os.makedirs(models_dir, exist_ok=True)
    local_path = os.path.join(models_dir, filename)

    if os.path.exists(local_path):
        sys.stderr.write(f"[Model Manager] Model already exists locally at: {local_path}\n")
        return local_path

    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub library is not installed in the virtual environment.")

    sys.stderr.write(f"[Model Manager] Downloading model '{model_id}/{filename}'...\n")
    downloaded_path = hf_hub_download(repo_id=model_id, filename=filename)
    
    # Copy file to the local models folder
    shutil.copy2(downloaded_path, local_path)
    sys.stderr.write(f"[Model Manager] Download complete. Model saved to: {local_path}\n")
    return local_path
