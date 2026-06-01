import os
import sys
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Bootstrapping local venv site-packages
plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, plugin_dir)
venv_paths = [
    d for d in [os.path.join(plugin_dir, "venv", "lib", p, "site-packages") for p in ["python3.14", "python3.13", "python3.12", "python3.11", "python3.10"]]
    if os.path.exists(d)
]
if venv_paths:
    sys.path.insert(0, venv_paths[0])

# Payload Schema
class BatchRequest(BaseModel):
    task_type: str
    model_id: str
    batch_payload: List[Any]
    options: Optional[Dict[str, Any]] = None


# Try to load environment variables from a .env file if it exists
def load_dotenv():
    # Check common locations for .env
    env_paths = [
        os.path.join(plugin_dir, ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), ".env"),
        ".env"
    ]
    for path in env_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            key, val = line.split("=", 1)
                            # Remove quotes from value if present
                            val = val.strip().strip('"\'')
                            os.environ[key.strip()] = val
                sys.stderr.write(f"[Server Config] Loaded environment variables from: {path}\n")
                break
            except Exception as e:
                sys.stderr.write(f"[Server Config] Failed to read .env file at {path}: {e}\n")

load_dotenv()

# Model Configuration Registry
MODELS_CONFIG = {
    "PaddleOCR_Manga": {
        "repo": "adambarbato/PaddleOCR-VL-For-Manga-GGUF",
        "file": "PaddleOCR-VL-For-Manga-BF16.gguf",
        "projector_repo": "adambarbato/PaddleOCR-VL-For-Manga-GGUF",
        "projector_file": "PaddleOCR-VL-For-Manga-mmproj-BF16.gguf",
        "local_name": "PaddleOCR-VL-For-Manga-BF16.gguf",
        "local_projector_name": "PaddleOCR-VL-For-Manga-mmproj-BF16.gguf",
        "handler_class": "Llava15ChatHandler",
        "n_ctx": 4096,
        "n_gpu_layers": -1
    },
    "olmOCR2_Q4": {
        "repo": "bartowski/allenai_olmOCR-2-7B-1025-GGUF",
        "file": "allenai_olmOCR-2-7B-1025-Q4_K_M.gguf",
        "projector_repo": "bartowski/allenai_olmOCR-2-7B-1025-GGUF",
        "projector_file": "mmproj-allenai_olmOCR-2-7B-1025-f16.gguf",
        "local_name": "olmOCR-2-7B-1025-Q4_K_M.gguf",
        "local_projector_name": "mmproj-allenai_olmOCR-2-7B-1025-f16.gguf",
        "handler_class": "Qwen25VLChatHandler",
        "n_ctx": 4096
    },
    "olmOCR2_Q6": {
        "repo": "bartowski/allenai_olmOCR-2-7B-1025-GGUF",
        "file": "allenai_olmOCR-2-7B-1025-Q6_K.gguf",
        "projector_repo": "bartowski/allenai_olmOCR-2-7B-1025-GGUF",
        "projector_file": "mmproj-allenai_olmOCR-2-7B-1025-f16.gguf",
        "local_name": "olmOCR-2-7B-1025-Q6_K.gguf",
        "local_projector_name": "mmproj-allenai_olmOCR-2-7B-1025-f16.gguf",
        "handler_class": "Qwen25VLChatHandler",
        "n_ctx": 4096
    },
    "olmOCR2_Q8": {
        "repo": "bartowski/allenai_olmOCR-2-7B-1025-GGUF",
        "file": "allenai_olmOCR-2-7B-1025-Q8_0.gguf",
        "projector_repo": "bartowski/allenai_olmOCR-2-7B-1025-GGUF",
        "projector_file": "mmproj-allenai_olmOCR-2-7B-1025-f16.gguf",
        "local_name": "olmOCR-2-7B-1025-Q8_0.gguf",
        "local_projector_name": "mmproj-allenai_olmOCR-2-7B-1025-f16.gguf",
        "handler_class": "Qwen25VLChatHandler",
        "n_ctx": 4096
    },
    "JP_Arbiter_8B": {
        "repo": "RumiaChannel/llm-jp-4-8b-thinking-uncensored-ara-gguf",
        "file": "llm-jp-4-8b-thinking-uncensored-ara.Q8_0.gguf",
        "projector_repo": None,
        "projector_file": None,
        "local_name": "llm-jp-4-8b-thinking-uncensored-Q8.gguf",
        "local_projector_name": None,
        "handler_class": "TextOnly",
        "n_ctx": 8192,
        "n_gpu_layers": -1
    },
    "DeepSeek": {
        "repo": None,
        "file": None,
        "projector_repo": None,
        "projector_file": None,
        "local_name": None,
        "local_projector_name": None,
        "handler_class": "DeepSeekAPI",
        "n_ctx": 4096,
        "model_name": "deepseek-chat"
    },
    "DeepSeek-V4-Flash": {
        "repo": None,
        "file": None,
        "projector_repo": None,
        "projector_file": None,
        "local_name": None,
        "local_projector_name": None,
        "handler_class": "DeepSeekAPI",
        "n_ctx": 4096,
        "model_name": "deepseek-v4-flash"
    },
    "DeepSeek-V4-Pro": {
        "repo": None,
        "file": None,
        "projector_repo": None,
        "projector_file": None,
        "local_name": None,
        "local_projector_name": None,
        "handler_class": "DeepSeekAPI",
        "n_ctx": 4096,
        "model_name": "deepseek-v4-pro"
    },
    "lama-manga": {
        "repo": "mayocream/lama-manga-onnx",
        "file": "lama-manga.onnx",
        "projector_repo": None,
        "projector_file": None,
        "local_name": "lama-manga.onnx",
        "local_projector_name": None,
        "handler_class": "Inpainting",
        "n_ctx": 0
    },
    "aot-inpainting": {
        "repo": "ogkalu/aot-inpainting",
        "file": "aot.onnx",
        "projector_repo": None,
        "projector_file": None,
        "local_name": "aot-inpainting.onnx",
        "local_projector_name": None,
        "handler_class": "Inpainting",
        "n_ctx": 0
    },
    "sd-inpainting": {
        "repo": "runwayml/stable-diffusion-inpainting",
        "file": None,
        "projector_repo": None,
        "projector_file": None,
        "local_name": None,
        "local_projector_name": None,
        "handler_class": "DiffusionInpainting",
        "n_ctx": 0
    },
    "anime-inpaint": {
        "repo": "Uminosachi/revAnimated_v121Inp-inpainting",
        "file": None,
        "projector_repo": None,
        "projector_file": None,
        "local_name": None,
        "local_projector_name": None,
        "handler_class": "DiffusionInpainting",
        "n_ctx": 0
    },
    "sdxl-inpainting": {
        "repo": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "file": None,
        "projector_repo": None,
        "projector_file": None,
        "local_name": None,
        "local_projector_name": None,
        "handler_class": "DiffusionInpainting",
        "n_ctx": 0
    }
}
