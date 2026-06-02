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
# Model Configuration Registry
class ModelMetadata(BaseModel):
    model_id: str
    display_name: str
    description: str
    tasks: List[str]
    modalities: List[str]
    handler_class: str
    download_info: Optional[Dict[str, Any]] = None
    n_ctx: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    model_name: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            if val is not None:
                return val
        if self.download_info and key in self.download_info:
            return self.download_info[key]
        return default

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            if val is not None:
                return val
        if self.download_info and key in self.download_info:
            return self.download_info[key]
        raise KeyError(key)

MODELS_CONFIG = {
    "PaddleOCR_Manga": ModelMetadata(
        model_id="PaddleOCR_Manga",
        display_name="PaddleOCR Manga (BF16)",
        description="Specialized Japanese text extractor, best for vertical manga text",
        tasks=["ocr_expert"],
        modalities=["image", "text"],
        handler_class="Llava15ChatHandler",
        download_info={
            "repo": "adambarbato/PaddleOCR-VL-For-Manga-GGUF",
            "file": "PaddleOCR-VL-For-Manga-BF16.gguf",
            "projector_repo": "adambarbato/PaddleOCR-VL-For-Manga-GGUF",
            "projector_file": "PaddleOCR-VL-For-Manga-mmproj-BF16.gguf",
            "local_name": "PaddleOCR-VL-For-Manga-BF16.gguf",
            "local_projector_name": "PaddleOCR-VL-For-Manga-mmproj-BF16.gguf",
        },
        n_ctx=4096,
        n_gpu_layers=-1
    ),
    "olmOCR2_Q4": ModelMetadata(
        model_id="olmOCR2_Q4",
        display_name="olmOCR 2 (Q4)",
        description="High-accuracy document OCR model (4-bit quantized), optimized for comics and scans",
        tasks=["ocr_expert"],
        modalities=["image", "text"],
        handler_class="Qwen25VLChatHandler",
        download_info={
            "repo": "bartowski/allenai_olmOCR-2-7B-1025-GGUF",
            "file": "allenai_olmOCR-2-7B-1025-Q4_K_M.gguf",
            "projector_repo": "bartowski/allenai_olmOCR-2-7B-1025-GGUF",
            "projector_file": "mmproj-allenai_olmOCR-2-7B-1025-f16.gguf",
            "local_name": "olmOCR-2-7B-1025-Q4_K_M.gguf",
            "local_projector_name": "mmproj-allenai_olmOCR-2-7B-1025-f16.gguf",
        },
        n_ctx=4096
    ),
    "olmOCR2_Q6": ModelMetadata(
        model_id="olmOCR2_Q6",
        display_name="olmOCR 2 (Q6)",
        description="High-accuracy document OCR model (6-bit quantized), balances speed and precision",
        tasks=["ocr_expert"],
        modalities=["image", "text"],
        handler_class="Qwen25VLChatHandler",
        download_info={
            "repo": "bartowski/allenai_olmOCR-2-7B-1025-GGUF",
            "file": "allenai_olmOCR-2-7B-1025-Q6_K.gguf",
            "projector_repo": "bartowski/allenai_olmOCR-2-7B-1025-GGUF",
            "projector_file": "mmproj-allenai_olmOCR-2-7B-1025-f16.gguf",
            "local_name": "olmOCR-2-7B-1025-Q6_K.gguf",
            "local_projector_name": "mmproj-allenai_olmOCR-2-7B-1025-f16.gguf",
        },
        n_ctx=4096
    ),
    "olmOCR2_Q8": ModelMetadata(
        model_id="olmOCR2_Q8",
        display_name="olmOCR 2 (Q8)",
        description="High-accuracy document OCR model (8-bit quantized), maximum precision for difficult text",
        tasks=["ocr_expert"],
        modalities=["image", "text"],
        handler_class="Qwen25VLChatHandler",
        download_info={
            "repo": "bartowski/allenai_olmOCR-2-7B-1025-GGUF",
            "file": "allenai_olmOCR-2-7B-1025-Q8_0.gguf",
            "projector_repo": "bartowski/allenai_olmOCR-2-7B-1025-GGUF",
            "projector_file": "mmproj-allenai_olmOCR-2-7B-1025-f16.gguf",
            "local_name": "olmOCR-2-7B-1025-Q8_0.gguf",
            "local_projector_name": "mmproj-allenai_olmOCR-2-7B-1025-f16.gguf",
        },
        n_ctx=4096
    ),
    "JP_Arbiter_8B": ModelMetadata(
        model_id="JP_Arbiter_8B",
        display_name="JP Arbiter 8B",
        description="Local Japanese-centric LLM (8B), ideal for translation and consensus arbitration",
        tasks=["ocr_arbiter", "translate"],
        modalities=["text"],
        handler_class="TextOnly",
        download_info={
            "repo": "RumiaChannel/llm-jp-4-8b-thinking-uncensored-ara-gguf",
            "file": "llm-jp-4-8b-thinking-uncensored-ara.Q8_0.gguf",
            "projector_repo": None,
            "projector_file": None,
            "local_name": "llm-jp-4-8b-thinking-uncensored-Q8.gguf",
            "local_projector_name": None,
        },
        n_ctx=8192,
        n_gpu_layers=-1
    ),
    "DeepSeek": ModelMetadata(
        model_id="DeepSeek",
        display_name="DeepSeek Chat (API)",
        description="DeepSeek remote language model, excellent general translator and text arbiter",
        tasks=["ocr_arbiter", "translate"],
        modalities=["text"],
        handler_class="DeepSeekAPI",
        model_name="deepseek-chat",
        n_ctx=4096
    ),
    "DeepSeek-V4-Flash": ModelMetadata(
        model_id="DeepSeek-V4-Flash",
        display_name="DeepSeek V4 Flash (API)",
        description="Fast remote model, optimized for quick translation and basic arbitration",
        tasks=["ocr_arbiter", "translate"],
        modalities=["text"],
        handler_class="DeepSeekAPI",
        model_name="deepseek-v4-flash",
        n_ctx=4096
    ),
    "DeepSeek-V4-Pro": ModelMetadata(
        model_id="DeepSeek-V4-Pro",
        display_name="DeepSeek V4 Pro (API)",
        description="Powerful remote model, best for complex translation and reasoning-heavy arbitration",
        tasks=["ocr_arbiter", "translate"],
        modalities=["text"],
        handler_class="DeepSeekAPI",
        model_name="deepseek-v4-pro",
        n_ctx=4096
    ),
    "lama-manga": ModelMetadata(
        model_id="lama-manga",
        display_name="LaMa Manga (ONNX)",
        description="Fast, local resolution-independent inpainting model trained specifically for manga",
        tasks=["inpaint"],
        modalities=["image"],
        handler_class="Inpainting",
        download_info={
            "repo": "mayocream/lama-manga-onnx",
            "file": "lama-manga.onnx",
            "projector_repo": None,
            "projector_file": None,
            "local_name": "lama-manga.onnx",
            "local_projector_name": None,
        },
        n_ctx=0
    ),
    "aot-inpainting": ModelMetadata(
        model_id="aot-inpainting",
        display_name="AOT Inpainting (ONNX)",
        description="Local ONNX inpainting model using Aggregated Contextual Transformations",
        tasks=["inpaint"],
        modalities=["image"],
        handler_class="Inpainting",
        download_info={
            "repo": "ogkalu/aot-inpainting",
            "file": "aot.onnx",
            "projector_repo": None,
            "projector_file": None,
            "local_name": "aot-inpainting.onnx",
            "local_projector_name": None,
        },
        n_ctx=0
    ),
    "sd-inpainting": ModelMetadata(
        model_id="sd-inpainting",
        display_name="Stable Diffusion Inpainting",
        description="Standard SD-based inpainting model for complex, detailed background fills",
        tasks=["inpaint"],
        modalities=["image"],
        handler_class="DiffusionInpainting",
        download_info={
            "repo": "runwayml/stable-diffusion-inpainting",
            "file": None,
            "projector_repo": None,
            "projector_file": None,
            "local_name": None,
            "local_projector_name": None,
        },
        n_ctx=0
    ),
    "anime-inpaint": ModelMetadata(
        model_id="anime-inpaint",
        display_name="Anime Inpaint (RevAnimated)",
        description="Stable Diffusion model fine-tuned for anime illustrations, best for color manga and art",
        tasks=["inpaint"],
        modalities=["image"],
        handler_class="DiffusionInpainting",
        download_info={
            "repo": "Uminosachi/revAnimated_v121Inp-inpainting",
            "file": None,
            "projector_repo": None,
            "projector_file": None,
            "local_name": None,
            "local_projector_name": None,
        },
        n_ctx=0
    ),
    "sdxl-inpainting": ModelMetadata(
        model_id="sdxl-inpainting",
        display_name="Stable Diffusion XL Inpainting",
        description="High-resolution SDXL-based inpainting for detailed large-canvas artwork",
        tasks=["inpaint"],
        modalities=["image"],
        handler_class="DiffusionInpainting",
        download_info={
            "repo": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            "file": None,
            "projector_repo": None,
            "projector_file": None,
            "local_name": None,
            "local_projector_name": None,
        },
        n_ctx=0
    )
}
