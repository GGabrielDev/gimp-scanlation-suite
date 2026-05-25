import os
import sys
import gc
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query

# Bootstrapping local venv site-packages so we can import llama-cpp and modules
plugin_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, plugin_dir)
venv_paths = [
    d for d in [os.path.join(plugin_dir, "venv", "lib", p, "site-packages") for p in ["python3.14", "python3.13", "python3.12", "python3.11", "python3.10"]]
    if os.path.exists(d)
]
if venv_paths:
    sys.path.insert(0, venv_paths[0])

app = FastAPI(title="Koharu Universal Remote Dispatch Server")

# Payload Schema
class BatchRequest(BaseModel):
    task_type: str
    model_id: str
    batch_payload: List[Any]
    options: Optional[Dict[str, Any]] = None

# Model Configuration Registry
MODELS_CONFIG = {
    "PaddleOCR": {
        "repo": "mradermacher/Fast-PaddleOCR-VL-1.5-GGUF",
        "file": "Fast-PaddleOCR-VL-1.5.Q4_K_M.gguf",
        "projector_repo": "PaddlePaddle/PaddleOCR-VL-1.5-GGUF",
        "projector_file": "PaddleOCR-VL-1.5-mmproj.gguf",
        "local_name": "PaddleOCR-VL-1.5-Q4_K_M.gguf",
        "local_projector_name": "PaddleOCR-VL-1.5-mmproj.gguf",
        "handler_class": "Llava15ChatHandler",
        "n_ctx": 4096
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
    }
}

# Global loaded models cache
_loaded_models = {}
_current_loaded_model_id = None
_n_gpu_layers_used = -1

def unload_model(model_id: str):
    """
    Cleans up a model and calls garbage collector to free up system VRAM/RAM.
    """
    global _loaded_models, _current_loaded_model_id
    if model_id in _loaded_models:
        sys.stderr.write(f"[Server Dispatcher] Unloading model '{model_id}' to free VRAM/RAM...\n")
        del _loaded_models[model_id]
        if _current_loaded_model_id == model_id:
            _current_loaded_model_id = None
        gc.collect()

def get_or_load_model(model_id: str, force_cpu: bool = False):
    """
    Retrieves the model from cache or lazy-loads it.
    Unloads any other loaded models to prevent RAM/VRAM issues.
    """
    global _loaded_models, _current_loaded_model_id, _n_gpu_layers_used
    
    if model_id not in MODELS_CONFIG:
        raise ValueError(f"Model ID '{model_id}' is not registered in the server config.")
        
    # Force unload other models
    other_models = [m for m in _loaded_models if m != model_id]
    for other in other_models:
        unload_model(other)
        
    # If forcing CPU and model was loaded on GPU, discard it
    if force_cpu and _n_gpu_layers_used != 0 and model_id in _loaded_models:
        unload_model(model_id)

    if model_id in _loaded_models:
        return _loaded_models[model_id]

    cfg = MODELS_CONFIG[model_id]
    from llama_cpp import Llama
    from modules import model_manager

    sys.stderr.write(f"[Server Dispatcher] Fetching weights for '{model_id}'...\n")
    model_path = model_manager.ensure_model_exists(cfg["repo"], cfg["file"], local_filename=cfg["local_name"])
    proj_path = model_manager.ensure_model_exists(cfg["projector_repo"], cfg["projector_file"], local_filename=cfg["local_projector_name"])

    # Load appropriate Chat Handler
    handler_class = cfg["handler_class"]
    sys.stderr.write(f"[Server Dispatcher] Loading Vision Projector ({handler_class})...\n")
    if handler_class == "Qwen25VLChatHandler":
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        chat_handler = Qwen25VLChatHandler(clip_model_path=proj_path, verbose=False)
    else:
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        chat_handler = Llava15ChatHandler(clip_model_path=proj_path, verbose=False)

    sys.stderr.write(f"[Server Dispatcher] Initializing LLM context (n_ctx={cfg['n_ctx']})...\n")
    
    if force_cpu:
        sys.stderr.write("[Server Dispatcher] Forcing CPU initialization (n_gpu_layers=0)...\n")
        llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=cfg["n_ctx"],
            logits_all=True,
            n_gpu_layers=0,
            verbose=False
        )
        _n_gpu_layers_used = 0
    else:
        try:
            sys.stderr.write("[Server Dispatcher] Trying GPU offloading (n_gpu_layers=-1)...\n")
            llm = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=cfg["n_ctx"],
                logits_all=True,
                n_gpu_layers=-1,
                verbose=False
            )
            _n_gpu_layers_used = -1
            sys.stderr.write("[Server Dispatcher] GPU initialization succeeded.\n")
        except Exception as e:
            sys.stderr.write(f"[Server Dispatcher] GPU loading failed: {e}. Falling back to CPU...\n")
            llm = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=cfg["n_ctx"],
                logits_all=True,
                n_gpu_layers=0,
                verbose=False
            )
            _n_gpu_layers_used = 0
            sys.stderr.write("[Server Dispatcher] CPU fallback initialization succeeded.\n")

    _loaded_models[model_id] = llm
    _current_loaded_model_id = model_id
    return llm

@app.get("/api/v1/models")
def list_models(task_type: str = Query(..., description="The type of pipeline task, e.g. ocr")):
    """
    Returns the list of available models supported by the server for the specified task type.
    """
    if task_type == "ocr":
        return {"models": list(MODELS_CONFIG.keys())}
    return {"models": []}

@app.post("/api/v1/dispatch")
def dispatch(request: BatchRequest):
    """
    Universal batch routing endpoint.
    Loads requested model on demand, processes the batch, and returns results.
    """
    global _n_gpu_layers_used
    if request.task_type != "ocr":
        raise HTTPException(status_code=400, detail=f"Unsupported task_type '{request.task_type}'")

    if request.model_id not in MODELS_CONFIG:
        raise HTTPException(status_code=400, detail=f"Model ID '{request.model_id}' is not registered on the server.")

    try:
        llm = get_or_load_model(request.model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # Build OCR target language prompt if provided
    options = request.options or {}
    target_lang = options.get("target_language")
    prompt = f"OCR: (Language: {target_lang})" if target_lang else "OCR:"

    sys.stderr.write(f"[Server Dispatcher] Processing batch of size {len(request.batch_payload)} with prompt='{prompt}'...\n")
    results = []

    for idx, item in enumerate(request.batch_payload):
        # Extract base64 image data string
        img_str = ""
        if isinstance(item, dict):
            img_str = item.get("image_data", "")
        else:
            img_str = str(item)

        if not img_str:
            results.append("")
            continue

        # Clean/prepend data URI header
        if not img_str.startswith("data:"):
            img_str = f"data:image/jpeg;base64,{img_str}"

        # Reset cache before inference to prevent overflow
        try:
            llm.reset()
        except Exception as r_err:
            sys.stderr.write(f"[Server Dispatcher] LLM reset error: {r_err}\n")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_str}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        try:
            # Perform inference
            try:
                response = llm.create_chat_completion(messages=messages)
            except Exception as inf_err:
                # CPU fallback retry
                if _n_gpu_layers_used != 0:
                    sys.stderr.write(f"[Server Dispatcher] GPU execution failed: {inf_err}. Re-routing to CPU...\n")
                    llm = get_or_load_model(request.model_id, force_cpu=True)
                    llm.reset()
                    response = llm.create_chat_completion(messages=messages)
                else:
                    raise inf_err

            text = response["choices"][0]["message"]["content"]
            results.append(text.strip())
        except Exception as e:
            sys.stderr.write(f"[Server Dispatcher] Error during OCR inference on item {idx}: {e}\n")
            results.append("")

    return results
