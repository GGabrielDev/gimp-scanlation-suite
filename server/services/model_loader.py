import os
import sys
import gc
from server.core.config import MODELS_CONFIG

_loaded_models = {}
_current_loaded_model_id = None
_n_gpu_layers_used = -1

def log_memory_status(message: str):
    try:
        import psutil
        mem = psutil.virtual_memory()
        sys.stderr.write(f"[Server Memory Monitor] {message} | RAM Free: {mem.available / 1024**3:.2f}GB / {mem.total / 1024**3:.2f}GB ({mem.percent}% used)\n")
    except Exception:
        sys.stderr.write(f"[Server Memory Monitor] {message}\n")


def unload_model(model_id: str):
    """
    Cleans up a model and calls garbage collector to free up system VRAM/RAM.
    """
    global _loaded_models, _current_loaded_model_id
    if model_id in _loaded_models:
        log_memory_status(f"Before unloading model '{model_id}'")
        sys.stderr.write(f"[Server Model Loader] Unloading model '{model_id}' to free VRAM/RAM...\n")
        del _loaded_models[model_id]
        if _current_loaded_model_id == model_id:
            _current_loaded_model_id = None
        gc.collect()
        
        # Reclaim PyTorch GPU cache if relevant
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        log_memory_status(f"After unloading model '{model_id}' and garbage collecting")


def get_or_load_model(model_id: str, force_cpu: bool = False):
    """
    Retrieves the model from cache or lazy-loads it.
    Unloads any other loaded models to prevent RAM/VRAM issues.
    """
    global _loaded_models, _current_loaded_model_id, _n_gpu_layers_used
    
    if model_id == "manga_ocr":
        # Force unload other models
        other_models = [m for m in _loaded_models if m != "manga_ocr"]
        for other in other_models:
            unload_model(other)
            
        if "manga_ocr" in _loaded_models:
            return _loaded_models["manga_ocr"]
            
        log_memory_status("Before initializing PyTorch manga-ocr")
        sys.stderr.write("[Server Model Loader] Initializing PyTorch manga-ocr...\n")
        from manga_ocr import MangaOcr
        mocr = MangaOcr()
        _loaded_models["manga_ocr"] = mocr
        _current_loaded_model_id = "manga_ocr"
        log_memory_status("After initializing PyTorch manga-ocr")
        return mocr


    # Bypass loading for API models
    if model_id in MODELS_CONFIG and MODELS_CONFIG[model_id].get("handler_class") == "DeepSeekAPI":
        for other in list(_loaded_models.keys()):
            unload_model(other)
        return None

    # Load inpainting model via ONNX runtime
    if model_id in MODELS_CONFIG and MODELS_CONFIG[model_id].get("handler_class") == "Inpainting":
        for other in list(_loaded_models.keys()):
            if other != model_id:
                unload_model(other)
        if model_id in _loaded_models:
            return _loaded_models[model_id]
        
        log_memory_status(f"Before loading ONNX inpainting model '{model_id}'")
        sys.stderr.write(f"[Server Model Loader] Initializing ONNX inpainting model '{model_id}'...\n")
        import onnxruntime as ort
        from modules import model_manager
        cfg = MODELS_CONFIG[model_id]
        model_path = model_manager.ensure_model_exists(cfg["repo"], cfg["file"], local_filename=cfg["local_name"])
        
        # Determine best available execution providers for ONNX Runtime
        providers = ["CPUExecutionProvider"]
        if not force_cpu:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "ROCmExecutionProvider" in available:
                providers = ["ROCmExecutionProvider", "CPUExecutionProvider"]
        sys.stderr.write(f"[Server Model Loader] Using ONNX Runtime providers: {providers}\n")
        session = ort.InferenceSession(model_path, providers=providers)
        _loaded_models[model_id] = session
        _current_loaded_model_id = model_id
        log_memory_status(f"After loading ONNX inpainting model '{model_id}'")
        return session

        
    # Load Diffusion inpainting model via HuggingFace diffusers
    if model_id in MODELS_CONFIG and MODELS_CONFIG[model_id].get("handler_class") == "DiffusionInpainting":
        for other in list(_loaded_models.keys()):
            if other != model_id:
                unload_model(other)
        if model_id in _loaded_models:
            return _loaded_models[model_id]
            
        log_memory_status(f"Before loading Diffusion Inpainting model '{model_id}'")
        sys.stderr.write(f"[Server Model Loader] Initializing Diffusion Inpainting model '{model_id}'...\n")
        try:
            from diffusers import AutoPipelineForInpainting
            import torch
        except ImportError:
            raise ImportError(
                "Failed to import 'diffusers' or 'torch'. Please ensure you have installed "
                "diffusers, transformers, accelerate, and torch in your server environment."
            )
            
        cfg = MODELS_CONFIG[model_id]
        try:
            try:
                pipe = AutoPipelineForInpainting.from_pretrained(
                    cfg["repo"],
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    safety_checker=None, # Uncensored support for SD 1.5 style models
                    requires_safety_checker=False,
                    low_cpu_mem_usage=True
                )
            except TypeError:
                # Fallback for SDXL which does not accept safety_checker parameters
                pipe = AutoPipelineForInpainting.from_pretrained(
                    cfg["repo"],
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True
                )
            
            # Enable attention slicing to drastically reduce peak RAM usage
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass

            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
            _loaded_models[model_id] = pipe
            _current_loaded_model_id = model_id
            log_memory_status(f"After loading Diffusion Inpainting model '{model_id}'")
            return pipe

        except Exception as e:
            sys.stderr.write(f"[Server Model Loader] Failed to load Diffusion model: {e}\n")
            raise e
        
    if model_id not in MODELS_CONFIG:
        raise ValueError(f"Model ID '{model_id}' is not registered in the server config.")
        
    # Force unload other models
    other_models = [m for m in _loaded_models if m != model_id]
    for other in other_models:
        unload_model(other)
        
    cfg = MODELS_CONFIG[model_id]
    if cfg.get("n_gpu_layers") == 0:
        force_cpu = True

    # If forcing CPU and model was loaded on GPU, discard it
    if force_cpu and _n_gpu_layers_used != 0 and model_id in _loaded_models:
        unload_model(model_id)

    if model_id in _loaded_models:
        return _loaded_models[model_id]

    from llama_cpp import Llama
    from modules import model_manager

    sys.stderr.write(f"[Server Model Loader] Fetching weights for '{model_id}'...\n")
    model_path = model_manager.ensure_model_exists(cfg["repo"], cfg["file"], local_filename=cfg["local_name"])

    # Bypass vision loading for TextOnly models
    handler_class = cfg["handler_class"]
    if handler_class == "TextOnly":
        proj_path = None
        chat_handler = None
    else:
        proj_path = model_manager.ensure_model_exists(cfg["projector_repo"], cfg["projector_file"], local_filename=cfg["local_projector_name"])
        sys.stderr.write(f"[Server Model Loader] Loading Vision Projector ({handler_class})...\n")
        if handler_class == "Qwen25VLChatHandler":
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
            chat_handler = Qwen25VLChatHandler(clip_model_path=proj_path, verbose=False)
        else:
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            chat_handler = Llava15ChatHandler(clip_model_path=proj_path, verbose=False)
            chat_handler.DEFAULT_SYSTEM_MESSAGE = None

    sys.stderr.write(f"[Server Model Loader] Initializing LLM context (n_ctx={cfg['n_ctx']})...\n")
    
    if force_cpu:
        sys.stderr.write("[Server Model Loader] Forcing CPU initialization (n_gpu_layers=0)...\n")
        llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=cfg["n_ctx"],
            logits_all=False,
            n_gpu_layers=0,
            verbose=False
        )
        _n_gpu_layers_used = 0
    else:
        try:
            n_layers = cfg.get("n_gpu_layers", -1)
            sys.stderr.write(f"[Server Model Loader] Trying GPU offloading (n_gpu_layers={n_layers})...\n")
            llm = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=cfg["n_ctx"],
                logits_all=False,
                n_gpu_layers=n_layers,
                verbose=False
            )
            _n_gpu_layers_used = n_layers
            sys.stderr.write("[Server Model Loader] GPU initialization succeeded.\n")
        except Exception as e:
            sys.stderr.write(f"[Server Model Loader] GPU loading failed: {e}. Falling back to CPU...\n")
            llm = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=cfg["n_ctx"],
                logits_all=False,
                n_gpu_layers=0,
                verbose=False
            )
            _n_gpu_layers_used = 0
            sys.stderr.write("[Server Model Loader] CPU fallback initialization succeeded.\n")

    _loaded_models[model_id] = llm
    _current_loaded_model_id = model_id
    return llm

def get_n_gpu_layers_used():
    global _n_gpu_layers_used
    return _n_gpu_layers_used

def set_n_gpu_layers_used(val):
    global _n_gpu_layers_used
    _n_gpu_layers_used = val
