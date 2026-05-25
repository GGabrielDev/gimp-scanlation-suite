import os
import sys
import glob
import io
import base64
from PIL import Image
import numpy as np

# Bootstrapping local venv site-packages so we can import llama-cpp
plugin_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
venv_paths = glob.glob(os.path.join(plugin_dir, "venv", "lib", "python*", "site-packages"))
if venv_paths:
    sys.path.insert(0, venv_paths[0])

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except ImportError as e:
    sys.stderr.write(f"[OCR Engine] Failed to import llama-cpp-python: {e}\n")
    Llama = None
    Llava15ChatHandler = None

_llm_instance = None
_n_gpu_layers_used = -1

def get_ocr_engine_instance(force_cpu=False):
    """
    Lazy loads and returns the Llama VLM model instance.
    If force_cpu is True, forces loading on CPU (n_gpu_layers=0).
    """
    global _llm_instance, _n_gpu_layers_used
    
    # If CPU is forced, but the active instance is on GPU, discard the GPU instance
    if force_cpu and _n_gpu_layers_used != 0:
        sys.stderr.write("[OCR Engine] Discarding GPU instance to load CPU instance...\n")
        _llm_instance = None

    if _llm_instance is not None:
        return _llm_instance

    if Llama is None or Llava15ChatHandler is None:
        raise RuntimeError("llama-cpp-python is not installed or import failed in the virtual environment.")

    # 1. Download/get paths to the models
    from modules import model_manager
    try:
        model_path, projector_path = model_manager.ensure_ocr_models_exist()
    except Exception as e:
        sys.stderr.write(f"[OCR Engine] Failed to get OCR models: {e}\n")
        raise

    # 2. Initialize
    n_ctx = 4096
    
    if force_cpu:
        sys.stderr.write("[OCR Engine] Forcing CPU initialization (n_gpu_layers=0)...\n")
        chat_handler = Llava15ChatHandler(clip_model_path=projector_path, verbose=False)
        _llm_instance = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=n_ctx,
            logits_all=True,
            n_gpu_layers=0,
            verbose=False
        )
        _n_gpu_layers_used = 0
        sys.stderr.write("[OCR Engine] Model initialized successfully on CPU.\n")
    else:
        sys.stderr.write("[OCR Engine] Trying GPU offloading (n_gpu_layers=-1)...\n")
        try:
            chat_handler = Llava15ChatHandler(clip_model_path=projector_path, verbose=False)
            _llm_instance = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=n_ctx,
                logits_all=True,
                n_gpu_layers=-1,
                verbose=False
            )
            _n_gpu_layers_used = -1
            sys.stderr.write("[OCR Engine] Model initialized successfully with GPU offloading.\n")
        except Exception as gpu_err:
            sys.stderr.write(f"[OCR Engine] GPU initialization failed: {gpu_err}. Falling back to CPU...\n")
            chat_handler = Llava15ChatHandler(clip_model_path=projector_path, verbose=False)
            _llm_instance = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=n_ctx,
                logits_all=True,
                n_gpu_layers=0,
                verbose=False
            )
            _n_gpu_layers_used = 0
            sys.stderr.write("[OCR Engine] Model initialized successfully on CPU (fallback).\n")

    return _llm_instance

def extract_text_from_crops(image_crops):
    """
    Takes a list of NumPy image crops (RGB u8 arrays) and runs OCR on each.
    Returns a list of extracted text strings.
    """
    global _n_gpu_layers_used
    llm = get_ocr_engine_instance()
    results = []
    
    for i, crop in enumerate(image_crops):
        sys.stderr.write(f"[OCR Engine] Processing crop {i+1}/{len(image_crops)}...\n")
        
        # Reset LLM state before each crop to clear the KV cache
        try:
            llm.reset()
        except Exception as reset_err:
            sys.stderr.write(f"[OCR Engine] Failed to reset LLM: {reset_err}\n")
            
        # Convert numpy array to PIL Image and then base64 JPEG
        try:
            pil_img = Image.fromarray(crop)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            data_uri = f"data:image/jpeg;base64,{img_str}"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": "OCR:"}
                    ]
                }
            ]
            
            # Run vision completion with automatic fallback
            try:
                response = llm.create_chat_completion(messages=messages)
            except Exception as e:
                if _n_gpu_layers_used != 0:
                    sys.stderr.write(f"[OCR Engine] GPU inference failed ({e}). Retrying on CPU...\n")
                    llm = get_ocr_engine_instance(force_cpu=True)
                    # Reset CPU model state
                    try:
                        llm.reset()
                    except Exception:
                        pass
                    # Retry inference
                    response = llm.create_chat_completion(messages=messages)
                else:
                    raise e
                    
            text = response["choices"][0]["message"]["content"]
            text = text.strip()
            sys.stderr.write(f"[OCR Engine] Extracted text: {text}\n")
            results.append(text)
        except Exception as e:
            sys.stderr.write(f"[OCR Engine] Error processing crop {i}: {e}\n")
            results.append("")
            
    return results
