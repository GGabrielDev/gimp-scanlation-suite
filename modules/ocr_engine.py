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

def get_ocr_engine_instance():
    """
    Lazy loads and returns the Llama VLM model instance.
    Falls back to CPU if GPU/Vulkan offloading fails.
    """
    global _llm_instance
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

    # 2. Try loading with GPU/Vulkan offloading first
    sys.stderr.write("[OCR Engine] Initializing PaddleOCR-VL model...\n")
    
    # PaddleOCR-VL needs enough context for the image embeddings and the prompt.
    n_ctx = 2048
    
    try:
        sys.stderr.write("[OCR Engine] Trying GPU offloading (n_gpu_layers=-1)...\n")
        chat_handler = Llava15ChatHandler(clip_model_path=projector_path, verbose=False)
        _llm_instance = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=n_ctx,
            logits_all=True,
            n_gpu_layers=-1,
            verbose=False
        )
        sys.stderr.write("[OCR Engine] Model initialized successfully with GPU offloading.\n")
    except Exception as gpu_err:
        sys.stderr.write(f"[OCR Engine] GPU initialization failed: {gpu_err}. Falling back to CPU...\n")
        try:
            chat_handler = Llava15ChatHandler(clip_model_path=projector_path, verbose=False)
            _llm_instance = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=n_ctx,
                logits_all=True,
                n_gpu_layers=0,
                verbose=False
            )
            sys.stderr.write("[OCR Engine] Model initialized successfully on CPU.\n")
        except Exception as cpu_err:
            sys.stderr.write(f"[OCR Engine] CPU fallback initialization also failed: {cpu_err}\n")
            raise cpu_err

    return _llm_instance

def extract_text_from_crops(image_crops):
    """
    Takes a list of NumPy image crops (RGB u8 arrays) and runs OCR on each.
    Returns a list of extracted text strings.
    """
    llm = get_ocr_engine_instance()
    results = []
    
    for i, crop in enumerate(image_crops):
        sys.stderr.write(f"[OCR Engine] Processing crop {i+1}/{len(image_crops)}...\n")
        
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
            
            # Run vision completion
            response = llm.create_chat_completion(messages=messages)
            text = response["choices"][0]["message"]["content"]
            text = text.strip()
            sys.stderr.write(f"[OCR Engine] Extracted text: {text}\n")
            results.append(text)
        except Exception as e:
            sys.stderr.write(f"[OCR Engine] Error processing crop {i}: {e}\n")
            results.append("")
            
    return results
