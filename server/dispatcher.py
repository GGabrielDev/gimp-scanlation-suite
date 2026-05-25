import os
import sys
import gc
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

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
        
        # Reclaim PyTorch GPU cache if relevant
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

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
            
        sys.stderr.write("[Server Dispatcher] Initializing PyTorch manga-ocr...\n")
        from manga_ocr import MangaOcr
        mocr = MangaOcr()
        _loaded_models["manga_ocr"] = mocr
        _current_loaded_model_id = "manga_ocr"
        return mocr
        
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
        return {"models": ["Ensemble"] + list(MODELS_CONFIG.keys())}
    return {"models": []}

@app.post("/api/v1/dispatch")
def dispatch(request: BatchRequest):
    """
    Universal batch routing endpoint.
    Loads requested model on demand, processes the batch, and returns results.
    """
    global _n_gpu_layers_used
    import io
    import base64
    from PIL import Image

    task = request.task_type
    model = request.model_id

    # Promote to ensemble_ocr if model is Ensemble
    if model == "Ensemble":
        task = "ensemble_ocr"

    if task not in ["ocr", "ensemble_ocr"]:
        raise HTTPException(status_code=400, detail=f"Unsupported task_type '{task}'")

    if task == "ocr":
        if model not in MODELS_CONFIG:
            raise HTTPException(status_code=400, detail=f"Model ID '{model}' is not registered on the server.")

        def event_generator():
            global _n_gpu_layers_used
            try:
                llm = get_or_load_model(model)
            except Exception as e:
                yield json.dumps({"type": "progress", "percentage": 0.0, "message": f"Error: Failed to load model: {e}"}) + "\n"
                raise RuntimeError(f"Failed to load model: {e}")

            options = request.options or {}
            target_lang = options.get("target_language")
            prompt = f"OCR: (Language: {target_lang})" if target_lang else "OCR:"

            sys.stderr.write(f"[Server Dispatcher] Processing single-model batch of size {len(request.batch_payload)} with prompt='{prompt}'...\n")
            results = []
            N = len(request.batch_payload)

            try:
                for idx, item in enumerate(request.batch_payload):
                    yield json.dumps({
                        "type": "progress",
                        "percentage": idx / N if N > 0 else 0.0,
                        "message": f"Processing crop {idx+1}/{N}..."
                    }) + "\n"

                    img_str = item.get("image_data", "") if isinstance(item, dict) else str(item)
                    if not img_str:
                        results.append("")
                        continue

                    if not img_str.startswith("data:"):
                        img_str = f"data:image/jpeg;base64,{img_str}"

                    try:
                        llm.reset()
                    except Exception as r_err:
                        sys.stderr.write(f"[Server Dispatcher] LLM reset error: {r_err}\n")

                    if MODELS_CONFIG[model].get("handler_class") == "Llava15ChatHandler":
                        user_text = f"You are a precise OCR engine. Transcribe all text in the image. Output ONLY the raw transcribed text. Do not translate, explain, or add conversational filler. If no text is visible, output nothing.\n\nPrompt: {prompt}"
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": img_str}},
                                    {"type": "text", "text": user_text}
                                ]
                            }
                        ]
                    else:
                        messages = [
                            {
                                "role": "system",
                                "content": "You are a precise OCR engine. Transcribe all text in the image. Output ONLY the raw transcribed text. Do not translate, explain, or add conversational filler. If no text is visible, output nothing."
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": img_str}},
                                    {"type": "text", "text": prompt}
                                ]
                            }
                        ]

                    try:
                        try:
                            response = llm.create_chat_completion(messages=messages)
                        except Exception as inf_err:
                            if _n_gpu_layers_used != 0:
                                sys.stderr.write(f"[Server Dispatcher] GPU execution failed: {inf_err}. Re-routing to CPU...\n")
                                llm = get_or_load_model(model, force_cpu=True)
                                llm.reset()
                                response = llm.create_chat_completion(messages=messages)
                            else:
                                raise inf_err

                        text = response["choices"][0]["message"]["content"]
                        results.append(text.strip())
                    except Exception as e:
                        sys.stderr.write(f"[Server Dispatcher] Error during OCR inference on item {idx}: {e}\n")
                        results.append("")

                yield json.dumps({
                    "type": "progress",
                    "percentage": 1.0,
                    "message": "Completed OCR batch."
                }) + "\n"
                yield json.dumps({"type": "result", "results": results}) + "\n"
            finally:
                unload_model(model)

        return StreamingResponse(event_generator(), media_type="application/x-ndjson")

    else:
        # ensemble_ocr Mixture of Experts pipeline
        options = request.options or {}
        source_lang = options.get("source_language") or "Japanese"
        target_lang = options.get("target_language") or "Japanese"
        material_type = options.get("material_type") or "manga"
        half_to_full = options.get("half_to_full")

        # Resolve prompt dictionary instructions
        from server.prompt_dictionary import PROMPT_DICTIONARY
        system_instruction = PROMPT_DICTIONARY.get(material_type, PROMPT_DICTIONARY["manga"])

        # Determine Arbiter VLM Model ID
        arbiter_model_id = model
        if arbiter_model_id == "Ensemble":
            arbiter_model_id = "olmOCR2_Q4" # default strong arbiter VLM

        if arbiter_model_id not in MODELS_CONFIG:
            raise HTTPException(status_code=400, detail=f"Arbiter VLM Model ID '{arbiter_model_id}' is not registered.")

        # Clean/convert batch payloads
        crops_base64 = []
        for item in request.batch_payload:
            img_str = item.get("image_data", "") if isinstance(item, dict) else str(item)
            if not img_str:
                crops_base64.append("")
                continue
            if not img_str.startswith("data:"):
                img_str = f"data:image/jpeg;base64,{img_str}"
            crops_base64.append(img_str)

        sys.stderr.write(f"[Server Dispatcher] Starting Ensemble OCR Consensus on {len(crops_base64)} crops (Arbiter={arbiter_model_id}, Type={material_type})...\n")

        def event_generator():
            global _n_gpu_layers_used
            N = len(crops_base64)
            if N == 0:
                yield json.dumps({"type": "progress", "percentage": 1.0, "message": "No crops to process."}) + "\n"
                yield json.dumps({"type": "result", "results": []}) + "\n"
                return

            # --- PASS 1: manga-ocr (PyTorch) ---
            results_a = []
            try:
                yield json.dumps({
                    "type": "progress",
                    "percentage": 0.0,
                    "message": "Initializing PyTorch manga-ocr..."
                }) + "\n"
                mocr = get_or_load_model("manga_ocr")
                for idx, img_b64 in enumerate(crops_base64):
                    pct = idx / (3 * N)
                    yield json.dumps({
                        "type": "progress",
                        "percentage": pct,
                        "message": f"Pass 1/3 (manga-ocr): Crop {idx+1}/{N}..."
                    }) + "\n"
                    
                    if not img_b64:
                        results_a.append("")
                        continue
                    try:
                        # Convert to PIL
                        header, data = img_b64.split(",", 1)
                        pil_img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
                        text_a = mocr(pil_img)
                        results_a.append(text_a.strip() if text_a else "")
                    except Exception as ex_a:
                        sys.stderr.write(f"[Server Dispatcher] Expert A (manga-ocr) error on crop {idx}: {ex_a}\n")
                        results_a.append("")
            finally:
                unload_model("manga_ocr")

            # --- PASS 2: PaddleOCR-VL-1.5 (llama.cpp) ---
            results_b = []
            try:
                yield json.dumps({
                    "type": "progress",
                    "percentage": N / (3 * N),
                    "message": "Initializing PaddleOCR..."
                }) + "\n"
                paddle = get_or_load_model("PaddleOCR")
                for idx, img_b64 in enumerate(crops_base64):
                    pct = (N + idx) / (3 * N)
                    yield json.dumps({
                        "type": "progress",
                        "percentage": pct,
                        "message": f"Pass 2/3 (PaddleOCR): Crop {idx+1}/{N}..."
                    }) + "\n"
                    
                    if not img_b64:
                        results_b.append("")
                        continue
                    try:
                        paddle.reset()
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": img_b64}},
                                    {
                                        "type": "text",
                                        "text": "You are a precise Japanese OCR engine. Transcribe all text in the image. Output ONLY the raw transcribed text. Do not explain.\n\nPrompt: OCR:"
                                    }
                                ]
                            }
                        ]
                        # Direct llama.cpp inference with CPU fallback
                        try:
                            response = paddle.create_chat_completion(messages=messages)
                        except Exception as inf_err:
                            if _n_gpu_layers_used != 0:
                                sys.stderr.write(f"[Server Dispatcher] Expert B GPU failed. Re-routing to CPU...\n")
                                paddle = get_or_load_model("PaddleOCR", force_cpu=True)
                                paddle.reset()
                                response = paddle.create_chat_completion(messages=messages)
                            else:
                                raise inf_err

                        text_b = response["choices"][0]["message"]["content"]
                        results_b.append(text_b.strip() if text_b else "")
                    except Exception as ex_b:
                        sys.stderr.write(f"[Server Dispatcher] Expert B (PaddleOCR) error on crop {idx}: {ex_b}\n")
                        results_b.append("")
            finally:
                unload_model("PaddleOCR")

            # --- PASS 3: Arbiter VLM Consensus ---
            final_results = []
            try:
                yield json.dumps({
                    "type": "progress",
                    "percentage": (2 * N) / (3 * N),
                    "message": f"Initializing Arbiter ({arbiter_model_id})..."
                }) + "\n"
                arbiter = get_or_load_model(arbiter_model_id)
                for idx, img_b64 in enumerate(crops_base64):
                    pct = (2 * N + idx) / (3 * N)
                    yield json.dumps({
                        "type": "progress",
                        "percentage": pct,
                        "message": f"Pass 3/3 (Arbiter VLM): Crop {idx+1}/{N}..."
                    }) + "\n"
                    
                    if not img_b64:
                        final_results.append("")
                        continue
                    
                    result_a = results_a[idx]
                    result_b = results_b[idx]

                    try:
                        arbiter.reset()
                        
                        user_prompt = (
                            f"Expert OCR transcriptions candidates:\n"
                            f"- Candidate A: {result_a}\n"
                            f"- Candidate B: {result_b}\n\n"
                            f"Source Language: {source_lang}\n"
                            f"Target Language: {target_lang}\n\n"
                            f"Verify against the image. Output ONLY the corrected final Japanese transcription. Do not explain, translate, or conversationalize."
                        )

                        if MODELS_CONFIG[arbiter_model_id].get("handler_class") == "Llava15ChatHandler":
                            user_text = f"{system_instruction}\n\n{user_prompt}"
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image_url", "image_url": {"url": img_b64}},
                                        {"type": "text", "text": user_text}
                                    ]
                                }
                            ]
                        else:
                            messages = [
                                {
                                    "role": "system",
                                    "content": system_instruction
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image_url", "image_url": {"url": img_b64}},
                                        {"type": "text", "text": user_prompt}
                                    ]
                                }
                            ]

                        try:
                            response = arbiter.create_chat_completion(messages=messages)
                        except Exception as inf_err:
                            if _n_gpu_layers_used != 0:
                                sys.stderr.write(f"[Server Dispatcher] Arbiter VLM GPU execution failed. Re-routing to CPU...\n")
                                arbiter = get_or_load_model(arbiter_model_id, force_cpu=True)
                                arbiter.reset()
                                response = arbiter.create_chat_completion(messages=messages)
                            else:
                                raise inf_err

                        text_final = response["choices"][0]["message"]["content"]
                        final_results.append(text_final.strip() if text_final else "")
                        sys.stderr.write(f"[Server Dispatcher] Ensemble crop {idx} final: '{result_a}' / '{result_b}' -> '{text_final.strip()}'\n")
                    except Exception as ex_c:
                        sys.stderr.write(f"[Server Dispatcher] Arbiter consensus error on crop {idx}: {ex_c}\n")
                        final_results.append(result_a or result_b or "") # fallback to any expert
            finally:
                unload_model(arbiter_model_id)

            yield json.dumps({
                "type": "progress",
                "percentage": 1.0,
                "message": "Completed Ensemble OCR Consensus."
            }) + "\n"
            yield json.dumps({"type": "result", "results": final_results}) + "\n"

        return StreamingResponse(event_generator(), media_type="application/x-ndjson")
