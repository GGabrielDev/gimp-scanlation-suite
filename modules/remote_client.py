import sys
import json
import requests

_metadata_cache = {}

def get_model_metadata(model_id):
    """
    Looks up a model's metadata from the global cache, falling back to local file if needed.
    """
    global _metadata_cache
    if model_id in _metadata_cache:
        return _metadata_cache[model_id]
        
    # Lazy load from fallback if cache is empty
    import os
    current_dir = os.path.dirname(os.path.realpath(__file__))
    fallback_file = os.path.join(current_dir, "models_fallback.json")
    if os.path.exists(fallback_file):
        try:
            with open(fallback_file, "r", encoding="utf-8") as f:
                fallback_data = json.load(f)
                for m in fallback_data:
                    _metadata_cache[m["model_id"]] = m
        except Exception:
            pass
            
    return _metadata_cache.get(model_id)

def get_available_models(task_type, api_url):
    """
    Queries the remote dispatcher server for a list of available models for a given task type.
    Falls back to the local models_fallback.json file if the server is offline or returns an error.
    """
    global _metadata_cache
    
    # 1. Try querying the remote server if api_url is provided
    if api_url:
        try:
            url = f"{api_url.rstrip('/')}/api/v1/models"
            sys.stderr.write(f"[Remote Client] Querying models from: {url} (task_type={task_type})...\n")
            response = requests.get(url, params={"task_type": task_type}, timeout=2.0)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("models")
                if isinstance(models, list) and len(models) > 0:
                    sys.stderr.write(f"[Remote Client] Discovered models from server.\n")
                    # Update cache
                    for m in models:
                        _metadata_cache[m["model_id"]] = m
                    return models
            sys.stderr.write(f"[Remote Client] Server returned status {response.status_code}. Using local fallback.\n")
        except Exception as e:
            sys.stderr.write(f"[Remote Client] Connection failed ({e}). Using local fallback.\n")
            
    # 2. Local fallback parsing
    import os
    current_dir = os.path.dirname(os.path.realpath(__file__))
    fallback_file = os.path.join(current_dir, "models_fallback.json")
    fallback_models = []
    if os.path.exists(fallback_file):
        try:
            with open(fallback_file, "r", encoding="utf-8") as f:
                fallback_data = json.load(f)
                # Cache all fallback models
                for m in fallback_data:
                    _metadata_cache[m["model_id"]] = m
                
                # Filter based on task_type
                if task_type == "ocr":
                    fallback_models = [m for m in fallback_data if "ocr_expert" in m.get("tasks", [])]
                elif task_type == "arbitration":
                    fallback_models = [m for m in fallback_data if "ocr_arbiter" in m.get("tasks", [])]
                elif task_type == "translate":
                    fallback_models = [m for m in fallback_data if "translate" in m.get("tasks", [])]
                elif task_type == "inpaint":
                    fallback_models = [m for m in fallback_data if "inpaint" in m.get("tasks", [])]
        except Exception as err:
            sys.stderr.write(f"[Remote Client] Failed to load models_fallback.json: {err}\n")
            
    return fallback_models

def dispatch_batch(task_type, model_id, batch_payload, api_url, options=None, progress_callback=None):
    """
    Sends an inference task payload to the remote dispatcher server and streams progress updates.
    """
    if not api_url:
        raise ValueError("Remote API URL must be specified.")
        
    url = f"{api_url.rstrip('/')}/api/v1/dispatch"
    payload = {
        "task_type": task_type,
        "model_id": model_id,
        "batch_payload": batch_payload,
        "options": options or {}
    }
    
    sys.stderr.write(f"[Remote Client] Dispatching '{task_type}' task (model='{model_id}') to: {url}...\n")
    
    # Remove timeout limit for lazy-loading larger models and streaming chunked responses on slow CPUs
    response = requests.post(url, json=payload, timeout=None, stream=True)
    
    if response.status_code != 200:
        error_msg = f"Server returned error code {response.status_code}"
        try:
            content = response.content.decode("utf-8")
            data = json.loads(content)
            detail = data.get("detail", "")
            if detail:
                error_msg += f": {detail}"
        except Exception:
            pass
        raise RuntimeError(error_msg)
        
    results = []
    try:
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if data.get("type") == "progress":
                    if progress_callback:
                        progress_callback(data.get("percentage", 0.0), data.get("message", ""))
                elif data.get("type") == "result":
                    results = data.get("results", [])
    except Exception as parse_err:
        raise RuntimeError(f"Failed to parse streaming server response: {parse_err}")
        
    return results
