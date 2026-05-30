import sys
import json
import requests

def get_available_models(task_type, api_url):
    """
    Queries the remote dispatcher server for a list of available models for a given task type.
    Falls back to a default list if the server is offline or returns an error.
    """
    fallback_models = ["PaddleOCR", "olmOCR2_Q4", "olmOCR2_Q6", "olmOCR2_Q8", "DeepSeek", "DeepSeek-V4-Flash", "DeepSeek-V4-Pro"]
    
    if not api_url:
        return fallback_models
        
    try:
        url = f"{api_url.rstrip('/')}/api/v1/models"
        sys.stderr.write(f"[Remote Client] Querying models from: {url} (task_type={task_type})...\n")
        response = requests.get(url, params={"task_type": task_type}, timeout=2.0)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("models")
            if isinstance(models, list) and len(models) > 0:
                sys.stderr.write(f"[Remote Client] Discovered models: {models}\n")
                return [str(m) for m in models]
                
        sys.stderr.write(f"[Remote Client] Server returned status {response.status_code}. Using fallback models.\n")
    except Exception as e:
        sys.stderr.write(f"[Remote Client] Connection failed ({e}). Using fallback models.\n")
        
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
    
    # 600 second timeout for lazy-loading larger models and streaming chunked responses
    response = requests.post(url, json=payload, timeout=600.0, stream=True)
    
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
