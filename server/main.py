import os
import sys

# Auto-configure ROCm GFX override for AMD APUs/GPUs (like BC-250 / gfx1013) if not already set.
# This must run before torch or any GPU/ROCm libraries are imported.
if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    # gfx1013 (Cyan Skillfish on BC-250) is RDNA-based and runs fine when overridden to 10.3.0 (gfx1030)
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

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

from server.core.config import MODELS_CONFIG, BatchRequest
from server.routes import ocr, inpaint, translate, detect

app = FastAPI(title="GIMP Scanlation Suite Remote Dispatch Server")

# Include the modular sub-routers
app.include_router(ocr.router, prefix="/api/v1")
app.include_router(inpaint.router, prefix="/api/v1")
app.include_router(translate.router, prefix="/api/v1")
app.include_router(detect.router, prefix="/api/v1")

@app.get("/api/v1/models")
def list_models(task_type: str = Query(..., description="The capability tag or task type to filter by, e.g. ocr_expert, ocr_arbiter, translate, inpaint")):
    """
    Returns the list of available models supported by the server for the specified task type or capability tag.
    """
    tag = task_type
    if task_type == "ocr":
        tag = "ocr_expert"
    elif task_type == "arbitration":
        tag = "ocr_arbiter"
        
    filtered = [v for v in MODELS_CONFIG.values() if tag in v.tasks]
    return {"models": filtered}

@app.post("/api/v1/dispatch")
def dispatch(request: BatchRequest):
    """
    Centralized dispatch route maintaining backward compatibility with the GIMP client.
    Delegates requests to the appropriate router logic.
    """
    task = request.task_type
    model = request.model_id

    # Promote to ensemble_ocr if model is Ensemble
    if model == "Ensemble":
        task = "ensemble_ocr"

    if task not in ["ocr", "ensemble_ocr", "inpaint", "translate", "detect"]:
        raise HTTPException(status_code=400, detail=f"Unsupported task_type '{task}'")

    if task in ["ocr", "ensemble_ocr"]:
        return ocr.dispatch_ocr(request)
    elif task == "inpaint":
        return inpaint.dispatch_inpaint(request)
    elif task == "translate":
        return translate.dispatch_translate(request)
    elif task == "detect":
        return detect.dispatch_detect(request)

if __name__ == "__main__":
    import uvicorn
    sys.stderr.write("[Server main] Starting uvicorn server...\n")
    uvicorn.run(app, host="0.0.0.0", port=7890)
