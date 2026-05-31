from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from server.core.config import BatchRequest, MODELS_CONFIG
from server.services.inpaint_service import run_inpaint_generator

router = APIRouter()

@router.post("/inpaint")
def dispatch_inpaint(request: BatchRequest):
    """
    Direct route handler for Inpainting.
    """
    model = request.model_id
    options = request.options or {}

    if model not in MODELS_CONFIG:
        raise HTTPException(status_code=400, detail=f"Inpainting model '{model}' is not registered on the server.")

    return StreamingResponse(
        run_inpaint_generator(model, request.batch_payload, options),
        media_type="application/x-ndjson"
    )
