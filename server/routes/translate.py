from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from server.core.config import BatchRequest, MODELS_CONFIG
from server.services.translate_service import run_translate_generator

router = APIRouter()

@router.post("/translate")
def dispatch_translate(request: BatchRequest):
    """
    Direct route handler for Translation.
    """
    model = request.model_id
    options = request.options or {}

    if model not in MODELS_CONFIG:
        raise HTTPException(status_code=400, detail=f"Translation model '{model}' is not registered on the server.")

    return StreamingResponse(
        run_translate_generator(model, request.batch_payload, options),
        media_type="application/x-ndjson"
    )
