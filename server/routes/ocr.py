from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from server.core.config import BatchRequest, MODELS_CONFIG
from server.services.ocr_service import run_single_ocr_generator, run_ensemble_ocr_generator

router = APIRouter()

@router.post("/ocr")
def dispatch_ocr(request: BatchRequest):
    """
    Direct route handler for OCR and Ensemble OCR.
    """
    model = request.model_id
    options = request.options or {}
    task = request.task_type

    if model == "Ensemble" or task == "ensemble_ocr":
        return StreamingResponse(
            run_ensemble_ocr_generator(model, request.batch_payload, options),
            media_type="application/x-ndjson"
        )
    else:
        if model not in MODELS_CONFIG:
            raise HTTPException(status_code=400, detail=f"Model ID '{model}' is not registered on the server.")
        return StreamingResponse(
            run_single_ocr_generator(model, request.batch_payload, options),
            media_type="application/x-ndjson"
        )
