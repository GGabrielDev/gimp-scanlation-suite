from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from server.core.config import BatchRequest
from server.services.detect_service import run_detect_generator

router = APIRouter()

@router.post("/detect")
def dispatch_detect(request: BatchRequest):
    """
    Direct route handler for Detection.
    """
    options = request.options or {}
    return StreamingResponse(
        run_detect_generator(request.model_id, request.batch_payload, options),
        media_type="application/x-ndjson"
    )
