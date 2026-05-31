import json

def run_detect_generator(model: str, batch_payload: list, options: dict):
    """
    Placeholder/Stub for remote bubble and text region detection.
    This service will handle server-side YOLO/PP-DocLayoutV3 detection in the future.
    """
    yield json.dumps({"type": "progress", "percentage": 0.0, "message": "Server-side detection is not yet implemented. Please use local detection."}) + "\n"
    yield json.dumps({"type": "result", "results": []}) + "\n"
