import json
import sys
import time

def run_detect_generator(model: str, batch_payload: list, options: dict):
    """
    Placeholder/Stub for remote bubble and text region detection.
    This service will handle server-side YOLO/PP-DocLayoutV3 detection in the future.
    """
    start_time = time.time()
    sys.stderr.write(
        f"[Server Detect Service] Received detection request | Model: {model} | "
        f"Payload size: {len(batch_payload)} | Options: {options}\n"
    )
    sys.stderr.write("[Server Detect Service] WARNING: Server-side detection is not yet implemented. Returning stub.\n")
    
    yield json.dumps({"type": "progress", "percentage": 0.0, "message": "Server-side detection is not yet implemented. Please use local detection."}) + "\n"
    yield json.dumps({"type": "result", "results": []}) + "\n"
    
    elapsed = time.time() - start_time
    sys.stderr.write(f"[Server Detect Service] Completed request in {elapsed:.4f} seconds.\n")
