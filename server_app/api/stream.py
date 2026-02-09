import base64
import warnings

import cv2
import numpy as np
from cv2 import error
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from models import models
from video_processing import FaceAnalysisPipeline

stream_router = APIRouter()


def convert_to_serializable(obj):
    """Recursively converts numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


@stream_router.websocket('/ws/stream')
async def stream(websocket: WebSocket):
    analyzer: FaceAnalysisPipeline = models['analyzer']
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()  # TODO add validation
            image_b64 = data.get("image")
            try:
                image_bytes = base64.b64decode(image_b64)
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except (ValueError, TypeError, error) as e:
                await websocket.send_json({"error": f"Failed to decode image: {str(e)}"})
                continue
            if img is None:
                warnings.warn('Could not decode img /ws/stream')
                continue
            new_img, results = analyzer.analyze(img)
            _, buffer = cv2.imencode('.jpg', new_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            results_serializable = convert_to_serializable(results)
            await websocket.send_json({
                'image': img_base64,
                'results': results_serializable
            })
    except WebSocketDisconnect:
        pass
