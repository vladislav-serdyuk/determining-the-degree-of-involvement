import asyncio
import base64
import warnings
from typing import Annotated
from uuid import uuid4, UUID

import cv2
import numpy as np
from cv2 import error
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, status, Query

from models import models
from services.room import RoomService, Client, RoomNotFoundError, ClientNotFoundError, get_room_service
from video_processing import FaceAnalysisPipeline, EngagementCalculator

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


@stream_router.websocket('/ws/rooms/{room_id}/stream')
async def stream(websocket: WebSocket, room_id: str,
                 name: Annotated[str | None, Query(max_length=30)] = None,
                 room_service: RoomService = Depends(get_room_service)):
    analyzer: FaceAnalysisPipeline = models['analyzer']
    engagement_calculator = EngagementCalculator()   # per-session экземпляр
    await websocket.accept()
    client: Client = Client(id_=uuid4(), name=name)
    await room_service.add_client(room_id, client)
    try:
        while True:
            data = await websocket.receive_json()
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

            # Обогатить каждое лицо данными по engagement
            for face_result in results:
                ear_data = face_result.get('ear')
                head_pose_data = face_result.get('head_pose')
                emotion = face_result.get('emotion', 'Neutral')
                confidence = face_result.get('confidence', 0.5)

                engagement = engagement_calculator.calculate(
                    emotion=emotion,
                    emotion_confidence=confidence,
                    ear_data=ear_data,
                    head_pose_data=head_pose_data
                )
                face_result['engagement'] = engagement

            client.src_frame = img
            client.prc_frame = new_img
            client.metrics = results
            _, buffer = cv2.imencode('.jpg', new_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            results_serializable = convert_to_serializable(results)
            await websocket.send_json({
                'image': img_base64,
                'results': results_serializable
            })
    except WebSocketDisconnect:
        engagement_calculator.reset()   # очистка при разрыве
    finally:
        await room_service.remove_client(room_id, client)


@stream_router.websocket('/ws/rooms/{room_id}/clients/{client_id}/output_stream')
async def client_stream(websocket: WebSocket, room_id: str, client_id: UUID,
                        room_service: RoomService = Depends(get_room_service)):
    await websocket.accept()
    try:
        client = await room_service.get_client(room_id, client_id)
    except (RoomNotFoundError, ClientNotFoundError) as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close(status.WS_1008_POLICY_VIOLATION)
        return
    try:
        while True:
            img = client.src_frame
            new_img = client.prc_frame
            results = client.metrics
            if img is None or new_img is None:
                await asyncio.sleep(0.05)
                continue
            _, buffer = cv2.imencode('.jpg', img)
            img_src_base64 = base64.b64encode(buffer).decode('utf-8')
            _, buffer = cv2.imencode('.jpg', new_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            results_serializable = convert_to_serializable(results)
            await websocket.send_json({
                'image_src': img_src_base64,
                'image': img_base64,
                'results': results_serializable
            })
            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        pass
