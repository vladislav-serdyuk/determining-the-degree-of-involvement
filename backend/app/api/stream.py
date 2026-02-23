import asyncio
import base64
import warnings
from dataclasses import asdict
from typing import Annotated
from uuid import uuid4, UUID

import cv2
import numpy as np
from cv2 import error
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, status, Query, Path

from app.services.room import RoomService, Client, RoomNotFoundError, ClientNotFoundError, get_room_service
from app.services.video_processing import get_face_analysis_pipeline_service, FaceAnalysisPipelineService
from services.video_processing.models import models

stream_router = APIRouter()


@stream_router.websocket('/ws/rooms/{room_id}/stream')
async def stream(websocket: WebSocket, room_service: Annotated[RoomService, Depends(get_room_service)],
                 analyzer_service: Annotated[FaceAnalysisPipelineService, Depends(get_face_analysis_pipeline_service)],
                 room_id: Annotated[str, Path(max_length=40)],
                 name: Annotated[str | None, Query(max_length=30)] = None):
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
            analyze_res = analyzer_service.analyze(client.id_, img)
            new_img = analyze_res.image
            results = analyze_res.metrics

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

            queue = client.get_frame_queue()
            frame_data = {
                'src': img,
                'prc': new_img,
                'results': results
            }
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(frame_data)
            _, buffer = cv2.imencode('.jpg', new_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            results_serializable = list(map(asdict, results))
            await websocket.send_json({
                'image': img_base64,
                'results': results_serializable
            })
    except WebSocketDisconnect:
        engagement_calculator.reset()   # очистка при разрыве
    finally:
        await room_service.remove_client(room_id, client)


@stream_router.websocket('/ws/rooms/{room_id}/clients/{client_id}/output_stream')
async def client_stream(websocket: WebSocket, room_id: Annotated[str, Path(max_length=40)], client_id: UUID,
                        room_service: Annotated[RoomService, Depends(get_room_service)]):
    await websocket.accept()
    try:
        client = await room_service.get_client(room_id, client_id)
    except (RoomNotFoundError, ClientNotFoundError) as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close(status.WS_1008_POLICY_VIOLATION)
        return
    try:
        while True:
            queue = client.get_frame_queue()
            try:
                frame_data = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            img = frame_data['src']
            new_img = frame_data['prc']
            results = frame_data['results']
            if img is None or new_img is None:
                continue
            _, buffer = cv2.imencode('.jpg', img)
            img_src_base64 = base64.b64encode(buffer).decode('utf-8')
            _, buffer = cv2.imencode('.jpg', new_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            results_serializable = list(map(asdict, results))
            await websocket.send_json({
                'image_src': img_src_base64,
                'image': img_base64,
                'results': results_serializable
            })

    except WebSocketDisconnect:
        pass
