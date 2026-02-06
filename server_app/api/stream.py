import base64
import tempfile
import warnings

import cv2
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from video_processing import AttentionAnalyzer, FaceDetector, EmotionRecognizer, \
    EyeAspectRatioAnalyzer, HeadPoseEstimator

stream_router = APIRouter()
face_detector = FaceDetector(min_detection_confidence=0.5)
emotion_recognizer = EmotionRecognizer(device='cuda' if torch.cuda.is_available() else 'cpu', window_size=15,
                                       confidence_threshold=0.55, ambiguity_threshold=0.15)
eye_analyzer = EyeAspectRatioAnalyzer()
head_analyzer = HeadPoseEstimator()
face_detector_and_emotion_recognizer = AttentionAnalyzer(face_detector, emotion_recognizer, eye_analyzer, head_analyzer)


@stream_router.websocket('/ws/stream')
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()  # TODO add validation
            image_b64 = data.get("image")
            image_bytes = base64.b64decode(image_b64)
            with tempfile.TemporaryFile() as tmp_img_file:
                tmp_img_file.write(image_bytes)
                img = cv2.imread(tmp_img_file.name)
            if img is None:
                await websocket.close(status.WS_1011_INTERNAL_ERROR)
                warnings.warn('Could not read img from fs in /ws/stream')
                continue
            new_img, results = face_detector_and_emotion_recognizer.analyze(img)
            with tempfile.TemporaryFile() as tmp_img_file:
                cv2.imwrite(tmp_img_file.name, new_img)
                new_img_bytes = tmp_img_file.read()
            img_base64 = base64.b64encode(new_img_bytes)
            await websocket.send_json({
                'image': img_base64,
                'results': results
            })
    except WebSocketDisconnect:
        pass
