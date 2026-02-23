from uuid import UUID

from cv2.typing import MatLike
from fastapi import Request, WebSocket, FastAPI

from .face_analysis_pipeline import FaceAnalysisPipeline, FaceAnalizResult, make_face_analysis_pipeline


class FaceAnalysisPipelineService:
    def __init__(self):
        self._analyzers: dict[UUID, FaceAnalysisPipeline] = {}

    def analyze(self, client_id: UUID, image: MatLike) -> FaceAnalizResult:
        if client_id not in self._analyzers:
            self._analyzers[client_id] = make_face_analysis_pipeline()
        return self._analyzers[client_id].analyze(image)


def get_face_analysis_pipeline_service(request: Request = None, websocket: WebSocket = None) -> FaceAnalysisPipelineService:
    if request is not None:
        app: FastAPI = request.app
    elif websocket is not None:
        app: FastAPI = websocket.app
    else:
        raise RuntimeError('get_face_analysis_pipeline_service expected "request" or "websocket" arg, got Nones')
    if not hasattr(app.state, 'face_analysis_pipeline_service'):
        app.state.face_analysis_pipeline_service = FaceAnalysisPipelineService()
    return app.state.face_analysis_pipeline_service
