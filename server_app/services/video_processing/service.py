from uuid import UUID
from functools import lru_cache

from cv2.typing import MatLike

from .face_analysis_pipeline import FaceAnalysisPipeline, FaceAnalizResult, make_face_analysis_pipeline


class FaceAnalysisPipelineService:
    def __init__(self):
        self._analyzers: dict[UUID, FaceAnalysisPipeline] = {}

    def analyze(self, client_id: UUID, image: MatLike) -> FaceAnalizResult:
        if client_id not in self._analyzers:
            self._analyzers[client_id] = make_face_analysis_pipeline()
        return self._analyzers[client_id].analyze(image)

@lru_cache()
def get_face_analysis_pipeline_service():
    return FaceAnalysisPipelineService()
