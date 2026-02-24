"""
Модуль сервиса пайплайна анализа лиц для FastAPI.
"""

from typing import cast
from uuid import UUID

from cv2.typing import MatLike
from fastapi import FastAPI, Request, WebSocket

from .face_analysis_pipeline import (
    FaceAnalizResult,
    FaceAnalysisPipeline,
    make_face_analysis_pipeline,
)


class FaceAnalysisPipelineService:
    """
    Сервис для управления анализом лиц и эмоций.

    Создает и хранит отдельный экземпляр FaceAnalysisPipeline для каждого клиента.
    """

    def __init__(self):
        """Инициализирует сервис."""
        self._analyzers: dict[UUID, FaceAnalysisPipeline] = {}

    def analyze(self, client_id: UUID, image: MatLike) -> FaceAnalizResult:
        """
        Анализирует изображение для конкретного клиента.

        Если анализатор для клиента еще не создан, создает новый.

        Args:
            client_id: ID клиента
            image: Изображение для анализа

        Returns:
            FaceAnalizResult: Результат анализа с обработанным изображением и метриками
        """
        if client_id not in self._analyzers:
            self._analyzers[client_id] = make_face_analysis_pipeline()
        return self._analyzers[client_id].analyze(image)


def get_face_analysis_pipeline_service(
    request: Request = None,  # type: ignore[assignment]
    websocket: WebSocket = None,  # type: ignore[assignment]
) -> FaceAnalysisPipelineService:
    """
    Получает экземпляр FaceAnalysisPipelineService из состояния приложения FastAPI.

    Если сервис еще не инициализирован, создает новый экземпляр.

    Args:
        request: Объект HTTP запроса FastAPI (опционально)
        websocket: Объект WebSocket соединения (опционально)

    Returns:
        FaceAnalysisPipelineService: Экземпляр сервиса анализа лиц

    Raises:
        RuntimeError: Если не передан ни request, ни websocket
    """
    app: FastAPI
    if request is not None:
        app = request.app
    elif websocket is not None:
        app = websocket.app
    else:
        raise RuntimeError('get_face_analysis_pipeline_service expected "request" or "websocket" arg, got Nones')
    if not hasattr(app.state, "face_analysis_pipeline_service"):
        app.state.face_analysis_pipeline_service = FaceAnalysisPipelineService()
    return cast(FaceAnalysisPipelineService, app.state.face_analysis_pipeline_service)
