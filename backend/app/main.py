"""
Модуль основного приложения FastAPI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.room import room_router
from app.api.stream import stream_router
from app.core.config import settings

app = FastAPI(
    title="API распознавания эмоций",
    description="REST API для детекции лиц и распознавания эмоций в реальном времени",
    version=settings.app_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@stream_router.get("/health")
async def health_check():
    """
    Проверка работоспособности сервиса.

    Returns:
        dict: Статус сервиса и версия приложения
    """
    return {"status": "healthy", "version": settings.app_version}


app.include_router(stream_router)
app.include_router(room_router)
