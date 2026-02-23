from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.room import room_router
from app.api.stream import stream_router
from app.core.config import settings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@stream_router.get('/health')
async def health_check():
    return {
        "status": "healthy",
        "version": settings.app_version
    }


app.include_router(stream_router)
app.include_router(room_router)
