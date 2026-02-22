from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.room import room_router
from api.stream import stream_router
from services.video_processing.models import load_models, close_models, models


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup и загрузка моделей
    load_models()
    yield
    # Shutdown и очистка ресурсов
    close_models()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "*"],  # Streamlit по умолчанию
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@stream_router.get('/health')
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(models) > 0,
        "version": "1.0.0"
    }


app.include_router(stream_router)
app.include_router(room_router)
