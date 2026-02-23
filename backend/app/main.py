from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.room import room_router
from api.stream import stream_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501",  # Streamlit по умолчанию
                   "http://localhost:63342"],  # PyCharm webserver default port for opening html files
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@stream_router.get('/health')
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


app.include_router(stream_router)
app.include_router(room_router)
