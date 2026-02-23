import asyncio
from dataclasses import dataclass
from uuid import UUID

from cv2.typing import MatLike
from fastapi import Request, WebSocket, FastAPI

from app.services.video_processing.face_analysis_pipeline import OneFaceMetricsAnalizResult


class RoomNotFoundError(Exception):
    pass


class ClientNotFoundError(Exception):
    pass


@dataclass
class Client:
    id_: UUID
    name: str | None = None
    src_frame: MatLike | None = None
    prc_frame: MatLike | None = None
    metrics: list[OneFaceMetricsAnalizResult] | None = None
    _frame_queue: asyncio.Queue | None = None

    def get_frame_queue(self) -> asyncio.Queue:
        if self._frame_queue is None:
            self._frame_queue = asyncio.Queue(maxsize=1)
        return self._frame_queue


@dataclass
class Room:
    id_: str
    clients: dict[UUID, Client]


class RoomService:
    def __init__(self):
        print('init')
        self._rooms: dict[str, Room] = {}
        self._lock = asyncio.Lock()

    async def get_rooms(self) -> list[Room]:
        async with self._lock:
            return list(self._rooms.values())

    async def add_client(self, room_id: str, client: Client) -> None:
        async with self._lock:
            if room_id not in self._rooms:
                self._rooms[room_id] = Room(id_=room_id, clients={})
            self._rooms[room_id].clients[client.id_] = client

    async def get_client(self, room_id: str, client_id: UUID) -> Client:
        async with self._lock:
            if room_id not in self._rooms:
                raise RoomNotFoundError(f"Room {room_id} not found")
            if client_id not in self._rooms[room_id].clients:
                raise ClientNotFoundError(f"Client {client_id} not found in room {room_id}")
            return self._rooms[room_id].clients[client_id]

    async def remove_client(self, room_id: str, client: Client) -> None:
        async with self._lock:
            if room_id not in self._rooms:
                return
            if client.id_ not in self._rooms[room_id].clients:
                return
            del self._rooms[room_id].clients[client.id_]
            if len(self._rooms[room_id].clients) == 0:
                del self._rooms[room_id]

    async def get_clients_in_room(self, room_id: str) -> list[Client]:
        async with self._lock:
            if room_id not in self._rooms:
                raise RoomNotFoundError(f"Room {room_id} not found")
            return list(self._rooms[room_id].clients.values())


def get_room_service(request: Request = None, websocket: WebSocket = None) -> RoomService:
    if request is not None:
        app: FastAPI = request.app
    elif websocket is not None:
        app: FastAPI = websocket.app
    else:
        raise RuntimeError('get_room_service expected "request" or "websocket" arg, got Nones')
    if not hasattr(app.state, 'room_service'):
        app.state.room_service = RoomService()
    return app.state.room_service
