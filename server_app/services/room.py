import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from uuid import UUID

from cv2.typing import MatLike


class RoomNotFoundError(Exception):
    pass


class ClientNotFoundError(Exception):
    pass


@lru_cache()
def get_room_service() -> "RoomService":
    return RoomService()


@dataclass
class Client:
    id_: UUID
    name: str | None = None
    src_frame: MatLike | None = None
    prc_frame: MatLike | None = None
    metrics: Any | None = None


@dataclass
class Room:
    id_: str
    clients: dict[UUID, Client]


class RoomService:
    def __init__(self):
        print('init')
        self.rooms: dict[str, Room] = {}
        self._lock = asyncio.Lock()

    async def get_rooms(self) -> list[Room]:
        async with self._lock:
            return list(self.rooms.values())

    async def add_client(self, room_id: str, client: Client) -> None:
        async with self._lock:
            if room_id not in self.rooms:
                self.rooms[room_id] = Room(id_=room_id, clients={})
            self.rooms[room_id].clients[client.id_] = client

    async def get_client(self, room_id: str, client_id: UUID) -> Client:
        async with self._lock:
            if room_id not in self.rooms:
                raise RoomNotFoundError(f"Room {room_id} not found")
            if client_id not in self.rooms[room_id].clients:
                raise ClientNotFoundError(f"Client {client_id} not found in room {room_id}")
            return self.rooms[room_id].clients[client_id]

    async def remove_client(self, room_id: str, client: Client) -> None:
        async with self._lock:
            if room_id not in self.rooms:
                return
            if client.id_ not in self.rooms[room_id].clients:
                return
            del self.rooms[room_id].clients[client.id_]
            if len(self.rooms[room_id].clients) == 0:
                del self.rooms[room_id]

    async def get_clients_in_room(self, room_id: str) -> list[Client]:
        async with self._lock:
            if room_id not in self.rooms:
                raise RoomNotFoundError(f"Room {room_id} not found")
            return list(self.rooms[room_id].clients.values())
