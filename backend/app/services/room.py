"""
Модуль сервиса управления комнатами и клиентами.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import cast
from uuid import UUID

from cv2.typing import MatLike
from fastapi import FastAPI, Request, WebSocket

from app.services.video_processing.face_analysis_pipeline import OneFaceMetricsAnalyzeResult

logger = logging.getLogger(__name__)


class RoomNotFoundError(Exception):
    """Исключение, выбрасываемое при отсутствии комнаты."""

    pass


class ClientNotFoundError(Exception):
    """Исключение, выбрасываемое при отсутствии клиента."""

    pass


@dataclass
class Client:
    """
    Представляет клиента в комнате для видеопотока.

    Attributes:
        id_: Уникальный идентификатор клиента
        name: Имя клиента (опционально)
        src_frame: Исходный кадр
        prc_frame: Обработанный кадр
        metrics: Результаты анализа лица
        _frame_queue: Очередь для передачи кадров
    """

    id_: UUID
    name: str | None = None
    src_frame: MatLike | None = None
    prc_frame: MatLike | None = None
    metrics: list[OneFaceMetricsAnalyzeResult] | None = None
    _frame_queue: asyncio.Queue | None = field(default=None, init=False, repr=False)
    _source_closed: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    def get_frame_queue(self) -> asyncio.Queue:
        """
        Получает очередь кадров для клиента.

        Returns:
            asyncio.Queue: Очередь для передачи видеокадров
        """
        if self._frame_queue is None:
            self._frame_queue = asyncio.Queue(maxsize=1)
        return self._frame_queue

    def get_source_closed(self) -> asyncio.Event:
        """
        Получает событие закрытия клиента

        Returns:
           asyncio.Event: Событие закрытия клиента
        """
        return self._source_closed


@dataclass
class Room:
    """
    Представляет комнату для группового видеопотока.

    Attributes:
        id_: Уникальный идентификатор комнаты
        clients: Словарь клиентов в комнате
    """

    id_: str
    clients: dict[UUID, Client]


class RoomService:
    """
    Сервис управления комнатами и клиентами.

    Обеспечивает создание, поиск и удаление комнат и клиентов,
    а также безопасный доступ к данным через asyncio.Lock.
    """

    def __init__(self):
        """Инициализирует сервис управления комнатами."""
        self._rooms: dict[str, Room] = {}
        self._lock = asyncio.Lock()

    async def get_rooms(self) -> list[Room]:
        """
        Получает список всех активных комнат.

        Returns:
            list[Room]: Список объектов Room
        """
        async with self._lock:
            return list(self._rooms.values())

    async def add_client(self, room_id: str, client: Client) -> None:
        """
        Добавляет клиента в комнату.

        Если комната не существует, она будет создана.

        Args:
            room_id: ID комнаты
            client: Объект клиента для добавления
        """
        async with self._lock:
            if room_id not in self._rooms:
                self._rooms[room_id] = Room(id_=room_id, clients={})
                logger.info(f"Room {room_id} created")
            self._rooms[room_id].clients[client.id_] = client
            logger.debug(f"Client {client.id_} added to room {room_id}")

    async def get_client(self, room_id: str, client_id: UUID) -> Client:
        """
        Получает клиента из комнаты.

        Args:
            room_id: ID комнаты
            client_id: ID клиента

        Returns:
            Client: Объект клиента

        Raises:
            RoomNotFoundError: Если комната не найдена
            ClientNotFoundError: Если клиент не найден в комнате
        """
        async with self._lock:
            if room_id not in self._rooms:
                raise RoomNotFoundError(f"Room {room_id} not found")
            if client_id not in self._rooms[room_id].clients:
                raise ClientNotFoundError(f"Client {client_id} not found in room {room_id}")
            return self._rooms[room_id].clients[client_id]

    async def remove_client(self, room_id: str, client: Client) -> None:
        """
        Удаляет клиента из комнаты.

        Если в комнате больше нет клиентов, комната также удаляется.

        Args:
            room_id: ID комнаты
            client: Объект клиента для удаления
        """
        async with self._lock:
            if room_id not in self._rooms:
                return
            if client.id_ not in self._rooms[room_id].clients:
                return
            del self._rooms[room_id].clients[client.id_]
            logger.debug(f"Client {client.id_} removed from room {room_id}")
            if len(self._rooms[room_id].clients) == 0:
                del self._rooms[room_id]
                logger.info(f"Room {room_id} deleted (empty)")

    async def get_clients_in_room(self, room_id: str) -> list[Client]:
        """
        Получает всех клиентов в указанной комнате.

        Args:
            room_id: ID комнаты

        Returns:
            list[Client]: Список клиентов в комнате

        Raises:
            RoomNotFoundError: Если комната не найдена
        """
        async with self._lock:
            if room_id not in self._rooms:
                raise RoomNotFoundError(f"Room {room_id} not found")
            return list(self._rooms[room_id].clients.values())


def get_room_service(
    request: Request = None,  # type: ignore[assignment]
    websocket: WebSocket = None,  # type: ignore[assignment]
) -> RoomService:
    """
    Получает экземпляр RoomService из состояния приложения FastAPI.

    Если сервис еще не инициализирован, создает новый экземпляр.

    Args:
        request: Объект HTTP запроса FastAPI (опционально)
        websocket: Объект WebSocket соединения (опционально)

    Returns:
        RoomService: Экземпляр сервиса управления комнатами

    Raises:
        RuntimeError: Если не передан ни request, ни websocket
    """
    app: FastAPI
    if request is not None:
        app = request.app
    elif websocket is not None:
        app = websocket.app
    else:
        raise RuntimeError('get_room_service expected "request" or "websocket" arg, got Nones')
    if not hasattr(app.state, "room_service"):
        app.state.room_service = RoomService()
    return cast(RoomService, app.state.room_service)
