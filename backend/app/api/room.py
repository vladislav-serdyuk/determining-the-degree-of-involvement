"""
Модуль REST эндпоинтов для управления комнатами.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.services.room import RoomNotFoundError, RoomService, get_room_service

room_router = APIRouter()


@room_router.get("/rooms")
async def get_rooms(room_service: Annotated[RoomService, Depends(get_room_service)]):
    """
    Получение списка всех активных комнат.

    Args:
        room_service: Сервис управления комнатами

    Returns:
        list: Список ID активных комнат
    """
    rooms = await room_service.get_rooms()
    return [room.id_ for room in rooms]


@room_router.get("/rooms/{room_id}/clients")
async def get_clients(
    room_id: str, room_service: Annotated[RoomService, Depends(get_room_service)]
):
    """
    Получение списка клиентов в указанной комнате.

    Args:
        room_id: ID комнаты
        room_service: Сервис управления комнатами

    Returns:
        list: Список кортежей (имя, ID) клиентов

    Raises:
        HTTPException 404: Если комната не найдена
    """
    try:
        clients = await room_service.get_clients_in_room(room_id)
        return [(item.name, item.id_) for item in clients]
    except RoomNotFoundError:
        raise HTTPException(status_code=404, detail=f"Room {room_id} not found")
