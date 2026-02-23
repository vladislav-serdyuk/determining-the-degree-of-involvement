from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.services.room import RoomService, RoomNotFoundError, get_room_service

room_router = APIRouter()


@room_router.get('/rooms')
async def get_rooms(room_service: Annotated[RoomService, Depends(get_room_service)]):
    rooms = await room_service.get_rooms()
    return [room.id_ for room in rooms]


@room_router.get('/rooms/{room_id}/clients')
async def get_clients(room_id: str, room_service: Annotated[RoomService, Depends(get_room_service)]):
    try:
        clients = await room_service.get_clients_in_room(room_id)
        return [(item.name, item.id_) for item in clients]
    except RoomNotFoundError:
        raise HTTPException(status_code=404, detail=f"Room {room_id} not found")
