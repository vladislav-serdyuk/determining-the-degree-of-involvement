from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import AsyncClient, ASGITransport

from services.room import RoomNotFoundError


@pytest.fixture
def mock_room_service():
    from services.room import RoomService
    mock_service = MagicMock(spec=RoomService)
    mock_service.get_rooms = AsyncMock(return_value=[])
    mock_service.get_clients_in_room = AsyncMock(return_value=[])
    return mock_service


@pytest.mark.asyncio
async def test_health_check():
    with patch('models.models', {'analyzer': MagicMock()}):
        with patch('models.load_models'):
            with patch('models.close_models'):
                from main import app
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.get("/health")
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "healthy"
                    assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_get_rooms_empty():
    from services.room import RoomService

    mock_service = MagicMock(spec=RoomService)
    mock_service.get_rooms = AsyncMock(return_value=[])

    with patch('models.models', {'analyzer': MagicMock()}):
        with patch('models.load_models'):
            with patch('models.close_models'):
                from main import app
                from api.room import get_room_service
                app.dependency_overrides[get_room_service] = lambda: mock_service

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.get("/rooms")
                    assert response.status_code == 200
                    assert response.json() == []

                app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_rooms_with_data():
    from services.room import RoomService, Room

    mock_service = MagicMock(spec=RoomService)
    mock_room = Room(id_="test_room", clients={})
    mock_service.get_rooms = AsyncMock(return_value=[mock_room])

    with patch('models.models', {'analyzer': MagicMock()}):
        with patch('models.load_models'):
            with patch('models.close_models'):
                from main import app
                from api.room import get_room_service
                app.dependency_overrides[get_room_service] = lambda: mock_service

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.get("/rooms")
                    assert response.status_code == 200
                    assert response.json() == ["test_room"]

                app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_clients_in_room_success():
    from services.room import RoomService, Client as RoomClient

    mock_service = MagicMock(spec=RoomService)
    test_client = RoomClient(id_=uuid4(), name="test_client")
    mock_service.get_clients_in_room = AsyncMock(return_value=[test_client])

    with patch('models.models', {'analyzer': MagicMock()}):
        with patch('models.load_models'):
            with patch('models.close_models'):
                from main import app
                from api.room import get_room_service
                app.dependency_overrides[get_room_service] = lambda: mock_service

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.get("/rooms/test_room/clients")
                    assert response.status_code == 200
                    data = response.json()
                    assert len(data) == 1
                    assert data[0][0] == "test_client"

                app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_clients_in_room_not_found():
    from services.room import RoomService

    mock_service = MagicMock(spec=RoomService)
    mock_service.get_clients_in_room = AsyncMock(side_effect=RoomNotFoundError("Room not found"))

    with patch('models.models', {'analyzer': MagicMock()}):
        with patch('models.load_models'):
            with patch('models.close_models'):
                from main import app
                from api.room import get_room_service
                app.dependency_overrides[get_room_service] = lambda: mock_service

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.get("/rooms/nonexistent/clients")
                    assert response.status_code == 404

                app.dependency_overrides.clear()
