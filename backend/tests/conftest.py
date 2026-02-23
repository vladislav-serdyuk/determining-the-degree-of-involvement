from uuid import uuid4

import pytest


@pytest.fixture
def room_service():
    from app.services.room import RoomService
    service = RoomService()
    return service


@pytest.fixture
def client():
    from app.services.room import Client
    return Client(id_=uuid4(), name="test_client")
