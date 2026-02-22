from uuid import uuid4

import pytest


@pytest.fixture
def room_service():
    from services.room import RoomService
    service = RoomService()
    return service


@pytest.fixture
def client():
    from services.room import Client
    return Client(id_=uuid4(), name="test_client")
