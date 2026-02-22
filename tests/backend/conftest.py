from unittest.mock import patch, MagicMock
from uuid import uuid4

import pytest


@pytest.fixture
def mock_models():
    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = (MagicMock(), [{'emotion': 'neutral', 'ear': 0.3}])
    return {'analyzer': mock_analyzer}


@pytest.fixture
def room_service():
    from services.room import RoomService
    service = RoomService()
    return service


@pytest.fixture
def client():
    from services.room import Client
    return Client(id_=uuid4(), name="test_client")


@pytest.fixture
def app(mock_models):
    with patch('models.models', mock_models):
        with patch('models.load_models'):
            with patch('models.close_models'):
                from main import app
                yield app
