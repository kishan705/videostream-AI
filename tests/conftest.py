import pytest
import os
from unittest.mock import MagicMock, patch
import fakeredis
from fastapi.testclient import TestClient

os.environ["API_SECRET_KEY"] = "test-token"

@pytest.fixture(autouse=True)
def mock_io():
    fake_r = fakeredis.FakeRedis(decode_responses=True)
    with patch("app.api.v1.videos.redis_client", fake_r), \
         patch("workers.tasks.redis_client", fake_r), \
         patch("app.api.v1.videos.get_vector_store") as mock_get_qdrant, \
         patch("app.api.v1.videos.celery_client.send_task") as mock_celery_send_task, \
         patch("workers.celery_app.celery_app.send_task") as mock_worker_celery_send_task, \
         patch("workers.tasks.get_vector_store") as mock_tasks_get_qdrant:
        
        mock_qdrant = MagicMock()
        mock_qdrant.search_similarity.return_value = [{"video_id": "11111111-1111-1111-1111-111111111111", "score": 0.99}]
        mock_qdrant.upsert_embeddings.return_value = True
        mock_get_qdrant.return_value = mock_qdrant
        mock_tasks_get_qdrant.return_value = mock_qdrant

        mock_async_result = MagicMock()
        mock_async_result.get.return_value = [0.1] * 1152
        mock_celery_send_task.return_value = mock_async_result
        mock_worker_celery_send_task.return_value = mock_async_result
        
        yield {
            "redis": fake_r,
            "qdrant": mock_qdrant,
            "celery": mock_celery_send_task,
            "worker_celery": mock_worker_celery_send_task
        }

@pytest.fixture
def fake_redis(mock_io):
    return mock_io["redis"]

@pytest.fixture
def mock_qdrant(mock_io):
    return mock_io["qdrant"]

@pytest.fixture
def mock_celery(mock_io):
    return mock_io["celery"]

@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)

@pytest.fixture
def sample_video_bytes():
    return b"\x00" * 1024

@pytest.fixture
def sample_video_id():
    return "11111111-1111-1111-1111-111111111111"

@pytest.fixture
def auth_header():
    return {"Authorization": "Bearer test-token"}
