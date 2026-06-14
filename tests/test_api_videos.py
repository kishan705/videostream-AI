import pytest
import io
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

@pytest.fixture(autouse=True)
def mock_dependencies():
    with patch("app.api.v1.videos.redis_client") as mock_redis, \
         patch("app.api.v1.videos.vector_store") as mock_qdrant, \
         patch("app.api.v1.videos.celery_client") as mock_celery:
        
        mock_redis.smembers.return_value = set()
        mock_redis.mget.return_value = []
        
        yield {
            "redis": mock_redis,
            "qdrant": mock_qdrant,
            "celery": mock_celery
        }

@pytest.fixture
def test_client(mock_dependencies):
    # Import app inside the fixture to ensure mocks are active before module initialization
    from app.main import app
    return TestClient(app)

@pytest.fixture
def auth_headers():
    secret = os.getenv("API_SECRET_KEY", "default_secret")
    return {"Authorization": f"Bearer {secret}"}

def test_upload_success(test_client, auth_headers):
    file_content = b"fake video content"
    files = {"file": ("test.mp4", io.BytesIO(file_content), "video/mp4")}
    data = {"title": "Test Video", "description": "Desc", "tags": "tag1,tag2"}
    
    response = test_client.post("/api/v1/videos/upload", headers=auth_headers, files=files, data=data)
    assert response.status_code == 202
    assert "video_id" in response.json()

def test_upload_invalid_mime(test_client, auth_headers):
    file_content = b"fake content"
    files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
    data = {"title": "Test Video"}
    
    response = test_client.post("/api/v1/videos/upload", headers=auth_headers, files=files, data=data)
    assert response.status_code == 400
    assert "Invalid video format" in response.json()["detail"]

def test_upload_exceed_size(test_client, auth_headers, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "10")
    file_content = b"fake video content that is larger than 10 bytes"
    files = {"file": ("test.mp4", io.BytesIO(file_content), "video/mp4")}
    data = {"title": "Test Video"}
    
    response = test_client.post("/api/v1/videos/upload", headers=auth_headers, files=files, data=data)
    assert response.status_code == 413

def test_search_valid_query(test_client, auth_headers, mock_dependencies):
    mock_task = mock_dependencies["celery"].send_task.return_value
    mock_task.get.return_value = [0.1] * 1152
    
    response = test_client.get("/api/v1/videos/search?query=test", headers=auth_headers)
    assert response.status_code == 200

def test_delete_existing_video(test_client, auth_headers, mock_dependencies):
    mock_dependencies["redis"].get.return_value = '{"id":"12345678-1234-1234-1234-123456789012"}'
    video_id = "12345678-1234-1234-1234-123456789012"
    response = test_client.delete(f"/api/v1/videos/{video_id}", headers=auth_headers)
    assert response.status_code == 200

def test_delete_non_existent_video(test_client, auth_headers, mock_dependencies):
    mock_dependencies["redis"].get.return_value = None
    video_id = "12345678-1234-1234-1234-123456789012"
    response = test_client.delete(f"/api/v1/videos/{video_id}", headers=auth_headers)
    # NOTE: This will fail until Issue #13 (Pass 3) is implemented
    assert response.status_code == 404

def test_delete_invalid_uuid(test_client, auth_headers):
    response = test_client.delete("/api/v1/videos/invalid-id", headers=auth_headers)
    assert response.status_code == 400
