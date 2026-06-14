import pytest
import io
import os
import json
from unittest.mock import patch

def test_upload_valid_mp4_returns_202(client, auth_header, sample_video_bytes):
    response = client.post("/api/v1/videos/upload", headers=auth_header, files={"file": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}, data={"title": "Test"})
    assert response.status_code == 202

def test_upload_invalid_extension_returns_400(client, auth_header, sample_video_bytes):
    response = client.post("/api/v1/videos/upload", headers=auth_header, files={"file": ("test.txt", io.BytesIO(sample_video_bytes), "text/plain")}, data={"title": "Test"})
    assert response.status_code == 400

def test_upload_invalid_mime_type_despite_mp4_extension_returns_400(client, auth_header, sample_video_bytes):
    response = client.post("/api/v1/videos/upload", headers=auth_header, files={"file": ("evil.mp4", io.BytesIO(sample_video_bytes), "application/octet-stream")}, data={"title": "Test"})
    assert response.status_code == 400

def test_upload_exceeds_size_limit_returns_413(client, auth_header, sample_video_bytes, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "10")
    response = client.post("/api/v1/videos/upload", headers=auth_header, files={"file": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}, data={"title": "Test"})
    assert response.status_code == 413

def test_upload_cleans_up_file_on_redis_failure(client, auth_header, sample_video_bytes, fake_redis):
    # Fake redis.set to raise exception
    fake_redis.set = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("Redis boom"))
    
    with patch('os.remove') as mock_remove, patch('os.path.exists', return_value=True):
        response = client.post("/api/v1/videos/upload", headers=auth_header, files={"file": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}, data={"title": "Test"})
        assert response.status_code == 500
        mock_remove.assert_called_once()

def test_upload_does_not_dispatch_celery_on_redis_failure(client, auth_header, sample_video_bytes, fake_redis, mock_celery):
    fake_redis.set = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("Redis boom"))
    with patch('os.remove'), patch('os.path.exists', return_value=True):
        client.post("/api/v1/videos/upload", headers=auth_header, files={"file": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}, data={"title": "Test"})
        mock_celery.assert_not_called()

def test_upload_metadata_contains_processing_status(client, auth_header, sample_video_bytes, fake_redis):
    response = client.post("/api/v1/videos/upload", headers=auth_header, files={"file": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}, data={"title": "Test"})
    video_id = response.json()["video_id"]
    meta = json.loads(fake_redis.get(f"video:metadata:{video_id}"))
    assert meta["status"] == "processing"

def test_upload_response_contains_video_id_as_uuid(client, auth_header, sample_video_bytes):
    response = client.post("/api/v1/videos/upload", headers=auth_header, files={"file": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}, data={"title": "Test"})
    assert "video_id" in response.json()
    import re
    assert re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", response.json()["video_id"])
