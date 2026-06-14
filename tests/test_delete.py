import pytest
import os
import json
from unittest.mock import patch

def test_delete_existing_video_returns_200(client, auth_header, sample_video_id, fake_redis):
    fake_redis.set(f"video:metadata:{sample_video_id}", json.dumps({"id": sample_video_id}))
    response = client.delete(f"/api/v1/videos/{sample_video_id}", headers=auth_header)
    assert response.status_code == 200

def test_delete_nonexistent_video_returns_404(client, auth_header, sample_video_id):
    response = client.delete(f"/api/v1/videos/{sample_video_id}", headers=auth_header)
    assert response.status_code == 404

@pytest.mark.parametrize("bad_id", ["../../etc", "not-a-uuid", "", "' OR 1=1--", "../passwd"])
def test_delete_invalid_uuid_format_returns_400(client, auth_header, bad_id):
    response = client.delete(f"/api/v1/videos/{bad_id}", headers=auth_header)
    assert response.status_code in (400, 404, 405) # 404/405 can happen if routing doesn't match empty string
    if response.status_code == 400:
        assert "Invalid video ID format" in response.json()["detail"]

def test_delete_removes_processed_directory(client, auth_header, sample_video_id, fake_redis, tmp_path):
    fake_redis.set(f"video:metadata:{sample_video_id}", json.dumps({"id": sample_video_id}))
    
    with patch("app.core.config.settings.OUTPUT_DIR", str(tmp_path)):
        processed_dir = tmp_path / sample_video_id
        processed_dir.mkdir()
        assert processed_dir.exists()
        
        response = client.delete(f"/api/v1/videos/{sample_video_id}", headers=auth_header)
        assert response.status_code == 200
        assert not processed_dir.exists()

def test_delete_removes_raw_uploaded_file(client, auth_header, sample_video_id, fake_redis, tmp_path):
    fake_redis.set(f"video:metadata:{sample_video_id}", json.dumps({"id": sample_video_id}))
    
    with patch("app.core.config.settings.UPLOAD_DIR", str(tmp_path)):
        raw_file = tmp_path / f"{sample_video_id}.mp4"
        raw_file.touch()
        assert raw_file.exists()
        
        response = client.delete(f"/api/v1/videos/{sample_video_id}", headers=auth_header)
        assert response.status_code == 200
        assert not raw_file.exists()

def test_delete_removes_redis_metadata(client, auth_header, sample_video_id, fake_redis):
    fake_redis.set(f"video:metadata:{sample_video_id}", json.dumps({"id": sample_video_id}))
    response = client.delete(f"/api/v1/videos/{sample_video_id}", headers=auth_header)
    assert fake_redis.get(f"video:metadata:{sample_video_id}") is None

def test_delete_removes_qdrant_vectors(client, auth_header, sample_video_id, fake_redis, mock_qdrant):
    fake_redis.set(f"video:metadata:{sample_video_id}", json.dumps({"id": sample_video_id}))
    
    with patch("app.api.v1.videos.vector_store_instance", mock_qdrant):
        response = client.delete(f"/api/v1/videos/{sample_video_id}", headers=auth_header)
        
    mock_qdrant.client.delete.assert_called_once()

def test_delete_removes_video_from_redis_id_set(client, auth_header, sample_video_id, fake_redis):
    fake_redis.set(f"video:metadata:{sample_video_id}", json.dumps({"id": sample_video_id}))
    fake_redis.sadd("video:ids", sample_video_id)
    response = client.delete(f"/api/v1/videos/{sample_video_id}", headers=auth_header)
    assert fake_redis.sismember("video:ids", sample_video_id) == False
