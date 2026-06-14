import pytest
import io
import json
from unittest.mock import patch

def test_video_id_added_to_set_on_upload(client, auth_header, sample_video_bytes, fake_redis):
    response = client.post("/api/v1/videos/upload", headers=auth_header, files={"file": ("test.mp4", io.BytesIO(sample_video_bytes), "video/mp4")}, data={"title": "Test"})
    video_id = response.json()["video_id"]
    assert fake_redis.sismember("video:ids", video_id) is True

def test_video_id_removed_from_set_on_delete(client, auth_header, sample_video_id, fake_redis):
    fake_redis.sadd("video:ids", sample_video_id)
    fake_redis.set(f"video:metadata:{sample_video_id}", json.dumps({"id": sample_video_id}))
    client.delete(f"/api/v1/videos/{sample_video_id}", headers=auth_header)
    assert fake_redis.sismember("video:ids", sample_video_id) is False

def test_list_videos_uses_smembers_not_keys(client, auth_header, fake_redis):
    fake_redis.keys = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("redis.keys() should not be called"))
    response = client.get("/api/v1/videos/", headers=auth_header)
    assert response.status_code == 200

def test_mget_used_for_bulk_metadata_fetch(client, auth_header, fake_redis):
    with patch.object(fake_redis, "mget", return_value=[]) as mock_mget:
        fake_redis.sadd("video:ids", "test")
        client.get("/api/v1/videos/", headers=auth_header)
        mock_mget.assert_called_once()
