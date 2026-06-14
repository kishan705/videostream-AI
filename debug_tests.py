import asyncio
import json
from unittest.mock import patch, MagicMock
from tests.conftest import mock_io, fake_redis, client, sample_video_bytes, sample_video_id, auth_header, mock_qdrant

def test_auth():
    import requests
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    headers = {"Authorization": "invalid-token-format"}
    response = c.post("/api/v1/videos/upload", headers=headers, files={"file": ("test.mp4", b"\x00"*1024, "video/mp4")}, data={"title": "Test"})
    print("AUTH STATUS CODE:", response.status_code)
    print("AUTH RESPONSE:", response.json())

def test_delete():
    from app.main import app
    from fastapi.testclient import TestClient
    from app.api.v1.videos import get_vector_store
    
    mock_q = MagicMock()
    mock_q.client = MagicMock()
    mock_q.collection_name = "test_collection"
    app.dependency_overrides[get_vector_store] = lambda: mock_q
    
    c = TestClient(app)
    c.headers.update({"Authorization": "Bearer test-token"})
    
    import fakeredis
    r = fakeredis.FakeRedis(decode_responses=True)
    r.set("video:metadata:11111111-1111-1111-1111-111111111111", json.dumps({"id": "11111111-1111-1111-1111-111111111111"}))
    with patch("app.api.v1.videos.redis_client", r):
        response = c.delete("/api/v1/videos/11111111-1111-1111-1111-111111111111")
        print("DELETE STATUS CODE:", response.status_code)
        print("DELETE RESPONSE:", response.json())
        print("DELETE CALLS:", mock_q.client.delete.call_count)

def test_worker():
    import fakeredis
    r = fakeredis.FakeRedis(decode_responses=True)
    video_id = "11111111-1111-1111-1111-111111111111"
    r.set(f"video:metadata:{video_id}", json.dumps({"id": video_id, "status": "processing"}))
    
    mock_q = MagicMock()
    mock_q.upsert_embeddings.return_value = False
    
    with patch("workers.tasks.redis_client", r), \
         patch("workers.tasks.get_ffmpeg_strategy") as mock_ffmpeg, \
         patch("workers.tasks.load_ai_engine") as mock_ai, \
         patch("app.api.v1.videos.get_vector_store") as mock_get_store:
         
        mock_get_store.return_value = mock_q
        mock_ffmpeg.return_value.extract_keyframes.return_value = [{"file_path": "frame1.jpg"}]
        mock_ai.return_value.get_image_batch_embeddings.return_value = [[0.1] * 1152]
        
        from workers.tasks import process_video_pipeline
        result = process_video_pipeline(video_id, "test.mp4")
        print("WORKER RESULT:", result)
        print("WORKER UPSERT CALLED:", mock_q.upsert_embeddings.call_count)
        meta = r.get(f"video:metadata:{video_id}")
        print("WORKER META:", meta)

if __name__ == "__main__":
    print("--- AUTH ---")
    test_auth()
    print("--- DELETE ---")
    test_delete()
    print("--- WORKER ---")
    test_worker()
