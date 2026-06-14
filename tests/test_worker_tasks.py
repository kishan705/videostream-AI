import pytest
import json
from unittest.mock import patch

def test_pipeline_updates_status_to_ready_on_success(fake_redis, sample_video_id, mock_qdrant):
    fake_redis.set(f"video:metadata:{sample_video_id}", json.dumps({"id": sample_video_id, "status": "processing"}))
    
    with patch("workers.tasks.get_ffmpeg_strategy") as mock_ffmpeg, \
         patch("workers.tasks.load_ai_engine") as mock_ai:
        
        mock_ffmpeg.return_value.extract_keyframes.return_value = [{"file_path": "frame1.jpg"}]
        mock_ai.return_value.get_image_batch_embeddings.return_value = [[0.1] * 1152]
        
        from workers.tasks import process_video_pipeline
        result = process_video_pipeline(sample_video_id, "test.mp4")
        
        assert result is True
        meta = json.loads(fake_redis.get(f"video:metadata:{sample_video_id}"))
        assert meta["status"] == "ready"

def test_pipeline_updates_status_to_failed_on_exception(fake_redis, sample_video_id, mock_qdrant):
    fake_redis.set(f"video:metadata:{sample_video_id}", json.dumps({"id": sample_video_id, "status": "processing"}))
    
    with patch("workers.tasks.get_ffmpeg_strategy") as mock_ffmpeg, \
         patch("workers.tasks.load_ai_engine") as mock_ai, \
         patch("workers.tasks.get_vector_store") as mock_get_store:
        
        mock_get_store.return_value = mock_qdrant
        mock_qdrant.upsert_embeddings.return_value = False
        mock_ffmpeg.return_value.extract_keyframes.return_value = [{"file_path": "frame1.jpg"}]
        mock_ai.return_value.get_image_batch_embeddings.return_value = [[0.1] * 1152]
        
        from workers.tasks import process_video_pipeline
        result = process_video_pipeline(sample_video_id, "test.mp4")
        
        assert result is False
        meta = json.loads(fake_redis.get(f"video:metadata:{sample_video_id}"))
        assert meta["status"] == "failed"
        assert "error" in meta
        assert "Vector DB embedding insertion failed." in meta["error"]

def test_pipeline_sets_error_message_on_failure(fake_redis, sample_video_id, mock_qdrant):
    test_pipeline_updates_status_to_failed_on_exception(fake_redis, sample_video_id, mock_qdrant)

def test_pipeline_does_not_leave_status_as_processing(fake_redis, sample_video_id, mock_qdrant):
    test_pipeline_updates_status_to_failed_on_exception(fake_redis, sample_video_id, mock_qdrant)
