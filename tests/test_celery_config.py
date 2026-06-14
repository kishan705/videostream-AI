import pytest
from app.api.v1.videos import celery_client as api_celery
from workers.celery_app import celery_app as worker_celery

def test_single_celery_instance_used_in_videos_module():
    assert api_celery is worker_celery

def test_celery_app_has_tasks_registered():
    assert "workers.tasks.process_video_pipeline" in worker_celery.tasks

def test_celery_app_has_generate_text_embedding_registered():
    assert "workers.tasks.generate_text_embedding" in worker_celery.tasks
