from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "video_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["workers.tasks"]
)

# Enterprise task processing configurations
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=1  # Recommended as 1 if running heavy ViT models locally to prevent memory thrashing
)