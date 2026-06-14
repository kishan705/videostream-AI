#!/bin/bash
# Start the Celery background worker process concurrently
celery -A workers.celery_app:celery_app worker --loglevel=info --concurrency=1 &

# Start the primary FastAPI gateway app on port 7860 (Hugging Face default)
uvicorn app.main:app --host 0.0.0.0 --port 7860