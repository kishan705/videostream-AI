# videostream-AI

A production-ready Video Search Platform leveraging multimodal neural processing to find exact moments inside videos using semantic text queries.

## Architecture
- **FastAPI**: High-performance async web framework for API routing.
- **Celery**: Background task worker for heavy processing (transcoding, feature extraction).
- **Redis**: Fast key-value store for Celery broker, backend, and metadata caching.
- **Qdrant**: Vector database for high-dimensional frame embeddings.
- **SigLIP 2**: State-of-the-art vision-language model for aligning text and image features.
- **FFmpeg**: Efficient video transcoding to HLS and keyframe extraction.

## Prerequisites
- Docker and Docker Compose
- Python 3.10+
- FFmpeg installed locally (if running without Docker)

## Environment Variables
Copy the `.env.example` to `.env` and fill in the required variables.

## Quick Start
To spin up the entire stack using Docker Compose:
```bash
docker-compose up -d --build
```
This will start the FastAPI app, Redis, and Qdrant.

## API Endpoints
- `POST /api/v1/videos/upload`: Upload a video for processing.
  - Requires `Authorization: Bearer <API_SECRET_KEY>` header.
  - Form Data: `file` (video), `title`, `description`, `tags`.
- `GET /api/v1/videos/search?query=...&top_k=20`: Search inside a video.
- `GET /api/v1/videos/global-search?query=...`: Search across the platform.
- `DELETE /api/v1/videos/{video_id}`: Delete a video and all associated data.

## Frontend Usage
The frontend is a vanilla JavaScript app. Simply open `frontend/index.html` in a browser or serve it via a local static file server. Note: The API_BASE points to the FastAPI instance running on port 7860.
