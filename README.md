# VideoStream.ai: Multimodal AI Video Search & Streaming

A full-stack, AI-powered video streaming and search platform. This system ingests raw video files, transcodes them into HLS (HTTP Live Streaming) format for seamless playback, and utilizes Google's state-of-the-art **SigLIP 2** vision-language model to enable deep semantic frame-by-frame video search.

## 🚀 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688.svg?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C.svg?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers-F9AB00.svg?logo=huggingface&logoColor=black)
![SigLIP 2](https://img.shields.io/badge/SigLIP_2-Vision_Language-8e44ad.svg)
![Celery](https://img.shields.io/badge/Celery-Task_Queue-37814A.svg?logo=celery&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-Cache_&_Broker-DC382D.svg?logo=redis&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_Database-1C1E26.svg)
![FFmpeg](https://img.shields.io/badge/FFmpeg-Video_Processing-007808.svg?logo=ffmpeg&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Infrastructure-2496ED.svg?logo=docker&logoColor=white)

## ✨ Core Features

*   **Multimodal AI Search:** Query specific moments inside videos using natural language. The system maps text queries to video frames using the `google/siglip2-so400m-patch16-256` model (1152-D vector space).
*   **Adaptive HLS Streaming:** Automatically transcodes `.mp4`, `.mkv`, and `.avi` uploads into segmented HLS (`.m3u8`) playlists for optimized, buffer-free web playback.
*   **Asynchronous Processing:** Video processing (transcoding, keyframe extraction, and ML inference) is handled in the background by Celery workers to keep the API highly responsive.
*   **Hardware Acceleration:** The ML pipeline dynamically scales to use CUDA (Nvidia GPUs), MPS (Apple Silicon), or standard CPU architectures.
*   **Persistent Cloud Metadata:** Video metadata is reliably stored and indexed using a Redis cloud database layer.

## 🧠 System Architecture

1.  **Ingestion:** User uploads a video via FastAPI. Raw file is saved, and a metadata record is created in Redis.
2.  **Orchestration:** FastAPI triggers a Celery task and immediately returns a 202 Accepted response.
3.  **Processing (Celery Worker):**
    *   **FFmpeg Strategy:** Transcodes the raw video to HLS and extracts keyframes at 1-second intervals.
    *   **ML Pipeline:** Pushes the extracted frames through the SigLIP 2 Vision Transformer to generate dense embeddings.
    *   **Vector DB:** Upserts the 1152-D frame embeddings and timestamp metadata into Qdrant.
4.  **Retrieval:** Users send text queries. The ML pipeline embeds the text, and Qdrant performs an Approximate Nearest Neighbor (ANN) search to return the exact video and timestamp.

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.10+
*   Docker & Docker Compose
*   FFmpeg installed on your host machine (`sudo apt install ffmpeg` or `brew install ffmpeg`)

### 1. Spin up the Infrastructure
Start the Redis broker and Qdrant vector database using the provided Docker Compose file:
```bash
docker-compose up -d
```

### 2. Install Dependencies
Create a virtual environment and install the required Python packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install fastapi uvicorn celery redis qdrant-client torch torchvision transformers pillow pydantic-settings ffmpeg-python
```

### 3. Start the Services
You need to run the API server and the Celery worker in separate terminal windows.

**Terminal 1: Start the FastAPI Server**
```bash
uvicorn app.main:app --reload --port 8000
```
*(Note: Adjust `app.main:app` based on your exact main file entry point).*

**Terminal 2: Start the Celery Worker**
```bash
celery -A workers.celery_app worker --loglevel=info
```

## 📡 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/videos/` | Retrieves all processed video cards from the platform. |
| `POST` | `/api/v1/videos/upload` | Ingests a new video, saves metadata, and triggers ML pipeline. |
| `GET` | `/api/v1/videos/search` | Deep neural search inside videos (returns specific timestamps). |
| `GET` | `/api/v1/videos/global-search` | Hybrid search (Text Metadata + Neural Scene Matching). |
| `DELETE` | `/api/v1/videos/{video_id}` | Cascading deletion of vectors, metadata, and physical files. |

## ⚙️ Configuration
Environment variables can be managed via a `.env` file (parsed by `pydantic-settings`). Key variables include:
*   `REDIS_URL`: Connection string for Redis.
*   `QDRANT_HOST` / `QDRANT_PORT`: Connection details for Qdrant.
*   `SIGLIP_MODEL_ID`: Hugging Face model identifier (default: `google/siglip2-so400m-patch16-256`).
