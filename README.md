![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)
![License MIT](https://img.shields.io/badge/license-MIT-blue)

Stop scanning hours of footage — videostream-AI finds exact moments inside your videos using multimodal neural processing and semantic text queries.

```json
// Request
GET /api/v1/videos/search?query=red+car+driving+away&top_k=1

// Response
{
  "video_id": "vid_8f92a",
  "timestamp": "00:12:45",
  "confidence_score": 0.94,
  "frame_url": "/processed/vid_8f92a/frame_1245.jpg"
}
```

## Why this exists

Video search traditionally relies on manual tagging, transcripts, or file names. I got tired of not being able to search raw visual content directly. videostream-AI embeds video frames into a high-dimensional vector space alongside text queries, letting you query visual events across terabytes of video without writing a single metadata tag. 

## Features

- Neural visual search: Query exact frames using SigLIP 2 instead of relying on transcripts.
- High-throughput architecture: Asynchronous processing via FastAPI and Celery.
- Distributed vector matching: Sub-millisecond similarity search powered by Qdrant.
- Automated media pipeline: FFmpeg extracts keyframes and transcodes to HLS automatically.
- Native UI included: Vanilla JS frontend out of the box to test semantic queries.

## Quick start

Prerequisites: Docker and Docker Compose.

```bash
git clone https://github.com/yourusername/videostream-AI.git
cd videostream-AI
cp .env.example .env
docker-compose up -d --build
```

The FastAPI backend is now running at `http://localhost:7860`. Open `frontend/index.html` in your browser to interact with the UI.

## Usage

**1. Index a video**
Upload a file to trigger frame extraction and vectorization.

```bash
curl -X POST http://localhost:7860/api/v1/videos/upload \
  -H "Authorization: Bearer sk_prod_0192837465" \
  -F "file=@/var/media/security_cam_01.mp4" \
  -F "title=Main Entrance"
```

**2. Search within a specific video**
Find the exact timestamp of an event inside the uploaded video.

```bash
curl -H "Authorization: Bearer sk_prod_0192837465" \
  "http://localhost:7860/api/v1/videos/search?video_id=vid_8f92a&query=person+walking+a+dog"
```

**3. Global semantic search**
Search across the entire platform's video database for a visual match.

```bash
curl -H "Authorization: Bearer sk_prod_0192837465" \
  "http://localhost:7860/api/v1/videos/global-search?query=fire+truck&top_k=5"
```

## Configuration

Set these in your `.env` file before booting the stack.

| Variable | Default | Description |
|---|---|---|
| `API_SECRET_KEY` | (none) | Required token for upload endpoints. |
| `SIGLIP_MODEL_ID` | `google/siglip2-so400m-patch16-256` | HuggingFace model path for embeddings. |
| `VECTOR_DIMENSION` | `1152` | Dimensionality for Qdrant index. |
| `FRAME_INTERVAL_SECONDS` | `5` | Gap between extracted keyframes. |
| `MAX_UPLOAD_SIZE` | `524288000` | Maximum video size in bytes. |

## How it works

When a video is uploaded, Celery delegates processing to FFmpeg, which extracts keyframes based on your configured interval. These frames pass through the SigLIP 2 vision-language model to generate high-dimensional embeddings that are stored in Qdrant. At search time, your text query is embedded into the same vector space, and Qdrant performs a cosine similarity search to return the precise timestamps that visually match your query.

## Contributing

Run `pip install -r requirements-dev.txt` to set up your local environment. Open a pull request with your changes, ensuring tests pass via `pytest`.

## License

MIT — do whatever you want.
