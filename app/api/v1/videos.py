import os
import json
import shutil
import uuid
import traceback
import redis
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from typing import List, Dict, Any
from celery import Celery
from qdrant_client.http import exceptions as qdrant_exceptions
from app.core.config import settings
from app.LLD.qdrant_strategy import QdrantVectorStoreStrategy

router = APIRouter()

# Initialize Permanent Cloud Redis Persistence Client
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# Lazy-loaded Celery Client Instance for task injection
celery_client = Celery("video_tasks", broker=settings.REDIS_URL, backend=settings.REDIS_URL)
vector_store = QdrantVectorStoreStrategy()


# --- HELPER STORAGE UTILITIES ---
def fetch_all_cloud_metadata() -> List[Dict[str, Any]]:
    """Retrieves all permanently saved metadata objects from Upstash Redis."""
    try:
        keys = redis_client.keys("video:metadata:*")
        if not keys:
            return []
        values = redis_client.mget(keys)
        return [json.loads(v) for v in values if v]
    except Exception as e:
        print(f"[REDIS STORAGE ERROR] Fetch failed: {str(e)}")
        return []


# --- 1. STATIC EXPLICIT GET/POST ROUTES ---

@router.get("/", response_model=List[Dict[str, Any]])
async def get_landing_page_feed():
    """Returns all permanently registered video cards from the Redis cloud database layer."""
    return fetch_all_cloud_metadata()


@router.post("/upload", status_code=202)
async def upload_video(
    file: UploadFile = File(...),
    title: str = Form(...),
    description: str = Form(""),
    tags: str = Form("")
):
    """Saves raw files locally, indexes vectors, and saves state to persistent cloud storage."""
    if not file.filename.endswith(('.mp4', '.mkv', '.avi')):
        raise HTTPException(status_code=400, detail="Invalid video format codec structure.")
    
    video_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    saved_filename = f"{video_id}{file_extension}"
    raw_file_path = os.path.join(settings.UPLOAD_DIR, saved_filename)
    
    try:
        with open(raw_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File system failure during ingestion: {str(e)}")

    video_metadata = {
        "id": video_id,
        "title": title,
        "description": description,
        "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
        "hls_playlist_url": f"/shared_storage/processed/{video_id}/playlist.m3u8",
        "status": "processing"
    }
    
    # Commit directly to your permanent Upstash Redis cluster
    try:
        redis_client.set(f"video:metadata:{video_id}", json.dumps(video_metadata))
    except Exception as e:
        print(f"[REDIS WRITE ERROR] Critical persistence failure: {str(e)}")

    celery_client.send_task(
        "workers.tasks.process_video_pipeline",
        args=[video_id, raw_file_path]
    )

    return {
        "message": "Video ingestion accepted. Saved to permanent cloud records.",
        "video_id": video_id,
        "status": "processing"
    }


@router.get("/search")
async def execute_multimodal_search(query: str = Query(...), top_k: int = 20):
    """Granular inside-video highlight matching using modern query_points."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Search text cannot be blank.")
        
    try:
        task = celery_client.send_task("workers.tasks.generate_text_embedding", args=[query])
        text_vector = task.get(timeout=10) 
        
        if not text_vector:
            raise HTTPException(status_code=502, detail="Neural embedding task failed.")

        response = vector_store.client.query_points(
            collection_name=vector_store.collection_name,
            query=text_vector,
            limit=top_k
        )
        
        search_results = response.points
        
        formatted_results = [
            {
                "video_id": hit.payload.get("video_id"),
                "timestamp_seconds": hit.payload.get("timestamp_seconds"),
                "score": hit.score
            }
            for hit in search_results
        ]
        
        return {"results": formatted_results}

    except qdrant_exceptions.UnexpectedResponse as q_err:
        print(f"[QDRANT DATABASE WARN] Collection not initialized yet: {str(q_err)}")
        return {"results": []}
        
    except Exception as e:
        print("\n💥!!! CRITICAL SEARCH EXCEPTION CAUGHT !!!💥")
        traceback.print_exc() 
        print("💥!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!💥\n")
        raise HTTPException(status_code=500, detail=f"Internal vector processing fault: {str(e)}")


@router.get("/global-search", response_model=List[Dict[str, Any]])
async def execute_global_platform_search(query: str = Query(...), top_k: int = 20):
    """Global hybrid search engine reading cross-references safely from cloud Redis clusters."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be blank.")
        
    query_lower = query.lower()
    discovered_video_map = {}

    # 1. Fetch live metadata array from Redis cache
    all_metadata = fetch_all_cloud_metadata()

    # 2. Relational Text Scanning
    for video in all_metadata:
        video_id = video.get("id")
        in_title = query_lower in video.get("title", "").lower()
        in_desc = query_lower in video.get("description", "").lower()
        in_tags = any(query_lower in tag.lower() for tag in video.get("tags", []))
        
        if (in_title or in_desc or in_tags) and video_id:
            discovered_video_map[video_id] = video

    # 3. Multimodal Neural Frame Scanning
    try:
        task = celery_client.send_task("workers.tasks.generate_text_embedding", args=[query])
        text_vector = task.get(timeout=5)
        
        if text_vector:
            response = vector_store.client.query_points(
                collection_name=vector_store.collection_name,
                query=text_vector,
                limit=top_k
            )
            
            # Map out database item objects using the Redis dictionary cache
            metadata_lookup = {v.get("id"): v for v in all_metadata if v.get("id")}
            for hit in response.points:
                v_id = hit.payload.get("video_id")
                if v_id and v_id not in discovered_video_map:
                    if v_id in metadata_lookup:
                        discovered_video_map[v_id] = metadata_lookup[v_id]
                        
    except Exception as e:
        print(f"[GLOBAL SEARCH WARN] Neural fallback active: {str(e)}")

    return list(discovered_video_map.values())


# --- 2. DYNAMIC WILDCARD ROUTES (BOTTOM ZONE) ---

@router.delete("/{video_id}", status_code=200)
async def delete_video(video_id: str):
    """Removes files, deletes vectors, and drops records from the cloud Redis instance."""
    try:
        redis_client.delete(f"video:metadata:{video_id}")
    except Exception as e:
        print(f"[REDIS DELETE ERROR] Record cleanup failed: {str(e)}")
    
    processed_dir = os.path.join(settings.OUTPUT_DIR, video_id)
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
        
    try:
        vector_store.client.delete(
            collection_name=vector_store.collection_name,
            points_selector=vector_store.client.models.Filter(
                must=[
                    vector_store.client.models.FieldCondition(
                        key="video_id",
                        match=vector_store.client.models.MatchValue(value=video_id)
                    )
                ]
            )
        )
    except Exception as e:
        print(f"[WARNING] Vector cascade clearance mismatch: {str(e)}")

    return {"message": f"Successfully purged assets for ID: {video_id}."}