import os
import json
import shutil
import uuid
import traceback
import redis
import asyncio
import re
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Dict, Any
from qdrant_client.http import exceptions as qdrant_exceptions
from qdrant_client import models as qdrant_models
from app.core.config import settings
from app.LLD.qdrant_strategy import QdrantVectorStoreStrategy

router = APIRouter()

security = HTTPBearer()
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "default_secret")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Initialize Permanent Cloud Redis Persistence Client
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# Lazy-loaded Celery Client Instance for task injection
from workers.celery_app import celery_app as celery_client

vector_store_instance = None
def get_vector_store():
    global vector_store_instance
    if vector_store_instance is None:
        vector_store_instance = QdrantVectorStoreStrategy()
    return vector_store_instance


# --- HELPER STORAGE UTILITIES ---
def fetch_all_cloud_metadata() -> List[Dict[str, Any]]:
    """Retrieves all permanently saved metadata objects from Upstash Redis."""
    try:
        video_ids = redis_client.smembers("video:ids")
        if not video_ids:
            return []
        keys = [f"video:metadata:{vid}" for vid in video_ids]
        values = redis_client.mget(keys)
        return [json.loads(v) for v in values if v]
    except Exception as e:
        print(f"[REDIS STORAGE ERROR] Fetch failed: {str(e)}")
        return []


# --- 1. STATIC EXPLICIT GET/POST ROUTES ---

@router.get("/", response_model=List[Dict[str, Any]], dependencies=[Depends(verify_token)])
async def get_landing_page_feed():
    """Returns all permanently registered video cards from the Redis cloud database layer."""
    return fetch_all_cloud_metadata()


@router.post("/upload", status_code=202, dependencies=[Depends(verify_token)])
async def upload_video(
    file: UploadFile = File(...),
    title: str = Form(...),
    description: str = Form(""),
    tags: str = Form("")
):
    """Saves raw files locally, indexes vectors, and saves state to persistent cloud storage."""
    ALLOWED_MIME_TYPES = ["video/mp4", "video/x-matroska", "video/avi", "video/x-msvideo"]
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid video format codec structure.")
    
    video_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    saved_filename = f"{video_id}{file_extension}"
    raw_file_path = os.path.join(settings.UPLOAD_DIR, saved_filename)
    
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 500 * 1024 * 1024))
    
    try:
        with open(raw_file_path, "wb") as buffer:
            total_size = 0
            while chunk := file.file.read(8192):
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE:
                    buffer.close()
                    if os.path.exists(raw_file_path):
                        os.remove(raw_file_path)
                    raise HTTPException(status_code=413, detail="File size exceeds limit.")
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(raw_file_path):
            os.remove(raw_file_path)
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
        redis_client.sadd("video:ids", video_id)
    except Exception as e:
        print(f"[REDIS WRITE ERROR] Critical persistence failure: {str(e)}")
        if os.path.exists(raw_file_path):
            os.remove(raw_file_path)
        raise HTTPException(status_code=500, detail="Failed to save metadata to Redis.")

    celery_client.send_task(
        "workers.tasks.process_video_pipeline",
        args=[video_id, raw_file_path]
    )

    return {
        "message": "Video ingestion accepted. Saved to permanent cloud records.",
        "video_id": video_id,
        "status": "processing"
    }


@router.get("/search", dependencies=[Depends(verify_token)])
async def execute_multimodal_search(
    query: str = Query(...), 
    top_k: int = Query(default=20, ge=1, le=100),
    vector_store: QdrantVectorStoreStrategy = Depends(get_vector_store)
):
    """Granular inside-video highlight matching using modern query_points."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Search text cannot be blank.")
        
    try:
        task = celery_client.send_task("workers.tasks.generate_text_embedding", args=[query])
        text_vector = await asyncio.get_event_loop().run_in_executor(None, lambda: task.get(timeout=10)) 
        
        if not text_vector:
            raise HTTPException(status_code=502, detail="Neural embedding task failed.")

        search_results = vector_store.search_similarity(
            query_vector=text_vector,
            top_k=top_k
        )
        
        return {"results": search_results}

    except qdrant_exceptions.UnexpectedResponse as q_err:
        print(f"[QDRANT DATABASE WARN] Collection not initialized yet: {str(q_err)}")
        return {"results": []}
        
    except Exception as e:
        print("\n💥!!! CRITICAL SEARCH EXCEPTION CAUGHT !!!💥")
        traceback.print_exc() 
        print("💥!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!💥\n")
        raise HTTPException(status_code=500, detail=f"Internal vector processing fault: {str(e)}")


@router.get("/global-search", response_model=List[Dict[str, Any]], dependencies=[Depends(verify_token)])
async def execute_global_platform_search(
    query: str = Query(...), 
    top_k: int = Query(default=20, ge=1, le=100),
    vector_store: QdrantVectorStoreStrategy = Depends(get_vector_store)
):
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
        text_vector = await asyncio.get_event_loop().run_in_executor(None, lambda: task.get(timeout=5))
        
        if text_vector:
            search_results = vector_store.search_similarity(
                query_vector=text_vector,
                top_k=top_k
            )
            
            # Map out database item objects using the Redis dictionary cache
            metadata_lookup = {v.get("id"): v for v in all_metadata if v.get("id")}
            for hit in search_results:
                v_id = hit.get("video_id")
                if v_id and v_id not in discovered_video_map:
                    if v_id in metadata_lookup:
                        discovered_video_map[v_id] = metadata_lookup[v_id]
                        
    except Exception as e:
        print(f"[GLOBAL SEARCH WARN] Neural fallback active: {str(e)}")

    return list(discovered_video_map.values())


# --- 2. DYNAMIC WILDCARD ROUTES (BOTTOM ZONE) ---

@router.delete("/{video_id}", status_code=200, dependencies=[Depends(verify_token)])
async def delete_video(video_id: str, vector_store: QdrantVectorStoreStrategy = Depends(get_vector_store)):
    """Removes files, deletes vectors, and drops records from the cloud Redis instance."""
    if not re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", video_id):
        raise HTTPException(status_code=400, detail="Invalid video ID format")
        
    meta = redis_client.get(f"video:metadata:{video_id}")
    if not meta:
        raise HTTPException(status_code=404, detail="Video not found")
        
    try:
        redis_client.delete(f"video:metadata:{video_id}")
        redis_client.srem("video:ids", video_id)
    except Exception as e:
        print(f"[REDIS DELETE ERROR] Record cleanup failed: {str(e)}")
    
    processed_dir = os.path.join(settings.OUTPUT_DIR, video_id)
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
        
    for ext in ['.mp4', '.mkv', '.avi']:
        raw_file = os.path.join(settings.UPLOAD_DIR, f"{video_id}{ext}")
        if os.path.exists(raw_file):
            os.remove(raw_file)
            break
        
    try:
        vector_store.client.delete(
            collection_name=vector_store.collection_name,
            points_selector=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="video_id",
                        match=qdrant_models.MatchValue(value=video_id)
                    )
                ]
            )
        )
    except Exception as e:
        print(f"[WARNING] Vector cascade clearance mismatch: {str(e)}")

    return {"message": f"Successfully purged assets for ID: {video_id}."}