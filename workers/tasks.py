import os
import json
import redis
from workers.celery_app import celery_app
from workers.ml_pipeline import Siglip2EmbeddingPipeline
from app.LLD.ffmpeg_strategy import LocalFFmpegStrategy
from app.LLD.qdrant_strategy import QdrantVectorStoreStrategy
from app.core.config import settings

redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# Global placeholders for safe, lazy runtime evaluation
ffmpeg_strategy = None
vector_store = None
ai_engine_pipeline = None

def get_ffmpeg_strategy():
    global ffmpeg_strategy
    if ffmpeg_strategy is None:
        ffmpeg_strategy = LocalFFmpegStrategy()
    return ffmpeg_strategy

def get_vector_store():
    global vector_store
    if vector_store is None:
        vector_store = QdrantVectorStoreStrategy()
    return vector_store

def load_ai_engine():
    global ai_engine_pipeline
    if ai_engine_pipeline is None:
        ai_engine_pipeline = Siglip2EmbeddingPipeline()
    return ai_engine_pipeline


@celery_app.task(name="workers.tasks.process_video_pipeline")
def process_video_pipeline(video_id: str, raw_file_path: str) -> bool:
    print(f"[WORKER CHOREOGRAPHER] Commencing processing pipeline layout for ID: {video_id}")
    output_dir = os.path.join(settings.OUTPUT_DIR, video_id)
    
    # Instantiate strategies inside the execution block instead of the import layer
    ffmpeg_engine = get_ffmpeg_strategy()
    db_vector_store = get_vector_store()
    ai_model = load_ai_engine()
    
    try:
        # Step 1: HLS transcoding
        playlist_path = ffmpeg_engine.transcode_to_hls(raw_file_path, output_dir)
        print(f"[WORKER] Transcoding complete: {playlist_path}")
        
        # Step 2: Keyframe extraction
        interval = int(os.getenv("FRAME_INTERVAL_SECONDS", 5))
        frames_metadata = ffmpeg_engine.extract_keyframes(raw_file_path, output_dir, interval_seconds=interval)
        print(f"[WORKER] Extracted {len(frames_metadata)} frames.")
        
        if not frames_metadata:
            return False
            
        # Step 3: SigLIP 2 Batch Matrix Encoding
        frame_paths = [item["file_path"] for item in frames_metadata]
        batch_size = 16
        all_computed_vectors = []
        
        for i in range(0, len(frame_paths), batch_size):
            chunk_paths = frame_paths[i:i + batch_size]
            chunk_vectors = ai_model.get_image_batch_embeddings(chunk_paths)
            all_computed_vectors.extend(chunk_vectors)
            
        # Step 4: Sync to Qdrant Space
        success = db_vector_store.upsert_embeddings(
            video_id=video_id,
            embeddings=all_computed_vectors,
            metadata=frames_metadata
        )
        
        if success:
            try:
                meta_str = redis_client.get(f"video:metadata:{video_id}")
                if meta_str:
                    meta = json.loads(meta_str)
                    meta["status"] = "ready"
                    redis_client.set(f"video:metadata:{video_id}", json.dumps(meta))
            except Exception as re_err:
                print(f"[REDIS UPDATE ERROR] {str(re_err)}")
                
        return success
            
    except Exception as e:
        print(f"[WORKER CRITICAL SHUTDOWN] Ingestion routine dropped: {str(e)}")
        try:
            meta_str = redis_client.get(f"video:metadata:{video_id}")
            if meta_str:
                meta = json.loads(meta_str)
                meta["status"] = "failed"
                meta["error"] = str(e)
                redis_client.set(f"video:metadata:{video_id}", json.dumps(meta))
        except Exception as re_err:
            pass
        return False


@celery_app.task(name="workers.tasks.generate_text_embedding")
def generate_text_embedding(query_text: str) -> list[float]:
    # Dynamic instantiation on user search trigger call
    ai_model = load_ai_engine()
    return ai_model.get_text_embedding(query_text)