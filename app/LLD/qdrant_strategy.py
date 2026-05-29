import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.LLD.interfaces import VectorStoreInterface
from app.core.config import settings

class QdrantVectorStoreStrategy(VectorStoreInterface):
    """
    Concrete implementation of VectorStoreInterface leveraging Qdrant DB.
    Encapsulates schema enforcement, item upsert workflows, and multi-modal calculations.
    """
    def __init__(self) -> None:
        # Establish client connection targeting the docker-compose service network
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection_name = "video_frames"
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """
        Idempotent schema controller checking for collection persistence 
        and initializing vector parameters if absent.
        """
        try:
            if not self.client.collection_exists(collection_name=self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.VECTOR_DIMENSION, 
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            # Crucial LLD practice: fail fast during object construction if state is compromised
            raise RuntimeError(f"Failed initializing Qdrant collection layer: {str(e)}")

    def upsert_embeddings(self, video_id: str, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> bool:
        """
        Transforms raw embeddings into structurally validated Point payloads 
        and updates the database via high-throughput vector chunk batching.
        """
        try:
            points = []
            for idx, (vector, meta) in enumerate(zip(embeddings, metadata)):
                # Inject parent relational identifier directly into metadata payload
                payload = {
                    "video_id": video_id,
                    "timestamp_seconds": meta.get("timestamp_seconds"),
                    "file_path": meta.get("file_path")
                }
                
                # Use deterministic UUID generation based on the namespace 
                # to prevent document duplication during re-processing jobs
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_id}_{idx}"))
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )
            
            # Execute transactional batch upsert operation
            self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points
            )
            return True
            
        except Exception as e:
            # Structured application reporting block
            print(f"[ERROR] Vector storage execution fault for Video {video_id}: {str(e)}")
            return False

    def search_similarity(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Performs an approximate nearest neighbor (ANN) search inside the multi-dimensional 
        vector space matching against text embeddings.
        """
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            
            # Translate raw database structures into standard clean payloads
            formatted_results = []
            for hit in search_results:
                formatted_results.append({
                    "score": hit.score,
                    "video_id": hit.payload.get("video_id"),
                    "timestamp_seconds": hit.payload.get("timestamp_seconds"),
                    "file_path": hit.payload.get("file_path")
                })
            return formatted_results
            
        except Exception as e:
            print(f"[ERROR] Vector distance analysis failed: {str(e)}")
            return []
        