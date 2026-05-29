from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VideoProcessingStrategy(ABC):
    """
    Abstract Strategy for processing raw videos (Transcoding + Frame Extraction).
    Enforces clean LLD decoupled from specific system binaries.
    """
    @abstractmethod
    def transcode_to_hls(self, input_path: str, output_dir: str) -> str:
        """Converts raw video (.mp4) into an HLS playlist (.m3u8)"""
        pass

    @abstractmethod
    def extract_keyframes(self, input_path: str, output_dir: str, interval_seconds: int) -> List[Dict[str, Any]]:
        """Extracts periodic frame assets along with their exact timestamps"""
        pass


class VectorStoreInterface(ABC):
    """
    Abstract Storage Interface for handling high-dimensional semantic search indexing.
    Decouples application logic from specific Vector DB clients (Qdrant/Milvus).
    """
    @abstractmethod
    def upsert_embeddings(self, video_id: str, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> bool:
        """Pushes batch frame vectors down into the vector space"""
        pass

    @abstractmethod
    def search_similarity(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Executes a high-dimensional nearest-neighbor lookup matching the query text vector"""
        pass