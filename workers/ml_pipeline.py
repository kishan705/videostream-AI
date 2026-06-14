import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from typing import List
from app.core.config import settings

class Siglip2EmbeddingPipeline:
    """
    High-performance Machine Learning inference pipeline for SigLIP 2 (1152-D).
    Optimized dynamically for both CUDA environments and Apple Silicon Neural Cores.
    """
    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"  
        else:
            self.device = "cpu"
        
        self.model_id = settings.SIGLIP_MODEL_ID
        
        print(f"[AI ENGINE] Booting SigLIP 2 SO400M on device accelerator: {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval() 
        print("[AI ENGINE] Multimodal network parameters successfully mapped and frozen.")

    def get_text_embedding(self, text: str) -> List[float]:
        """Maps raw user text queries down into the shared 1152-D spatial map."""
        with torch.no_grad():
            inputs = self.processor(text=[text], padding="max_length", return_tensors="pt").to(self.device)
            outputs = self.model.get_text_features(**inputs)
            
            # --- LLD Safe Extraction Guard ---
            # If the response is wrapped inside BaseModelOutputWithPooling, extract the core pooler tensor
            text_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            
            # Perform explicit L2 normalization to enable accurate Cosine Distance math inside Qdrant
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.squeeze(0).cpu().tolist()

    def get_image_batch_embeddings(self, image_paths: List[str]) -> List[List[float]]:
        """Extracts dense visual embeddings across structural frame lists concurrently."""
        images = []
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    images.append(img.convert("RGB").copy())
            except Exception as e:
                print(f"[AI ENGINE] Error reading frame asset {path}: {str(e)}")
                
        if not images:
            return []

        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model.get_image_features(**inputs)
            
            # --- LLD Safe Extraction Guard ---
            # Safely unpack the raw frame feature matrix tensor from the container wrapper
            image_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().tolist()