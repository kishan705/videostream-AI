import pytest
from unittest.mock import patch, MagicMock

def test_image_file_handles_closed_after_processing():
    from workers.ml_pipeline import Siglip2EmbeddingPipeline
    with patch("workers.ml_pipeline.torch"), patch("workers.ml_pipeline.AutoProcessor"), patch("workers.ml_pipeline.AutoModel"):
        pipeline = Siglip2EmbeddingPipeline()
        
    with patch("workers.ml_pipeline.Image.open") as mock_open:
        mock_img = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_img
        
        pipeline.get_image_batch_embeddings(["path1.jpg", "path2.jpg"])
        assert mock_open.return_value.__exit__.call_count == 2

def test_embedding_output_shape_is_correct():
    from workers.ml_pipeline import Siglip2EmbeddingPipeline
    with patch("workers.ml_pipeline.torch") as mock_torch, \
         patch("workers.ml_pipeline.AutoProcessor"), \
         patch("workers.ml_pipeline.AutoModel"):
        
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.no_grad = MagicMock()
        
        pipeline = Siglip2EmbeddingPipeline()
        
        mock_tensor = MagicMock()
        mock_tensor.norm.return_value = 1.0
        mock_tensor.__truediv__.return_value = mock_tensor
        mock_tensor.squeeze.return_value = mock_tensor
        mock_tensor.cpu.return_value = mock_tensor
        mock_tensor.tolist.return_value = [0.1] * 1152
        
        pipeline.model.get_text_features.return_value = mock_tensor
        res = pipeline.get_text_embedding("test")
        assert len(res) == 1152

def test_processing_continues_if_one_frame_fails_to_open():
    from workers.ml_pipeline import Siglip2EmbeddingPipeline
    with patch("workers.ml_pipeline.torch"), patch("workers.ml_pipeline.AutoProcessor"), patch("workers.ml_pipeline.AutoModel"):
        pipeline = Siglip2EmbeddingPipeline()
        
    with patch("workers.ml_pipeline.Image.open") as mock_open:
        mock_img = MagicMock()
        
        def side_effect(path):
            if path == "fail.jpg":
                raise OSError("fail")
            mock_ctx = MagicMock()
            mock_ctx.__enter__.return_value = mock_img
            return mock_ctx
            
        mock_open.side_effect = side_effect
        
        pipeline.get_image_batch_embeddings(["fail.jpg", "pass.jpg"])
