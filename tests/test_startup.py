import pytest
from unittest.mock import patch

def test_qdrant_not_connected_at_import_time():
    with patch("app.LLD.qdrant_strategy.QdrantClient") as mock_client:
        mock_client.side_effect = Exception("Should not be called at import")
        import app.api.v1.videos

def test_get_vector_store_returns_same_instance():
    from app.api.v1.videos import get_vector_store
    instance1 = get_vector_store()
    instance2 = get_vector_store()
    assert instance1 is instance2

@pytest.mark.asyncio
async def test_lifespan_creates_upload_directory(tmp_path):
    import os
    from app.main import lifespan, app, settings
    with patch("app.core.config.settings.UPLOAD_DIR", str(tmp_path / "uploads")), \
         patch("app.core.config.settings.OUTPUT_DIR", str(tmp_path / "outputs")):
         
        async with lifespan(app):
            assert os.path.exists(str(tmp_path / "uploads"))

@pytest.mark.asyncio
async def test_lifespan_creates_output_directory(tmp_path):
    import os
    from app.main import lifespan, app, settings
    with patch("app.core.config.settings.UPLOAD_DIR", str(tmp_path / "uploads")), \
         patch("app.core.config.settings.OUTPUT_DIR", str(tmp_path / "outputs")):
         
        async with lifespan(app):
            assert os.path.exists(str(tmp_path / "outputs"))

def test_no_on_event_decorator_in_main():
    with open("app/main.py", "r") as f:
        source = f.read()
    assert "@app.on_event" not in source
