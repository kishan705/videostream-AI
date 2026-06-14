"""
Shared test fixtures.

Creates a lightweight FastAPI test app that mirrors the real CORS configuration
from app.main but without pulling in Qdrant, Redis, Celery, or torch —
so the CORS security tests can run instantly on any CI machine.
"""
import os
import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import AsyncClient, ASGITransport


# ---------------------------------------------------------------------------
# Helpers — rebuild CORS exactly the way app/main.py does
# ---------------------------------------------------------------------------

def _build_app(env_origins: str | None = None) -> FastAPI:
    """
    Factory that constructs a minimal FastAPI app using the same CORS
    logic as app/main.py.  Accepts an optional override for the
    ALLOWED_ORIGINS env-var so we can test multiple configurations.
    """
    if env_origins is not None:
        os.environ["ALLOWED_ORIGINS"] = env_origins
    elif "ALLOWED_ORIGINS" not in os.environ:
        # Match the production default
        os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"

    allowed_origins = os.environ["ALLOWED_ORIGINS"].split(",")

    test_app = FastAPI()

    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # Minimal endpoint so we can exercise real HTTP requests
    @test_app.get("/")
    async def health():
        return {"status": "ok"}

    @test_app.post("/upload")
    async def upload():
        return {"status": "uploaded"}

    @test_app.delete("/item/{item_id}")
    async def delete_item(item_id: str):
        return {"deleted": item_id}

    return test_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_app():
    """App with default ALLOWED_ORIGINS (http://localhost:3000)."""
    old = os.environ.pop("ALLOWED_ORIGINS", None)
    app = _build_app()
    yield app
    # Restore env
    if old is not None:
        os.environ["ALLOWED_ORIGINS"] = old
    else:
        os.environ.pop("ALLOWED_ORIGINS", None)


@pytest.fixture()
def multi_origin_app():
    """App with multiple allowed origins set via env var."""
    old = os.environ.pop("ALLOWED_ORIGINS", None)
    app = _build_app("https://prod.example.com,https://staging.example.com")
    yield app
    if old is not None:
        os.environ["ALLOWED_ORIGINS"] = old
    else:
        os.environ.pop("ALLOWED_ORIGINS", None)


@pytest.fixture()
def single_origin_app():
    """App with a single custom production origin."""
    old = os.environ.pop("ALLOWED_ORIGINS", None)
    app = _build_app("https://myapp.example.com")
    yield app
    if old is not None:
        os.environ["ALLOWED_ORIGINS"] = old
    else:
        os.environ.pop("ALLOWED_ORIGINS", None)
