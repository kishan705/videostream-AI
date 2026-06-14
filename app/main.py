import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # <-- 1. Add this critical import
from app.api.v1.videos import router as video_router
from app.core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    print("[BOOTSTRAP] System shared storage directories verified successfully.")
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="LLD-compliant production-ready Video Search Platform",
    lifespan=lifespan
)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

# Serve only the processed videos folder to prevent raw file exposure
os.makedirs("shared_storage/processed", exist_ok=True)
app.mount("/shared_storage/processed", StaticFiles(directory="shared_storage/processed"), name="shared_storage_processed")





app.include_router(video_router, prefix="/api/v1/videos", tags=["Videos"])

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME}