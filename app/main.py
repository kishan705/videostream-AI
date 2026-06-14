import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # <-- 1. Add this critical import
from app.api.v1.videos import router as video_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="LLD-compliant production-ready Video Search Platform"
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

# <-- 2. Add this mount command right here! 
# This tells FastAPI to serve anything inside the physical 'shared_storage' folder over the network.
app.mount("/shared_storage", StaticFiles(directory="shared_storage"), name="shared_storage")


@app.on_event("startup")
def configure_storage_directories():
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    print("[BOOTSTRAP] System shared storage directories verified successfully.")


app.include_router(video_router, prefix="/api/v1/videos", tags=["Videos"])

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME}