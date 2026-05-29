from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Upgraded Video Search Platform (SigLIP 2 SO400M)"
    REDIS_URL: str = "redis://localhost:6379/0"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    SIGLIP_MODEL_ID: str = "google/siglip2-so400m-patch16-256"
    VECTOR_DIMENSION: int = 1152 
    
    UPLOAD_DIR: str = "./shared_storage/uploads"
    OUTPUT_DIR: str = "./shared_storage/processed"

    class Config:
        env_file = ".env"

settings = Settings()