"""
SatQuery AI - Central configuration.
All settings are read from environment variables (.env file supported).
Every setting has a safe local-dev default so the backend boots with
zero configuration -- swap in real values as they become available.
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    APP_NAME: str = "SatQuery AI"
    ENV: str = "development"
    DEBUG: bool = True

    # --- Storage ---
    UPLOAD_DIR: str = "./storage/uploads"
    PROCESSED_DIR: str = "./storage/processed"
    DEMO_DATA_DIR: str = "../demo_data"
    MAX_UPLOAD_MB: int = 200

    # --- Database (SQLite by default, zero config) ---
    DATABASE_URL: str = "sqlite:///./satquery.db"

    # --- LLM synthesis layer ---
    # Runs 100% locally offline using local PyTorch/HF pipeline or structured template
    LLM_PROVIDER: str = "local"          # "local" | "anthropic" | "openai" | "none"
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"

    # --- Vision engines ---
    # VQA_MOCK_MODE=False uses real PyTorch models + rasterio + SSIM
    VQA_MOCK_MODE: bool = False
    VQA_MODEL_PATH: Optional[str] = None       # e.g. ./checkpoints/qwen2vl-rs-lora
    CHANGE_MODEL_PATH: Optional[str] = None    # e.g. ./checkpoints/vista-cdvqa
    FUSION_MODEL_PATH: Optional[str] = None    # e.g. ./checkpoints/optical-sar-fusion

    # --- Geospatial ---
    TARGET_UTM_CRS: str = "EPSG:32644"        # UTM 44N, adjust to your AOI
    TILE_SIZE: int = 512
    TILE_OVERLAP: int = 64
    CHANGE_THRESHOLD: float = 0.35
    CLOUD_COVER_SAR_SWITCH_THRESHOLD: float = 0.4  # above this, up-weight SAR branch

    # --- Bhoonidhi (ISRO data access) ---
    BHOONIDHI_USER: Optional[str] = None
    BHOONIDHI_PASSWORD: Optional[str] = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
