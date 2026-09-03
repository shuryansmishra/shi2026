"""
SatQuery AI - Central configuration.
All settings are read from environment variables (.env file supported).
Every setting has a safe local-dev default so the backend boots with
zero configuration -- swap in real values as they become available.
"""
import os
import tempfile
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_storage_dir(subdir: str) -> str:
    """Redirect to /tmp on serverless environments (Vercel / Lambda) where ./ is read-only."""
    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        base = os.path.join(tempfile.gettempdir(), "satquery_storage")
    else:
        base = "./storage"
    return os.path.join(base, subdir)


def _resolve_db_url() -> str:
    """Redirect SQLite db to /tmp on serverless environments."""
    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return f"sqlite:///{tempfile.gettempdir()}/satquery.db"
    return "sqlite:///./satquery.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    APP_NAME: str = "SatQuery AI"
    ENV: str = "development"
    DEBUG: bool = True

    # --- Security & CORS ---
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://localhost:8000,https://*.vercel.app,https://*.onrender.com"
    ALLOWED_EXTENSIONS: set = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    # --- Storage ---
    UPLOAD_DIR: str = _resolve_storage_dir("uploads")
    PROCESSED_DIR: str = _resolve_storage_dir("processed")
    PUBLIC_STORAGE_DIR: str = _resolve_storage_dir("public")
    DEMO_DATA_DIR: str = "../demo_data"
    MAX_UPLOAD_MB: int = 200

    # --- Database (SQLite by default, zero config) ---
    DATABASE_URL: str = _resolve_db_url()

    # --- LLM synthesis layer ---
    # Runs 100% locally offline using local PyTorch/HF pipeline or structured template
    LLM_PROVIDER: str = "local"          # "local" | "anthropic" | "openai" | "none"
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"

    # --- Vision engines ---
    # VQA_MOCK_MODE=False uses real PyTorch models + rasterio + SSIM / Qwen
    # Default is True (safe zero-config start). Override via .env: VQA_MOCK_MODE=false
    VQA_MOCK_MODE: bool = True
    VQA_MODEL_PATH: Optional[str] = None       # e.g. ./checkpoints/qwen2.5-vl-lora
    CHANGE_MODEL_PATH: Optional[str] = None    # e.g. ./checkpoints/vista-cdvqa
    FUSION_MODEL_PATH: Optional[str] = None    # e.g. ./checkpoints/optical-sar-fusion

    # --- Qwen 2.5 VL Integration ---
    QWEN_MODEL_ID: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    QWEN_REMOTE_URL: Optional[str] = None      # e.g. https://xxxx.ngrok-free.app or modal/HF endpoint
    QWEN_USE_4BIT: bool = True
    QWEN_DEVICE: str = "auto"

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
