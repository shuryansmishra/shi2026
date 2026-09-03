"""
Vercel Serverless Entrypoint for SatQuery AI
---------------------------------------------
This file is the ASGI adapter Vercel's Python runtime uses to serve
the FastAPI backend as serverless functions.

It runs with VQA_MOCK_MODE=True (no GPU, no large models).
All ML inference is mocked deterministically. The backend codebase
handles missing torch/rasterio gracefully via try/except guards.

For real ML inference (local development):
  cd backend && python -m uvicorn main:app --reload --port 8000
"""
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path Setup — add backend/ to sys.path so `from main import app` resolves
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve().parent        # /project/api/
_root = _here.parent                           # /project/
_backend = _root / "backend"                   # /project/backend/

for _p in [str(_backend), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Force mock mode & serverless flag BEFORE any backend import
# ---------------------------------------------------------------------------
os.environ["VERCEL"] = "1"
os.environ["VQA_MOCK_MODE"] = "True"

# ---------------------------------------------------------------------------
# Import the FastAPI app — all heavy deps (torch, rasterio) are wrapped in
# try/except inside vision_models.py so they fail silently when absent.
# ---------------------------------------------------------------------------
try:
    from main import app  # noqa: E402
except ImportError:
    from backend.main import app  # noqa: E402

__all__ = ["app"]
